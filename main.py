"""Ana çalışma döngüsü, FPS sayacı ve OpenCV görselleştirme.

Faz 3 akışı: kamera (veya ``--video`` dosyası) → ``FaceAligner`` (256x256) →
``OpticalFlowExtractor`` (u, v, strain) → ``SpottingEngine`` (16 karelik ring
buffer + apex eşiği) → ``STSTNet`` sınıflandırması (ONNX Runtime / PyTorch) →
görselleştirme + log.

``models/ststnet.onnx`` mevcutsa ``--weights`` verilmeden otomatik yüklenir;
sınıf adları ``models/ststnet.classes.txt`` dosyasından okunur. Tahmin (sınıf
adı + güven skoru) hem olay banner'ında hem kalıcı HUD satırında gösterilir ve
loga yazılır.

STSTNet çıkarımı tek iş parçacıklı ``ThreadPoolExecutor`` üzerinde arka planda
çalışır: bir ``SpotEvent`` tetiklendiğinde ana video döngüsü tahmini beklemez,
60 FPS akmaya devam eder; tahmin tamamlandığında HUD, banner ve log güncellenir.
Önceki tahmin hâlâ sürerken gelen yeni olay için sınıflandırma atlanır (kuyruk
şişmesini önlemek amacıyla); spotting olayı yine kaydedilir.

Kullanım::

    python main.py                       # models/ststnet.onnx varsa otomatik
    python main.py --video kayit.mp4 --weights models/ststnet.pt
    python main.py --video kayit.mp4 --realtime
    python main.py --no-display --duration 10
    python main.py --render-stride 3     # görselleştirme her 3 karede bir

Gerçek zamanlı bütçe (16.6 ms/kare) için: landmark tespiti 480px kopyada,
optical flow 128x128'de, görselleştirme her ``render_stride`` karede bir
çalıştırılır.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera_stream import CameraStream, FramePacket
from src.preprocessing.face_aligner import ROI_SIZE, AlignResult, FaceAligner
from src.preprocessing.optical_flow import FlowResult, OpticalFlowExtractor
from src.pipeline.spotting_engine import SpottingConfig, SpottingEngine

logger = logging.getLogger("main")

WINDOW_NAME = "Micro-Expression 60 FPS | Faz 3"
_QUIT_KEYS = (27, ord("q"), ord("Q"))
_BANNER_SECONDS = 2.0
_DEFAULT_CLASSES = "negative,positive,surprise"

# ``--weights`` verilmezse bu sırayla aranır (ONNX Runtime tercih edilir).
_DEFAULT_WEIGHTS = (Path("models") / "ststnet.onnx", Path("models") / "ststnet.pt")
_DEFAULT_CLASS_FILE = Path("models") / "ststnet.classes.txt"


# ---------------------------------------------------------------------- #
# Video dosyası kaynağı
# ---------------------------------------------------------------------- #
class VideoFileStream:
    """Video dosyasını sıralı (thread'siz) okuyan kare kaynağı.

    Kamera akışının aksine video dosyasında kare atlanmaz; her ``read()``
    bir sonraki kareyi döndürür. Zaman damgası, dosyanın FPS değerinden
    türetilir; böylece spotting süreleri video zamanıyla tutarlıdır.

    Args:
        path: Video dosyası yolu.
        realtime: ``True`` ise kareler dosyanın doğal hızında okunur;
            ``False`` ise işlemci hızında mümkün olduğunca hızlı okunur.
    """

    def __init__(self, path: str | Path, realtime: bool = False) -> None:
        self.path = Path(path)
        self.realtime = realtime

        self._cap: cv2.VideoCapture | None = None
        self._index: int = 0
        self._fps: float = 60.0
        self._started: float = 0.0
        self._read_times: deque[float] = deque(maxlen=180)

    def start(self) -> None:
        """Videoyu açar ve metadata'sını okur.

        Raises:
            RuntimeError: Dosya açılamazsa.
        """
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Video dosyası açılamadı: {self.path}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
        self._index = 0
        self._started = time.perf_counter()
        logger.info(
            "Video hazır: %s | %dx%d @ %.1f FPS | realtime=%s",
            self.path,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self._fps,
            self.realtime,
        )

    def stop(self) -> None:
        """Video kaynağını serbest bırakır."""
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def read(self) -> FramePacket | None:
        """Bir sonraki kareyi döndürür; video bittiyse ``None``."""
        cap = self._cap
        if cap is None:
            return None

        if self.realtime:
            target = self._started + self._index / self._fps
            wait = target - time.perf_counter()
            if wait > 0.0:
                time.sleep(wait)

        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        timestamp_ms = int(round(self._index * 1000.0 / self._fps))
        self._index += 1
        self._read_times.append(time.perf_counter())
        return FramePacket(
            frame=frame, timestamp_ms=timestamp_ms, frame_id=self._index
        )

    def measured_fps(self) -> float:
        """Dosya okuma hızının son ~3 saniyedeki duvar saati FPS'i."""
        times = list(self._read_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0.0 else 0.0

    @property
    def capture_fps(self) -> float:
        """Dosyanın bildirdiği FPS değeri."""
        return self._fps


# ---------------------------------------------------------------------- #
# STSTNet sınıflandırıcı (PyTorch / ONNX Runtime)
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class Recognition:
    """Tek bir klip için sınıflandırma sonucu."""

    label: str
    confidence: float
    probabilities: np.ndarray
    latency_ms: float


class ExpressionRecognizer:
    """STSTNet ağırlıklarıyla duygu/AU sınıflandırıcısı.

    ``.pt``/``.pth`` dosyaları PyTorch ile, ``.onnx`` dosyaları ONNX Runtime
    ile yüklenir. ONNX tarafında FP16 modeller otomatik algılanır ve
    CUDA/CPU provider'larından uygun olanı kullanılır.

    Args:
        weights_path: Ağırlık dosyası.
        class_names: Çıktı indekslerine karşılık gelen sınıf adları.
        device: ``"auto"``, ``"cpu"`` veya ``"cuda"``.

    Raises:
        RuntimeError: Ağırlıklar yüklenemez veya mimariyle uyuşmazsa.
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names: list[str],
        device: str = "auto",
    ) -> None:
        if not class_names:
            raise ValueError("En az bir sınıf adı gerekir.")

        self.weights_path = Path(weights_path)
        self.class_names = list(class_names)
        self.requested_device = device
        self.backend = "?"

        self._session = None  # onnxruntime.InferenceSession
        self._input_name = ""
        self._onnx_fp16 = False
        self._torch = None
        self._model = None
        self._device = "cpu"

        if self.weights_path.suffix.lower() == ".onnx":
            self._init_onnx()
        else:
            self._init_torch()
        logger.info("Sınıflandırıcı hazır: %s | backend=%s", self.weights_path, self.backend)

    @property
    def enabled(self) -> bool:
        """Sınıflandırıcı kullanılabilir mi?"""
        return self._session is not None or self._model is not None

    def _init_torch(self) -> None:
        """PyTorch modelini ve ağırlıklarını yükler."""
        import torch

        from src.models.ststnet import build_ststnet

        if self.requested_device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = self.requested_device
            if resolved == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "--device cuda istendi ancak CUDA kullanılamıyor."
                )

        self._torch = torch
        self._device = resolved
        self._model = build_ststnet(
            self.weights_path,
            num_classes=len(self.class_names),
            device=resolved,
        )
        self.backend = f"torch:{resolved}"

    def _init_onnx(self) -> None:
        """ONNX Runtime oturumunu oluşturur."""
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        wanted = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.requested_device in ("auto", "cuda")
            else ["CPUExecutionProvider"]
        )
        providers = [provider for provider in wanted if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(self.weights_path), providers=providers
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._onnx_fp16 = "float16" in model_input.type
        self.backend = f"onnx:{providers[0]}"

    def predict(self, clip: np.ndarray) -> Recognition:
        """``(1, 3, T, H, W)`` klibi sınıflandırır.

        Args:
            clip: ``SpotEvent.to_model_input()`` çıktısı.

        Returns:
            Etiket, güven ve tüm sınıf olasılıkları.
        """
        started = time.perf_counter()

        if self._session is not None:
            tensor = clip.astype(np.float16 if self._onnx_fp16 else np.float32)
            outputs = self._session.run(None, {self._input_name: tensor})
            probabilities = _to_probabilities(np.asarray(outputs[0]).reshape(-1))
        else:
            torch = self._torch
            tensor = torch.from_numpy(clip).to(self._device)
            with torch.inference_mode():
                logits = self._model(tensor)
            probabilities = _to_probabilities(
                logits.detach().cpu().float().numpy().reshape(-1)
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        index = int(np.argmax(probabilities))
        label = (
            self.class_names[index]
            if index < len(self.class_names)
            else f"class_{index}"
        )
        return Recognition(
            label=label,
            confidence=float(probabilities[index]),
            probabilities=probabilities,
            latency_ms=latency_ms,
        )


def _to_probabilities(values: np.ndarray) -> np.ndarray:
    """Logit veya olasılık dizisini normalize edilmiş olasılığa çevirir."""
    values = values.astype(np.float32)
    if values.size == 0:
        return values
    if float(values.min()) >= 0.0 and abs(float(values.sum()) - 1.0) <= 1e-3:
        return values
    shifted = values - float(values.max())
    exponent = np.exp(shifted)
    total = float(exponent.sum())
    if total <= 0.0:
        return np.full_like(values, 1.0 / values.size)
    return (exponent / total).astype(np.float32)


def _format_probabilities(
    class_names: list[str], probabilities: np.ndarray
) -> str:
    """Sınıf olasılıklarını ``negative=0.012 positive=0.973 ...`` biçiminde yazar."""
    parts: list[str] = []
    for index, probability in enumerate(probabilities):
        name = class_names[index] if index < len(class_names) else f"class_{index}"
        parts.append(f"{name}={float(probability):.3f}")
    return " ".join(parts)


# Arka plan tahmin işinin sonucu: (olay numarası, apex_ofset_ms, tahmin).
RecognitionJob = tuple[int, int, Recognition]


def _classify_in_background(
    recognizer: ExpressionRecognizer,
    model_input: np.ndarray,
    event_index: int,
    apex_relative_ms: int,
) -> RecognitionJob:
    """STSTNet çıkarımını arka plan thread'inde çalıştırır.

    Ana video döngüsünün 16.6 ms bütçesini korumak için çağrılır. Girdi,
    çağrı öncesinde ana thread'de üretilmiş bağımsız bir kopyadır.

    Args:
        recognizer: Yüklü sınıflandırıcı (tek worker kullandığı için yarış yok).
        model_input: ``(1, 3, T, H, W)`` klip tensörü.
        event_index: Tahminin ait olduğu spot olay numarası.
        apex_relative_ms: Onset'e göre apex ofseti (ms).

    Returns:
        ``(event_index, apex_relative_ms, Recognition)`` üçlüsü.
    """
    return event_index, apex_relative_ms, recognizer.predict(model_input)


# ---------------------------------------------------------------------- #
# Ölçüm yardımcıları
# ---------------------------------------------------------------------- #
class FpsMeter:
    """Üstel hareketli ortalama ile işlem FPS'i ölçer."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value = 0.0
        self._last: float | None = None

    def tick(self) -> None:
        """Her işlenen karede çağrılır."""
        now = time.perf_counter()
        if self._last is not None:
            delta = now - self._last
            if delta > 0.0:
                instant = 1.0 / delta
                self.value = (
                    instant
                    if self.value == 0.0
                    else self.alpha * instant + (1.0 - self.alpha) * self.value
                )
        self._last = now


class LatencyStats:
    """Aşama bazlı gecikme örneklerini tutar ve yüzdelik üretir."""

    def __init__(self, maxlen: int = 240) -> None:
        self._maxlen = maxlen
        self._samples: dict[str, deque[float]] = {}

    def add(self, stage: str, milliseconds: float) -> None:
        """Bir aşamanın süresini (ms) kaydeder."""
        bucket = self._samples.get(stage)
        if bucket is None:
            bucket = deque(maxlen=self._maxlen)
            self._samples[stage] = bucket
        bucket.append(milliseconds)

    def percentile(self, stage: str, q: float) -> float:
        """Aşamanın ``q`` yüzdelik gecikmesini (ms) döndürür."""
        bucket = self._samples.get(stage)
        if not bucket:
            return 0.0
        return float(np.percentile(np.fromiter(bucket, dtype=np.float64), q))

    def summary_line(self) -> str:
        """Tüm aşamalar için ``p50/p95`` metni üretir."""
        parts: list[str] = []
        for stage in self._samples:
            p50 = self.percentile(stage, 50)
            p95 = self.percentile(stage, 95)
            parts.append(f"{stage} p50={p50:.1f}ms p95={p95:.1f}ms")
        return " | ".join(parts)


# ---------------------------------------------------------------------- #
# Görselleştirme
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class SpotOverlay:
    """Spotting bilgilerinin HUD'da gösterimi için taşıyıcı."""

    energy: float
    threshold: float
    buffer_fill: int
    buffer_size: int
    event_count: int
    event_text: str | None = None
    recognition_text: str | None = None
    active: bool = False


def _heatmap(
    values: np.ndarray, percentile: float = 99.0, size: int = ROI_SIZE
) -> np.ndarray:
    """``float32`` akış/strain dizisini JET ısı haritasına çevirir.

    Girdi (ör. 128x128 flow çıktısı) hedef karo boyutundan küçükse sonuç
    ``size x size`` boyutuna ölçeklenir.
    """
    magnitude = np.abs(values)
    if magnitude.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    high = float(np.percentile(magnitude, percentile))
    if high <= 1e-9:
        scaled = np.zeros(magnitude.shape, dtype=np.uint8)
    else:
        scaled = np.clip(magnitude / high, 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    if heat.shape[0] != size or heat.shape[1] != size:
        heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)
    return heat


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """Okunabilirlik için konturlu metin yazar."""
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )


def render(
    frame: np.ndarray,
    align_result: AlignResult | None,
    flow_result: FlowResult | None,
    fps: float,
    stats: LatencyStats,
    spot: SpotOverlay | None = None,
) -> np.ndarray:
    """2x2 görselleştirme tuvali üretir: hizalı yüz + u + v + strain."""
    tile = ROI_SIZE
    canvas = np.zeros((tile * 2, tile * 2, 3), dtype=np.uint8)

    if align_result is not None and align_result.aligned is not None:
        canvas[0:tile, 0:tile] = align_result.aligned
    else:
        canvas[0:tile, 0:tile] = cv2.resize(
            frame, (tile, tile), interpolation=cv2.INTER_AREA
        )

    if flow_result is not None:
        canvas[0:tile, tile : tile * 2] = _heatmap(flow_result.u)
        canvas[tile : tile * 2, 0:tile] = _heatmap(flow_result.v)
        canvas[tile : tile * 2, tile : tile * 2] = _heatmap(flow_result.strain)
    else:
        for row, col in ((0, 1), (1, 0), (1, 1)):
            canvas[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile] = 0

    _put_label(canvas, "Aligned 256x256", (8, 22))
    _put_label(canvas, "u (horizontal)", (tile + 8, 22))
    _put_label(canvas, "v (vertical)", (8, tile + 22))
    _put_label(canvas, "Optical strain", (tile + 8, tile + 22))

    if align_result is None or not align_result.detected:
        _put_label(canvas, "Yuz yok / Face not found", (8, tile + tile // 2))

    if spot is not None:
        spot_line = (
            f"Spot e={spot.energy:.4f} thr={spot.threshold:.4f} "
            f"buf={spot.buffer_fill}/{spot.buffer_size} evt={spot.event_count}"
        )
        _put_label(canvas, spot_line, (8, tile * 2 - 30))

        if spot.recognition_text:
            # Son STSTNet tahmini banner süresi dolsa da ekranda kalır.
            _put_label(canvas, spot.recognition_text, (tile + 8, tile + 44))

        if spot.event_text:
            cv2.rectangle(
                canvas, (0, tile * 2 - 74), (tile * 2, tile * 2 - 44), (0, 0, 0), -1
            )
            _put_label(canvas, spot.event_text, (8, tile * 2 - 54))
            cv2.rectangle(canvas, (0, 0), (tile * 2 - 1, tile * 2 - 1), (0, 0, 255), 3)
        elif spot.active:
            cv2.rectangle(canvas, (0, 0), (tile * 2 - 1, tile * 2 - 1), (0, 215, 255), 2)

    detail = stats.summary_line()
    hud = f"FPS {fps:5.1f} | {detail}" if detail else f"FPS {fps:5.1f}"
    _put_label(canvas, hud, (8, tile * 2 - 10))
    return canvas


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description=(
            "60 FPS gerçek zamanlı mikro mimik analizi — Faz 3 "
            "(spotting + STSTNet ONNX/PyTorch sınıflandırma)"
        )
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Kamera indeksi")
    parser.add_argument("--width", type=int, default=1280, help="Kamera genişliği")
    parser.add_argument("--height", type=int, default=720, help="Kamera yüksekliği")
    parser.add_argument("--fps", type=int, default=60, help="Kamera FPS hedefi")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Kamera yerine kullanılacak video dosyası",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Video dosyasını doğal hızında oynat (varsayılan: hızlı işle)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "STSTNet ağırlıkları (.pt/.pth/.onnx); verilmezse "
            "models/ststnet.onnx, yoksa models/ststnet.pt otomatik aranır"
        ),
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help=(
            "Virgülle ayrılmış sınıf adları (model çıkış sırasıyla). "
            "Verilmezse <ağırlık>.classes.txt veya models/ststnet.classes.txt, "
            f"o da yoksa varsayılan: {_DEFAULT_CLASSES}"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="STSTNet çıkarım cihazı",
    )
    parser.add_argument(
        "--no-recognition",
        action="store_true",
        help="STSTNet sınıflandırmasını kapat",
    )
    parser.add_argument(
        "--no-spotting",
        action="store_true",
        help="Spotting motorunu kapat (sadece flow görselleştirmesi)",
    )
    parser.add_argument(
        "--spot-threshold",
        type=float,
        default=None,
        help="Sabit apex enerji eşiği (varsayılan: uyarlamalı medyan+MAD)",
    )
    parser.add_argument("--no-display", action="store_true", help="Görselleştirmeyi kapat")
    parser.add_argument(
        "--render-stride",
        type=int,
        default=3,
        help=(
            "Görselleştirme/imshow her kaç karede bir çalışsın "
            "(1 = her kare, 3 = varsayılan)"
        ),
    )
    parser.add_argument("--mirror", action="store_true", help="Görüntüyü aynala")
    parser.add_argument(
        "--duration", type=float, default=0.0, help="Saniye cinsinden çalışma süresi (0=sonsuz)"
    )
    parser.add_argument(
        "--no-eye-mask", action="store_true", help="Göz kapağı maskelemeyi kapat"
    )
    parser.add_argument(
        "--cv-threads", type=int, default=2, help="OpenCV thread sayısı (latency stabilitesi)"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------- #
# Bileşen kurulumu
# ---------------------------------------------------------------------- #
def _open_source(args: argparse.Namespace) -> CameraStream | VideoFileStream:
    """Video dosyası veya kamera kaynağını açar.

    Kamera açılışında varsayılan backend (``CAP_ANY``) başarısız olursa
    Windows'ta otomatik olarak ``cv2.CAP_DSHOW`` ile yeniden denenir. Bu,
    bazı USB kamera sürücülerinde varsayılan MSMF backend'inin takılması
    veya kamerayı hiç açamaması durumunu kurtarır.

    Raises:
        RuntimeError: Kaynak açılamazsa.
    """
    if args.video:
        source = VideoFileStream(args.video, realtime=args.realtime)
        source.start()
        return source

    backends = [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        backends.append(cv2.CAP_DSHOW)

    last_error: Exception | None = None
    for attempt, backend in enumerate(backends):
        camera = CameraStream(
            src=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=backend,
        )
        try:
            camera.start()
            if attempt > 0:
                logger.info(
                    "Kamera %s backend'i ile açıldı (fallback).",
                    "CAP_DSHOW" if backend == cv2.CAP_DSHOW else backend,
                )
            return camera
        except RuntimeError as exc:
            last_error = exc
            camera.stop()
            if attempt + 1 < len(backends):
                logger.warning(
                    "Kamera varsayılan backend ile açılamadı (%s); "
                    "cv2.CAP_DSHOW ile yeniden denenecek.",
                    exc,
                )

    logger.error("%s", last_error)
    logger.error(
        "Kamera kullanılamıyor. Video dosyasıyla çalıştırmak için: --video <yol>"
    )
    raise RuntimeError(str(last_error))


def _build_spotter(args: argparse.Namespace) -> SpottingEngine | None:
    """CLI ayarlarına göre spotting motorunu oluşturur."""
    if args.no_spotting:
        logger.info("Spotting devre dışı (--no-spotting).")
        return None
    config = SpottingConfig(fixed_threshold=args.spot_threshold)
    engine = SpottingEngine(config)
    logger.info(
        "Spotting hazır: buffer=%d kare | eşik=%s",
        config.buffer_size,
        f"sabit {config.fixed_threshold:.4f}"
        if config.fixed_threshold is not None
        else "uyarlamalı (medyan + 4*MAD)",
    )
    return engine


def _resolve_weights(args: argparse.Namespace) -> Path | None:
    """``--weights`` verilmediyse varsayılan model dosyalarını arar.

    ``models/ststnet.onnx`` (ONNX Runtime) ``models/ststnet.pt``'ye tercih
    edilir. Hiçbiri yoksa ``None`` döner.
    """
    if args.weights:
        return Path(args.weights)
    for candidate in _DEFAULT_WEIGHTS:
        if candidate.exists():
            logger.info("Varsayılan STSTNet ağırlığı bulundu: %s", candidate)
            return candidate
    return None


def _load_class_names(weights: Path, explicit: str | None) -> list[str]:
    """Sınıf adlarını CLI → sidecar dosyası → varsayılan sırasıyla çözer."""
    if explicit:
        names = [name.strip() for name in explicit.split(",") if name.strip()]
        if names:
            return names

    sidecar_candidates = (
        weights.with_suffix(".classes.txt"),
        _DEFAULT_CLASS_FILE,
    )
    for candidate in sidecar_candidates:
        if not candidate.is_file():
            continue
        names = [
            line.strip()
            for line in candidate.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if names:
            logger.info(
                "Sınıf adları okundu: %s (%s)", ", ".join(names), candidate
            )
            return names

    return [name.strip() for name in _DEFAULT_CLASSES.split(",") if name.strip()]


def _build_recognizer(args: argparse.Namespace) -> ExpressionRecognizer | None:
    """Varsa STSTNet ağırlıklarını yükleyip sınıflandırıcıyı oluşturur.

    ``models/ststnet.onnx`` varsa ``--weights`` gerekmeden ONNX Runtime ile
    gerçek tahmin üretilir. Model dosyası yoksa yalnızca spotting çalışır.
    """
    if args.no_recognition:
        logger.info("Sınıflandırma devre dışı (--no-recognition).")
        return None

    weights = _resolve_weights(args)
    if weights is None:
        logger.warning(
            "STSTNet ağırlığı bulunamadı (models/ststnet.onnx veya "
            "models/ststnet.pt). Eğitmek için: python train.py --dummy. "
            "Yalnızca spotting aktif."
        )
        return None
    if not weights.exists():
        logger.warning(
            "Ağırlık dosyası bulunamadı: %s; yalnızca spotting aktif.", weights
        )
        return None

    class_names = _load_class_names(weights, args.classes)
    try:
        recognizer = ExpressionRecognizer(weights, class_names, device=args.device)
    except Exception:
        logger.exception(
            "Sınıflandırıcı yüklenemedi; yalnızca spotting ile devam ediliyor."
        )
        return None

    logger.info(
        "STSTNet sınıflandırıcı: %s | backend=%s | sınıflar: %s",
        weights,
        recognizer.backend,
        ", ".join(class_names),
    )
    return recognizer


# ---------------------------------------------------------------------- #
# Ana döngü
# ---------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    """Kaynak → hizalama → flow → spotting → (arka plan STSTNet) → görselleştirme.

    STSTNet tahmini ayrı bir thread'de çalıştığı için ana döngü hiçbir zaman
    çıkarım süresi (ör. ~43 ms) kadar beklemez.
    """
    cv2.setNumThreads(max(1, args.cv_threads))
    render_stride = max(1, int(args.render_stride))

    source = _open_source(args)
    aligner: FaceAligner | None = None
    spotter: SpottingEngine | None = None
    recognizer: ExpressionRecognizer | None = None
    executor: ThreadPoolExecutor | None = None
    pending: Future[RecognitionJob] | None = None

    processed = 0
    event_count = 0
    stats = LatencyStats()
    started = time.perf_counter()
    last_recognition: Recognition | None = None

    try:
        spotter = _build_spotter(args)
        recognizer = _build_recognizer(args)
        if recognizer is not None:
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ststnet"
            )
            logger.info(
                "STSTNet çıkarımı arka planda çalışacak "
                "(ThreadPoolExecutor, max_workers=1); ana döngü beklemez."
            )
        aligner = FaceAligner()
        flow_extractor = OpticalFlowExtractor(mask_eyes=not args.no_eye_mask)

        prev_gray: np.ndarray | None = None
        fps_meter = FpsMeter()
        last_frame_id = -1
        banner_text: str | None = None
        banner_until = 0.0
        last_report = time.perf_counter()

        while True:
            now = time.perf_counter()
            if args.duration > 0 and now - started >= args.duration:
                break

            packet = source.read()
            if packet is None:
                if isinstance(source, VideoFileStream):
                    logger.info("Video dosyası bitti.")
                    break
                time.sleep(0.001)
                continue
            if packet.frame_id == last_frame_id:
                # Kamera thread'i yeni kare üretmediyse aynı kare tekrar işlenmez.
                time.sleep(0.001)
                continue
            last_frame_id = packet.frame_id

            loop_start = time.perf_counter()
            frame = cv2.flip(packet.frame, 1) if args.mirror else packet.frame

            align_start = time.perf_counter()
            align_result = aligner.process(frame, packet.timestamp_ms)
            align_end = time.perf_counter()

            flow_result: FlowResult | None = None
            flow_start = time.perf_counter()
            if align_result.detected and align_result.aligned is not None:
                gray = OpticalFlowExtractor.to_gray(align_result.aligned)
                flow_result = flow_extractor.extract(prev_gray, gray, align_result.landmarks)
                prev_gray = gray
            else:
                # Yüz kaybolduğunda akış referansı sıfırlanır; yanlış eşleşme olmaz.
                prev_gray = None
            flow_end = time.perf_counter()

            spot_start = time.perf_counter()
            spot_event = (
                spotter.update(flow_result, packet.timestamp_ms)
                if spotter is not None
                else None
            )
            spot_end = time.perf_counter()

            # Arka plan tahmini bittiyse sonucu burada (ana thread'de) işle:
            # banner, HUD ve log güncellenir; video döngüsü hiç bekletilmedi.
            # Yeni olaydan önce toplanır ki tamamlanmış bir Future kaybolmasın.
            if pending is not None and pending.done():
                try:
                    done_index, apex_relative, recognition = pending.result()
                except Exception:
                    logger.exception(
                        "STSTNet arka plan çıkarımı başarısız; "
                        "sınıflandırma kapatılıyor."
                    )
                    recognizer = None
                else:
                    last_recognition = recognition
                    stats.add("ststnet", recognition.latency_ms)
                    probability_text = _format_probabilities(
                        recognizer.class_names if recognizer is not None else [],
                        recognition.probabilities,
                    )
                    banner_text = (
                        f"#{done_index} MIKRO MIMIK | {recognition.label} "
                        f"({recognition.confidence:.2f}) | apex +{apex_relative} ms"
                    )
                    banner_until = time.perf_counter() + _BANNER_SECONDS
                    logger.info(
                        "Spot #%d tahmin tamamlandı | %s guven=%.2f | "
                        "%.1f ms (arka plan) | olasılıklar: %s",
                        done_index,
                        recognition.label,
                        recognition.confidence,
                        recognition.latency_ms,
                        probability_text,
                    )
                pending = None

            if spot_event is not None:
                event_count += 1
                apex_relative = max(0, spot_event.apex_ms - spot_event.onset_ms)
                banner_text = (
                    f"#{event_count} MIKRO MIMIK | apex +{apex_relative} ms | "
                    f"enerji {spot_event.peak_energy:.4f}"
                )
                banner_until = time.perf_counter() + _BANNER_SECONDS

                if recognizer is not None and executor is not None:
                    if pending is not None and not pending.done():
                        # Önceki tahmin sürerken yeni iş kuyruğa eklenirse
                        # çıkarım birikintisi oluşur; bu olay yalnızca loglanır.
                        prediction_note = "önceki tahmin sürüyor, atlandı"
                        logger.warning(
                            "STSTNet önceki tahmin hâlâ sürüyor; #%d için "
                            "sınıflandırma atlandı (ana döngü bloke edilmedi).",
                            event_count,
                        )
                    else:
                        pending = executor.submit(
                            _classify_in_background,
                            recognizer,
                            spot_event.to_model_input(),
                            event_count,
                            apex_relative,
                        )
                        prediction_note = "arka planda"
                else:
                    prediction_note = "tahmin yok"

                logger.info(
                    "Spot #%d | onset=%d ms apex=%d ms offset=%d ms | "
                    "peak=%.5f thr=%.5f | tahmin: %s",
                    event_count,
                    spot_event.onset_ms,
                    spot_event.apex_ms,
                    spot_event.offset_ms,
                    spot_event.peak_energy,
                    spot_event.threshold,
                    prediction_note,
                )

            render_start = time.perf_counter()
            # Görselleştirme her `render_stride` karede bir yapılır; analiz
            # döngüsü imshow/waitKey maliyetini taşımadan akar.
            if not args.no_display and processed % render_stride == 0:
                overlay: SpotOverlay | None = None
                if spotter is not None:
                    overlay = SpotOverlay(
                        energy=spotter.last_energy,
                        threshold=spotter.current_threshold,
                        buffer_fill=spotter.buffer_fill,
                        buffer_size=spotter.config.buffer_size,
                        event_count=event_count,
                        event_text=(
                            banner_text
                            if time.perf_counter() < banner_until
                            else None
                        ),
                        recognition_text=(
                            f"STSTNet: {last_recognition.label} "
                            f"({last_recognition.confidence:.2f}) "
                            f"{last_recognition.latency_ms:.1f}ms"
                            if last_recognition is not None
                            else None
                        ),
                        active=spotter.is_active,
                    )
                canvas = render(
                    frame, align_result, flow_result, fps_meter.value, stats, overlay
                )
                cv2.imshow(WINDOW_NAME, canvas)
                if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                    break
            render_end = time.perf_counter()

            fps_meter.tick()
            stats.add("align", (align_end - align_start) * 1000.0)
            stats.add("flow", (flow_end - flow_start) * 1000.0)
            stats.add("spot", (spot_end - spot_start) * 1000.0)
            stats.add("render", (render_end - render_start) * 1000.0)
            stats.add("total", (render_end - loop_start) * 1000.0)
            processed += 1

            now = time.perf_counter()
            if now - last_report >= 1.0:
                logger.info(
                    "FPS %.1f | kaynak %.1f | %s",
                    fps_meter.value,
                    source.measured_fps(),
                    stats.summary_line(),
                )
                last_report = now
    finally:
        if executor is not None:
            # Devam eden arka plan tahmini varsa tamamlanmasını bekle; kaynaklar
            # serbest bırakılmadan önce thread temiz kapanır.
            executor.shutdown(wait=True)
        source.stop()
        if aligner is not None:
            aligner.close()
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-9)
    logger.info(
        "Toplam %d kare, %.1f s, ortalama %.1f FPS, %d spot",
        processed,
        elapsed,
        processed / elapsed,
        event_count,
    )
    if stats.summary_line():
        logger.info("Gecikme özeti: %s", stats.summary_line())


def main(argv: list[str] | None = None) -> int:
    """Program giriş noktası."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        run(args)
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")
    except Exception:
        logger.exception("Beklenmeyen hata")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
