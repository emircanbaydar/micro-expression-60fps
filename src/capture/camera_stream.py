"""Thread'li, MJPG/60 FPS yapılandırılmış kamera okuyucusu.

Kamera I/O ana işlem döngüsünü bloklamamalıdır. Bu modül okumayı ayrı bir
daemon thread'de yapar ve tüketiciye her zaman **en güncel** kareyi sunar.
Kuyruk tutulmadığı için tüketici yavaşlasa bile gecikme birikmez; eski kareler
sessizce ezilir.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 30


@dataclass(slots=True)
class FramePacket:
    """Kamera thread'inden tüketiciye aktarılan tek kare.

    Attributes:
        frame: BGR kare (tüketiciye kopyalanmış).
        timestamp_ms: Akış başlangıcından itibaren monoton artan zaman (ms).
        frame_id: Thread tarafından artırılan kare sayacı.
    """

    frame: np.ndarray
    timestamp_ms: int
    frame_id: int


class CameraStream:
    """60 FPS MJPG kamera akışını ayrı thread'de okuyan sınıf.

    Args:
        src: Kamera indeksi (varsayılan ``0``) veya video dosyası yolu.
        width: İstenen kare genişliği.
        height: İstenen kare yüksekliği.
        fps: İstenen kamera FPS değeri (CLAUDE.md: 60).
        fourcc: FOURCC kodu; USB bant genişliği için ``"MJPG"``.
        backend: OpenCV video backend'i (Windows'ta ``CAP_DSHOW``).
    """

    def __init__(
        self,
        src: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        fourcc: str = "MJPG",
        backend: int = cv2.CAP_DSHOW,
    ) -> None:
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.backend = backend

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._latest_frame: np.ndarray | None = None
        self._latest_ts_ms: int = -1
        self._frame_id: int = 0
        self._start_ns: int = 0
        self._frame_times: deque[float] = deque(maxlen=180)

    # ------------------------------------------------------------------ #
    # Yaşam döngüsü
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Kamerayı yapılandırır ve okuma thread'ini başlatır.

        Raises:
            RuntimeError: Kamera açılamazsa.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._cap = self._open_capture()
        self._start_ns = time.monotonic_ns()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, name="camera-reader", daemon=True
        )
        self._thread.start()
        logger.info(
            "Kamera hazır: istenen %dx%d@%d %s | gerçek %dx%d@%.1f %s",
            self.width,
            self.height,
            self.fps,
            self.fourcc,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self._cap.get(cv2.CAP_PROP_FPS),
            self._fourcc_to_str(int(self._cap.get(cv2.CAP_PROP_FOURCC))),
        )
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if 0.0 < actual_fps < self.fps * 0.9:
            logger.warning(
                "Kamera istenen FPS'e ulaşamadı (%.1f). MJPG/FOURCC ayarını veya "
                "çözünürlüğü düşürmeyi deneyin.",
                actual_fps,
            )

    def _open_capture(self) -> cv2.VideoCapture:
        """Kamerayı MJPG/60 FPS olacak şekilde açar ve doğrular."""
        cap = cv2.VideoCapture(self.src, self.backend)
        if not cap.isOpened() and self.backend != cv2.CAP_ANY:
            logger.warning("Backend %s ile açılamadı, CAP_ANY denenecek.", self.backend)
            cap.release()
            cap = cv2.VideoCapture(self.src, cv2.CAP_ANY)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Kamera açılamadı (src={self.src!r}). Başka bir uygulama kullanıyor olabilir."
            )

        # FOURCC, çözünürlük ve FPS sırası USB bant genişliği için kritiktir.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def stop(self) -> None:
        """Okuma thread'ini durdurur ve kamerayı serbest bırakır."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        logger.info("Kamera kapatıldı.")

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Tüketici API'si
    # ------------------------------------------------------------------ #
    def read(self) -> FramePacket | None:
        """En güncel kareyi döndürür; henüz kare yoksa ``None``.

        Kare, kilit altında kopyalanır. Böylece thread aynı anda yeni kare
        yazarken yırtılmış (torn) görüntü okunmaz.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return FramePacket(
                frame=self._latest_frame.copy(),
                timestamp_ms=self._latest_ts_ms,
                frame_id=self._frame_id,
            )

    def measured_fps(self) -> float:
        """Okuma thread'inin son ~3 saniyedeki gerçek FPS'i."""
        with self._lock:
            times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        if span <= 0.0:
            return 0.0
        return (len(times) - 1) / span

    @property
    def capture_fps(self) -> float:
        """Sürücünün bildirdiği kamera FPS'i (kamera yoksa 0)."""
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap is not None else 0.0

    @property
    def is_running(self) -> bool:
        """Okuma thread'i çalışıyor mu?"""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Thread gövdesi
    # ------------------------------------------------------------------ #
    def _reader_loop(self) -> None:
        """Sürekli okuyup en güncel kareyi paylaşan thread gövdesi."""
        failures = 0
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.warning("Kamera %d kare üst üste okunamadı; yeniden açılıyor.", failures)
                    failures = 0
                    try:
                        cap.release()
                        self._cap = self._open_capture()
                    except RuntimeError:
                        logger.exception("Kamera yeniden açılamadı.")
                        self._stop_event.set()
                        break
                time.sleep(0.002)
                continue

            failures = 0
            ts_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
            now = time.perf_counter()
            with self._lock:
                if ts_ms <= self._latest_ts_ms:
                    ts_ms = self._latest_ts_ms + 1
                self._latest_frame = frame
                self._latest_ts_ms = int(ts_ms)
                self._frame_id += 1
                self._frame_times.append(now)

    @staticmethod
    def _fourcc_to_str(value: int) -> str:
        """OpenCV FOURCC tam sayısını okunabilir metne çevirir."""
        if value <= 0:
            return "?"
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip()
