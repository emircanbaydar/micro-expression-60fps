"""MediaPipe FaceLandmarker (VIDEO modu) + 3 noktalı affine hizalama.

Yüz, göz merkezleri ve burun ucu referans alınarak 256x256 sabit bir ROI'ye
hizalanır. Landmark tespiti, gerçek zamanlı bütçeyi korumak için genişliği
``DETECT_WIDTH`` piksele indirgenmiş bir kopya üzerinde yapılır; MediaPipe
normalize koordinat döndürdüğü için bu koordinatlar doğrudan orijinal kare
boyutuyla çarpılıp affine hizalamada kullanılır. Ardışık kareler arasında
optical flow hesaplanacağı için landmark konumları EMA ile yumuşatılır;
tespit kaybolduğunda son geçerli transform korunur.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models") / "face_landmarker.task"

ROI_SIZE = 256

# Landmark tespitinin çalıştırıldığı indirgenmiş görüntü genişliği (px).
# 1280x720 karede ~2.7x daha az piksel işlenir; normalize landmark'lar
# orijinal kare boyutuna geri ölçeklendiği için hizalama doğruluğu korunur.
DETECT_WIDTH = 480

# MediaPipe Face Mesh indeksleri
LEFT_EYE_INDICES = (33, 133)
RIGHT_EYE_INDICES = (362, 263)
NOSE_TIP_INDEX = 1
_MIN_LANDMARKS = 468

# 256x256 hedef şablon: sol göz, sağ göz, burun ucu
_TEMPLATE_POINTS = np.array(
    [
        [0.35 * ROI_SIZE, 0.42 * ROI_SIZE],
        [0.65 * ROI_SIZE, 0.42 * ROI_SIZE],
        [0.50 * ROI_SIZE, 0.55 * ROI_SIZE],
    ],
    dtype=np.float32,
)

_MAX_LOST_FRAMES = 15


@dataclass(slots=True)
class AlignResult:
    """Hizalama çıktısı.

    Attributes:
        aligned: 256x256x3 BGR hizalanmış yüz; yüz yoksa ``None``.
        landmarks: Hizalanmış ROI koordinatlarında ``(N, 2)`` landmark dizisi.
        transform: Görüntüyü hizalayan 2x3 affine matrisi.
        detected: Bu karede yüz tespit edildi mi?
    """

    aligned: np.ndarray | None
    landmarks: np.ndarray | None
    transform: np.ndarray | None
    detected: bool


class FaceAligner:
    """MediaPipe tabanlı yüz tespiti ve 256x256 affine hizalama.

    Args:
        model_path: ``face_landmarker.task`` model yolu; yoksa otomatik indirilir.
        smoothing: Landmark EMA ağırlığı (0 = yumuşatma yok, 0.9 = çok güçlü).
        min_detection_confidence: Yüz tespit eşiği.
        min_tracking_confidence: Takip eşiği.
        detect_width: Landmark tespitinin yapılacağı indirgenmiş genişlik (px).
        """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        smoothing: float = 0.6,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        detect_width: int = DETECT_WIDTH,
    ) -> None:
        if detect_width < 2:
            raise ValueError("detect_width en az 2 olmalıdır.")
        self.model_path = Path(model_path)
        self.detect_width = int(detect_width)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

        self._ensure_model()
        self._landmarker = self._create_landmarker()

        self._smoothed_src: np.ndarray | None = None
        self._last_transform: np.ndarray | None = None
        self._last_ts_ms: int = -1
        self._lost_frames: int = 0

    # ------------------------------------------------------------------ #
    # Model yönetimi
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        """Model dosyası yoksa resmi MediaPipe deposundan indirir.

        İndirme geçici bir dosyaya yapılır ve ``os.replace`` ile atomik olarak
        taşınır; yarım kalan indirme model sanılmasın diye.
        """
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.model_path.with_name(self.model_path.name + ".tmp")
        logger.info("FaceLandmarker modeli indiriliyor: %s", MODEL_URL)
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            os.replace(tmp_path, self.model_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        size_mb = self.model_path.stat().st_size / 1e6
        logger.info("Model hazır: %s (%.1f MB)", self.model_path, size_mb)

    def _create_landmarker(self) -> vision.FaceLandmarker:
        """Güncel ``mediapipe.tasks`` VIDEO API'siyle landmarker oluşturur."""
        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self._min_detection_confidence,
            min_face_presence_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ #
    # Ana API
    # ------------------------------------------------------------------ #
    def process(self, bgr: np.ndarray, timestamp_ms: int) -> AlignResult:
        """Bir BGR kareyi hizalar.

        Args:
            bgr: ``(H, W, 3)`` BGR kare.
            timestamp_ms: Monoton artan zaman damgası (ms).

        Returns:
            Hizalama sonucu. Yüz bulunamazsa ``detected=False`` ve
            ``aligned=None`` döner; ``transform`` son geçerli değeri taşır.
        """
        height, width = bgr.shape[:2]

        # Tespit küçük kopyada yapılır; warpAffine ise orijinal çözünürlükte.
        detect_bgr = self._downscale(bgr, width, height)
        rgb = cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        ts_ms = int(timestamp_ms)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            self._lost_frames += 1
            if self._lost_frames > _MAX_LOST_FRAMES:
                self._smoothed_src = None
                self._last_transform = None
            return AlignResult(
                aligned=None,
                landmarks=None,
                transform=self._last_transform,
                detected=False,
            )

        self._lost_frames = 0
        # MediaPipe koordinatları normalize (0..1) döndürür; indirgenmiş kopya
        # yerine orijinal kare boyutuyla ölçeklenerek tam çözünürlükte hizalama
        # yapılır.
        landmarks_px = np.array(
            [(lm.x * width, lm.y * height) for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
        if len(landmarks_px) < _MIN_LANDMARKS:
            return AlignResult(None, None, self._last_transform, False)

        src_points = np.array(
            [
                landmarks_px[list(LEFT_EYE_INDICES)].mean(axis=0),
                landmarks_px[list(RIGHT_EYE_INDICES)].mean(axis=0),
                landmarks_px[NOSE_TIP_INDEX],
            ],
            dtype=np.float32,
        )
        smoothed_src = self._smooth(src_points)

        transform, _ = cv2.estimateAffinePartial2D(
            smoothed_src, _TEMPLATE_POINTS, method=cv2.LMEDS
        )
        if transform is None:
            transform = cv2.getAffineTransform(smoothed_src, _TEMPLATE_POINTS)

        aligned = cv2.warpAffine(
            bgr,
            transform,
            (ROI_SIZE, ROI_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aligned_landmarks = self._apply_transform(transform, landmarks_px)

        self._last_transform = transform
        return AlignResult(
            aligned=aligned,
            landmarks=aligned_landmarks,
            transform=transform,
            detected=True,
        )

    def close(self) -> None:
        """Landmarker kaynaklarını serbest bırakır."""
        self._landmarker.close()

    # ------------------------------------------------------------------ #
    # Yardımcılar
    # ------------------------------------------------------------------ #
    def _downscale(self, bgr: np.ndarray, width: int, height: int) -> np.ndarray:
        """Tespit için kareyi ``detect_width`` genişliğine indirger.

        Kare zaten hedeften darsa kopya oluşturmadan aynen döner. En-boy
        oranı korunur; MediaPipe normalize koordinat verdiği için ölçek
        sonrası landmark'lar orijinal boyuta sorunsuz geri taşınır.
        """
        if width <= self.detect_width:
            return bgr
        scale = self.detect_width / float(width)
        detect_height = max(2, int(round(height * scale)))
        return cv2.resize(
            bgr,
            (self.detect_width, detect_height),
            interpolation=cv2.INTER_AREA,
        )

    def _smooth(self, src_points: np.ndarray) -> np.ndarray:
        """Referans noktalarını EMA ile yumuşatır (flow kararlılığı için)."""
        if self._smoothed_src is None:
            self._smoothed_src = src_points.copy()
        else:
            self._smoothed_src = (
                self.smoothing * self._smoothed_src
                + (1.0 - self.smoothing) * src_points
            ).astype(np.float32)
        return self._smoothed_src

    @staticmethod
    def _apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        """``(N, 2)`` noktaları 2x3 affine matrisiyle dönüştürür."""
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([points, ones])
        return (homogeneous @ transform.T).astype(np.float32)
