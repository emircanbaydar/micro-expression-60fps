"""Farnebäck dense optical flow, optical strain ve göz kapağı maskeleme.

Girdi, ``FaceAligner`` tarafından üretilen 256x256 hizalanmış yüz ROI'sinin
gri tonlamalı halidir. Farnebäck hesabı gerçek zamanlı bütçeye sığması için
``FLOW_SIZE`` (128x128) çözünürlükte yapılır; ``u``, ``v`` ve strain çıktıları
bu boyuttadır ve doğrudan STSTNet tensör boyutuyla uyumludur. Göz kapağı
bölgesi maskelenerek göz kırpma kaynaklı yapay hareket gürültüsü uzaklaştırılır.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Optical flow / strain hesaplama çözünürlüğü (kare).
FLOW_SIZE = 128

# MediaPipe yüz konturu indeksleri (16'şar noktalık kapalı göz çevrimleri)
RIGHT_EYE_CONTOUR = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
)
LEFT_EYE_CONTOUR = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
)
_MIN_LANDMARKS = 468


@dataclass(slots=True)
class FlowResult:
    """Tek kare çifti için hareket çıktıları.

    Tüm diziler ``FLOW_SIZE x FLOW_SIZE`` (128x128) boyutundadır.

    Attributes:
        u: Yatay akış bileşeni, ``float32 (H, W)``.
        v: Dikey akış bileşeni, ``float32 (H, W)``.
        strain: Optical strain büyüklüğü, ``float32 (H, W)``.
        magnitude: ``sqrt(u^2 + v^2)``, ``float32 (H, W)``.
        eye_mask: Göz kapağı maskesi (``True`` = maskeli) veya ``None``.
    """

    u: np.ndarray
    v: np.ndarray
    strain: np.ndarray
    magnitude: np.ndarray
    eye_mask: np.ndarray | None = None


class OpticalFlowExtractor:
    """Farnebäck optical flow + strain hesaplayıcı.

    Args:
        pyr_scale: Görüntü piramidi ölçek faktörü.
        levels: Piramit seviye sayısı.
        win_size: Pencere boyutu.
        iterations: Seviye başına iterasyon sayısı.
        poly_n: Polinom komşuluğu boyutu.
        poly_sigma: Polinom yaklaşımı Gauss standart sapması.
        mask_eyes: Göz kapağı bölgesini maskele.
        eye_mask_kernel: Maske genişletme çekirdek boyutu (tek sayı).
        flow_size: Akış hesabının yapılacağı kare kenar uzunluğu (px).
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        win_size: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        mask_eyes: bool = True,
        eye_mask_kernel: int = 13,
        flow_size: int = FLOW_SIZE,
    ) -> None:
        if flow_size < 16:
            raise ValueError("flow_size en az 16 olmalıdır.")
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.win_size = win_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.mask_eyes = mask_eyes
        self.flow_size = int(flow_size)
        self._eye_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (eye_mask_kernel, eye_mask_kernel)
        )

    # ------------------------------------------------------------------ #
    # Ana API
    # ------------------------------------------------------------------ #
    def extract(
        self,
        prev_gray: np.ndarray | None,
        curr_gray: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> FlowResult:
        """İki gri kare arasındaki akışı ve strain'i hesaplar.

        Args:
            prev_gray: Önceki gri kare; ``None`` ise sıfır akış döner.
            curr_gray: Güncel gri kare (ör. hizalanmış 256x256 ROI).
            landmarks: ``curr_gray`` ile aynı koordinat uzayındaki
                ``(N, 2)`` landmark'lar; göz maskesi flow boyutuna ölçeklenir.

        Returns:
            Maske uygulanmış, ``FLOW_SIZE`` çözünürlüklü ``FlowResult``.
        """
        if prev_gray is None:
            zeros = np.zeros((self.flow_size, self.flow_size), dtype=np.float32)
            return FlowResult(
                u=zeros, v=zeros.copy(), strain=zeros.copy(), magnitude=zeros.copy()
            )

        source_width = curr_gray.shape[1]
        prev_small = self._resize(prev_gray)
        curr_small = self._resize(curr_gray)

        u, v = self._calc_farneback(prev_small, curr_small)
        eye_mask = (
            self._build_eye_mask(
                landmarks,
                (self.flow_size, self.flow_size),
                scale=self.flow_size / float(source_width),
            )
            if self.mask_eyes
            else None
        )

        if eye_mask is not None:
            # Maske sınırında yapay türev oluşmaması için akış önce sıfırlanır.
            u[eye_mask] = 0.0
            v[eye_mask] = 0.0

        strain = self._compute_strain(u, v)
        if eye_mask is not None:
            strain[eye_mask] = 0.0

        magnitude = cv2.magnitude(u, v)
        return FlowResult(
            u=u, v=v, strain=strain, magnitude=magnitude, eye_mask=eye_mask
        )

    @staticmethod
    def to_gray(bgr: np.ndarray) -> np.ndarray:
        """Hizalanmış BGR kareyi gri tona çevirir (flow girdisi)."""
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------ #
    # Hesaplama
    # ------------------------------------------------------------------ #
    def _resize(self, gray: np.ndarray) -> np.ndarray:
        """Gri kareyi flow çözünürlüğüne indirger.

        Zaten hedef boyuttaysa kopya oluşturmadan aynen döner.
        """
        if gray.shape[0] == self.flow_size and gray.shape[1] == self.flow_size:
            return gray
        return cv2.resize(
            gray,
            (self.flow_size, self.flow_size),
            interpolation=cv2.INTER_AREA,
        )

    def _calc_farneback(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Farnebäck dense optical flow (``u``, ``v``)."""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            self.pyr_scale,
            self.levels,
            self.win_size,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            0,
        )
        u = np.ascontiguousarray(flow[..., 0], dtype=np.float32)
        v = np.ascontiguousarray(flow[..., 1], dtype=np.float32)
        return u, v

    @staticmethod
    def _compute_strain(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Optical strain büyüklüğünü hesaplar.

        ``exx = du/dx``, ``eyy = dv/dy``, ``exy = 0.5 * (dv/dx + du/dy)`` ve
        büyüklük ``sqrt(exx^2 + eyy^2 + 2 * exy^2)``. OpenCV Sobel 3x3 çekirdeği
        türevin 8 katını verdiğinden sonuç 8'e bölünür.
        """
        ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3) / 8.0

        exx = ux
        eyy = vy
        exy = 0.5 * (vx + uy)
        return np.sqrt(exx * exx + eyy * eyy + 2.0 * exy * exy).astype(np.float32)

    def _build_eye_mask(
        self,
        landmarks: np.ndarray | None,
        shape: tuple[int, int],
        scale: float = 1.0,
    ) -> np.ndarray | None:
        """Göz kapağı konturlarından boolean maske üretir.

        Args:
            landmarks: Kaynak koordinat uzayındaki ``(N, 2)`` landmark'lar.
            shape: Üretilecek maskenin ``(h, w)`` boyutu.
            scale: Landmark koordinatlarını maske uzayına taşıyan ölçek.

        Landmark sayısı yetersizse (yüz yok / kısmi tespit) ``None`` döner ve
        maskeleme atlanır.
        """
        if landmarks is None or len(landmarks) < _MIN_LANDMARKS:
            return None

        mask = np.zeros(shape, dtype=np.uint8)
        scaled = landmarks.reshape(-1, 2).astype(np.float32) * float(scale)
        for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
            points = scaled[list(contour)]
            if points.shape[0] < 3:
                continue
            hull = cv2.convexHull(points.astype(np.int32).reshape(-1, 1, 2))
            cv2.fillPoly(mask, [hull], 255)

        mask = cv2.dilate(mask, self._eye_kernel, iterations=1)
        return mask.astype(bool)
