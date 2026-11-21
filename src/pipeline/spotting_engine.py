"""16 karelik ring buffer ve apex eşik tetikleyicisi.

Mikro mimik spotting akışı her karede ``OpticalFlowExtractor`` çıktısını
(``u``, ``v``, ``strain``) alır. Kare başına hareket enerjisi (ortalama mutlak
strain) uyarlamalı bir eşikle karşılaştırılır:

1. Enerji eşiği aştığında bir **aday** açılır (onset).
2. Aday boyunca en yüksek enerji ve buna karşılık gelen **apex** karesi izlenir.
3. Enerji, tepe değerin ``release_ratio`` katının altına ``confirm_frames``
   kare boyunca düşerse (veya aday ``max_duration_ms`` süresini aşarsa) olay
   sonlandırılır ve 16 karelik ring buffer'dan klip üretilir.

Eşik varsayılan olarak son karelerin medyan + ``baseline_k * 1.4826 * MAD``
değerinden türetilir; böylece kamera/kimlik bazlı parlaklık ve gürültü farkları
elle ayar gerektirmez. ``fixed_threshold`` verilirse sabit eşik kullanılır.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.preprocessing.optical_flow import FlowResult

logger = logging.getLogger(__name__)

_MAD_TO_STD = 1.4826


@dataclass(slots=True)
class SpottingConfig:
    """Spotting davranışını belirleyen parametreler.

    Args:
        buffer_size: Ring buffer uzunluğu (kare). STSTNet zaman boyutu.
        min_energy: Uyarlamalı eşiğin alt sınırı; gürültü tabanını engeller.
        baseline_k: Medyan + k * sağlam standart sapma eşiği.
        baseline_window: Baz çizgisi için tutulan kare sayısı (~2 sn @ 60 FPS).
        baseline_min_samples: Baz çizgisi için gereken minimum kare sayısı.
        release_ratio: Apex sonrası "bitti" sayılma oranı (tepe * oran).
        confirm_frames: Serbest bırakmanın onaylanması için gereken kare sayısı.
        min_duration_ms: Geçerli olay için minimum onset→offset süresi.
        max_duration_ms: Adayın zorla kapatılacağı maksimum süre.
        cooldown_ms: Olay sonrası yeni aday açılmadan beklenecek süre.
        fixed_threshold: Verilirse uyarlamalı eşik yerine bu değer kullanılır.
    """

    buffer_size: int = 16
    min_energy: float = 0.005
    baseline_k: float = 4.0
    baseline_window: int = 120
    baseline_min_samples: int = 20
    release_ratio: float = 0.5
    confirm_frames: int = 2
    min_duration_ms: int = 40
    max_duration_ms: int = 250
    cooldown_ms: int = 250
    fixed_threshold: float | None = None

    def __post_init__(self) -> None:
        """Parametreleri doğrular."""
        if self.buffer_size < 4:
            raise ValueError("buffer_size en az 4 olmalıdır.")
        if self.baseline_window < self.baseline_min_samples:
            raise ValueError("baseline_window, baseline_min_samples'tan küçük olamaz.")
        if not 0.0 < self.release_ratio < 1.0:
            raise ValueError("release_ratio (0, 1) aralığında olmalıdır.")
        if self.min_energy < 0.0:
            raise ValueError("min_energy negatif olamaz.")
        if self.baseline_k <= 0.0:
            raise ValueError("baseline_k pozitif olmalıdır.")
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames en az 1 olmalıdır.")
        if self.min_duration_ms < 0 or self.min_duration_ms >= self.max_duration_ms:
            raise ValueError("0 <= min_duration_ms < max_duration_ms olmalıdır.")
        if self.cooldown_ms < 0:
            raise ValueError("cooldown_ms negatif olamaz.")
        if self.fixed_threshold is not None and self.fixed_threshold <= 0.0:
            raise ValueError("fixed_threshold pozitif olmalıdır.")


@dataclass(slots=True)
class _BufferedFrame:
    """Ring buffer'da tutulan tek karenin hareket verisi."""

    timestamp_ms: int
    energy: float
    u: np.ndarray
    v: np.ndarray
    strain: np.ndarray


@dataclass(slots=True)
class SpotEvent:
    """Tespit edilen bir mikro mimik klibi.

    Attributes:
        u: ``(T, H, W)`` yatay akış dizisi.
        v: ``(T, H, W)`` dikey akış dizisi.
        strain: ``(T, H, W)`` optical strain dizisi.
        timestamps_ms: Kare zaman damgaları (``buffer_size`` uzunluğunda).
        onset_ms: Enerjinin eşiği aştığı zaman.
        apex_ms: En yüksek enerjili karenin zamanı.
        offset_ms: Olayın sonlandığı (serbest bırakma) zaman.
        apex_index: Klip içindeki apex kare indeksi.
        peak_energy: Klip içindeki en yüksek hareket enerjisi.
        threshold: Olayı tetikleyen eşik değeri.
    """

    u: np.ndarray
    v: np.ndarray
    strain: np.ndarray
    timestamps_ms: tuple[int, ...]
    onset_ms: int
    apex_ms: int
    offset_ms: int
    apex_index: int
    peak_energy: float
    threshold: float

    @property
    def num_frames(self) -> int:
        """Klipteki kare sayısı."""
        return int(self.u.shape[0])

    @property
    def duration_ms(self) -> int:
        """Onset→offset süresi (ms)."""
        return int(self.offset_ms - self.onset_ms)

    def to_model_input(
        self, *, standardize: bool = True, eps: float = 1e-5
    ) -> np.ndarray:
        """Klibi STSTNet girdisine dönüştürür.

        Args:
            standardize: Her kanalı kendi ortalama/standart sapmasıyla
                standartlaştır (flow ölçeği kameraya göre değişebilir).
            eps: Sıfıra bölmeyi engelleyen küçük sabit.

        Returns:
            ``(1, 3, T, H, W)`` ``float32`` tensör; kanal sırası
            ``u``, ``v``, ``strain``.
        """
        clip = np.stack((self.u, self.v, self.strain), axis=0).astype(np.float32)
        if standardize:
            mean = clip.mean(axis=(1, 2, 3), keepdims=True)
            std = clip.std(axis=(1, 2, 3), keepdims=True)
            clip = (clip - mean) / np.maximum(std, eps)
        return clip[np.newaxis, ...]


class SpottingEngine:
    """Ring buffer tabanlı, apex eşiğiyle tetiklenen mikro mimik tespiti.

    Args:
        config: Spotting parametreleri; ``None`` ise varsayılanlar kullanılır.
    """

    def __init__(self, config: SpottingConfig | None = None) -> None:
        self.config = config or SpottingConfig()
        cfg = self.config

        self._buffer: deque[_BufferedFrame] = deque(maxlen=cfg.buffer_size)
        self._energies: deque[float] = deque(maxlen=cfg.baseline_window)

        self._shape: tuple[int, int] | None = None
        self._blank: FlowResult | None = None
        self._last_ts: int = -1
        self._last_energy: float = 0.0
        self._last_threshold: float = (
            cfg.fixed_threshold if cfg.fixed_threshold is not None else cfg.min_energy
        )

        self._active: bool = False
        self._onset_ts: int = 0
        self._peak_energy: float = 0.0
        self._below_count: int = 0
        self._cooldown_until_ms: int = -(10**12)

    # ------------------------------------------------------------------ #
    # Durum sorguları (HUD / teşhis için)
    # ------------------------------------------------------------------ #
    @property
    def buffer_fill(self) -> int:
        """Ring buffer'daki mevcut kare sayısı."""
        return len(self._buffer)

    @property
    def is_active(self) -> bool:
        """Aktif bir aday olay var mı?"""
        return self._active

    @property
    def last_energy(self) -> float:
        """Son işlenen karenin hareket enerjisi."""
        return self._last_energy

    @property
    def current_threshold(self) -> float:
        """Son karede kullanılan eşik değeri."""
        return self._last_threshold

    # ------------------------------------------------------------------ #
    # Ana API
    # ------------------------------------------------------------------ #
    def update(self, flow: FlowResult | None, timestamp_ms: int) -> SpotEvent | None:
        """Yeni bir kare işler ve olay tamamlanmışsa döndürür.

        Args:
            flow: ``OpticalFlowExtractor`` çıktısı. Yüz yoksa ``None``
                geçilebilir; bu durumda sıfır hareketli kare eklenir ve aktif
                aday iptal edilir.
            timestamp_ms: Monoton artan zaman damgası (ms).

        Returns:
            Olay serbest bırakma anında tamamlandıysa :class:`SpotEvent`,
            aksi halde ``None``.
        """
        ts = int(timestamp_ms)
        if ts <= self._last_ts:
            # Aynı zaman damgası iki kez gelirse süre hesabı bozulmasın.
            ts = self._last_ts + 1
        self._last_ts = ts

        missing = flow is None
        if missing:
            flow = self._blank_flow()
            if flow is None:
                # Henüz geçerli bir kare boyutu öğrenilmedi.
                self._active = False
                self._below_count = 0
                return None
        else:
            self._shape = flow.strain.shape[:2]

        energy = float(np.mean(np.abs(flow.strain), dtype=np.float64))
        self._last_energy = energy
        self._buffer.append(
            _BufferedFrame(
                timestamp_ms=ts,
                energy=energy,
                u=np.array(flow.u, dtype=np.float32, copy=True),
                v=np.array(flow.v, dtype=np.float32, copy=True),
                strain=np.array(flow.strain, dtype=np.float32, copy=True),
            )
        )
        self._energies.append(energy)

        if missing:
            # Yüz kaybolduğunda yarım kalan aday güvenilmez; iptal edilir.
            self._active = False
            self._below_count = 0
            return None

        threshold = self._compute_threshold()
        self._last_threshold = threshold

        if not self._active:
            if ts < self._cooldown_until_ms:
                return None
            if energy >= threshold:
                self._active = True
                self._onset_ts = ts
                self._peak_energy = energy
                self._below_count = 0
                logger.debug(
                    "Spot adayı: ts=%d enerji=%.5f eşik=%.5f", ts, energy, threshold
                )
            return None

        if energy > self._peak_energy:
            self._peak_energy = energy
        if energy < self._peak_energy * self.config.release_ratio:
            self._below_count += 1
        else:
            self._below_count = 0

        duration_ms = ts - self._onset_ts
        released = self._below_count >= self.config.confirm_frames
        overdue = duration_ms >= self.config.max_duration_ms
        if not (released or overdue):
            return None

        event = self._finalize(ts, threshold)
        self._active = False
        self._below_count = 0
        self._cooldown_until_ms = ts + self.config.cooldown_ms
        return event

    def reset(self) -> None:
        """Tüm durumu (buffer, baz çizgisi, aday) sıfırlar."""
        self._buffer.clear()
        self._energies.clear()
        self._active = False
        self._below_count = 0
        self._peak_energy = 0.0
        self._last_energy = 0.0
        self._cooldown_until_ms = -(10**12)
        self._last_ts = -1
        self._shape = None
        self._blank = None

    # ------------------------------------------------------------------ #
    # Yardımcılar
    # ------------------------------------------------------------------ #
    def _blank_flow(self) -> FlowResult | None:
        """Yüz yokken kullanılacak sıfır hareketli kare (önbellekli)."""
        if self._shape is None:
            return None
        if self._blank is None or self._blank.strain.shape != self._shape:
            zeros = np.zeros(self._shape, dtype=np.float32)
            self._blank = FlowResult(
                u=zeros, v=zeros.copy(), strain=zeros.copy(), magnitude=zeros.copy()
            )
        return self._blank

    def _compute_threshold(self) -> float:
        """Uyarlamalı (medyan + k*MAD) veya sabit eşiği döndürür."""
        cfg = self.config
        if cfg.fixed_threshold is not None:
            return cfg.fixed_threshold
        if len(self._energies) < cfg.baseline_min_samples:
            return cfg.min_energy

        values = np.fromiter(self._energies, dtype=np.float64, count=len(self._energies))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_std = _MAD_TO_STD * mad
        return max(cfg.min_energy, median + cfg.baseline_k * robust_std)

    def _finalize(self, ts: int, threshold: float) -> SpotEvent | None:
        """Adayı klibe dönüştürür; geçersizse ``None`` döndürür."""
        cfg = self.config
        duration_ms = ts - self._onset_ts
        if duration_ms < cfg.min_duration_ms:
            logger.debug("Aday çok kısa (%.0f ms); atlandı.", duration_ms)
            return None
        if self._peak_energy < threshold:
            logger.debug("Aday tepe enerjisi eşiğin altında; atlandı.")
            return None

        frames = list(self._buffer)
        if not frames:
            return None
        if len(frames) < cfg.buffer_size:
            # Isınma evresinde eksik kareler ilk kare tekrarlanarak doldurulur;
            # böylece klibin sonu her zaman güncel kareye hizalanır.
            frames = [frames[0]] * (cfg.buffer_size - len(frames)) + frames

        energies = np.fromiter(
            (frame.energy for frame in frames), dtype=np.float64, count=len(frames)
        )
        apex_index = int(np.argmax(energies))
        timestamps = tuple(frame.timestamp_ms for frame in frames)

        event = SpotEvent(
            u=np.stack([frame.u for frame in frames], axis=0),
            v=np.stack([frame.v for frame in frames], axis=0),
            strain=np.stack([frame.strain for frame in frames], axis=0),
            timestamps_ms=timestamps,
            onset_ms=self._onset_ts,
            apex_ms=timestamps[apex_index],
            offset_ms=ts,
            apex_index=apex_index,
            peak_energy=float(energies[apex_index]),
            threshold=float(threshold),
        )
        logger.debug(
            "Spot olayı: onset=%d apex=%d offset=%d süre=%d ms tepe=%.5f eşik=%.5f",
            event.onset_ms,
            event.apex_ms,
            event.offset_ms,
            event.duration_ms,
            event.peak_energy,
            event.threshold,
        )
        return event
