"""CASME II / SAMM tabanlı mikro mimik veri kümesi okuyucu.

Her örnek; bir mikro mimik klibinin Onset→Apex→Offset aralığından örneklenmiş
``temporal_length`` karesinden, çalışma zamanındaki (``main.py``) akışla aynı
biçimde üretilen ``(3, T, H, W)`` tensörüdür:

* kanal 0: yatay optical flow ``u``
* kanal 1: dikey optical flow ``v``
* kanal 2: optical strain

Akış, ardışık örneklenmiş kare çiftleri arasında Farnebäck ile hesaplanır ve
ilk kareye sıfır akış atanır; bu, ``SpottingEngine`` ring buffer'ının ürettiği
kliple birebir aynı yapıdır. Her kanal örnek bazında standartlaştırılır
(``SpotEvent.to_model_input`` ile aynı ortalama/std mantığı).

Beklenen klasör düzeni (CASME II / SAMM ortak)::

    data/
    └── <subject>/<clip>/img1.jpg ... imgN.jpg

Sınıf etiketi şu sırayla çözümlenir:

1. ``manifest`` CSV'sindeki ``label`` sütunu,
2. klip klasöründeki ``label.txt`` / ``emotion.txt`` / ``class.txt`` dosyası,
3. üst klasör adının bir sınıf adıyla birebir eşleşmesi
   (ör. ``.../negative/clip01``),
4. CASME II kod öneki (``HA`` → positive, ``DI``/``RE``/``FE``/``SA``/``AN`` →
   negative, ``SU`` → surprise). ``EP`` bir **Episode** önekidir, duygu
   sınıfı belirtmez; bu nedenle tek başına etiket olarak kullanılmaz.
5. CASME II kodlama Excel'i (``CASME2-coding-20140508.xlsx``) ``Estimated
   Emotion`` sütunu: ``happiness`` → positive, ``surprise`` → surprise,
   ``disgust``/``repression``/``fear``/``sadness`` → negative, diğerleri
   (``others``) atlanır.

Etiketi çözümlenemeyen klipler atlanır. SAMM kliplerinin kodlanmış etiketleri
klasör adında bulunmadığından SAMM için ``manifest`` kullanılması önerilir.

Manifest CSV örneği (``onset``/``apex``/``offset`` 1 tabanlı kare numaralarıdır;
verilmezse onset = ilk kare, offset = son kare, apex = onset karesine göre en
çok değişen kare olarak tahmin edilir)::

    frames_dir,label,onset,apex,offset
    sub01/EP02_01f,positive,1,45,90
    sub02/SU08_01,surprise,3,40,78

Veri yoksa ``dummy=True`` ile eğitim boru hattını (DataLoader → model → kayıt →
ONNX) sınamak için sınıfa göre şekillendirilmiş sentetik klipler üretilir.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.optical_flow import FLOW_SIZE, OpticalFlowExtractor

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
DEFAULT_CLASSES: tuple[str, ...] = ("negative", "positive", "surprise")
DEFAULT_TEMPORAL_LENGTH = 16

_LABEL_FILES = ("label.txt", "emotion.txt", "class.txt", "label.csv")
_FRAMES_KEYS = ("frames_dir", "frames", "clip_dir", "path", "dir", "folder")
_LABEL_KEYS = ("label", "class", "emotion", "category")
_ONSET_KEYS = ("onset", "onset_frame", "onset_idx", "onset_index")
_APEX_KEYS = ("apex", "apex_frame", "apex_idx", "apex_index")
_OFFSET_KEYS = ("offset", "offset_frame", "offset_idx", "offset_index")

_APEX_SCAN_FRAMES = 24
_APEX_THUMB_SIZE = 64

_NATURAL_SORT_RE = re.compile(r"(\d+)")
_PREFIX_RE = re.compile(r"^([A-Za-z]{2})(?=$|[0-9_])")

# CASME II kod önekleri → kanonik sınıf adları.
# ``EP`` burada **yoktur**: CASME II'de ``EP02_01f`` gibi dosya adlarındaki
# ``EP`` öneki "Episode" anlamına gelir ve duygu sınıfı hakkında bilgi vermez
# (ör. ``EP19_05f`` negatif, ``EP02_01f`` pozitiftir). Bu kliplerin etiketi
# manifest veya kodlama Excel'inden çözülmelidir; aksi halde atlanır.
_PREFIX_CLASSES: dict[str, str] = {
    "HA": "positive",
    "DI": "negative",
    "RE": "negative",
    "FE": "negative",
    "SA": "negative",
    "AN": "negative",
    "SU": "surprise",
}

# Serbest metin etiketleri (label dosyası / klasör adı) → kanonik sınıf.
_TEXT_ALIASES: dict[str, str] = {
    "positive": "positive",
    "pos": "positive",
    "happy": "positive",
    "happiness": "positive",
    "ha": "positive",
    "joy": "positive",
    "di": "negative",
    "disgust": "negative",
    "negative": "negative",
    "neg": "negative",
    "sad": "negative",
    "sadness": "negative",
    "sa": "negative",
    "fear": "negative",
    "fe": "negative",
    "anger": "negative",
    "an": "negative",
    "repression": "negative",
    "re": "negative",
    "contempt": "negative",
    "su": "surprise",
    "surprise": "surprise",
    "surprised": "surprise",
    "sp": "surprise",
}


@dataclass(slots=True)
class ClipSample:
    """Diskteki bir mikro mimik klibi için çözümlenmiş metadata.

    Attributes:
        clip_dir: Karelerin bulunduğu klasör.
        frames: Doğal sıralanmış kare yolları.
        label: Sınıf indeksi (``classes`` listesindeki konum).
        class_name: Sınıf adı.
        onset: Onset kare indeksi (klip içinde 0 tabanlı).
        apex: Apex kare indeksi.
        offset: Offset kare indeksi.
        subject: Üst klasör adı (denek kimliği).
        clip_name: Klip klasörü adı.
    """

    clip_dir: Path
    frames: tuple[Path, ...]
    label: int
    class_name: str
    onset: int
    apex: int
    offset: int
    subject: str
    clip_name: str


@dataclass(slots=True)
class CodingEntry:
    """CASME II kodlama Excel'indeki tek klip satırı.

    Attributes:
        label: ``Estimated Emotion`` sütunundan çözümlenen sınıf adı.
        frames: Varsa ``onset``/``apex``/``offset`` (1 tabanlı kare numaraları).
    """

    label: str
    frames: dict[str, int]


# CASME II ``Estimated Emotion`` değerleri → kanonik sınıf adları.
_EMOTION_TO_CLASS: dict[str, str] = {
    "happiness": "positive",
    "surprise": "surprise",
    "disgust": "negative",
    "repression": "negative",
    "fear": "negative",
    "sadness": "negative",
}
# ``others`` ve tanınmayan değerler atlanır (veri kümesine dahil edilmez).
_CODING_SUBJECT_KEYS = ("subject", "sub", "subject id")
_CODING_FILENAME_KEYS = ("filename", "file", "clip", "clip name")
_CODING_EMOTION_KEYS = ("estimated emotion", "emotion", "estimatedemotion")
_CODING_FRAME_KEYS = {
    "onset": ("onsetframe", "onset frame", "onset"),
    "apex": ("apexframe", "apex frame", "apex"),
    "offset": ("offsetframe", "offset frame", "offset"),
}


def load_casme2_coding(
    path: str | Path, classes: Sequence[str] = DEFAULT_CLASSES
) -> dict[tuple[str, str], CodingEntry]:
    """CASME II kodlama Excel'ini (``Estimated Emotion``) okuyup haritalar.

    Args:
        path: ``CASME2-coding-20140508.xlsx`` (veya aynı sütunlara sahip
            ``.csv``) dosyası.
        classes: Sınıf adları; yalnızca bu adlara çözümlenen satırlar alınır.

    Returns:
        ``(normalize_denek, normalize_klip_adı) → CodingEntry`` haritası.
        ``others`` ve tanınmayan duygular dahil edilmez.

    Raises:
        FileNotFoundError: Dosya yoksa.
        ValueError: Gerekli sütunlar bulunamazsa.
        RuntimeError: Excel okunamazsa.
    """
    coding_path = Path(path)
    if not coding_path.is_file():
        raise FileNotFoundError(f"Kodlama dosyası bulunamadı: {coding_path}")

    rows = _read_coding_rows(coding_path)
    if not rows:
        raise ValueError(f"Kodlama dosyası boş: {coding_path}")

    header = [str(value).strip().lower() if value is not None else "" for value in rows[0]]
    column = {name: index for index, name in enumerate(header) if name}

    def find(keys: Sequence[str]) -> int | None:
        for key in keys:
            if key in column:
                return column[key]
        return None

    subject_col = find(_CODING_SUBJECT_KEYS)
    filename_col = find(_CODING_FILENAME_KEYS)
    emotion_col = find(_CODING_EMOTION_KEYS)
    if subject_col is None or filename_col is None or emotion_col is None:
        raise ValueError(
            "Kodlama dosyası Subject/Filename/Estimated Emotion sütunlarını "
            f"içermeli: {coding_path}"
        )
    frame_cols = {
        key: find(candidates) for key, candidates in _CODING_FRAME_KEYS.items()
    }

    lowered = [name.casefold() for name in classes]
    result: dict[tuple[str, str], CodingEntry] = {}
    skipped_other = 0
    for row in rows[1:]:
        if row is None or all(value in (None, "") for value in row):
            continue
        subject = _normalize_subject(_cell(row, subject_col))
        clip_name = _cell(row, filename_col)
        emotion = _cell(row, emotion_col).casefold()
        if not subject or not clip_name:
            continue
        class_name = _EMOTION_TO_CLASS.get(emotion)
        if class_name is None or class_name.casefold() not in lowered:
            skipped_other += 1
            continue
        frames: dict[str, int] = {}
        for key, col in frame_cols.items():
            if col is None:
                continue
            try:
                frames[key] = int(float(_cell(row, col)))
            except (TypeError, ValueError):
                continue
        result[(subject, clip_name.casefold())] = CodingEntry(
            label=class_name, frames=frames
        )

    counts: dict[str, int] = {}
    for entry in result.values():
        counts[entry.label] = counts.get(entry.label, 0) + 1
    logger.info(
        "CASME II kodlaması okundu: %d klip | %s | atlanan (others/geçersiz)=%d",
        len(result),
        " | ".join(f"{name}={count}" for name, count in sorted(counts.items())),
        skipped_other,
    )
    return result


def verify_manifest_against_coding(
    manifest: str | Path,
    coding_xlsx: str | Path,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> dict[str, object]:
    """Manifest etiketlerini kodlama Excel'iyle karşılaştırır.

    Returns:
        ``total``, ``counts`` (sınıf dağılımı), ``missing`` (Excel'de olup
        manifest'te olmayan), ``extra`` (manifest'te olup Excel'de olmayan)
        ve ``mismatched`` (etiket farkları) anahtarlarını içeren özet.
    """
    coding = load_casme2_coding(coding_xlsx, classes)
    manifest_labels: dict[tuple[str, str], str] = {}
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        columns = {name.strip().lower(): name for name in reader.fieldnames or []}
        frames_key = _first_present(columns, _FRAMES_KEYS)
        label_key = _first_present(columns, _LABEL_KEYS)
        if frames_key is None or label_key is None:
            raise ValueError("Manifest frames_dir ve label sütunlarını içermeli.")
        for row in reader:
            raw = (row.get(frames_key) or "").strip()
            if not raw:
                continue
            clip_path = Path(raw)
            key = _coding_key(clip_path.parent.name, clip_path.name)
            manifest_labels[key] = (row.get(label_key) or "").strip()

    counts: dict[str, int] = {}
    for label in manifest_labels.values():
        counts[label] = counts.get(label, 0) + 1
    missing = sorted(f"{subject}/{clip}" for subject, clip in coding if (subject, clip) not in manifest_labels)
    extra = sorted(f"{subject}/{clip}" for subject, clip in manifest_labels if (subject, clip) not in coding)
    mismatched = sorted(
        (subject, clip, coding[(subject, clip)].label, label)
        for (subject, clip), label in manifest_labels.items()
        if (subject, clip) in coding and coding[(subject, clip)].label != label
    )
    return {
        "total": len(manifest_labels),
        "counts": counts,
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
    }


def _read_coding_rows(path: Path) -> list[tuple[object, ...]]:
    """Kodlama dosyasını (``.xlsx``/``.csv``) satır demetleri olarak okur."""
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [tuple(row) for row in csv.reader(handle)]
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - ortam bağımlılığı
        raise RuntimeError(
            "Excel okumak için openpyxl gerekir: pip install openpyxl"
        ) from exc
    try:
        workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
        sheet = workbook.active
        rows = [tuple(row) for row in sheet.iter_rows(values_only=True)]
        workbook.close()
    except Exception as exc:
        raise RuntimeError(f"Kodlama Excel'i okunamadı ({path}): {exc}") from exc
    return rows


def _cell(row: Sequence[object], index: int) -> str:
    """Satırdaki hücreyi güvenli biçimde boşlukları kırpılmış metne çevirir."""
    if index >= len(row):
        return ""
    value = row[index]
    return "" if value is None else str(value).strip()


def _normalize_subject(text: str) -> str:
    """Denek kimliğini normalize eder (``sub01``, ``01`` ve ``1`` aynı olur)."""
    cleaned = re.sub(r"[^a-z0-9]", "", text.casefold())
    if not cleaned:
        return ""
    cleaned = cleaned.removeprefix("sub")
    return cleaned.lstrip("0") or "0"


def _coding_key(subject: str, clip_name: str) -> tuple[str, str]:
    """Kodlama haritası için ``(denek, klip)`` anahtarı üretir."""
    return _normalize_subject(subject), clip_name.strip().casefold()


def _find_coding_xlsx(root: Path, manifest_path: Path | None) -> Path | None:
    """``root`` (veya manifest klasörü) altında CASME II kodlama dosyası arar."""
    search_dirs = []
    if manifest_path is not None:
        search_dirs.append(manifest_path.parent)
    search_dirs.append(root)
    for directory in search_dirs:
        try:
            candidates = sorted(directory.glob("CASME2-coding*.xlsx"))
        except OSError:
            continue
        for candidate in candidates:
            logger.info("CASME II kodlama Excel'i otomatik bulundu: %s", candidate)
            return candidate
    return None


class MicroExpressionDataset(Dataset[tuple[torch.Tensor, int]]):
    """Onset-Apex-Offset kliplerinden ``(u, v, strain)`` tensörleri üretir.

    Args:
        root: Veri kümesi kök klasörü (CASME II / SAMM). ``dummy=True`` ise
            yok sayılır.
        classes: Sınıf adları; model çıkış sırası bu listeyle aynıdır.
        temporal_length: Klip başına örneklenen kare sayısı (STSTNet ``T``).
        image_size: Akış/tensör uzamsal boyutu (varsayılan 128).
        dummy: Veri yerine sentetik klipler üret (eğitim boru hattı testi).
        dummy_samples: ``dummy=True`` iken üretilecek örnek sayısı.
        dummy_seed: Sentetik verinin tekrarlanabilirlik tohumu.
        align: Kareleri ``FaceAligner`` ile 256x256 ROI'ye hizala. CASME II /
            SAMM klipleri zaten kırpılmış yüzler içerdiğinden varsayılan
            ``False``; göz maskeleme için ``True`` gerekir.
        mask_eyes: Göz kapağı bölgesini akıştan maskele (``align=True`` iken
            landmark'lardan üretilir).
        manifest: İsteğe bağlı CSV metadata dosyası.
        coding_xlsx: İsteğe bağlı CASME II kodlama Excel'i (``Estimated
            Emotion`` sütunu). Etiket önceliği: manifest → Excel → klasör
            etiketi → kod öneki. ``None`` ve ``root`` altında
            ``CASME2-coding*.xlsx`` varsa otomatik bulunur.

    Raises:
        ValueError: Parametreler geçersizse.
        FileNotFoundError: ``root`` veya ``manifest`` bulunamazsa.
        RuntimeError: Veri kökünde etiketlenebilir klip yoksa.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        classes: Sequence[str] = DEFAULT_CLASSES,
        temporal_length: int = DEFAULT_TEMPORAL_LENGTH,
        image_size: int = FLOW_SIZE,
        dummy: bool = False,
        dummy_samples: int = 64,
        dummy_seed: int = 0,
        align: bool = False,
        mask_eyes: bool = False,
        manifest: str | Path | None = None,
        coding_xlsx: str | Path | None = None,
    ) -> None:
        class_names = [name.strip() for name in classes if name.strip()]
        if not class_names:
            raise ValueError("En az bir sınıf adı gerekir.")
        if temporal_length < 2:
            raise ValueError("temporal_length en az 2 olmalıdır.")
        if image_size < 16:
            raise ValueError("image_size en az 16 olmalıdır.")
        if dummy_samples < 1:
            raise ValueError("dummy_samples en az 1 olmalıdır.")

        self.class_names = class_names
        self.temporal_length = int(temporal_length)
        self.image_size = int(image_size)
        self.dummy = bool(dummy)
        self.dummy_seed = int(dummy_seed)
        self._dummy_samples = int(dummy_samples)
        self.align = bool(align)
        self.mask_eyes = bool(mask_eyes)
        self.samples: list[ClipSample] = []

        # MediaPipe landmarker her süreçte (DataLoader worker) ayrı tutulur.
        self._aligner = None
        self._aligner_pid: int | None = None
        self._align_ts: int = 0
        self._align_failed: bool = False

        if self.mask_eyes and not self.align:
            logger.warning(
                "mask_eyes=True ancak align=False: landmark olmadığından göz "
                "maskesi uygulanmayacak."
            )

        if self.dummy:
            logger.info(
                "Sentetik veri modu: %d örnek | sınıflar: %s | T=%d, %dx%d",
                self._dummy_samples,
                ", ".join(self.class_names),
                self.temporal_length,
                self.image_size,
                self.image_size,
            )
            return

        if root is None:
            raise ValueError("dummy=False için root zorunludur.")
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(
                f"Veri kökü bulunamadı: {root_path}. Sentetik veriyle denemek "
                f"için --dummy kullanın."
            )

        manifest_path = Path(manifest) if manifest is not None else None
        coding_path = (
            Path(coding_xlsx)
            if coding_xlsx is not None
            else _find_coding_xlsx(root_path, manifest_path)
        )
        coding = (
            load_casme2_coding(coding_path, self.class_names)
            if coding_path is not None
            else {}
        )
        self.samples = _scan_clips(
            root_path, self.class_names, manifest_path, coding
        )
        if not self.samples:
            raise RuntimeError(
                f"{root_path} altında etiketlenebilir mikro mimik klibi "
                f"bulunamadı. Klasör düzenini kontrol edin veya --dummy ile "
                f"sentetik veri üretin."
            )

        counts = {name: 0 for name in self.class_names}
        for sample in self.samples:
            counts[sample.class_name] += 1
        logger.info(
            "Veri kümesi hazır: %d klip | %s",
            len(self.samples),
            " | ".join(f"{name}={count}" for name, count in counts.items()),
        )

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    @property
    def num_classes(self) -> int:
        """Sınıf sayısı."""
        return len(self.class_names)

    def __len__(self) -> int:
        """Örnek sayısı."""
        return self._dummy_samples if self.dummy else len(self.samples)

    @property
    def labels(self) -> list[int]:
        """Tüm örneklerin sınıf indeksleri (katmanlı bölme/sınıf ağırlığı için)."""
        if self.dummy:
            return [
                index % len(self.class_names) for index in range(self._dummy_samples)
            ]
        return [sample.label for sample in self.samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Bir klibi ``((3, T, H, W) float32, sınıf indeksi)`` olarak döndürür."""
        if self.dummy:
            return self._dummy_item(int(index))

        sample = self.samples[int(index)]
        indices = _sample_indices(
            len(sample.frames),
            sample.onset,
            sample.apex,
            sample.offset,
            self.temporal_length,
        )

        aligner = self._get_aligner() if self.align else None
        extractor = OpticalFlowExtractor(
            mask_eyes=self.mask_eyes, flow_size=self.image_size
        )

        u_frames: list[np.ndarray] = []
        v_frames: list[np.ndarray] = []
        strain_frames: list[np.ndarray] = []
        prev_gray: np.ndarray | None = None

        for position in indices:
            gray, landmarks = self._load_gray(
                sample.frames[position], aligner=aligner
            )
            flow = extractor.extract(prev_gray, gray, landmarks)
            u_frames.append(flow.u)
            v_frames.append(flow.v)
            strain_frames.append(flow.strain)
            prev_gray = gray

        clip = np.stack(
            (
                np.stack(u_frames, axis=0),
                np.stack(v_frames, axis=0),
                np.stack(strain_frames, axis=0),
            ),
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(_standardize(clip)), sample.label

    # ------------------------------------------------------------------ #
    # Kare okuma / hizalama
    # ------------------------------------------------------------------ #
    def _load_gray(
        self, path: Path, *, aligner: object | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Kareyi gri tona çevirir; hizalama açıksa ROI ve landmark döndürür.

        Okunamayan karede sıfır görüntü döner; eğitim tek bir bozuk dosya
        yüzünden durmaz.
        """
        bgr = _read_bgr(path)
        if bgr is None:
            logger.warning("Kare okunamadı, sıfır görüntü kullanılıyor: %s", path)
            return (
                np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                None,
            )

        if aligner is not None:
            try:
                self._align_ts += 33
                result = aligner.process(bgr, self._align_ts)  # type: ignore[attr-defined]
                if result.detected and result.aligned is not None:
                    return (
                        OpticalFlowExtractor.to_gray(result.aligned),
                        result.landmarks,
                    )
            except Exception:  # pragma: no cover - bozuk karede güvenli düşüş
                logger.debug("Hizalama başarısız: %s", path, exc_info=True)

        gray = OpticalFlowExtractor.to_gray(
            cv2.resize(
                bgr,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
        )
        return gray, None

    def _get_aligner(self) -> object | None:
        """DataLoader worker başına tek bir ``FaceAligner`` oluşturur.

        MediaPipe nesneleri süreçler arasında paylaşılamadığı için her worker
        kendi landmarker'ını tembel olarak kurar. Model indirilemezse veya
        oluşturulamazsa hizalama sessizce kapatılır.
        """
        pid = os.getpid()
        if self._align_failed:
            return None
        if self._aligner is not None and self._aligner_pid == pid:
            return self._aligner

        try:
            from src.preprocessing.face_aligner import FaceAligner

            self._aligner = FaceAligner()
            self._aligner_pid = pid
            self._align_ts = 0
            logger.info("FaceAligner hazır (pid=%d).", pid)
        except Exception:
            logger.exception(
                "FaceAligner oluşturulamadı; kareler yeniden boyutlandırılarak "
                "kullanılacak."
            )
            self._align_failed = True
            self._aligner = None
        return self._aligner

    # ------------------------------------------------------------------ #
    # Sentetik veri
    # ------------------------------------------------------------------ #
    def _dummy_item(self, index: int) -> tuple[torch.Tensor, int]:
        """Sınıfa göre şekillendirilmiş, tekrarlanabilir sentetik klip üretir."""
        rng = np.random.default_rng(self.dummy_seed + index)
        temporal = self.temporal_length
        size = self.image_size
        label = index % len(self.class_names)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        center_x = 0.30 + 0.20 * label
        blob = np.exp(
            -(
                ((xx / size) - center_x) ** 2 + ((yy / size) - 0.50) ** 2
            )
            / (2.0 * 0.08**2)
        ).astype(np.float32)

        pulse = np.sin(
            np.linspace(0.0, np.pi, temporal, dtype=np.float32)
        )[:, None, None]
        amplitude = 0.6 + 0.15 * label
        u = amplitude * blob[None, ...] * pulse
        v = -0.5 * amplitude * blob[None, ...] * pulse
        strain = np.abs(u) + np.abs(v)

        noise = rng.standard_normal((3, temporal, size, size)).astype(np.float32)
        smoothed = np.stack(
            [
                cv2.GaussianBlur(frame, (0, 0), 1.2)
                for frame in noise.reshape(-1, size, size)
            ]
        ).reshape(3, temporal, size, size)

        clip = np.stack((u, v, strain), axis=0).astype(np.float32)
        clip = clip + 0.05 * smoothed
        return torch.from_numpy(_standardize(clip)), label


# ---------------------------------------------------------------------- #
# Klip keşfi ve etiket çözümleme
# ---------------------------------------------------------------------- #
def _scan_clips(
    root: Path,
    classes: Sequence[str],
    manifest_path: Path | None,
    coding: dict[tuple[str, str], "CodingEntry"] | None = None,
) -> list[ClipSample]:
    """Kök klasörden (veya manifest'ten) etiketli klipleri toplar."""
    coding = coding or {}
    records: list[tuple[Path, str | None, dict[str, str]]] = []
    if manifest_path is not None:
        records.extend(_read_manifest(manifest_path, root))
    else:
        records.extend((clip_dir, None, {}) for clip_dir in _iter_clip_dirs(root))

    samples: list[ClipSample] = []
    skipped = 0
    for clip_dir, manifest_label, meta in records:
        frames = _list_frames(clip_dir)
        if len(frames) < 2:
            skipped += 1
            logger.debug("Klip en az 2 kare içermeli, atlandı: %s", clip_dir)
            continue

        entry = coding.get(_coding_key(clip_dir.parent.name, clip_dir.name))
        label = _resolve_class_index(manifest_label, classes)
        if label is None and entry is not None:
            label = _resolve_class_index(entry.label, classes, allow_prefix=False)
        if label is None:
            label = _resolve_clip_label(clip_dir, classes)
        if label is None:
            skipped += 1
            logger.debug("Sınıf etiketi çözümlenemedi, atlandı: %s", clip_dir)
            continue

        count = len(frames)
        # Manifest onset/apex/offset değerleri Excel'deki kodlamaya göre
        # önceliklidir; manifest boşsa Excel'deki kare numaraları kullanılır.
        #
        # Excel'deki Onset/Apex/OffsetFrame numaraları **orijinal kayda**
        # göredir. CASME II ``Cropped`` klasörleri ise onset karesinden
        # başlayan klip dilimini içerir (kare sayısı = offset - onset + 1).
        # Bu nedenle yerel indeks: ``local = frame - onset``.
        if entry is not None:
            reference = entry.frames.get("onset")
            mapped: dict[str, int] = {}
            if reference is not None:
                for key in ("onset", "apex", "offset"):
                    value = entry.frames.get(key)
                    if value is None:
                        continue
                    local = value - reference
                    if not 0 <= local < count:
                        mapped = {}
                        logger.debug(
                            "Kodlama kare indeksleri klasörle uyuşmuyor "
                            "(%s: %s, kare sayısı=%d); apex tahmini kullanılacak.",
                            clip_dir.name,
                            entry.frames,
                            count,
                        )
                        break
                    mapped[key] = local
            for key, local in mapped.items():
                if not meta.get(key):
                    meta[key] = str(local + 1)  # _to_index 1 tabanlı bekler

        onset = _to_index(meta.get("onset"), count)
        offset = _to_index(meta.get("offset"), count)
        onset = 0 if onset is None else onset
        offset = count - 1 if offset is None else offset
        if onset > offset:
            onset, offset = offset, onset

        apex = _to_index(meta.get("apex"), count)
        if apex is None:
            apex = _estimate_apex(frames, onset, offset)
        apex = min(max(apex, onset), offset)

        samples.append(
            ClipSample(
                clip_dir=clip_dir,
                frames=tuple(frames),
                label=label,
                class_name=classes[label],
                onset=onset,
                apex=apex,
                offset=offset,
                subject=clip_dir.parent.name,
                clip_name=clip_dir.name,
            )
        )

    if skipped:
        logger.warning(
            "%d klip eksik kare veya çözümlenemeyen etiket nedeniyle atlandı.",
            skipped,
        )
    return samples


def _iter_clip_dirs(root: Path) -> Iterator[Path]:
    """Doğrudan görüntü içeren tüm klasörleri (kök dahil) üretir."""
    if _list_frames(root):
        yield root
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        if _list_frames(directory):
            yield directory


def _list_frames(directory: Path) -> list[Path]:
    """Klasördeki görüntü dosyalarını doğal sırayla listeler."""
    try:
        frames = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    except OSError:
        return []
    return sorted(frames, key=_natural_sort_key)


def _natural_sort_key(path: Path) -> list[object]:
    """``img2.jpg`` < ``img10.jpg`` olacak biçimde doğal sıralama anahtarı."""
    parts = _NATURAL_SORT_RE.split(path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _read_manifest(
    path: Path, root: Path
) -> Iterator[tuple[Path, str | None, dict[str, str]]]:
    """Manifest CSV'sini ``(klip_klasörü, etiket, meta)`` üçlülerine açar.

    ``frames_dir`` yolu göreliyse ``root`` altında çözümlenir. Manifest
    bulunamaz veya gerekli sütunları içermezse hata yükseltilir.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Manifest bulunamadı: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"Manifest boş veya başlıksız: {path}")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        frames_key = _first_present(columns, _FRAMES_KEYS)
        if frames_key is None:
            raise ValueError(
                f"Manifest 'frames_dir' (veya path/clip_dir) sütunu içermeli: {path}"
            )
        label_key = _first_present(columns, _LABEL_KEYS)
        onset_key = _first_present(columns, _ONSET_KEYS)
        apex_key = _first_present(columns, _APEX_KEYS)
        offset_key = _first_present(columns, _OFFSET_KEYS)

        for row in reader:
            raw_path = (row.get(frames_key) or "").strip()
            if not raw_path:
                continue
            clip_dir = Path(raw_path)
            if not clip_dir.is_absolute():
                clip_dir = root / clip_dir
            label = (row.get(label_key) or "").strip() if label_key else None
            meta = {
                key: (row.get(column) or "").strip()
                for key, column in (
                    ("onset", onset_key),
                    ("apex", apex_key),
                    ("offset", offset_key),
                )
                if column is not None
            }
            yield clip_dir, label or None, meta


def _first_present(columns: dict[str, str], candidates: Sequence[str]) -> str | None:
    """Normalize edilmiş sütun haritasında ilk bulunan adayın asıl adını döndürür."""
    for candidate in candidates:
        original = columns.get(candidate)
        if original is not None:
            return original
    return None


def _to_index(value: str | None, count: int) -> int | None:
    """1 tabanlı kare numarasını 0 tabanlı indekse çevirir; geçersizse ``None``."""
    if value is None:
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return 0
    return min(number - 1, count - 1)


def _resolve_clip_label(clip_dir: Path, classes: Sequence[str]) -> int | None:
    """Klip klasöründen etiketi şu sırayla çözer: dosya → üst klasör → ad."""
    for file_name in _LABEL_FILES:
        label_file = clip_dir / file_name
        if label_file.is_file():
            index = _resolve_class_index(_read_label_file(label_file), classes)
            if index is not None:
                return index

    for ancestor in (clip_dir.parent, clip_dir.parent.parent):
        if ancestor == ancestor.parent:
            break
        index = _resolve_class_index(ancestor.name, classes, allow_prefix=False)
        if index is not None:
            return index

    return _resolve_class_index(clip_dir.name, classes)


def _read_label_file(path: Path) -> str | None:
    """Etiket dosyasındaki ilk anlamlı satırı döndürür."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        cleaned = line.strip().strip("\"'")
        if not cleaned:
            continue
        if cleaned.lower() in {"label", "class", "emotion", "category"}:
            continue
        return cleaned
    return None


def _resolve_class_index(
    text: str | None, classes: Sequence[str], *, allow_prefix: bool = True
) -> int | None:
    """Serbest metni sınıf adına (ve indeksine) çevirir.

    Eşleşme sırası: birebir sınıf adı → tam metin takma adı → kelime bazlı
    takma ad → CASME II 2 harfli kod öneki (``allow_prefix`` ise).
    """
    if not text:
        return None
    lowered = [name.casefold() for name in classes]
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    if not normalized:
        return None

    if normalized in lowered:
        return lowered.index(normalized)

    alias = _TEXT_ALIASES.get(normalized)
    if alias is not None and alias.casefold() in lowered:
        return lowered.index(alias.casefold())

    for token in normalized.split():
        if token in lowered:
            return lowered.index(token)
        alias = _TEXT_ALIASES.get(token)
        if alias is not None and alias.casefold() in lowered:
            return lowered.index(alias.casefold())

    if allow_prefix:
        prefix = _prefix_code(text)
        alias = _PREFIX_CLASSES.get(prefix) if prefix else None
        if alias is not None and alias.casefold() in lowered:
            return lowered.index(alias.casefold())
    return None


def _prefix_code(text: str) -> str | None:
    """CASME II kod önekini döndürür (``EP02_01f`` → ``EP``); yoksa ``None``.

    Yalnızca iki harf + (dosya sonu | rakam/alt çizgi) kalıbı kabul edilir;
    böylece ``sub01`` gibi denek klasörleri yanlışlıkla ``SU`` (surprise)
    sayılmaz.
    """
    match = _PREFIX_RE.match(text.strip())
    return match.group(1).upper() if match else None


def _estimate_apex(frames: Sequence[Path], onset: int, offset: int) -> int:
    """Onset karesine göre en çok değişen kareyi apex olarak tahmin eder.

    Tüm klipleri okumak yerine aralık boyunca en fazla ``_APEX_SCAN_FRAMES``
    kare, küçük gri kopyalar üzerinden karşılaştırılır.
    """
    low = min(onset, offset)
    high = max(onset, offset)
    if high - low < 4:
        return (low + high) // 2

    positions = np.unique(
        np.linspace(low, high, min(_APEX_SCAN_FRAMES, high - low + 1))
        .round()
        .astype(int)
    )
    reference = _read_gray_thumb(frames[int(positions[0])])
    if reference is None:
        return (low + high) // 2

    best_index = int(positions[0])
    best_score = -1.0
    for position in positions:
        thumb = _read_gray_thumb(frames[int(position)])
        if thumb is None:
            continue
        score = float(np.mean(np.abs(thumb - reference)))
        if score > best_score:
            best_score = score
            best_index = int(position)
    return best_index


def _read_gray_thumb(path: Path) -> np.ndarray | None:
    """Karşılaştırma için kareyi küçük, gri ``float32`` diziye indirger."""
    image = _read_bgr(path)
    if image is None:
        return None
    small = cv2.resize(
        image,
        (_APEX_THUMB_SIZE, _APEX_THUMB_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _read_bgr(path: Path) -> np.ndarray | None:
    """Unicode yolları da destekleyen, hata toleranslı görüntü okuyucu."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image if image is not None and image.size > 0 else None


def _sample_indices(
    frame_count: int,
    onset: int,
    apex: int,
    offset: int,
    temporal_length: int,
) -> list[int]:
    """Onset→offset aralığından ``temporal_length`` kare indeksi seçer.

    Kareler doğrusal olarak örneklenir; apex her zaman klibe dahil edilir ve
    ilk örnek onset karesidir (akış referansı).
    """
    onset = min(max(onset, 0), frame_count - 1)
    offset = min(max(offset, 0), frame_count - 1)
    if onset > offset:
        onset, offset = offset, onset
    if offset == onset:
        return [onset] * temporal_length

    apex = min(max(apex, onset), offset)
    indices = np.rint(
        np.linspace(onset, offset, temporal_length)
    ).astype(np.int64)
    indices[0] = onset
    indices[int(np.argmin(np.abs(indices - apex)))] = apex
    np.clip(indices, 0, frame_count - 1, out=indices)
    indices.sort()
    return [int(index) for index in indices]


def _standardize(clip: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Her kanalı kendi ortalama/std değeriyle standartlaştırır.

    ``SpotEvent.to_model_input`` ile aynı dönüşüm; böylece eğitim ve canlı
    çıkarım aynı ölçekte çalışır.
    """
    mean = clip.mean(axis=(1, 2, 3), keepdims=True)
    std = clip.std(axis=(1, 2, 3), keepdims=True)
    return ((clip - mean) / np.maximum(std, eps)).astype(np.float32)

