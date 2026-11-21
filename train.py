"""STSTNet eğitim betiği ve otomatik ONNX dışa aktarımı.

Kullanım::

    # Sentetik veriyle boru hattını sına (veri yoksa)
    python train.py --dummy --epochs 5

    # CASME II / SAMM klasöründen eğit
    python train.py --data-root data --epochs 50

    # Manifest (onset/apex/offset + etiket) ile eğit
    python train.py --data-root data --manifest data/casme2.csv

Eğitim sonunda:

* ``models/ststnet.pt`` — ``model_state_dict`` + sınıf adları + eğitim
  yapılandırmasını içeren checkpoint,
* ``models/ststnet.onnx`` — canlı çıkarım için ONNX modeli (``main.py``
  tarafından otomatik bulunur),
* ``models/ststnet.classes.txt`` — sınıf adları (``main.py`` etiketleri bu
  dosyadan okur; ``--classes`` ile elle geçersiz kılınabilir).

Veri kümesi beklentisi ve etiket çözümleme sırası için
:mod:`src.training.dataset` belgelerine bakın.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.models.ststnet import STSTNet
from src.training.dataset import (
    DEFAULT_CLASSES,
    DEFAULT_TEMPORAL_LENGTH,
    MicroExpressionDataset,
)

logger = logging.getLogger("train")

DEFAULT_WEIGHTS = Path("models") / "ststnet.pt"
DEFAULT_ONNX = Path("models") / "ststnet.onnx"


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description=(
            "STSTNet eğitimi: Onset-Apex-Offset klipleri → (u, v, strain) "
            "tensörleri → .pt ağırlığı ve .onnx modeli"
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="CASME II / SAMM kök klasörü (varsayılan: data)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Klip yolu + etiket (+ onset/apex/offset) içeren CSV dosyası",
    )
    parser.add_argument(
        "--coding-xlsx",
        type=str,
        default=None,
        help=(
            "CASME II kodlama Excel'i (Estimated Emotion sütunu). Verilmezse "
            "data kökünde CASME2-coding*.xlsx otomatik aranır."
        ),
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help="Virgülle ayrılmış sınıf adları (çıktı sırası korunur)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Eğitim epoch sayısı"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch boyutu"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Öğrenme oranı (AdamW)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Ağırlık cezası"
    )
    parser.add_argument(
        "--temporal-length",
        type=int,
        default=DEFAULT_TEMPORAL_LENGTH,
        help="Klip kare sayısı (varsayılan: 16)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Akış/tensör uzamsal boyutu (varsayılan: 128)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Doğrulama oranı (0 = doğrulama yok)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker sayısı (Windows'ta 0 önerilir)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Optical flow tensörlerini RAM önbelleğine alma; her epoch'ta "
            "Farnebäck akışını yeniden hesapla (yavaş, varsayılan: önbellek açık)"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Eğitim cihazı",
    )
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Veri yerine sentetik klipler üret (boru hattı testi)",
    )
    parser.add_argument(
        "--dummy-samples",
        type=int,
        default=96,
        help="--dummy modunda eğitim örneği sayısı",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Kareleri FaceAligner ile 256x256 ROI'ye hizala",
    )
    parser.add_argument(
        "--mask-eyes",
        action="store_true",
        help="Göz kapağı bölgesini akıştan maskele (--align önerilir)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Kaydedilecek .pt yolu (varsayılan: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(DEFAULT_ONNX),
        help=f"Dışa aktarılacak .onnx yolu (varsayılan: {DEFAULT_ONNX})",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="ONNX modelini FP16 olarak dışa aktar (GPU provider önerilir)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="ONNX dışa aktarımını atla",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------- #
# Yardımcılar
# ---------------------------------------------------------------------- #
def _parse_classes(raw: str) -> list[str]:
    """Virgülle ayrılmış sınıf listesini temizler."""
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("En az bir sınıf adı gerekir.")
    return names


def _resolve_device(requested: str) -> torch.device:
    """``auto``/``cpu``/``cuda`` seçimini somut ``torch.device``'a çevirir."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi ancak kullanılamıyor.")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    """Python/NumPy/PyTorch rastgelelik tohumlarını sabitler."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _HorizontalFlipDataset(Dataset[tuple[Tensor, int]]):
    """Eğitim sırasında %50 olasılıkla yatay çevirme uygulayan sarmalayıcı.

    Optical flow bağlamında yatay çevirme iki eksende de tutarlı olmalıdır:
    genişlik ekseni (son boyut) çevrilir ve yatay akış kanalının (kanal 0,
    ``u``) işareti terslenir. Böylece sahne aynalanırken akış vektörleri de
    doğru yönü korur.

    Args:
        base: Sarılacak veri kümesi.
        probability: Çevirme olasılığı.
    """

    def __init__(self, base: Dataset[tuple[Tensor, int]], probability: float = 0.5) -> None:
        self.base = base
        self.probability = probability

    @property
    def labels(self) -> list[int]:
        """Altta yatan veri kümesinin etiketleri."""
        return list(getattr(self.base, "labels", []))

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        clip, label = self.base[index]
        if float(torch.rand(())) < self.probability:
            clip = torch.flip(clip, dims=[3])
            clip[0] = -clip[0]  # yatay akış kanalı: u → -u
        return clip, label


class CachedClipDataset(Dataset[tuple[Tensor, int]]):
    """RAM'de tutulan önceden hesaplanmış ``(3, T, H, W)`` klipleri sunar.

    Farnebäck flow her epoch'ta yeniden hesaplanmadığı için eğitim belirgin
    biçimde hızlanır.

    Args:
        clips: Klip tensörleri (paylaşılan referanslar; kopyalanmaz).
        labels: Her klip için sınıf indeksi.
    """

    def __init__(self, clips: Sequence[Tensor], labels: Sequence[int]) -> None:
        if len(clips) != len(labels):
            raise ValueError("clips ve labels uzunlukları eşit olmalıdır.")
        self._clips = list(clips)
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._clips[int(index)], self.labels[int(index)]


def _materialize(
    dataset: Dataset[tuple[Tensor, int]], description: str
) -> tuple[list[Tensor], list[int]]:
    """Tüm klipleri bir kez hesaplayıp RAM'de listeler.

    Args:
        dataset: Mikro mimik veri kümesi.
        description: Log mesajında kullanılacak açıklama.

    Returns:
        ``(klipler, etiketler)``.
    """
    clips: list[Tensor] = []
    labels: list[int] = []
    total = len(dataset)  # type: ignore[arg-type]
    total_bytes = 0
    started = time.perf_counter()
    report_every = max(1, total // 10)
    for index in range(total):
        clip, label = dataset[index]
        clip = clip.contiguous()
        clips.append(clip)
        labels.append(int(label))
        total_bytes += clip.numel() * clip.element_size()
        if (index + 1) % report_every == 0 or index + 1 == total:
            logger.info(
                "%s önbelleğe alınıyor: %d/%d (%.0f MB)",
                description,
                index + 1,
                total,
                total_bytes / (1024 * 1024),
            )
    logger.info(
        "%s RAM önbelleği hazır: %d klip | %.1f MB | %.1f s",
        description,
        len(clips),
        total_bytes / (1024 * 1024),
        time.perf_counter() - started,
    )
    return clips, labels


def _stratified_split(
    labels: Sequence[int], val_split: float, num_classes: int, seed: int
) -> tuple[list[int], list[int]]:
    """Sınıf bazlı (katmanlı) eğitim/doğrulama indeksleri üretir.

    Her sınıftan en az bir örnek eğitimde kalacak biçimde yaklaşık
    ``val_split`` oranında doğrulama örneği ayrılır; ``negative`` çoğunluk
    sınıfı nedeniyle doğrulamanın çarpıklaşması engellenir.

    Args:
        labels: Tüm örneklerin sınıf indeksleri.
        val_split: Doğrulama oranı (0, 1).
        num_classes: Sınıf sayısı.
        seed: Tekrarlanabilir karıştırma tohumu.

    Returns:
        ``(eğitim indeksleri, doğrulama indeksleri)``.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {class_index: [] for class_index in range(num_classes)}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_index in sorted(by_class):
        indices = by_class[class_index]
        rng.shuffle(indices)
        count = len(indices)
        if count == 0:
            continue
        if count == 1:
            train_indices.extend(indices)
            continue
        val_count = int(round(count * val_split))
        val_count = max(1, min(count - 1, val_count))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _build_datasets(
    args: argparse.Namespace, class_names: Sequence[str]
) -> tuple[Dataset[tuple[Tensor, int]], Dataset[tuple[Tensor, int]] | None]:
    """Eğitim/doğrulama veri kümelerini oluşturur.

    Gerçek veride tüm flow tensörleri RAM önbelleğine alınır ve bölme sınıf
    bazlı katmanlı yapılır. ``--dummy`` modunda ayrı tohumlarla iki sentetik
    küme üretilir. Eğitim kümesi %50 yatay çevirme ile sarılır.
    """
    common = {
        "classes": class_names,
        "temporal_length": args.temporal_length,
        "image_size": args.image_size,
        "align": args.align,
        "mask_eyes": args.mask_eyes,
    }

    if args.dummy:
        if args.num_workers != 0:
            logger.warning("--dummy modunda num_workers=0 kullanılacak.")
        train_base: Dataset[tuple[Tensor, int]] = MicroExpressionDataset(
            dummy=True,
            dummy_samples=args.dummy_samples,
            dummy_seed=args.seed,
            **common,
        )
        val_base: Dataset[tuple[Tensor, int]] | None = MicroExpressionDataset(
            dummy=True,
            dummy_samples=max(8, args.dummy_samples // 4),
            dummy_seed=args.seed + 10_000,
            **common,
        )
        return _HorizontalFlipDataset(train_base), val_base

    dataset = MicroExpressionDataset(
        root=args.data_root,
        manifest=args.manifest,
        coding_xlsx=args.coding_xlsx,
        **common,
    )
    all_labels = [sample.label for sample in dataset.samples]

    if args.no_cache:
        if args.val_split <= 0.0 or len(dataset) < 2:
            logger.info("Doğrulama bölmesi yok; tüm %d örnek eğitimde.", len(dataset))
            return _HorizontalFlipDataset(dataset), None
        train_indices, val_indices = _stratified_split(
            all_labels, args.val_split, len(class_names), args.seed
        )
        return _SubsetDataset(dataset, train_indices, augment=True), _SubsetDataset(
            dataset, val_indices
        )

    clips, labels = _materialize(dataset, "Eğitim verisi")

    if args.val_split <= 0.0 or len(dataset) < 2:
        logger.info("Doğrulama bölmesi yok; tüm %d örnek eğitimde.", len(dataset))
        return _HorizontalFlipDataset(CachedClipDataset(clips, labels)), None

    train_indices, val_indices = _stratified_split(
        labels, args.val_split, len(class_names), args.seed
    )
    train_dataset = CachedClipDataset(
        [clips[index] for index in train_indices],
        [labels[index] for index in train_indices],
    )
    val_dataset = CachedClipDataset(
        [clips[index] for index in val_indices],
        [labels[index] for index in val_indices],
    )
    class_counts = {
        class_names[class_index]: train_dataset.labels.count(class_index)
        for class_index in range(len(class_names))
    }
    logger.info(
        "Katmanlı bölme: eğitim=%d | doğrulama=%d | eğitim sınıf dağılımı: %s",
        len(train_dataset),
        len(val_dataset),
        " | ".join(f"{name}={count}" for name, count in class_counts.items()),
    )
    return _HorizontalFlipDataset(train_dataset), val_dataset


class _SubsetDataset(Dataset[tuple[Tensor, int]]):
    """``--no-cache`` modunda indeks alt kümesi + isteğe bağlı çevirme sunar."""

    def __init__(
        self, base: Dataset[tuple[Tensor, int]], indices: Sequence[int], *, augment: bool = False
    ) -> None:
        self.base = base
        self.indices = list(indices)
        self.augment = augment
        self.labels = [int(getattr(base, "samples")[index].label) for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        clip, label = self.base[self.indices[int(index)]]
        if self.augment and float(torch.rand(())) < 0.5:
            clip = torch.flip(clip, dims=[3])
            clip[0] = -clip[0]
        return clip, label


def _class_weights(
    labels: Sequence[int], num_classes: int, device: torch.device
) -> Tensor:
    """Ters frekans sınıf ağırlıkları üretir (çoğunluk sınıfına yığılmayı önler).

    ``w_c = N / (C * n_c)``; hiç örneği olmayan sınıf ağırlığı 0 alır.
    """
    counts = np.bincount(np.asarray(list(labels), dtype=np.int64), minlength=num_classes)
    total = float(len(labels))
    weights = np.array(
        [
            total / (num_classes * count) if count > 0 else 0.0
            for count in counts
        ],
        dtype=np.float64,
    )
    logger.info(
        "Sınıf ağırlıkları: %s",
        ", ".join(f"{index}={weight:.3f}" for index, weight in enumerate(weights)),
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """DataLoader oluşturur (Windows'ta worker'lar spawn ile başlar)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Bir epoch eğitir; ortalama kayıp ve doğruluğu döndürür."""
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    batches = len(loader)
    report_every = max(1, batches // 4)

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.shape[0])
        running_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch_size

        if batch_index % report_every == 0 or batch_index == batches:
            logger.info(
                "epoch %d/%d | batch %d/%d | loss=%.4f | acc=%.3f",
                epoch,
                total_epochs,
                batch_index,
                batches,
                running_loss / max(seen, 1),
                correct / max(seen, 1),
            )

    return running_loss / max(seen, 1), correct / max(seen, 1)


@dataclass(slots=True)
class EvalMetrics:
    """Bir değerlendirme geçişinin metrikleri."""

    loss: float
    accuracy: float
    uar: float
    macro_f1: float
    confusion: np.ndarray
    recalls: list[float]


def _metrics_from_confusion(confusion: np.ndarray) -> tuple[float, float, list[float]]:
    """Confusion matrix'ten UAR, Macro-F1 ve sınıf bazlı recall üretir.

    UAR (Unweighted Average Recall) yalnızca gerçek etikette en az bir
    örneği olan sınıfların recall ortalamasıdır. Macro-F1 ise destek veya
    tahmini olan sınıfların F1 ortalamasıdır. Dönen ``recalls`` listesi sınıf
    indeksleriyle hizalıdır; doğrulamada hiç örneği olmayan sınıf ``nan``'dır.
    """
    num_classes = confusion.shape[0]
    recalls: list[float] = []
    valid_recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = float(confusion[class_index, class_index])
        support = float(confusion[class_index, :].sum())
        predicted = float(confusion[:, class_index].sum())
        if support > 0:
            recall = true_positive / support
            recalls.append(recall)
            valid_recalls.append(recall)
        else:
            recalls.append(float("nan"))
        if support > 0 or predicted > 0:
            precision = true_positive / predicted if predicted > 0 else 0.0
            recall_for_f1 = true_positive / support if support > 0 else 0.0
            denominator = precision + recall_for_f1
            f1_scores.append(
                2.0 * precision * recall_for_f1 / denominator
                if denominator > 0
                else 0.0
            )
    uar = float(np.mean(valid_recalls)) if valid_recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return uar, macro_f1, recalls


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> EvalMetrics:
    """Modeli değerlendirir; kayıp, doğruluk, UAR, Macro-F1 ve confusion.

    Args:
        model: Değerlendirilecek model.
        loader: Değerlendirme verisi.
        criterion: Kayıp fonksiyonu.
        device: Cihaz.
        num_classes: Confusion matrix boyutu.

    Returns:
        :class:`EvalMetrics` özeti.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = int(targets.shape[0])
        running_loss += float(loss) * batch_size
        predictions = logits.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        seen += batch_size
        np.add.at(
            confusion,
            (targets.detach().cpu().numpy(), predictions.detach().cpu().numpy()),
            1,
        )

    uar, macro_f1, recalls = _metrics_from_confusion(confusion)
    return EvalMetrics(
        loss=running_loss / max(seen, 1),
        accuracy=correct / max(seen, 1),
        uar=uar,
        macro_f1=macro_f1,
        confusion=confusion,
        recalls=recalls,
    )


def _confusion_summary(confusion: np.ndarray, class_names: Sequence[str]) -> str:
    """Confusion matrix'i okunaklı bir tabloya çevirir."""
    name_width = max((len(name) for name in class_names), default=6)
    header = "".ljust(name_width + 12) + "".join(
        name.rjust(max(6, len(name) + 2)) for name in class_names
    )
    lines = [header]
    for index, name in enumerate(class_names):
        row = "gerçek ".ljust(7) + name.ljust(name_width + 5)
        row += "".join(
            str(int(value)).rjust(max(6, len(column) + 2))
            for value, column in zip(confusion[index], class_names)
        )
        lines.append(row)
    return "\n".join(lines)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    class_names: Sequence[str],
    args: argparse.Namespace,
    epoch: int,
    accuracy: float,
    uar: float | None = None,
    macro_f1: float | None = None,
) -> None:
    """Checkpoint'i atomik olarak (geçici dosya + replace) kaydeder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(class_names),
            "num_classes": len(class_names),
            "temporal_length": args.temporal_length,
            "image_size": args.image_size,
            "epoch": epoch,
            "accuracy": accuracy,
            "uar": uar,
            "macro_f1": macro_f1,
            "config": vars(args),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def _write_class_names(path: Path, class_names: Sequence[str]) -> None:
    """Sınıf adlarını satır başına bir ad olacak biçimde yazar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def _export_onnx(
    model: STSTNet, output_path: Path, args: argparse.Namespace
) -> bool:
    """ONNX modelini dışa aktarır; başarı durumunu döndürür."""
    try:
        model.export_onnx(
            output_path,
            temporal_length=args.temporal_length,
            spatial_size=args.image_size,
            fp16=args.fp16,
        )
        _cleanup_stale_external_data(output_path)
        return True
    except Exception:
        logger.exception(
            "ONNX dışa aktarımı başarısız oldu. .pt ağırlığı korundu; "
            "yeniden denemek için --no-export kaldırın veya onnx paketini "
            "kontrol edin."
        )
        return False


def _cleanup_stale_external_data(onnx_path: Path) -> None:
    """Yeni model dış veri kullanmıyorsa eski ``.onnx.data`` dosyasını siler.

    Küçük modeller tek dosyaya gömülür; önceki dışa aktarımdan kalan
    ``ststnet.onnx.data`` gibi artıklar kafa karıştırmasın.
    """
    data_path = Path(str(onnx_path) + ".data")
    if not data_path.exists():
        return
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=False)
        uses_external = any(
            tensor.data_location == onnx.TensorProto.EXTERNAL
            for tensor in model.graph.initializer
        )
    except Exception:  # pragma: no cover - onnx yoksa dokunma
        return
    if not uses_external:
        data_path.unlink(missing_ok=True)
        logger.info("Eski ONNX dış veri dosyası temizlendi: %s", data_path)


# ---------------------------------------------------------------------- #
# Ana akış
# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Eğitimi çalıştırır, ağırlıkları kaydeder ve ONNX'e aktarır."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.epochs < 1:
        logger.error("--epochs en az 1 olmalıdır.")
        return 2
    if args.batch_size < 1:
        logger.error("--batch-size en az 1 olmalıdır.")
        return 2

    try:
        class_names = _parse_classes(args.classes)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        device = _resolve_device(args.device)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    if args.image_size != 128:
        logger.warning(
            "Canlı çıkarım optical flow boyutu 128x128'dir; image_size=%d "
            "seçildiğinde main.py klip boyutuyla uyuşmayabilir.",
            args.image_size,
        )
    if args.temporal_length != DEFAULT_TEMPORAL_LENGTH:
        logger.warning(
            "Spotting ring buffer %d kare üretir; temporal_length=%d eğitim "
            "dağılımından farklı olabilir.",
            DEFAULT_TEMPORAL_LENGTH,
            args.temporal_length,
        )

    _set_seed(args.seed)
    logger.info(
        "Eğitim başlıyor | cihaz=%s | sınıflar=%s | T=%d | %dx%d",
        device,
        ", ".join(class_names),
        args.temporal_length,
        args.image_size,
        args.image_size,
    )

    try:
        train_dataset, val_dataset = _build_datasets(args, class_names)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if args.dummy else args.num_workers,
        device=device,
    )
    val_loader = (
        _make_loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0 if args.dummy else args.num_workers,
            device=device,
        )
        if val_dataset is not None
        else None
    )

    model = STSTNet(num_classes=len(class_names)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "STSTNet hazır: %.1fK parametre | %d sınıf",
        parameter_count / 1_000.0,
        len(class_names),
    )

    train_labels = list(getattr(train_dataset, "labels", []))
    if train_labels:
        class_weight = _class_weights(train_labels, len(class_names), device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        logger.warning("Sınıf ağırlıkları hesaplanamadı; dengelenmemiş kayıp.")
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    weights_path = Path(args.out)
    best_metric = -1.0
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: EvalMetrics | None = None
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs,
        )

        if val_loader is not None:
            metrics = evaluate(
                model, val_loader, criterion, device, len(class_names)
            )
            metric = metrics.uar
            extra = (
                f" | val loss={metrics.loss:.4f} acc={metrics.accuracy:.3f} "
                f"UAR={metrics.uar:.3f} macroF1={metrics.macro_f1:.3f}"
            )
        else:
            metrics = None
            metric = train_accuracy
            extra = ""

        scheduler.step()
        elapsed = time.perf_counter() - epoch_started
        logger.info(
            "epoch %d/%d tamamlandı | train loss=%.4f acc=%.3f%s | %.1f s",
            epoch,
            args.epochs,
            train_loss,
            train_accuracy,
            extra,
            elapsed,
        )

        # En iyi checkpoint UAR skoruna göre seçilir ve anında kaydedilir.
        if metric > best_metric:
            best_metric = metric
            best_accuracy = metrics.accuracy if metrics is not None else train_accuracy
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            _save_checkpoint(
                weights_path,
                model,
                class_names=class_names,
                args=args,
                epoch=best_epoch,
                accuracy=best_accuracy,
                uar=metrics.uar if metrics is not None else None,
                macro_f1=metrics.macro_f1 if metrics is not None else None,
            )
            logger.info(
                "Yeni en iyi checkpoint kaydedildi: %s (epoch=%d, UAR=%.3f)",
                weights_path,
                best_epoch,
                best_metric,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    _save_checkpoint(
        weights_path,
        model,
        class_names=class_names,
        args=args,
        epoch=best_epoch or args.epochs,
        accuracy=best_accuracy,
        uar=best_metrics.uar if best_metrics is not None else None,
        macro_f1=best_metrics.macro_f1 if best_metrics is not None else None,
    )
    classes_path = weights_path.with_suffix(".classes.txt")
    _write_class_names(classes_path, class_names)
    logger.info(
        "Ağırlıklar kaydedildi: %s (en iyi epoch=%d, UAR=%.3f)",
        weights_path,
        best_epoch or args.epochs,
        best_metric,
    )

    if best_metrics is not None:
        recall_parts: list[str] = []
        for name, recall in zip(class_names, best_metrics.recalls):
            value = "nan" if np.isnan(recall) else f"{recall:.3f}"
            recall_parts.append(f"{name}={value}")
        recall_text = " | ".join(recall_parts)
        logger.info(
            "Doğrulama sınıf bazlı recall (epoch=%d): %s",
            best_epoch,
            recall_text,
        )
        logger.info(
            "Doğrulama UAR=%.3f | Macro-F1=%.3f | Accuracy=%.3f",
            best_metrics.uar,
            best_metrics.macro_f1,
            best_metrics.accuracy,
        )
        logger.info(
            "Confusion matrix (satır=gerçek, sütun=tahmin):\n%s",
            _confusion_summary(best_metrics.confusion, class_names),
        )

    export_ok = True
    if args.no_export:
        logger.info("ONNX dışa aktarımı atlandı (--no-export).")
    else:
        onnx_path = Path(args.onnx)
        export_ok = _export_onnx(model, onnx_path, args)
        if export_ok:
            if onnx_path.with_suffix(".classes.txt") != classes_path:
                _write_class_names(onnx_path.with_suffix(".classes.txt"), class_names)
            logger.info("ONNX modeli hazır: %s", onnx_path)

    total_elapsed = time.perf_counter() - started
    logger.info(
        "Eğitim tamamlandı | %.1f s | en iyi UAR=%.3f | doğruluk=%.3f | "
        "sınıflar: %s",
        total_elapsed,
        best_metric,
        best_accuracy,
        ", ".join(class_names),
    )
    if export_ok and not args.no_export:
        logger.info(
            "Canlı çalıştırma: python main.py  (models/ststnet.onnx otomatik "
            "yüklenir; --weights ile başka dosya seçilebilir)"
        )
    return 0 if export_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
