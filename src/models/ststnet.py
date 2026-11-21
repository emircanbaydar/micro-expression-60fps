"""STSTNet (Shallow Triple Stream 3D-CNN) PyTorch mimarisi.

Model, mikro mimik kliplerini ``(B, 3, T, H, W)`` biçiminde bekler:

* kanal 0: yatay optical flow ``u``
* kanal 1: dikey optical flow ``v``
* kanal 2: optical strain

Her akış bileşeni kendi sığ 3D-CNN dalında (triple stream) işlenir, elde
edilen özellikler birleştirilir ve doğrusal sınıflandırıcıya verilir. Uzamsal
boyut :class:`torch.nn.AdaptiveAvgPool3d` ile toplandığı için model farklı
ROI çözünürlükleriyle (örn. 256x256) çalışabilir; zaman boyutu ``T`` korunur.

Canlı çıkarım için :func:`build_ststnet` ile ``.pt`` ağırlıkları yüklenebilir;
:meth:`STSTNet.export_onnx` ile ONNX (isteğe bağlı FP16) modeli üretilebilir.
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

IN_CHANNELS = 3
_STREAM_CHANNELS = (16, 32)
_FEATURE_DIM = _STREAM_CHANNELS[-1] * IN_CHANNELS
_CHECKPOINT_KEYS = ("model_state_dict", "state_dict", "model")


class _TripleStream(nn.Module):
    """Tek bir flow bileşeni için sığ 3D-CNN dalı.

    Girdi ``(B, 1, T, H, W)``; çıktı uzamsal/zamansal olarak havuzlanmış
    ``(B, 32)`` özellik vektörüdür. Zaman boyutu küçültülmez, çünkü 16 karelik
    klibin tamamı temsil edilmelidir.
    """

    def __init__(self) -> None:
        super().__init__()
        first, second = _STREAM_CHANNELS
        self.features = nn.Sequential(
            nn.Conv3d(1, first, kernel_size=3, padding=1),
            nn.BatchNorm3d(first),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(first, second, kernel_size=3, padding=1),
            nn.BatchNorm3d(second),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, 1, T, H, W)`` girdisini ``(B, 32)`` özelliğe indirger."""
        return self.pool(self.features(x)).flatten(1)


class STSTNet(nn.Module):
    """Shallow Triple Stream 3D-CNN (STSTNet).

    Args:
        num_classes: Sınıf (duygu / AU kombinasyonu) sayısı.
        dropout: Sınıflandırıcı öncesi dropout oranı.
        in_channels: Akış kanalı sayısı; mimari gereği 3 (u, v, strain).
    """

    def __init__(
        self, num_classes: int = 3, dropout: float = 0.5, in_channels: int = IN_CHANNELS
    ) -> None:
        super().__init__()
        if in_channels != IN_CHANNELS:
            raise ValueError(f"STSTNet {IN_CHANNELS} kanal bekler (u, v, strain).")
        if num_classes < 1:
            raise ValueError("num_classes en az 1 olmalıdır.")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.streams = nn.ModuleList(
            [_TripleStream() for _ in range(IN_CHANNELS)]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(_FEATURE_DIM, num_classes),
        )
        self._init_weights()

    # ------------------------------------------------------------------ #
    # İleri/geri yayılım
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        """Klibi sınıf logit'lerine dönüştürür.

        Args:
            x: ``(B, 3, T, H, W)`` veya tek örnek için ``(3, T, H, W)``
                tensör. ``float32`` (veya CPU'da ``float16``) olmalıdır.

        Returns:
            ``(B, num_classes)`` logit tensörü.

        Raises:
            ValueError: Girdi boyutu veya kanal sayısı uyumsuzsa.
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() != 5:
            raise ValueError(
                f"Beklenen girdi (B, 3, T, H, W); alınan: {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Beklenen {self.in_channels} kanal; alınan {x.shape[1]}."
            )

        features = [
            stream(x[:, index : index + 1]) for index, stream in enumerate(self.streams)
        ]
        return self.classifier(torch.cat(features, dim=1))

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Softmax olasılıkları döndürür (değerlendirme/çıkarım)."""
        was_training = self.training
        self.eval()
        try:
            return torch.softmax(self.forward(x), dim=1)
        finally:
            self.train(was_training)

    # ------------------------------------------------------------------ #
    # Dışa aktarım
    # ------------------------------------------------------------------ #
    def export_onnx(
        self,
        output_path: str | Path,
        *,
        temporal_length: int = 16,
        spatial_size: int = 256,
        opset_version: int = 18,
        fp16: bool = False,
    ) -> Path:
        """Modeli ONNX Runtime ile çalıştırılabilecek biçimde dışa aktarır.

        PyTorch'un güncel sürümleri opset 17'de kaldırılan/taşınan semboller
        için uyarı üretir; bu nedenle varsayılan opset 18'dir. İstenen opset
        yüklü PyTorch tarafından desteklenmiyorsa en yüksek kararlı opset'e
        otomatik geri düşülür.

        Args:
            output_path: Yazılacak ``.onnx`` dosyası.
            temporal_length: Örnek girdideki kare sayısı.
            spatial_size: Örnek girdinin uzamsal kenar uzunluğu.
            opset_version: ONNX opset sürümü (varsayılan 18).
            fp16: Yarım duyarlıklı (FP16) model dışa aktar.

        Returns:
            Yazılan dosyanın yolu.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        model = copy.deepcopy(self).cpu().eval()
        dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            model = model.half()
        dummy = torch.randn(
            1,
            IN_CHANNELS,
            temporal_length,
            spatial_size,
            spatial_size,
            dtype=dtype,
        )

        def _export(opset: int) -> None:
            # ``dynamo=False``: TorchScript tabanlı kararlı dışa aktarıcı.
            # PyTorch 2.9+ varsayılanı olan yeni exporter, bazı konsol
            # kodlamalarında (ör. Windows cp1254) Unicode nedeniyle çöker ve
            # ``dynamic_axes`` desteği farklıdır; bu modelde legacy yol
            # güvenilirdir. Opset 18 ile opset 17 uyarısı ortadan kalkar.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    model,
                    dummy,
                    str(output),
                    input_names=["input"],
                    output_names=["logits"],
                    opset_version=opset,
                    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                    dynamo=False,
                    verbose=False,
                )

        try:
            _export(opset_version)
        except (RuntimeError, ValueError) as exc:
            fallback = _max_stable_opset()
            if fallback <= opset_version:
                raise
            logger.warning(
                "ONNX opset %d dışa aktarılamadı (%s); en yüksek kararlı "
                "opset %d ile yeniden denenecek.",
                opset_version,
                exc,
                fallback,
            )
            _export(fallback)
            opset_version = fallback

        logger.info(
            "ONNX modeli dışa aktarıldı: %s (opset=%d, fp16=%s)",
            output,
            opset_version,
            fp16,
        )
        if fp16:
            logger.info(
                "FP16 modeli paylaşırken onnxruntime-gpu (CUDAExecutionProvider) "
                "kullanılması önerilir."
            )
        return output

    def _init_weights(self) -> None:
        """Kaiming/BN/doğrusal katman başlangıç ağırlıkları."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)


def _max_stable_opset() -> int:
    """Yüklü PyTorch'un desteklediği en yüksek ONNX opset'ini döndürür."""
    try:
        from torch.onnx import _constants  # type: ignore[attr-defined]

        return int(getattr(_constants, "ONNX_MAX_OPSET", 20))
    except Exception:  # pragma: no cover - PyTorch sürümüne bağlı
        return 20


def build_ststnet(
    weights_path: str | Path | None = None,
    *,
    num_classes: int = 3,
    device: str | torch.device = "cpu",
) -> STSTNet:
    """STSTNet oluşturur ve varsa ağırlıkları yükler.

    Args:
        weights_path: ``.pt``/``.pth`` ağırlık dosyası. Ham ``state_dict`` ya
            da ``model_state_dict``/``state_dict``/``model`` anahtarlarından
            birini içeren sözlük olabilir.
        num_classes: Modelin sınıf sayısı.
        device: ``"cpu"``, ``"cuda"`` veya ``torch.device``.

    Returns:
        Değerlendirme moduna alınmış model.

    Raises:
        RuntimeError: Ağırlıklar mimariyle uyuşmazsa.
    """
    model = STSTNet(num_classes=num_classes).to(device)

    if weights_path is not None:
        path = Path(weights_path)
        state = _load_checkpoint(path, device)
        if isinstance(state, dict):
            for key in _CHECKPOINT_KEYS:
                candidate = state.get(key)
                if isinstance(candidate, dict):
                    state = candidate
                    break
        try:
            model.load_state_dict(state)
        except (RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"Ağırlıklar mimariyle uyuşmuyor ({path}): {exc}. "
                f"--classes sayısı modelin çıkış boyutuyla eşleşmeli."
            ) from exc
        logger.info("STSTNet ağırlıkları yüklendi: %s", path)

    model.eval()
    return model


def _load_checkpoint(path: Path, device: str | torch.device) -> object:
    """Checkpoint'i güvenli (weights_only) biçimde yükler.

    ``weights_only=True`` desteklenmeyen eski PyTorch sürümlerinde veya
    tensör dışı meta veri içeren güvenilir dosyalarda tam yükleme denenir.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # torch < 2.0: weights_only parametresi yok.
        return torch.load(path, map_location=device)
    except Exception as exc:
        logger.warning(
            "weights_only=True ile yüklenemedi (%s); tam pickle yüklemesi deneniyor.",
            exc,
        )
        return torch.load(path, map_location=device, weights_only=False)
