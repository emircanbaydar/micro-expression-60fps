"""Mikro mimik analizi için FastAPI servisi.

Mevcut gerçek zamanlı boru hattını (``FaceAligner`` → ``OpticalFlowExtractor``
→ ``SpottingEngine`` → ONNX ``ExpressionRecognizer``) HTTP ve WebSocket
üzerinden sunar. Model, uygulama yaşam döngüsünde (``lifespan``) **bir kez**
ONNX Runtime ile yüklenir; PyTorch bağımlılığı kullanılmaz.

Uç noktalar:
    * ``GET /health`` — model durumu, sınıf listesi ve ONNX provider'ı.
    * ``POST /analyze-video`` — yüklenen videoyu kare kare işler, tespit
      edilen mikro mimikleri ve video meta verilerini döndürür.
    * ``WebSocket /ws/stream`` — istemciden gelen JPEG/PNG karelerini
      gerçek zamanlı işler; yeni bir spot tamamlandığında JSON gönderir.

Çalıştırma::

    uvicorn api:app --host 0.0.0.0 --port 8000

    # veya
    python api.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import ExpressionRecognizer
from src.preprocessing.face_aligner import FaceAligner
from src.preprocessing.optical_flow import FlowResult, OpticalFlowExtractor
from src.pipeline.spotting_engine import SpotEvent, SpottingEngine

logger = logging.getLogger("api")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = BASE_DIR / "models" / "ststnet.onnx"
DEFAULT_CLASSES = BASE_DIR / "models" / "ststnet.classes.txt"
_FALLBACK_CLASSES = ("negative", "positive", "surprise")
_UPLOAD_CHUNK = 1024 * 1024


# ---------------------------------------------------------------------- #
# Yanıt modelleri
# ---------------------------------------------------------------------- #
class SpotPrediction(BaseModel):
    """Tespit edilen tek bir mikro mimik ve sınıflandırma sonucu."""

    spot_id: int = Field(..., description="İstek/akış içindeki 1 tabanlı spot sırası")
    onset_ms: int
    apex_ms: int
    offset_ms: int
    label: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    """``GET /health`` yanıtı."""

    status: str
    ready: bool
    classes: list[str]
    provider: str | None = None
    model: str | None = None


class AnalyzeResponse(BaseModel):
    """``POST /analyze-video`` yanıtı."""

    ready: bool
    classes: list[str]
    spots: list[SpotPrediction]
    fps: float
    total_frames: int
    processing_ms: float


# ---------------------------------------------------------------------- #
# Model yükleme
# ---------------------------------------------------------------------- #
def _read_class_names(path: Path) -> list[str]:
    """Sınıf adlarını sidecar dosyasından okur; yoksa varsayılana düşer."""
    if path.is_file():
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if names:
            return names
    logger.warning("Sınıf dosyası okunamadı (%s); varsayılan sınıflar kullanılıyor.", path)
    return list(_FALLBACK_CLASSES)


def _provider_name(recognizer: ExpressionRecognizer) -> str | None:
    """``ExpressionRecognizer.backend`` (``onnx:<provider>``) değerinden provider'ı çıkarır."""
    backend = recognizer.backend
    return backend.split(":", 1)[1] if ":" in backend else backend


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Sunucu açılışında ONNX modelini ve sınıf adlarını bir kez yükler."""
    class_names = _read_class_names(DEFAULT_CLASSES)
    recognizer: ExpressionRecognizer | None = None
    try:
        if not DEFAULT_ONNX.is_file():
            raise FileNotFoundError(f"ONNX modeli bulunamadı: {DEFAULT_ONNX}")
        recognizer = ExpressionRecognizer(DEFAULT_ONNX, class_names, device="auto")
        logger.info(
            "API modeli hazır: %s | backend=%s | sınıflar: %s",
            DEFAULT_ONNX,
            recognizer.backend,
            ", ".join(class_names),
        )
    except Exception:
        logger.exception(
            "ONNX modeli yüklenemedi; /health ready=false dönecek ve analiz "
            "uç noktaları 503 verecek."
        )
        recognizer = None

    app.state.recognizer = recognizer
    app.state.class_names = class_names
    app.state.provider = _provider_name(recognizer) if recognizer is not None else None
    try:
        yield
    finally:
        app.state.recognizer = None


app = FastAPI(
    title="Mikro Mimik Analiz API",
    description="60 FPS mikro mimik spotting + STSTNet ONNX sınıflandırma servisi",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------- #
# Ortak kare boru hattı
# ---------------------------------------------------------------------- #
class FramePipeline:
    """Tek bir video/akış için hizalama → flow → spotting → sınıflandırma.

    Durum tuttuğu için (MediaPipe takibi, optical flow referansı, spotting
    buffer'ı) her istek veya WebSocket bağlantısı kendi örneğini oluşturur.
    """

    def __init__(
        self,
        recognizer: ExpressionRecognizer,
        class_names: list[str],
        *,
        mask_eyes: bool = True,
    ) -> None:
        self.recognizer = recognizer
        self.class_names = list(class_names)
        self.aligner = FaceAligner()
        self.flow = OpticalFlowExtractor(mask_eyes=mask_eyes)
        self.spotter = SpottingEngine()
        self.prev_gray: np.ndarray | None = None

    def process(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> tuple[FlowResult | None, SpotEvent | None]:
        """Bir BGR kareyi işler; tamamlanan spot olayını (varsa) döndürür."""
        align_result = self.aligner.process(frame, timestamp_ms)

        flow_result: FlowResult | None = None
        if align_result.detected and align_result.aligned is not None:
            gray = OpticalFlowExtractor.to_gray(align_result.aligned)
            flow_result = self.flow.extract(
                self.prev_gray, gray, align_result.landmarks
            )
            self.prev_gray = gray
        else:
            # Yüz kaybolduğunda akış referansı sıfırlanır (main.py ile aynı).
            self.prev_gray = None

        event = self.spotter.update(flow_result, timestamp_ms)
        return flow_result, event

    def predict(self, event: SpotEvent, spot_id: int) -> SpotPrediction:
        """Spot klibini STSTNet ile sınıflandırıp yanıt modeline çevirir."""
        recognition = self.recognizer.predict(event.to_model_input())
        probabilities = {
            self.class_names[index]
            if index < len(self.class_names)
            else f"class_{index}": float(probability)
            for index, probability in enumerate(recognition.probabilities)
        }
        return SpotPrediction(
            spot_id=spot_id,
            onset_ms=int(event.onset_ms),
            apex_ms=int(event.apex_ms),
            offset_ms=int(event.offset_ms),
            label=recognition.label,
            confidence=float(recognition.confidence),
            probabilities=probabilities,
        )

    def close(self) -> None:
        """MediaPipe kaynaklarını serbest bırakır."""
        self.aligner.close()


def _processing_fps(capture: cv2.VideoCapture) -> float:
    """Video FPS değerini okur; geçersizse 30.0 döndürür."""
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 0.0 and np.isfinite(fps) else 30.0


def _analyze_video_file(
    path: Path, recognizer: ExpressionRecognizer, class_names: list[str]
) -> dict[str, Any]:
    """Video dosyasını kare kare boru hattından geçirir (senkron, thread'de çağrılır)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Video açılamadı: {path}")

    fps = _processing_fps(capture)
    pipeline = FramePipeline(recognizer, class_names)
    spots: list[SpotPrediction] = []
    frame_index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            _, event = pipeline.process(frame, timestamp_ms)
            if event is not None:
                spots.append(pipeline.predict(event, len(spots) + 1))
            frame_index += 1
    finally:
        capture.release()
        pipeline.close()

    return {
        "ready": True,
        "classes": list(class_names),
        "spots": [spot.model_dump() for spot in spots],
        "fps": round(fps, 3),
        "total_frames": frame_index,
        "processing_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


# ---------------------------------------------------------------------- #
# Uç noktalar
# ---------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, summary="Servis ve model durumu")
def health(request: Request) -> HealthResponse:
    """Model hazır mı, sınıflar ve ONNX provider bilgisini döndürür."""
    recognizer: ExpressionRecognizer | None = request.app.state.recognizer
    return HealthResponse(
        status="ok" if recognizer is not None else "degraded",
        ready=recognizer is not None,
        classes=list(request.app.state.class_names),
        provider=request.app.state.provider,
        model=str(DEFAULT_ONNX) if recognizer is not None else None,
    )


@app.post(
    "/analyze-video",
    response_model=AnalyzeResponse,
    summary="Yüklenen videodaki mikro mimikleri analiz et",
)
async def analyze_video(
    request: Request, file: UploadFile = File(..., description="Video dosyası")
) -> AnalyzeResponse:
    """Yüklenen videoyu geçici dosyaya alıp boru hattından geçirir.

    Tespit edilen her mikro mimik için zaman aralığı, etiket, güven ve sınıf
    olasılıkları döndürülür. Geçici dosya her durumda silinir.
    """
    recognizer: ExpressionRecognizer | None = request.app.state.recognizer
    if recognizer is None:
        raise HTTPException(status_code=503, detail="ONNX modeli hazır değil.")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    file_descriptor, temp_name = tempfile.mkstemp(prefix="mex_", suffix=suffix)
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as handle:
            while chunk := await file.read(_UPLOAD_CHUNK):
                handle.write(chunk)
        if temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Yüklenen dosya boş.")

        try:
            result = await asyncio.to_thread(
                _analyze_video_file,
                temp_path,
                recognizer,
                list(request.app.state.class_names),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()

    return AnalyzeResponse(**result)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """JPEG/PNG karelerini gerçek zamanlı işleyen WebSocket akışı.

    İstemci her kareyi ikili (bytes) olarak gönderir. Yeni bir spot
    tamamlandığında ``{"type": "spot", ...}`` JSON'u döner; bağlantı
    kurulduğunda ``{"type": "connected", ...}`` bilgisi gönderilir.
    """
    await websocket.accept()
    recognizer: ExpressionRecognizer | None = websocket.app.state.recognizer
    class_names = list(websocket.app.state.class_names)

    if recognizer is None:
        await websocket.send_json(
            {"type": "error", "message": "ONNX modeli hazır değil."}
        )
        await websocket.close(code=1011)
        return

    await websocket.send_json(
        {
            "type": "connected",
            "ready": True,
            "classes": class_names,
            "provider": websocket.app.state.provider,
        }
    )

    pipeline = FramePipeline(recognizer, class_names)
    started = time.perf_counter()
    spot_count = 0
    try:
        while True:
            payload = await websocket.receive_bytes()
            frame = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                await websocket.send_json(
                    {"type": "error", "message": "Kare çözümlenemedi (JPEG/PNG bekleniyor)."}
                )
                continue

            timestamp_ms = int((time.perf_counter() - started) * 1000.0)
            _, event = await asyncio.to_thread(
                pipeline.process, frame, timestamp_ms
            )
            if event is None:
                continue

            spot_count += 1
            prediction = await asyncio.to_thread(pipeline.predict, event, spot_count)
            await websocket.send_json({"type": "spot", **prediction.model_dump()})
    except WebSocketDisconnect:
        logger.info("WebSocket istemcisi ayrıldı (%d spot).", spot_count)
    except Exception:
        logger.exception("WebSocket akışı beklenmeyen hata ile kapandı.")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass
    finally:
        pipeline.close()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
