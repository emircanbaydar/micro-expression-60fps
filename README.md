# Micro-Expression 60 FPS

Real-time facial micro-expression spotting and classification (negative, positive, surprise) powered by MediaPipe, optical flow, and STSTNet (ONNX).

## Run Locally
`ash
pip install -r requirements.txt
python main.py
`

## Run FastAPI Server
`ash
pip install -r requirements-server.txt
uvicorn api:app --host 0.0.0.0 --port 8000
`
