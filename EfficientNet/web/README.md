# Dog Breed EfficientNet — Web UI

Local React frontend + FastAPI backend for the trained EfficientNet-B0
(`EfficientNet/outputs/best_efficientnet_b0.pt`).

## Quick start (single port)

```bash
cd EfficientNet/web
chmod +x start.sh
./start.sh
```

Open: http://127.0.0.1:8000

## Manual

### Backend

```bash
cd EfficientNet
source .venv/bin/activate
pip install -r web/backend/requirements.txt
cd web/frontend && npm install && npm run build && cd ../backend
uvicorn app:app --host 127.0.0.1 --port 8000
```

### Dev frontend (hot reload on 5173)

```bash
cd EfficientNet/web/frontend
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

## Usage

- **Single image:** choose one `.jpg` / `.png` → Predict
- **Batch / folder:** pick a folder with breed-named subfolders for accuracy
