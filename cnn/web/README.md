# Dog Breed CNN — Web UI

Local React frontend + FastAPI backend for the trained SmallCNN
(`cnn/outputs/best_small_cnn.pt`).

## Prerequisites

- Node.js 18+
- Python venv at `cnn/.venv` (already has PyTorch)

## 1. Backend (port 8000)

```bash
cd cnn
source .venv/bin/activate
pip install -r web/backend/requirements.txt
cd web/backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Check: http://127.0.0.1:8000/health

## 2. Frontend (port 5173)

```bash
cd cnn/web/frontend
npm install
npm run dev
```

Open: http://127.0.0.1:5173

## Or use the helper script

```bash
cd cnn/web
chmod +x start.sh
./start.sh
```

## Usage

- **Single image:** choose one `.jpg` / `.png` → Predict
- **Batch / folder:** choose `cnn/input_dog` (or any folder). If files live under
  breed-named subfolders (`Cardigan/…`, `French_Bulldog/…`), the UI shows
  accuracy against those folder labels.

API base URL defaults to `http://127.0.0.1:8000`. Override with:

```bash
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```
