# Single-container deploy: React UI + FastAPI + PyTorch CNN
FROM python:3.11-slim-bookworm AS frontend
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_URL=
RUN npm run build

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
  && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY backend/ /app/backend/
COPY --from=frontend /app/frontend/dist /app/frontend/dist
# Checkpoint + split stats (required at runtime)
COPY ../outputs/best_small_cnn.pt /app/outputs/best_small_cnn.pt
COPY ../outputs/split_meta.json /app/outputs/split_meta.json

# Remap paths expected by app.py (CNN_ROOT = parents[2] from backend/app.py → /app)
# backend is at /app/backend, parents[2] = /  — adjust layout:
# Put tree as /app/cnn/web/backend and /app/cnn/outputs
WORKDIR /srv
RUN mkdir -p /srv/cnn/web /srv/cnn/outputs \
  && mv /app/backend /srv/cnn/web/backend \
  && mv /app/frontend /srv/cnn/web/frontend \
  && mv /app/outputs/best_small_cnn.pt /srv/cnn/outputs/ \
  && mv /app/outputs/split_meta.json /srv/cnn/outputs/

WORKDIR /srv/cnn/web/backend
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
