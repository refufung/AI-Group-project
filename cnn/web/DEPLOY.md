# Live demo deploy

## Quick public URL (this Mac — Cloudflare Tunnel)

Requires the unified app running locally on port 8000 (API + built UI):

```bash
cd cnn/web/frontend && VITE_API_URL= npm run build
cd ../backend
source ../../.venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8000
# other terminal:
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflare prints a `https://*.trycloudflare.com` link — share that.

## Permanent hosting (Docker)

Build from `cnn/` (parent of `web/`) with a fixed Dockerfile context — see `Dockerfile` notes.
Suggested hosts: [Render](https://render.com), [Railway](https://railway.app), [Fly.io](https://fly.io).

Vercel alone cannot run the PyTorch API (size / runtime limits). Use Docker for the full stack, or Vercel (frontend only) + a separate API host.

## Vercel (frontend only)

```bash
cd cnn/web/frontend
vercel login
VITE_API_URL=https://YOUR-API-HOST npm run build
vercel --prod
```

Point `VITE_API_URL` at your deployed FastAPI URL.
