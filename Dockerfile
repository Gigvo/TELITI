# TELITI — single-image deployment (API + built frontend)
#
# Two stages: build the React bundle with Node, then serve it as static files from
# the same FastAPI process. One container, one port, no CORS and no reverse proxy
# to configure — which is what makes a demo survive an unfamiliar network.
#
#   docker build -t teliti .
#   docker run -p 8000:8000 teliti
#   open http://localhost:8000
#
# The model artifacts are NOT baked in. Mount them:
#   docker run -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts:ro" teliti

# ---------------------------------------------------------------------------
# Stage 1 — frontend
# ---------------------------------------------------------------------------
FROM node:22-slim AS web

WORKDIR /build

# Copy manifests first so the dependency layer caches independently of source.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# PYTHONDONTWRITEBYTECODE: no .pyc clutter on a read-only-ish layer.
# PYTHONUNBUFFERED: logs appear immediately, which matters when the only view of
# a failing demo is `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TELITI_STATIC_DIR=/app/web_dist

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY ml/ ./ml/
COPY data/reference/ ./data/reference/

# Locale resources live in data/reference and are discovered at start-up. An image
# built without them still runs — the affected rules report themselves unassessed
# rather than silently clean (see api/rules/base.py).

COPY --from=web /build/dist ./web_dist

# Run as a non-root user. The service takes untrusted text from the internet.
RUN useradd --create-home --uid 10001 teliti && chown -R teliti:teliti /app
USER teliti

EXPOSE 8000

# /health reports model_loaded and locales_available, so an unhealthy container is
# distinguishable from a merely-stubbed one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
