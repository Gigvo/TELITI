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
# The model artifacts ARE baked in, because a managed host (Railway, Render, Fly,
# HF Spaces) has no volume to mount and `artifacts/` is gitignored — a model that
# is not in the image is not on the server. The training checkpoint is excluded
# in .dockerignore; only the ~545 MB of inference weights ship.
#
# This means the image must be built on a machine that HAS the artifacts. If your
# host builds from the git repo instead, it will produce an image whose /health
# reports model_loaded=false. See docs/DEPLOYMENT.md.

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

# The user is created BEFORE the large copies so each COPY can set ownership as
# it writes. A `chown -R` afterwards would rewrite every file into a fresh layer,
# and since layers are additive that means shipping the 545 MB of weights twice —
# once owned by root, once by teliti. The image nearly doubles for no reason.
RUN useradd --create-home --uid 10001 teliti

# gosu drops privileges in the entrypoint. The container starts as root purely to
# fix ownership of a mounted volume, then hands off to `teliti` — see
# docker-entrypoint.sh for why that ordering is forced on us.
#
# `su` would work but stays in the process tree as PID 1, swallowing SIGTERM;
# gosu execs and gets out of the way.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gosu \
 && rm -rf /var/lib/apt/lists/* \
 && gosu nobody true

COPY requirements.txt ./

# torch first, from the CPU index. On Linux the default PyPI wheel is the CUDA
# build — ~2.5 GB of GPU kernels this container will never run. Installing it
# ahead of the rest means the later resolve sees torch already satisfied and does
# not pull the CUDA one back in as a transitive dependency of transformers.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Weights before source. This is the largest layer (~545 MB) and the one that
# changes least often; a code edit below it then rebuilds in seconds instead of
# re-sending half a gigabyte. Reversing these two lines is the difference between
# a 10-second rebuild and a 5-minute one.
COPY --chown=teliti:teliti artifacts/ ./artifacts/

COPY --chown=teliti:teliti api/ ./api/
COPY --chown=teliti:teliti ml/ ./ml/
COPY --chown=teliti:teliti data/reference/ ./data/reference/

# Locale resources live in data/reference and are discovered at start-up. An image
# built without them still runs — the affected rules report themselves unassessed
# rather than silently clean (see api/rules/base.py).

COPY --chown=teliti:teliti --from=web /build/dist ./web_dist
# --chmod because the build context is Windows, which has no execute bit to
# preserve. Without it the copied script lands as 0644 and the container dies at
# start with "permission denied".
COPY --chown=teliti:teliti --chmod=0755 docker-entrypoint.sh /app/

# Appeals are appended here. Ownership is set at build time for the no-volume
# case; when a volume IS mounted it arrives root-owned and the entrypoint fixes
# it, because a mount replaces whatever the image had at that path.
#
# No VOLUME instruction: it would make every `docker run` create an anonymous
# volume even in local testing, and it does not help the mounted case at all.
RUN mkdir -p /app/data/reports && chown -R teliti:teliti /app/data

# NOTE: no `USER teliti` here. The entrypoint starts as root, chowns any mounted
# volume, and then drops to teliti via gosu before exec'ing uvicorn. The server
# process never runs as root — verify with `docker exec <id> ps -o user,args`.

EXPOSE 8000

# /health reports model_loaded and locales_available, so an unhealthy container is
# distinguishable from a merely-stubbed one.
#
# start-period is 90s, not 10s: the model is loaded eagerly at start-up and 541 MB
# of weights take ~30s to read on a slow managed disk. At 10s the probe fails
# three times before the app has ever been ready, the platform concludes the
# container is broken, restarts it, and the same thing happens forever. Failures
# during start-period do not count against the retry budget.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
