#!/bin/sh
#
# TELITI container entrypoint.
#
# Two problems are solved here that a Dockerfile CMD cannot solve alone.
#
# --- 1. Managed hosts choose the port ---------------------------------------
#
# Railway, Render, Cloud Run and Heroku inject $PORT and route external traffic
# to it. A container hardcoding 8000 listens on the wrong socket, the platform's
# probe never connects, and the deploy fails with "no open ports detected" —
# which reads like a networking problem and is actually this.
#
# `CMD ["uvicorn", ..., "--port", "8000"]` in exec form cannot fix it: exec form
# runs no shell, so $PORT would reach uvicorn as five literal characters.
#
# Hugging Face Spaces is the exception — it expects the port named by `app_port`
# in the Space README and sets no $PORT. deploy/huggingface-space-README.md
# declares 8000 to match the default below.
#
# --- 2. Mounted volumes arrive owned by root --------------------------------
#
# Appeals (POST /api/v1/report) append to a JSONL file. On a host with
# persistent storage that path is a mount, and a mount replaces whatever the
# image had there — including its ownership. The server runs as uid 10001, so
# without the chown below every appeal fails with PermissionError and returns
# 500, after the UI has already told the user their report was received.
#
# That ownership fix requires root, which is why this script starts as root and
# drops privileges itself rather than the Dockerfile setting USER.

set -eu

PORT="${PORT:-8000}"
APP_USER="teliti"

# One worker, deliberately.
#
# Each worker is a separate process with its own copy of the model — roughly
# 600 MB resident. Two workers on a 2 GB instance is an out-of-memory kill under
# any real load. Scale by running more containers, not more workers per one.
#
# Inference is CPU-bound and releases the GIL inside torch, so a single worker
# with FastAPI's threadpool already overlaps requests reasonably.
WORKERS="${WEB_CONCURRENCY:-1}"

# Torch defaults to one thread per core and then fights itself: on a shared vCPU
# the threads contend and p95 latency gets worse, not better. Capped unless the
# operator has an opinion.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# Where appeals are written. Kept in step with api/feedback.py's default so the
# directory prepared here is the one actually used.
REPORTS_PATH="${TELITI_REPORTS:-/app/data/reports/corrections.jsonl}"
REPORTS_DIR="$(dirname "${REPORTS_PATH}")"

if [ "$(id -u)" = "0" ]; then
    # Best-effort: a read-only filesystem is a legitimate way to run this if you
    # do not need appeals, so failing to prepare the directory must not stop the
    # service from starting. It is logged loudly instead — a silent 500 on the
    # appeal route is much harder to diagnose than a line in the boot log.
    if mkdir -p "${REPORTS_DIR}" 2>/dev/null && chown -R "${APP_USER}:${APP_USER}" "${REPORTS_DIR}" 2>/dev/null; then
        echo "Appeal storage ready at ${REPORTS_PATH}"
    else
        echo "WARNING: cannot write ${REPORTS_DIR}."
        echo "WARNING: analysis will work; submitting an appeal will return 500."
    fi
    DROP="gosu ${APP_USER}"
else
    # Already non-root: some hosts (and `docker run --user`) impose their own uid.
    # Nothing to fix and no privileges to drop.
    if [ ! -w "${REPORTS_DIR}" ]; then
        echo "WARNING: ${REPORTS_DIR} is not writable by uid $(id -u); appeals will fail."
    fi
    DROP=""
fi

echo "TELITI starting on 0.0.0.0:${PORT} (workers=${WORKERS}, threads=${OMP_NUM_THREADS}, uid=$(id -u))"
echo "Model loads eagerly at start-up; first readiness may take ~30s."

# exec: uvicorn replaces this shell as PID 1, so SIGTERM from the platform
# reaches it directly. Without exec the shell holds PID 1, ignores the signal,
# and every deploy waits out a 30-second kill timeout before dying hard.
#
# ${DROP} is intentionally unquoted — it is either empty or a two-word command,
# and quoting would turn the empty case into an empty argument.
# shellcheck disable=SC2086
exec ${DROP} uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --timeout-keep-alive 65 \
    --proxy-headers \
    --forwarded-allow-ips '*'
