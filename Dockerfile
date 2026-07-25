# ── Stage 1: builder ──────────────────────────────────────────────
# Install Python deps in a virtualenv so the final image stays lean.
FROM python:3.14-slim AS builder

WORKDIR /app

# System deps needed to build wheels (some packages have C extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .

# Install into an isolated venv that we copy into the final stage.
# --require-hashes is incompatible with >= version pins.  If you pin
# exact versions with hashes (pip freeze --require-hashes), enable it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.14-slim

WORKDIR /app

# Chromium + Playwright system dependencies (the heavy stuff).
# --no-install-recommends keeps the list reasonable while still
# pulling in everything chromium needs for headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Playwright / Chromium runtime
    libnss3 libnspr4 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    libcups2 libdbus-1-3 \
    # Convenience
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the pre-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Chromium browser that Playwright / crawl4ai will use.
# --with-deps pulls in any remaining system libs and verifies the
# browser actually works inside the container.
RUN python -m playwright install chromium --with-deps

# ── Application ──────────────────────────────────────────────────
COPY . .

# Ensure artifact directories exist and are writable (non-root later).
RUN mkdir -p schemas .cache runs output

# Streamlit's default port.
EXPOSE 8501

# Streamlit listens on all interfaces inside the container, disables
# the file watcher (it doesn't work well in Docker), and opts out of
# the telemetry / email nag on first launch.
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run as non-root for safety.  1000:1000 is the typical first user.
# Files copied by COPY (above) are root-owned by default, so we chown
# everything to the app user including the artifact directories.
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Streamlit has no built-in auth.  If this container is reachable from
# anything other than localhost, put a reverse proxy with basic auth
# or an OAuth sidecar in front of it.  Anyone who can hit port 8501
# can run LLM calls on your API keys.
ENTRYPOINT ["streamlit", "run", "app.py"]
