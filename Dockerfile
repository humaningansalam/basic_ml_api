FROM debian:bookworm-slim

ARG VERSION
ARG INSTALL_DEV=false

ENV VERSION=$VERSION

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    gcc \
    g++ \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --python 3.11 --no-default-groups --no-install-project && \
    if [ "$INSTALL_DEV" = "true" ]; then \
        uv sync --frozen --python 3.11 --group dev --no-install-project; \
    fi

COPY . .

ENV PATH="/usr/src/app/.venv/bin:$PATH"

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).close()"]

CMD ["gunicorn", "--no-control-socket", "-w", "1", "-b", "0.0.0.0:5000", "src.main:create_app()"]
