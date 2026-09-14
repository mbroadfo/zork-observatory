# Jericho ships a compiled Frotz and builds against Linux headers, so the core
# runs here rather than on the host. The mock engine needs none of this — it is
# only the real-game path that requires the container.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[jericho]"

# Game files are mounted, never baked in — they are copyrighted.
VOLUME ["/app/roms", "/app/traces"]

EXPOSE 8000
CMD ["observatory", "serve", "--host", "0.0.0.0", "--port", "8000"]
