FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PALACE_MODE=hosted

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what the hosted read path imports. No build/extract/distill code and no
# known_secrets: the server never sees a transcript or a credential file; it
# serves the index the laptop already scrubbed and uploaded.
COPY mcp_server.py hosted.py index.py store.py scrub.py telemetry.py instrumentation.py ./

# Non-root. The index volume's mount point is created here, owned by appuser,
# because Docker seeds an empty named volume from the image path including its
# ownership; without this the volume arrives root-owned and every upload fails.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data && chown -R appuser:appuser /app /data

USER appuser
EXPOSE 8000
CMD ["python", "mcp_server.py"]
