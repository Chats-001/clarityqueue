FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLARITYQUEUE_ARTIFACT_DIR=/app/artifacts \
    CLARITYQUEUE_EVENT_DB=/app/runtime/events.sqlite3

WORKDIR /app
COPY pyproject.toml README.md ./
COPY clarityqueue ./clarityqueue
COPY artifacts ./artifacts
RUN pip install --no-cache-dir . && \
    addgroup --system clarityqueue && \
    adduser --system --ingroup clarityqueue clarityqueue && \
    mkdir -p /app/runtime && chown -R clarityqueue:clarityqueue /app/runtime

USER clarityqueue
EXPOSE 8000
CMD ["uvicorn", "clarityqueue.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

