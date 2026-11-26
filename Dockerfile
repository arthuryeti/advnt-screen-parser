FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OCR_MODEL_DIR=/opt/rapidocr-models \
    HF_HOME=/opt/hf-cache

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE LICENSE-MODEL ./
COPY src ./src

RUN pip install --no-cache-dir ".[all]" \
    && python -c "from advnt_screen_parser import ScreenParser; ScreenParser().warm_up()" \
    && useradd --create-home --uid 10001 parser \
    && chown -R parser:parser /app /opt/rapidocr-models /opt/hf-cache

USER parser

EXPOSE 8000

CMD ["uvicorn", "advnt_screen_parser.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
