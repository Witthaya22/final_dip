FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp/Ultralytics PORT=7860
WORKDIR /home/user/app

RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user web ./web
COPY --chown=user weights ./weights

EXPOSE 7860
CMD uvicorn web.backend.app:app --host 0.0.0.0 --port ${PORT}
