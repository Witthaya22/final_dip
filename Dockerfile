# =============================================================
#  Dockerfile : สร้าง container สำหรับ deploy เว็บ (หน้าบ้าน + หลังบ้าน)
#  ใช้กับ Hugging Face Spaces (SDK = Docker) หรือ server ไหนก็ได้ที่รัน Docker
#     docker build -t final-dip .
#     docker run -p 7860:7860 final-dip      -> เปิด http://localhost:7860
# =============================================================
FROM python:3.11-slim

# ไลบรารีระบบที่ OpenCV ต้องใช้
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces รันด้วย user id 1000 -> สร้าง user ไว้ให้มีสิทธิ์เขียนไฟล์
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp/Ultralytics PORT=7860
WORKDIR /home/user/app

# ติดตั้ง PyTorch แบบ CPU (เล็กกว่าแบบ GPU มาก) แล้วค่อยติดตั้งที่เหลือ
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# คัดลอกเฉพาะที่เว็บต้องใช้: โค้ดเว็บ + weight ที่เทรนแล้ว
COPY --chown=user web ./web
COPY --chown=user weights ./weights

EXPOSE 7860
CMD uvicorn web.backend.app:app --host 0.0.0.0 --port ${PORT}
