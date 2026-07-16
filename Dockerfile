# AI Видео Транскрибатор Docker образ с поддержкой GPU
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Устанавливаем Python 3.10 и зависимости
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    curl \
    ca-certificates \
    gcc \
    g++ \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем переменные окружения для CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda-12.2/bin:${PATH}

# Обновляем pip и устанавливаем зависимости
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Создаем директорию для временных файлов и кеша
RUN mkdir -p temp /app/cache/huggingface

# Устанавливаем переменные окружения
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WHISPER_MODEL_SIZE=large
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16
ENV UPLOAD_MAX_MB=2000
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface/hub
ENV PYTORCH_TRANSFORMERS_CACHE=/app/cache/huggingface/hub
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PRODUCTION_MODE=true

# Открываем порт
EXPOSE 8000

# Проверка работоспособности
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Команда запуска
CMD ["python3", "start.py", "--prod"]