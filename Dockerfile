# AI Видео Транскрибатор Docker образ — соответствует локальному окружению (Python 3.12), зависимости из requirements.txt
FROM python:3.12-slim-bookworm

WORKDIR /app

# Системные зависимости (FFmpeg: для загрузки по ссылке и локальной загрузки с транскодированием)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Сначала обновляем pip, затем устанавливаем зависимости из requirements.txt
# (поведение соответствует локальной команде `pip install -r requirements.txt`, берутся последние совместимые версии)
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Копируем файлы проекта
COPY . .

# Создаем директорию для временных файлов
RUN mkdir -p temp

# Устанавливаем переменные окружения
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WHISPER_MODEL_SIZE=base
ENV UPLOAD_MAX_MB=200

# Открываем порт
EXPOSE 8000

# Проверка работоспособности
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Команда запуска
CMD ["python3", "start.py", "--prod"]