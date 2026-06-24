#!/bin/bash

# Скрипт установки AI Видео Транскрибатора

echo "🚀 Скрипт установки AI Видео Транскрибатора"
echo "=========================="

# Проверка версии Python
echo "Проверка окружения Python..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
if [[ -z "$python_version" ]]; then
    echo "❌ Python3 не найден. Пожалуйста, установите Python 3.8 или выше"
    exit 1
fi
echo "✅ Версия Python: $python_version"

# Проверка pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 не найден. Пожалуйста, установите pip"
    exit 1
fi
echo "✅ pip установлен"

# Установка зависимостей Python
echo ""
echo "Установка зависимостей Python..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Установка зависимостей Python завершена"
else
    echo "❌ Ошибка установки зависимостей Python"
    exit 1
fi

# Проверка FFmpeg
echo ""
echo "Проверка FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg установлен"
else
    echo "⚠️  FFmpeg не найден, пробуем установить..."
    
    # Определение операционной системы
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        else
            echo "❌ Не удалось автоматически установить FFmpeg. Пожалуйста, установите вручную"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ Пожалуйста, установите Homebrew, затем выполните: brew install ffmpeg"
        fi
    else
        echo "❌ Неподдерживаемая операционная система. Пожалуйста, установите FFmpeg вручную"
    fi
fi

# Создание необходимых директорий
echo ""
echo "Создание необходимых директорий..."
mkdir -p temp static
echo "✅ Директории созданы"

# Установка прав на выполнение
chmod +x start.py

echo ""
echo "🎉 Установка завершена!"
echo ""
echo "Инструкция по использованию:"
echo "  1. (Опционально) Настройте API-ключ OpenAI для включения функции интеллектуального саммари"
echo "     export OPENAI_API_KEY=ваш_api_ключ_здесь"
echo ""
echo "  2. Запустите сервис:"
echo "     python3 start.py"
echo ""
echo "  3. Откройте браузер и перейдите по адресу: http://localhost:8000"
echo ""
echo "Поддерживаемые видеоплатформы:"
echo "  - YouTube"
echo "  - Bilibili"
echo "  - Другие платформы, поддерживаемые yt-dlp"