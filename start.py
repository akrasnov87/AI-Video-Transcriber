#!/usr/bin/env python3
"""
Скрипт запуска AI Видео Транскрибатора
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Проверка установленных зависимостей"""
    import sys
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn", 
        "yt-dlp": "yt_dlp",
        "faster-whisper": "faster_whisper",
        "openai": "openai"
    }
    
    missing_packages = []
    for display_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(display_name)
    
    if missing_packages:
        print("❌ Отсутствуют следующие зависимости:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nВыполните следующую команду для установки зависимостей:")
        print("source venv/bin/activate && pip install -r requirements.txt")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def check_ffmpeg():
    """Проверка установки FFmpeg"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        print("✅ FFmpeg установлен")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg не найден")
        print("Пожалуйста, установите FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: загрузите с официального сайта https://ffmpeg.org/download.html")
        return False

def setup_environment():
    """Настройка переменных окружения"""
    # Настройка конфигурации OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Предупреждение: переменная окружения OPENAI_API_KEY не установлена")
        print("Пожалуйста, установите переменную: export OPENAI_API_KEY=ваш_api_ключ_здесь")
        return False
    
    print("✅ OpenAI API Key установлен")
    
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://oneapi.basevec.com/v1"
        print("✅ OpenAI Base URL установлен")
    
    # Установка других настроек по умолчанию
    if not os.getenv("WHISPER_MODEL_SIZE"):
        os.environ["WHISPER_MODEL_SIZE"] = "base"
    
    print("🔑 OpenAI API настроен, функция создания саммари доступна")
    return True

def main():
    """Основная функция"""
    # Проверка использования производственного режима (отключение горячей перезагрузки)
    production_mode = "--prod" in sys.argv or os.getenv("PRODUCTION_MODE") == "true"
    
    print("🚀 Проверка запуска AI Видео Транскрибатора")
    if production_mode:
        print("🔒 Производственный режим - горячая перезагрузка отключена")
    else:
        print("🔧 Режим разработки - горячая перезагрузка включена")
    print("=" * 50)
    
    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    # Проверка FFmpeg
    if not check_ffmpeg():
        print("⚠️  FFmpeg не установлен, это может повлиять на обработку некоторых видеоформатов")
    
    # Настройка окружения
    setup_environment()
    
    print("\n🎉 Проверка запуска завершена!")
    print("=" * 50)
    
    # Запуск сервера
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"\n🌐 Запуск сервера...")
    print(f"   Адрес: http://localhost:{port}")
    print(f"   Нажмите Ctrl+C для остановки сервиса")
    print("=" * 50)
    
    try:
        # Переход в директорию backend и запуск сервиса
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        # Включение горячей перезагрузки только в режиме разработки
        if not production_mode:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Сервис остановлен")
    except Exception as e:
        print(f"\n❌ Ошибка запуска: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()