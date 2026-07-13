#!/usr/bin/env python3
"""
Скрипт запуска AI Видео Транскрибатора
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

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
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Предупреждение: переменная окружения OPENAI_API_KEY не установлена")
        print("Пожалуйста, установите переменную: export OPENAI_API_KEY=ваш_api_ключ_здесь")
        return False
    
    print("✅ OpenAI API Key установлен")
    
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
        print("✅ OpenAI Base URL установлен")
    
    if not os.getenv("WHISPER_MODEL_SIZE"):
        os.environ["WHISPER_MODEL_SIZE"] = "base"
    
    print("🔑 OpenAI API настроен, функция создания саммари доступна")
    return True

def main():
    """Основная функция"""

    # Загружаем .env файл
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"📄 Загружен .env файл: {env_file}")
    else:
        print("ℹ️  Файл .env не найден, используются системные переменные")

    production_mode = "--prod" in sys.argv or os.getenv("PRODUCTION_MODE") == "true"
    
    print("🚀 Проверка запуска AI Видео Транскрибатора")
    if production_mode:
        print("🔒 Производственный режим - горячая перезагрузка отключена")
    else:
        print("🔧 Режим разработки - горячая перезагрузка включена")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_ffmpeg():
        print("⚠️  FFmpeg не установлен, это может повлиять на обработку некоторых видеоформатов")
    
    setup_environment()
    
    print("\n🎉 Проверка запуска завершена!")
    print("=" * 50)
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"\n🌐 Запуск сервера...")
    print(f"   Адрес: http://localhost:{port}")
    print(f"   Нажмите Ctrl+C для остановки сервиса")
    print("=" * 50)
    
    try:
        root_dir = Path(__file__).parent
        
        # ═══════════════════════════════════════════════════════════
        # ✅ ФИКС: Проверяем оба возможных расположения main.py
        # ═══════════════════════════════════════════════════════════
        module_name = None
        
        # Проверяем backend/main.py (основной вариант)
        if (root_dir / "backend" / "main.py").exists():
            module_name = "backend.main"
            print(f"📄 Найден backend/main.py")
        # Проверяем main.py в корне (запасной вариант)
        elif (root_dir / "main.py").exists():
            module_name = "main"
            print(f"📄 Найден main.py в корне")
        else:
            print("❌ Файл main.py не найден!")
            print(f"   Проверены пути:")
            print(f"   - {root_dir / 'backend' / 'main.py'}")
            print(f"   - {root_dir / 'main.py'}")
            sys.exit(1)
        
        # Формируем команду
        cmd = [
            sys.executable, "-m", "uvicorn", f"{module_name}:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if not production_mode:
            cmd.append("--reload")
        
        # ═══════════════════════════════════════════════════════════
        # ✅ ФИКС: PYTHONPATH для импорта модулей
        # ═══════════════════════════════════════════════════════════
        env = os.environ.copy()
        python_paths = [str(root_dir)]
        if (root_dir / "backend").exists():
            python_paths.append(str(root_dir / "backend"))
        
        env["PYTHONPATH"] = ":".join(python_paths)
        
        print(f"📂 PYTHONPATH: {env['PYTHONPATH']}")
        print(f"🚀 Запуск: {' '.join(cmd)}")
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n\n👋 Сервис остановлен")
    except Exception as e:
        print(f"\n❌ Ошибка запуска: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()