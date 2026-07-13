#!/usr/bin/env python3
"""
Воркер для транскрибации в отдельном процессе.
Запускается один раз для каждого файла и завершается, освобождая память.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcriber import Transcriber


def main():
    parser = argparse.ArgumentParser(description='Whisper транскрибатор (отдельный процесс)')
    parser.add_argument('audio_path', help='Путь к аудиофайлу')
    parser.add_argument('output_file', help='Путь к файлу для сохранения результата')
    parser.add_argument('--language', default=None, help='Язык транскрипции (опционально)')
    parser.add_argument('--simple-format', action='store_true', help='Простой формат без Markdown')
    parser.add_argument('--model-size', default=os.getenv('WHISPER_MODEL_SIZE', 'base'), 
                       help='Размер модели Whisper')
    parser.add_argument('--device', default=os.getenv('WHISPER_DEVICE', 'cpu'),
                       help='Устройство: cuda или cpu')
    parser.add_argument('--compute-type', default=os.getenv('WHISPER_COMPUTE_TYPE', 'int8'),
                       help='Тип вычислений: float16, int8 и т.д.')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"🚀 Запуск воркера для: {args.audio_path}")
        logger.info(f"📊 Модель: {args.model_size}, устройство: {args.device}, тип: {args.compute_type}")
        
        # Проверка существования аудиофайла
        if not os.path.exists(args.audio_path):
            raise Exception(f"Аудиофайл не найден: {args.audio_path}")
        
        # Создаем экземпляр транскрибатора
        transcriber = Transcriber(
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type
        )
        
        # ═══════════════════════════════════════════════════════════
        # ✅ ФИКС: Убираем await, так как метод теперь синхронный
        # ═══════════════════════════════════════════════════════════
        logger.info("🎙️ Начало транскрибации...")
        result = transcriber.transcribe(
            audio_path=args.audio_path,
            language=args.language,
            simple_format=args.simple_format
        )
        
        # Сохраняем результат
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        logger.info(f"✅ Транскрибация завершена, результат сохранен: {args.output_file}")
        
        # Освобождаем память
        logger.info("🧹 Освобождение памяти...")
        
        # 1. Удаляем модель из памяти
        if hasattr(transcriber, 'model') and transcriber.model is not None:
            del transcriber.model
            
        # 2. Удаляем сам объект транскрибатора
        del transcriber
        
        # 3. Принудительный вызов сборщика мусора
        import gc
        gc.collect()
        
        logger.info("✅ Память освобождена")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в воркере: {str(e)}")
        
        # Записываем ошибку в выходной файл
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Ошибка транскрибации: {str(e)}")
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()