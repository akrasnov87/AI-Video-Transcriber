import os
from faster_whisper import WhisperModel
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Transcriber:
    """Аудио транскрибатор, использующий Faster-Whisper для преобразования речи в текст"""
    
    def __init__(
        self, 
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        """
        Инициализация транскрибатора
        
        Args:
            model_size: Размер модели Whisper (tiny, base, small, medium, large)
            device: Устройство для запуска ("cpu" или "cuda")
            compute_type: Тип вычислений ("int8", "float16", "float32")
        """
        self.model_size = model_size
        self.model = None
        self.last_detected_language = None
        
        # Чтение настроек из переменных окружения (если не переданы явно)
        self.device = device or os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        
        # Автоматический выбор compute_type для GPU, если не указан явно
        if self.device == "cuda" and self.compute_type == "int8":
            self.compute_type = "float16"  # Рекомендуемый тип для GPU
            logger.info(f"Автоматически установлен compute_type=float16 для GPU")
        
        logger.info(f"Whisper будет использовать: device={self.device}, compute_type={self.compute_type}, model={self.model_size}")
        
    def _load_model(self):
        """Отложенная загрузка модели"""
        if self.model is None:
            logger.info(f"Загрузка модели Whisper: {self.model_size} на {self.device}")
            try:
                self.model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
                logger.info(f"Модель успешно загружена на {self.device}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели на {self.device}: {str(e)}")
                
                # Fallback на CPU если GPU не доступен
                if self.device == "cuda":
                    logger.warning("GPU не доступен, пробую загрузить на CPU...")
                    try:
                        self.model = WhisperModel(
                            self.model_size, 
                            device="cpu", 
                            compute_type="int8"
                        )
                        self.device = "cpu"
                        self.compute_type = "int8"
                        logger.info("Модель загружена на CPU (fallback)")
                    except Exception as e2:
                        raise Exception(f"Ошибка загрузки модели на CPU: {str(e2)}")
                else:
                    raise Exception(f"Ошибка загрузки модели: {str(e)}")
    
    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Транскрипция аудиофайла
        
        Args:
            audio_path: Путь к аудиофайлу
            language: Указание языка (опционально, если не указан — автоматическое определение)
            
        Returns:
            Текст транскрипции (в формате Markdown)
        """
        try:
            # Проверка существования файла
            if not os.path.exists(audio_path):
                raise Exception(f"Аудиофайл не найден: {audio_path}")
            
            # Загрузка модели
            self._load_model()
            
            logger.info(f"Начало транскрипции: {audio_path}")
            
            # Вызов транскрипции в отдельном потоке для избежания блокировки
            import asyncio
            def _do_transcribe():
                return self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    temperature=[0.0, 0.2, 0.4],  # Стратегия с возрастающей температурой
                    # Более надежная настройка: VAD и пороговые значения для снижения повторений от шума/тишины
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 900,  # Длительность тишины для определения паузы
                        "speech_pad_ms": 300  # Отступы вокруг речи
                    },
                    no_speech_threshold=0.7,  # Порог отсутствия речи
                    compression_ratio_threshold=2.3,  # Порог сжатия для обнаружения повторений
                    log_prob_threshold=-1.0,  # Порог логарифмической вероятности
                    # Предотвращение каскадных повторений
                    condition_on_previous_text=False
                )
            segments, info = await asyncio.to_thread(_do_transcribe)
            
            detected_language = info.language
            self.last_detected_language = detected_language  # Сохранение определенного языка
            logger.info(f"Определенный язык: {detected_language}")
            logger.info(f"Вероятность определения языка: {info.language_probability:.2f}")
            
            # Сборка результата транскрипции
            transcript_lines = []
            transcript_lines.append("# Video Transcription")
            transcript_lines.append("")
            transcript_lines.append(f"**Detected Language:** {detected_language}")
            transcript_lines.append(f"**Language Probability:** {info.language_probability:.2f}")
            transcript_lines.append("")
            transcript_lines.append("## Transcription Content")
            transcript_lines.append("")
            
            # Добавление временных меток и текста
            for segment in segments:
                start_time = self._format_time(segment.start)
                end_time = self._format_time(segment.end)
                text = segment.text.strip()
                
                transcript_lines.append(f"**[{start_time} - {end_time}]**")
                transcript_lines.append("")
                transcript_lines.append(text)
                transcript_lines.append("")
            
            transcript_text = "\n".join(transcript_lines)
            logger.info("Транскрипция завершена")
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {str(e)}")
            raise Exception(f"Ошибка транскрипции: {str(e)}")
    
    def _format_time(self, seconds: float) -> str:
        """
        Преобразование секунд в формат часы:минуты:секунды
        
        Args:
            seconds: Количество секунд
            
        Returns:
            Отформатированная строка времени
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_supported_languages(self) -> list:
        """
        Получение списка поддерживаемых языков
        """
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
    
    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        """
        Получение определенного языка
        
        Args:
            transcript_text: Текст транскрипции (опционально, для извлечения информации о языке из текста)
            
        Returns:
            Код определенного языка
        """
        # Если сохраненный язык есть, возвращаем его
        if self.last_detected_language:
            return self.last_detected_language
        
        # Если предоставлен текст транскрипции, пытаемся извлечь информацию о языке
        if transcript_text and "**Detected Language:**" in transcript_text:
            lines = transcript_text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang if lang else None
        
        return None