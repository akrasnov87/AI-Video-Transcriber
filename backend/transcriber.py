import os
from faster_whisper import WhisperModel
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Добавляем отдельный логгер для прогресса
progress_logger = logging.getLogger('progress')
if not progress_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | 🔄 %(message)s', datefmt='%H:%M:%S'))
    progress_logger.addHandler(handler)

class Transcriber:
    """Аудио транскрибатор, использующий Faster-Whisper для преобразования речи в текст"""
    
    def __init__(
        self, 
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        self.model_size = model_size
        self.model = None
        self.last_detected_language = None
        
        self.device = device or os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        
        if self.device == "cuda" and self.compute_type == "int8":
            self.compute_type = "float16"
            logger.info(f"Автоматически установлен compute_type=float16 для GPU")
        
        logger.info(f"Whisper будет использовать: device={self.device}, compute_type={self.compute_type}, model={self.model_size}")
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Загрузка модели Whisper: {self.model_size} на {self.device}")
            try:
                self.model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
                logger.info(f"✅ Модель успешно загружена на {self.device}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели на {self.device}: {str(e)}")
                
                if self.device == "cuda":
                    logger.warning("⚠️ GPU не доступен, пробую загрузить на CPU...")
                    try:
                        self.model = WhisperModel(
                            self.model_size, 
                            device="cpu", 
                            compute_type="int8"
                        )
                        self.device = "cpu"
                        self.compute_type = "int8"
                        logger.info("✅ Модель загружена на CPU (fallback)")
                    except Exception as e2:
                        raise Exception(f"Ошибка загрузки модели на CPU: {str(e2)}")
                else:
                    raise Exception(f"Ошибка загрузки модели: {str(e)}")
    
    async def transcribe(self, audio_path: str, language: Optional[str] = None, simple_format: bool = False) -> str:
        try:
            if not os.path.exists(audio_path):
                raise Exception(f"Аудиофайл не найден: {audio_path}")
            
            self._load_model()
            
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            progress_logger.info(f"🎙️ Начало транскрипции: {os.path.basename(audio_path)} ({file_size_mb:.1f} МБ)")
            
            import asyncio
            def _do_transcribe():
                return self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    temperature=[0.0, 0.2, 0.4],
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 900,
                        "speech_pad_ms": 300
                    },
                    no_speech_threshold=0.7,
                    compression_ratio_threshold=2.3,
                    log_prob_threshold=-1.0,
                    condition_on_previous_text=False
                )
            
            # Засекаем время
            start_time = __import__('time').time()
            
            segments, info = await asyncio.to_thread(_do_transcribe)
            
            elapsed = __import__('time').time() - start_time
            
            detected_language = info.language
            self.last_detected_language = detected_language
            
            # Собираем сегменты для подсчета
            segment_count = 0
            transcript_lines = []
            
            if simple_format:
                for segment in segments:
                    start_time_seg = self._format_time_full(segment.start)
                    end_time_seg = self._format_time_full(segment.end)
                    text = segment.text.strip()
                    transcript_lines.append(f"[{start_time_seg} → {end_time_seg}]: {text}")
                    segment_count += 1
                    # Показываем прогресс каждые 10 сегментов
                    if segment_count % 10 == 0:
                        progress_logger.info(f"🎙️ Транскрипция: {segment_count} сегментов обработано...")
                transcript_text = "\n".join(transcript_lines)
            else:
                transcript_lines.append("# Video Transcription")
                transcript_lines.append("")
                transcript_lines.append(f"**Detected Language:** {detected_language}")
                transcript_lines.append(f"**Language Probability:** {info.language_probability:.2f}")
                transcript_lines.append("")
                transcript_lines.append("## Transcription Content")
                transcript_lines.append("")
                
                for segment in segments:
                    start_time_seg = self._format_time(segment.start)
                    end_time_seg = self._format_time(segment.end)
                    text = segment.text.strip()
                    
                    transcript_lines.append(f"**[{start_time_seg} - {end_time_seg}]**")
                    transcript_lines.append("")
                    transcript_lines.append(text)
                    transcript_lines.append("")
                    segment_count += 1
                    if segment_count % 10 == 0:
                        progress_logger.info(f"🎙️ Транскрипция: {segment_count} сегментов обработано...")
                
                transcript_text = "\n".join(transcript_lines)
            
            progress_logger.info(f"✅ Транскрипция завершена: {segment_count} сегментов, {elapsed:.1f}с")
            logger.info(f"🎙️ Язык: {detected_language}, вероятность: {info.language_probability:.2f}")
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"❌ Ошибка транскрипции: {str(e)}")
            raise Exception(f"Ошибка транскрипции: {str(e)}")
    
    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def _format_time_full(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_supported_languages(self) -> list:
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
    
    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        if self.last_detected_language:
            return self.last_detected_language
        
        if transcript_text and "**Detected Language:**" in transcript_text:
            lines = transcript_text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang if lang else None
        
        return None