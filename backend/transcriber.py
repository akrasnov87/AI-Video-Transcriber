import os
import time
import gc
import asyncio
import logging
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Добавляем отдельный логгер для прогресса
progress_logger = logging.getLogger('progress')
if not progress_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | 🔄 %(message)s', datefmt='%H:%M:%S'))
    progress_logger.addHandler(handler)


class Transcriber:
    """
    Аудио транскрибатор на базе Faster-Whisper.

    Особенности жизненного цикла модели:
      • Модель загружается лениво при первом вызове transcribe().
      • Модель НЕ выгружается после каждой задачи — это позволяет
        экономить 5–15 с (CPU) или до минуты (GPU) на каждом следующем запросе.
      • Если модель простаивает дольше WHISPER_IDLE_UNLOAD_SECONDS,
        фоновый воркер освобождает память. По умолчанию 1800 с (30 мин).
      • Отключить idle-выгрузку можно, установив WHISPER_IDLE_UNLOAD_SECONDS=0.
      • Принудительная выгрузка — метод unload_now() (используется,
        например, при shutdown сервера).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        self.model_size = model_size
        self.model = None
        self.last_detected_language = None

        self.device = device or os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        # --- Настройки жизненного цикла модели ---
        # True  — держим модель в памяти между задачами (по умолчанию)
        # False — старое поведение: выгружать после каждой транскрипции
        self.keep_model_loaded = (
            os.getenv("WHISPER_KEEP_MODEL_LOADED", "true").strip().lower()
            in ("1", "true", "yes", "on")
        )

        # Сколько секунд модель может простаивать до авто-выгрузки.
        # 0 — никогда не выгружать (держать всегда).
        try:
            self.idle_unload_seconds = int(
                os.getenv("WHISPER_IDLE_UNLOAD_SECONDS", "1800")
            )
        except ValueError:
            self.idle_unload_seconds = 1800

        self._last_used_at: float = 0.0     # monotonic timestamp последнего использования
        self._active_jobs: int = 0          # сколько транскрипций сейчас идёт
        self._load_lock = asyncio.Lock()    # защита от одновременной загрузки
        self._idle_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Проверка CUDA доступности
        if self.device == "cuda":
            cuda_found = self._check_cuda_available()
            if not cuda_found:
                logger.warning("⚠️ CUDA не найдена, переключение на CPU")
                self.device = "cpu"
                self.compute_type = "int8"

        # Для CUDA автоматически используем float16
        if self.device == "cuda" and self.compute_type == "int8":
            self.compute_type = "float16"
            logger.info("Автоматически установлен compute_type=float16 для GPU")

        logger.info(
            "Whisper будет использовать: device=%s, compute_type=%s, model=%s | "
            "keep_model_loaded=%s, idle_unload=%ss",
            self.device, self.compute_type, self.model_size,
            self.keep_model_loaded, self.idle_unload_seconds,
        )

        # Запускаем idle-выгрузчик (если он вообще нужен)
        self._start_idle_unloader_if_needed()

    # ------------------------------------------------------------------
    # CUDA detection
    # ------------------------------------------------------------------
    def _check_cuda_available(self) -> bool:
        """Проверка доступности CUDA через различные методы."""
        import ctypes
        import glob

        cuda_paths = [
            os.environ.get('CUDA_PATH', '/usr/local/cuda'),
            '/usr/local/cuda-12',
            '/usr/local/cuda-12.2',
            '/usr/local/cuda-12.1',
            '/usr/local/cuda-12.0',
            '/usr/lib/x86_64-linux-gnu',
        ]

        for cuda_path in cuda_paths:
            lib_paths = [
                f"{cuda_path}/lib64/libcudart.so",
                f"{cuda_path}/lib64/libcudart.so.12",
                f"{cuda_path}/lib/libcudart.so",
                f"{cuda_path}/lib/libcudart.so.12",
            ]
            if os.path.exists(cuda_path):
                pattern = f"{cuda_path}/**/libcudart.so*"
                found = glob.glob(pattern, recursive=True)
                if found:
                    lib_paths.extend(found[:5])

            for lib_path in lib_paths:
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path)
                        logger.info(f"✅ CUDA библиотека найдена: {lib_path}")
                        return True
                    except Exception:
                        continue

        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"✅ NVIDIA GPU обнаружен: {result.stdout.strip()}")
                return True
        except Exception:
            pass

        logger.warning("⚠️ CUDA библиотека не найдена")
        return False

    # ------------------------------------------------------------------
    # Управление моделью
    # ------------------------------------------------------------------
    async def _ensure_model_loaded(self) -> None:
        """Загружает модель, если она ещё не загружена. Потокобезопасно."""
        if self.model is not None:
            return
        async with self._load_lock:
            # Проверяем ещё раз под локом — возможно, кто-то уже загрузил
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Синхронная загрузка модели (выполняется в отдельном потоке)."""
        logger.info(f"🔄 Загрузка модели Whisper: {self.model_size} на {self.device}")
        try:
            compute = "float16" if self.device == "cuda" else self.compute_type
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute,
            )
            logger.info(
                "✅ Модель загружена: device=%s, compute_type=%s",
                self.device, compute,
            )
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели на {self.device}: {e}")
            if self.device == "cuda":
                logger.warning("⚠️ GPU недоступен, fallback на CPU...")
                try:
                    self.model = WhisperModel(
                        self.model_size, device="cpu", compute_type="int8",
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    logger.info("✅ Модель загружена на CPU (fallback)")
                except Exception as e2:
                    raise Exception(f"Ошибка загрузки модели на CPU: {e2}")
            else:
                raise Exception(f"Ошибка загрузки модели: {e}")

    def _unload_model(self) -> None:
        """Выгрузка модели из памяти."""
        if self.model is None:
            return
        logger.info("🧹 Выгрузка модели Whisper из памяти...")
        try:
            del self.model
            self.model = None
            gc.collect()
            progress_logger.info("✅ Модель Whisper выгружена из памяти")
            logger.info("✅ Модель Whisper успешно выгружена")
        except Exception as e:
            logger.error(f"⚠️ Ошибка при выгрузке модели: {e}")

    def unload_now(self) -> None:
        """Публичный метод принудительной выгрузки (например, при shutdown)."""
        self._unload_model()

    # ------------------------------------------------------------------
    # Idle-выгрузчик
    # ------------------------------------------------------------------
    def _start_idle_unloader_if_needed(self) -> None:
        """Запускает фоновую задачу авто-выгрузки модели, если она нужна."""
        if not self.keep_model_loaded:
            logger.info("Idle-выгрузка отключена: модель будет выгружаться после каждой задачи")
            return
        if self.idle_unload_seconds <= 0:
            logger.info("Idle-выгрузка отключена: модель будет храниться постоянно")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Нет активного loop — отложим до первого transcribe()
            logger.debug("Event loop не запущен, idle-выгрузчик будет запущен позже")
            self._idle_task = None
            return

        self._idle_task = loop.create_task(self._idle_unloader())
        logger.info(
            "Idle-выгрузчик модели запущен: порог %s с",
            self.idle_unload_seconds,
        )

    async def _idle_unloader(self) -> None:
        """Фоновая задача: выгружает модель, если она долго не использовалась."""
        interval = 60  # проверка раз в минуту
        try:
            while not self._shutdown:
                await asyncio.sleep(interval)
                if self.model is None:
                    continue
                if self._active_jobs > 0:
                    # Кто-то сейчас работает — не трогаем
                    continue
                idle = time.monotonic() - self._last_used_at
                if idle >= self.idle_unload_seconds:
                    logger.info(
                        "Модель простаивала %.0f с (порог %s) — выгружаем",
                        idle, self.idle_unload_seconds,
                    )
                    self._unload_model()
        except asyncio.CancelledError:
            logger.info("Idle-выгрузчик модели остановлен")
            raise

    def shutdown(self) -> None:
        """Останавливает idle-выгрузчик и освобождает модель."""
        self._shutdown = True
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._unload_model()

    # ------------------------------------------------------------------
    # Транскрипция
    # ------------------------------------------------------------------
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        simple_format: bool = False,
    ) -> str:
        if not os.path.exists(audio_path):
            raise Exception(f"Аудиофайл не найден: {audio_path}")

        # Ленивый запуск idle-выгрузчика, если не удалось запустить в __init__
        if (
            self._idle_task is None
            and self.keep_model_loaded
            and self.idle_unload_seconds > 0
        ):
            try:
                loop = asyncio.get_running_loop()
                self._idle_task = loop.create_task(self._idle_unloader())
                logger.info("Idle-выгрузчик модели запущен (ленивая инициализация)")
            except RuntimeError:
                pass

        await self._ensure_model_loaded()

        self._active_jobs += 1
        try:
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            progress_logger.info(
                f"🎙️ Начало транскрипции: {os.path.basename(audio_path)} "
                f"({file_size_mb:.1f} МБ)"
            )

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
                        "speech_pad_ms": 300,
                    },
                    no_speech_threshold=0.7,
                    compression_ratio_threshold=2.3,
                    log_prob_threshold=-1.0,
                    condition_on_previous_text=False,
                )

            start_time = time.time()
            segments, info = await asyncio.to_thread(_do_transcribe)
            elapsed = time.time() - start_time

            detected_language = info.language
            self.last_detected_language = detected_language

            segment_count = 0
            transcript_lines = []

            if simple_format:
                for segment in segments:
                    start_time_seg = self._format_time_full(segment.start)
                    end_time_seg = self._format_time_full(segment.end)
                    text = segment.text.strip()
                    transcript_lines.append(f"[{start_time_seg} → {end_time_seg}]: {text}")
                    segment_count += 1
                    if segment_count % 10 == 0:
                        progress_logger.info(
                            f"🎙️ Транскрипция: {segment_count} сегментов обработано..."
                        )
                transcript_text = "\n".join(transcript_lines)
            else:
                transcript_lines.append("# Video Transcription")
                transcript_lines.append("")
                transcript_lines.append(f"**Detected Language:** {detected_language}")
                transcript_lines.append(
                    f"**Language Probability:** {info.language_probability:.2f}"
                )
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
                        progress_logger.info(
                            f"🎙️ Транскрипция: {segment_count} сегментов обработано..."
                        )

                transcript_text = "\n".join(transcript_lines)

            progress_logger.info(
                f"✅ Транскрипция завершена: {segment_count} сегментов, {elapsed:.1f}с"
            )
            logger.info(
                f"🎙️ Язык: {detected_language}, "
                f"вероятность: {info.language_probability:.2f}"
            )

            return transcript_text

        except Exception as e:
            logger.error(f"❌ Ошибка транскрипции: {e}")
            raise Exception(f"Ошибка транскрипции: {e}")
        finally:
            # Снимаем счётчик и обновляем метку последнего использования.
            self._active_jobs = max(0, self._active_jobs - 1)
            self._last_used_at = time.monotonic()

            # Если пользователь хочет старое поведение — выгружаем сразу.
            # Иначе модель останется в памяти и будет выгружена idle-таймером.
            if not self.keep_model_loaded:
                self._unload_model()

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------
    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _format_time_full(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_supported_languages(self) -> list:
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no",
        ]

    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        if self.last_detected_language:
            return self.last_detected_language
        if transcript_text and "**Detected Language:**" in transcript_text:
            for line in transcript_text.split('\n'):
                if "**Detected Language:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang if lang else None
        return None