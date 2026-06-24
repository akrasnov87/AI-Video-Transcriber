import os
import re
import shutil
import uuid
import asyncio
import subprocess
import yt_dlp
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Обработчик видео, использующий yt-dlp для загрузки и конвертации видео"""
    
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',  # Приоритет загрузки лучшего аудиоисточника
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # Преобразование в моно 16 кГц на этапе извлечения (экономит место и стабильно)
                'preferredcodec': 'm4a',
                'preferredquality': '192'
            }],
            # Глобальные параметры FFmpeg: моно + частота 16 кГц + faststart
            'postprocessor_args': ['-ac', '1', '-ar', '16000', '-movflags', '+faststart'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,  # Принудительно загружать только одно видео, не плейлист
        }

    async def normalize_local_media_to_m4a(self, input_path: Path, output_dir: Path) -> str:
        """
        Преобразование локально загруженного аудио/видео в моно 16 кГц AAC m4a для Faster-Whisper.
        Параметры согласованы с постобработкой yt-dlp.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        out_path = output_dir / f"upload_norm_{unique_id}.m4a"

        cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", str(input_path.resolve()),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(out_path.resolve()),
        ]

        def _run():
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                raise Exception(f"Ошибка преобразования FFmpeg: {err[:800]}")
            if not out_path.exists():
                raise Exception("FFmpeg не создал выходной файл")

        await asyncio.to_thread(_run)
        return str(out_path)
    
    async def fetch_subtitles(self, url: str, output_dir: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Попытка получить текст субтитров с платформы (значительно быстрее загрузки аудио).

        Возвращает:
            (subtitle_markdown, video_title, language_code)
            subtitle_markdown = None означает отсутствие доступных субтитров.
        """
        import asyncio

        output_dir.mkdir(exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        sub_dir = output_dir / f"subs_{unique_id}"

        try:
            # 1. Быстрая проверка: получение информации о видео и доступности субтитров (без загрузки)
            check_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
            with yt_dlp.YoutubeDL(check_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, False)

            video_title = info.get("title", "unknown")
            manual_subs: dict = info.get("subtitles") or {}
            auto_caps: dict = info.get("automatic_captions") or {}

            # Фильтрация non-speech треков (например, live_chat)
            manual_langs = [k for k in manual_subs if not k.startswith("live_chat")]
            auto_langs = [k for k in auto_caps if not k.startswith("live_chat")]

            if not manual_langs and not auto_langs:
                logger.info(f"Видео не имеет доступных субтитров: {url}")
                return None, video_title, None

            # Приоритет ручным субтитрам, затем автоматическим
            prefer_manual = bool(manual_langs)
            candidate_langs = manual_langs if prefer_manual else auto_langs

            # Выбор языка по приоритету: английский > упрощенный китайский > традиционный китайский > другие
            _priority = ["en", "en-orig", "zh-Hans", "zh-Hant", "zh", "ja", "ko", "fr", "de", "es"]
            prefer_lang = next(
                (lang for lang in _priority if lang in candidate_langs),
                candidate_langs[0],
            )
            logger.info(
                f"Обнаружены {'ручные' if prefer_manual else 'автоматические'} субтитры, выбран язык: {prefer_lang}"
                f" (доступно {len(candidate_langs)} вариантов)"
            )

            # 2. Загрузка только субтитров (без аудио/видео)
            sub_dir.mkdir(exist_ok=True)
            dl_opts = {
                "writesubtitles": prefer_manual,
                "writeautomaticsub": not prefer_manual,
                "subtitlesformat": "vtt/srt/best",
                "subtitleslangs": [prefer_lang],
                "skip_download": True,
                "outtmpl": str(sub_dir / "sub.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])

            # 3. Поиск загруженного файла субтитров
            sub_files = list(sub_dir.glob("*.vtt")) + list(sub_dir.glob("*.srt"))
            if not sub_files:
                logger.warning("Файл субтитров не найден после загрузки, переход в аудиорежим")
                return None, video_title, None

            sub_file = sub_files[0]

            # Извлечение кода языка из имени файла (например, sub.en.vtt → en)
            stem_parts = sub_file.stem.split(".")
            file_lang = stem_parts[-1] if len(stem_parts) > 1 else prefer_lang

            # 4. Парсинг файла субтитров
            if sub_file.suffix == ".vtt":
                entries = self._parse_vtt(str(sub_file))
            else:
                entries = self._parse_srt(str(sub_file))

            if not entries:
                logger.warning("Результат парсинга субтитров пуст, переход в аудиорежим")
                return None, video_title, None

            # 5. Форматирование в Markdown, совместимый с выводом Whisper
            formatted = self._format_subtitle_entries(entries, file_lang)
            logger.info(f"Субтитры успешно получены: lang={file_lang}, {len(entries)} записей")
            return formatted, video_title, file_lang

        except Exception as e:
            logger.warning(f"Ошибка получения субтитров (переход к загрузке аудио): {e}")
            return None, None, None
        finally:
            if sub_dir.exists():
                try:
                    shutil.rmtree(str(sub_dir))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Вспомогательные методы парсинга субтитров
    # ------------------------------------------------------------------

    def _parse_vtt(self, filepath: str) -> list:
        """Парсинг WebVTT субтитров, возвращает список уникальных записей.

        Специальная обработка формата YouTube автоматических субтитров:
        одна фраза может быть разбита на несколько cue с последовательным добавлением слов,
        сохраняется только финальная версия каждой группы.
        """
        raw_entries = []
        seen_texts: set = set()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Ошибка чтения VTT файла: {e}")
            return []

        # Удаление заголовка WEBVTT, разделение на блоки по пустым строкам
        content = re.sub(r"^WEBVTT[^\n]*\n", "", content)
        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->\s*"
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            raw_text = " ".join(text_lines)
            # Удаление HTML/VTT тегов (включая теги временных меток YouTube)
            text = re.sub(r"<[^>]+>", "", raw_text)
            text = (
                text.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&nbsp;", " ")
                    .replace("&#39;", "'")
                    .replace("&quot;", '"')
                    .strip()
            )
            # Объединение лишних пробелов
            text = re.sub(r"\s+", " ", text).strip()

            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            raw_entries.append({"start": start_str, "end": end_str, "text": text})

        # ── Вторичная дедупликация: фильтрация промежуточных состояний YouTube ──
        # Если текст записи i является началом записи i+1, то запись i — промежуточная, удаляем.
        # Также удаляем пустые/односимвольные записи.
        if not raw_entries:
            return []

        entries = []
        for i, entry in enumerate(raw_entries):
            text = entry["text"]
            if len(text) < 2:
                continue
            # Проверка, является ли текущая запись началом следующих (признак последовательного добавления)
            is_intermediate = False
            for j in range(i + 1, min(i + 4, len(raw_entries))):
                next_text = raw_entries[j]["text"]
                if next_text.startswith(text) and len(next_text) > len(text):
                    is_intermediate = True
                    break
            if not is_intermediate:
                entries.append(entry)

        return entries

    def _parse_srt(self, filepath: str) -> list:
        """Парсинг SRT субтитров, возвращает список уникальных записей."""
        entries = []
        seen_texts: set = set()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Ошибка чтения SRT файла: {e}")
            return []

        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}:\d{2}[.,]\d+)\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d+)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            text = " ".join(text_lines)
            text = re.sub(r"<[^>]+>", "", text).strip()

            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            entries.append({"start": start_str, "end": end_str, "text": text})

        return entries

    def _normalize_time(self, time_str: str) -> str:
        """Преобразование HH:MM:SS.mmm или MM:SS.mmm в формат MM:SS."""
        time_str = re.sub(r"[.,]\d+$", "", time_str)
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{h * 60 + m:02d}:{s:02d}"
        elif len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return f"{m:02d}:{s:02d}"
        return time_str

    def _format_subtitle_entries(self, entries: list, language: str) -> str:
        """Форматирование записей субтитров в Markdown, совместимый с выводом Whisper."""
        lines = [
            "# Video Transcription",
            "",
            f"**Detected Language:** {language}",
            "**Language Probability:** 1.00",
            "",
            "## Transcription Content",
            "",
        ]
        for entry in entries:
            lines.append(f"**[{entry['start']} - {entry['end']}]**")
            lines.append("")
            lines.append(entry["text"])
            lines.append("")
        return "\n".join(lines)

    async def download_and_convert(
        self,
        url: str,
        output_dir: Path,
        prefetched_title: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Загрузка видео и преобразование в формат m4a.

        prefetched_title: если вызывающий код уже получил информацию о видео через fetch_subtitles,
        можно передать заголовок, чтобы избежать повторного запроса extract_info.
        """
        try:
            # Создание выходной директории
            output_dir.mkdir(exist_ok=True)
            
            # Генерация уникального имени файла
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
            # Обновление опций yt-dlp
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = output_template
            
            logger.info(f"Начало загрузки видео: {url}")
            
            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if prefetched_title:
                    # Заголовок и длительность уже получены в fetch_subtitles
                    video_title = prefetched_title
                    expected_duration = 0
                    logger.info(f"Использование предварительно полученного заголовка (пропуск extract_info): {video_title}")
                else:
                    # Получение информации о видео (в отдельном потоке для избежания блокировки)
                    info = await asyncio.to_thread(ydl.extract_info, url, False)
                    video_title = info.get('title', 'unknown')
                    expected_duration = info.get('duration') or 0
                    logger.info(f"Заголовок видео: {video_title}")
                
                # Загрузка видео (в отдельном потоке)
                await asyncio.to_thread(ydl.download, [url])
            
            # Поиск созданного m4a файла
            audio_file = str(output_dir / f"audio_{unique_id}.m4a")
            
            if not os.path.exists(audio_file):
                # Если m4a не найден, проверяем другие форматы
                for ext in ['webm', 'mp4', 'mp3', 'wav']:
                    potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                else:
                    raise Exception("Аудиофайл не найден")
            
            # Проверка длительности, при значительном расхождении с оригиналом — попытка исправления
            try:
                import subprocess, shlex
                probe_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(audio_file)}"
                out = subprocess.check_output(probe_cmd, shell=True).decode().strip()
                actual_duration = float(out) if out else 0.0
            except Exception as _:
                actual_duration = 0.0
            
            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"Аномальная длительность аудио: ожидалось {expected_duration}с, получено {actual_duration}с. Попытка перепаковки..."
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                    fix_cmd = f"ffmpeg -y -i {shlex.quote(audio_file)} -vn -c:a aac -b:a 160k -movflags +faststart {shlex.quote(fixed_path)}"
                    subprocess.check_call(fix_cmd, shell=True)
                    # Замена на исправленный файл
                    audio_file = fixed_path
                    # Повторная проверка
                    out2 = subprocess.check_output(probe_cmd.replace(shlex.quote(audio_file.rsplit('.',1)[0]+'.m4a'), shlex.quote(audio_file)), shell=True).decode().strip()
                    actual_duration2 = float(out2) if out2 else 0.0
                    logger.info(f"Перепаковка завершена, новая длительность ≈ {actual_duration2:.2f}с")
                except Exception as e:
                    logger.error(f"Ошибка перепаковки: {e}")
            
            logger.info(f"Аудиофайл сохранен: {audio_file}")
            return audio_file, video_title
            
        except Exception as e:
            logger.error(f"Ошибка загрузки видео: {str(e)}")
            raise Exception(f"Ошибка загрузки видео: {str(e)}")
    
    def get_video_info(self, url: str) -> dict:
        """
        Получение информации о видео
        
        Args:
            url: Ссылка на видео
            
        Returns:
            Словарь с информацией о видео
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"Ошибка получения информации о видео: {str(e)}")
            raise Exception(f"Ошибка получения информации о видео: {str(e)}")