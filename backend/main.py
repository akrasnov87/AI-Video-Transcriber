from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional
import aiofiles
import uuid
import json
import re
import openai
import subprocess
import tempfile
import sys
import threading
import gc

# ═══════════════════════════════════════════════════════════════
# ✅ УБРАЛИ: import torch
# ═══════════════════════════════════════════════════════════════

from video_processor import VideoProcessor
from summarizer import Summarizer
from translator import Translator

# ============================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

progress_logger = logging.getLogger('progress')
progress_logger.setLevel(logging.INFO)
if not progress_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | 🔄 %(message)s', datefmt='%H:%M:%S'))
    progress_logger.addHandler(handler)

# ============================================================
# ПУТИ И КОНФИГУРАЦИЯ
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

VERSION_FILE = PROJECT_ROOT / "version"
def get_version() -> str:
    try:
        if VERSION_FILE.exists():
            with open(VERSION_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except:
        pass
    return "1.0.0"

# ============================================================
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ============================================================
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "2000"))
UPLOAD_ALLOWED_EXT = frozenset({".txt", ".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mkv", ".ogg", ".flac"})

logger.info(f"🚀 Whisper: model={WHISPER_MODEL_SIZE}, device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE_TYPE}")

# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================
app = FastAPI(title="AI Видео Транскрибатор", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

video_processor = VideoProcessor()
summarizer = Summarizer()
translator = Translator()

# ============================================================
# ХРАНЕНИЕ СОСТОЯНИЯ ЗАДАЧ
# ============================================================
TASKS_FILE = TEMP_DIR / "tasks.json"
tasks_lock = threading.Lock()

def load_tasks():
    try:
        if TASKS_FILE.exists():
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_tasks(tasks_data):
    try:
        with tasks_lock:
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния задач: {e}")

tasks = load_tasks()
processing_urls = set()
active_tasks = {}
sse_connections = {}

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
async def broadcast_task_update(task_id: str, task_data: dict):
    if task_id in sse_connections:
        connections_to_remove = []
        for queue in sse_connections[task_id]:
            try:
                await queue.put(json.dumps(task_data, ensure_ascii=False))
            except:
                connections_to_remove.append(queue)
        for queue in connections_to_remove:
            sse_connections[task_id].remove(queue)
        if not sse_connections[task_id]:
            del sse_connections[task_id]

def _sanitize_title_for_filename(title: str) -> str:
    if not title:
        return "untitled"
    safe = re.sub(r"[^\w\-\s]", "", title)
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    return safe[:80] or "untitled"

def _txt_to_raw_transcript_markdown(body: str) -> str:
    text = body.strip() if body.strip() else "(empty)"
    return "\n".join([
        "# Video Transcription",
        "",
        "**Detected Language:**",
        "**Language Probability:** —",
        "",
        "## Transcription Content",
        "",
        text,
    ])

# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ: ЗАПУСК ВОРКЕРА
# ============================================================
async def _run_worker_transcription(
    audio_path: str,
    language: Optional[str] = None,
    simple_format: bool = False
) -> str:
    """Запуск транскрибации в отдельном процессе (worker.py)."""
    
    with tempfile.NamedTemporaryFile(
        mode='w+',
        suffix='.txt',
        delete=False,
        encoding='utf-8'
    ) as tmp:
        result_file = tmp.name
    
    try:
        worker_path = PROJECT_ROOT / "backend" / "worker.py"
        if not worker_path.exists():
            logger.warning(f"⚠️ worker.py не найден, используем прямой вызов Transcriber")
            from transcriber import Transcriber
            transcriber = Transcriber(
                model_size=WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            result = await transcriber.transcribe(
                audio_path=audio_path,
                language=language,
                simple_format=simple_format
            )
            del transcriber
            gc.collect()
            # ═══════════════════════════════════════════════════════════
            # ✅ УБРАЛИ: try/except для torch
            # ═══════════════════════════════════════════════════════════
            return result
        
        cmd = [
            "python3",
            str(worker_path),
            audio_path,
            result_file,
            "--model-size", WHISPER_MODEL_SIZE,
            "--device", WHISPER_DEVICE,
            "--compute-type", WHISPER_COMPUTE_TYPE
        ]
        
        if language and language != "auto":
            cmd.extend(["--language", language])
        if simple_format:
            cmd.append("--simple-format")
        
        logger.info(f"🚀 Запуск воркера: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            logger.error(f"❌ Ошибка воркера (код {process.returncode}): {error_msg}")
            raise Exception(f"Ошибка транскрибации: {error_msg[:200]}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result = f.read()
        
        if not result or result.strip() == "":
            raise Exception("Результат транскрибации пуст")
        
        logger.info("✅ Транскрибация в воркере успешно завершена")
        return result
        
    except Exception as e:
        logger.error(f"❌ Ошибка в _run_worker_transcription: {str(e)}")
        raise
    finally:
        try:
            if os.path.exists(result_file):
                os.unlink(result_file)
        except:
            pass

# ============================================================
# ОСНОВНОЙ КОНВЕЙЕР ОБРАБОТКИ
# ============================================================
async def _run_post_extract_pipeline(
    task_id: str,
    raw_script: str,
    video_title: str,
    source_ref: str,
    summary_language: str,
    request_summarizer: Summarizer,
    dedup_url: Optional[str] = None,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
    simple_format: bool = False,
) -> None:
    short_id = task_id.replace("-", "")[:6]
    safe_title = _sanitize_title_for_filename(video_title)

    progress_logger.info(f"📝 Задача {short_id}: Сохранение сырой транскрипции...")
    try:
        raw_md_filename = f"raw_{safe_title}_{short_id}.md"
        raw_md_path = TEMP_DIR / raw_md_filename
        with open(raw_md_path, "w", encoding="utf-8") as f:
            f.write((raw_script or "") + f"\n\nsource: {source_ref}\n")
        tasks[task_id].update({"raw_script_file": raw_md_filename})
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
    except Exception as e:
        logger.error(f"Ошибка сохранения исходной транскрипции: {e}")

    progress_logger.info(f"📝 Задача {short_id}: Оптимизация транскрипции (55%)...")
    tasks[task_id].update({"progress": 55, "message": "Оптимизация текста транскрипции..."})
    save_tasks(tasks)
    await broadcast_task_update(task_id, tasks[task_id])

    if simple_format:
        script = raw_script
    else:
        script = await request_summarizer.optimize_transcript(raw_script)

    script_with_title = f"# {video_title}\n\n{script}\n\nsource: {source_ref}\n"

    detected_language = translator.infer_language_code(raw_script)
    detected_language = translator.normalize_lang_code(detected_language) or detected_language

    logger.info(f"📝 Задача {short_id}: Определенный язык: {detected_language}, язык резюме: {summary_language}")

    translation_content = None
    translation_filename = None
    translation_path = None

    eff_key = (api_key or "").strip()
    eff_base = (model_base_url or "").strip().rstrip("/")
    if eff_key:
        request_translator = Translator(
            api_key=eff_key,
            base_url=eff_base or None,
            model=model_id or None,
        )
    else:
        request_translator = translator

    need_translation = translator.languages_differ_for_translation(
        detected_language, summary_language
    )

    if need_translation:
        progress_logger.info(f"🌍 Задача {short_id}: Перевод {detected_language} -> {summary_language} (70%)...")
        tasks[task_id].update({"progress": 70, "message": "Создание перевода..."})
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        translation_content = await request_translator.translate_text(
            script, summary_language, detected_language
        )
        translation_with_title = f"# {video_title}\n\n{translation_content}\n\nsource: {source_ref}\n"
        translation_filename = f"translation_{safe_title}_{short_id}.md"
        translation_path = TEMP_DIR / translation_filename
        async with aiofiles.open(translation_path, "w", encoding="utf-8") as f:
            await f.write(translation_with_title)
        progress_logger.info(f"🌍 Задача {short_id}: Перевод завершен")

    progress_logger.info(f"📊 Задача {short_id}: Создание резюме (80%)...")
    tasks[task_id].update({"progress": 80, "message": "Создание резюме..."})
    save_tasks(tasks)
    await broadcast_task_update(task_id, tasks[task_id])

    summary = await request_summarizer.summarize(script, summary_language, video_title)
    summary_with_source = summary + f"\n\nsource: {source_ref}\n"

    script_filename = f"transcript_{task_id}.md"
    script_path = TEMP_DIR / script_filename
    async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
        await f.write(script_with_title)

    new_script_filename = f"transcript_{safe_title}_{short_id}.md"
    new_script_path = TEMP_DIR / new_script_filename
    try:
        if script_path.exists():
            script_path.rename(new_script_path)
            script_path = new_script_path
    except Exception:
        pass

    summary_filename = f"summary_{safe_title}_{short_id}.md"
    summary_path = TEMP_DIR / summary_filename
    async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
        await f.write(summary_with_source)

    task_result = {
        "status": "completed",
        "progress": 100,
        "message": "Обработка завершена!",
        "video_title": video_title,
        "script": script_with_title,
        "summary": summary_with_source,
        "script_path": str(script_path),
        "summary_path": str(summary_path),
        "short_id": short_id,
        "safe_title": safe_title,
        "detected_language": detected_language,
        "summary_language": summary_language,
        "simple_format": simple_format,
    }

    if translation_content and translation_path:
        task_result.update({
            "translation": translation_with_title,
            "translation_path": str(translation_path),
            "translation_filename": translation_filename,
        })

    tasks[task_id].update(task_result)
    save_tasks(tasks)
    progress_logger.info(f"✅ Задача {short_id}: Обработка завершена!")
    await broadcast_task_update(task_id, tasks[task_id])

    if dedup_url:
        processing_urls.discard(dedup_url)
    if task_id in active_tasks:
        del active_tasks[task_id]

# ============================================================
# ЭНДПОИНТЫ API
# ============================================================
@app.get("/")
async def read_root():
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))

@app.post("/api/models")
async def list_models(
    base_url: str = Form(default=""),
    api_key: str = Form(default=""),
):
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
    effective_url = base_url.rstrip("/") or os.getenv("OPENAI_BASE_URL") or None

    if not effective_key:
        raise HTTPException(status_code=400, detail="Требуется API ключ")

    try:
        client = openai.OpenAI(api_key=effective_key, base_url=effective_url)
        resp = await asyncio.to_thread(client.models.list)
        models = [{"id": m.id, "name": getattr(m, "name", m.id)} for m in resp.data]
        models.sort(key=lambda x: x["id"])
        return {"data": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/process-video")
async def process_video(
    url: str = Form(default=""),
    summary_language: str = Form(default="zh"),
    transcription_language: str = Form(default="auto"),
    simple_format: str = Form(default="false"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
    file: Optional[UploadFile] = File(None),
):
    try:
        simple_format_bool = simple_format.lower() == "true"
        
        if file is not None and (file.filename or "").strip():
            raw_name = file.filename or "upload.bin"
            if ".." in raw_name or "/" in raw_name or "\\" in raw_name:
                raise HTTPException(status_code=400, detail="Недопустимое имя файла")
            safe_name = os.path.basename(raw_name)
            ext = Path(safe_name).suffix.lower()
            if ext not in UPLOAD_ALLOWED_EXT:
                raise HTTPException(
                    status_code=400,
                    detail=f"Неподдерживаемый тип файла: {ext or '(none)'}",
                )

            max_bytes = UPLOAD_MAX_MB * 1024 * 1024
            task_id = str(uuid.uuid4())
            unique_stem = task_id.replace("-", "")[:12]
            dest = TEMP_DIR / f"upload_{unique_stem}{ext}"

            total = 0
            with open(dest, "wb") as out_f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        try:
                            dest.unlink(missing_ok=True)
                        except:
                            pass
                        raise HTTPException(
                            status_code=413,
                            detail=f"Файл превышает лимит {UPLOAD_MAX_MB} МБ",
                        )
                    out_f.write(chunk)

            if total == 0:
                try:
                    dest.unlink(missing_ok=True)
                except:
                    pass
                raise HTTPException(status_code=400, detail="Пустой файл")

            video_title = _sanitize_title_for_filename(Path(safe_name).stem) or "upload"
            
            tasks[task_id] = {
                "status": "processing",
                "progress": 0,
                "message": "Начало обработки загруженного файла...",
                "script": None,
                "summary": None,
                "error": None,
                "url": f"upload:{safe_name}",
                "transcription_language": transcription_language,
                "simple_format": simple_format_bool,
            }
            save_tasks(tasks)

            task = asyncio.create_task(
                process_upload_task(
                    task_id, dest, safe_name, video_title, ext,
                    summary_language, transcription_language,
                    simple_format_bool, api_key, model_base_url, model_id
                )
            )
            active_tasks[task_id] = task
            return {"task_id": task_id, "message": "Задача создана, обработка выполняется..."}

        stripped = (url or "").strip()
        if not stripped:
            raise HTTPException(
                status_code=400,
                detail="Укажите URL видео или загрузите файл",
            )

        url = stripped
        logger.info(f"🔗 Обработка URL: {url}")

        if url in processing_urls:
            for tid, task in tasks.items():
                if task.get("url") == url:
                    logger.info(f"ℹ️ URL уже обрабатывается: {url}, задача: {tid}")
                    return {"task_id": tid, "message": "Это видео уже обрабатывается, пожалуйста, подождите..."}
            
        task_id = str(uuid.uuid4())
        short_id = task_id.replace("-", "")[:6]
        progress_logger.info(f"📝 Задача {short_id}: Создана для URL: {url[:60]}...")
        
        processing_urls.add(url)
        
        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Начало обработки видео...",
            "script": None,
            "summary": None,
            "error": None,
            "url": url,
            "transcription_language": transcription_language,
            "simple_format": simple_format_bool,
        }
        save_tasks(tasks)
        
        task = asyncio.create_task(
            process_video_task(
                task_id, url, summary_language, transcription_language,
                simple_format_bool, api_key, model_base_url, model_id
            )
        )
        active_tasks[task_id] = task
        
        return {"task_id": task_id, "message": "Задача создана, обработка выполняется..."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка обработки видео: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

async def process_video_task(
    task_id: str,
    url: str,
    summary_language: str,
    transcription_language: str = "auto",
    simple_format: bool = False,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
):
    short_id = task_id.replace("-", "")[:6]
    try:
        progress_logger.info(f"📝 Задача {short_id}: Проверка наличия субтитров (10%)...")
        tasks[task_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Проверка наличия субтитров..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        await asyncio.sleep(0.1)

        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(f"📝 Задача {short_id}: Использование API Key с фронтенда")
        else:
            request_summarizer = summarizer

        subtitle_text, sub_title, sub_lang = await video_processor.fetch_subtitles(url, TEMP_DIR)

        if subtitle_text:
            video_title = sub_title
            raw_script = subtitle_text

            progress_logger.info(f"✅ Задача {short_id}: Субтитры получены ({sub_lang}) (40%)")
            tasks[task_id].update({
                "progress": 40,
                "message": f"Субтитры получены ({sub_lang}), обработка текста..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
        else:
            progress_logger.info(f"🎵 Задача {short_id}: Субтитры не найдены, загрузка аудио (15%)...")
            tasks[task_id].update({
                "progress": 15,
                "message": "Субтитры не найдены, загрузка аудио видео..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            progress_logger.info(f"🎵 Задача {short_id}: Загрузка аудио...")
            audio_path, video_title = await video_processor.download_and_convert(
                url, TEMP_DIR, prefetched_title=sub_title or None
            )
            progress_logger.info(f"✅ Задача {short_id}: Аудио загружено ({os.path.getsize(audio_path) / (1024*1024):.1f} МБ)")

            tasks[task_id].update({
                "progress": 35,
                "message": "Аудио загружено, подготовка к транскрипции..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            progress_logger.info(f"🎙️ Задача {short_id}: Транскрипция аудио (Whisper) (40%)...")
            tasks[task_id].update({
                "progress": 40,
                "message": "Транскрипция аудио (Whisper)..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            trans_lang = None if transcription_language == "auto" else transcription_language
            logger.info(f"🎙️ Задача {short_id}: Язык транскрипции: {trans_lang if trans_lang else 'автоопределение'}")
            logger.info(f"📄 Задача {short_id}: Формат транскрипции: {'простой' if simple_format else 'Markdown'}")
            
            raw_script = await _run_worker_transcription(
                audio_path, 
                language=trans_lang, 
                simple_format=simple_format
            )
            progress_logger.info(f"✅ Задача {short_id}: Транскрипция завершена")

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=url,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            dedup_url=url,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
            simple_format=simple_format,
        )

    except Exception as e:
        logger.error(f"❌ Задача {short_id}: Ошибка: {str(e)}")
        processing_urls.discard(url)
        
        if task_id in active_tasks:
            del active_tasks[task_id]
            
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"Ошибка обработки: {str(e)}"
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

@app.post("/api/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    summary_language: str = Form(default="zh"),
    transcription_language: str = Form(default="auto"),
    simple_format: str = Form(default="false"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
):
    simple_format_bool = simple_format.lower() == "true"
    return await process_video(
        url="",
        summary_language=summary_language,
        transcription_language=transcription_language,
        simple_format=str(simple_format_bool).lower(),
        api_key=api_key,
        model_base_url=model_base_url,
        model_id=model_id,
        file=file
    )

async def process_upload_task(
    task_id: str,
    saved_path: Path,
    original_name: str,
    video_title: str,
    ext_lower: str,
    summary_language: str,
    transcription_language: str = "auto",
    simple_format: bool = False,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
):
    short_id = task_id.replace("-", "")[:6]
    source_ref = f"upload:{original_name}"
    try:
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(f"📝 Задача {short_id}: Использование API Key с фронтенда")
        else:
            request_summarizer = summarizer

        if ext_lower == ".txt":
            progress_logger.info(f"📄 Задача {short_id}: Чтение текстового файла (20%)...")
            tasks[task_id].update({
                "progress": 20,
                "message": "Чтение текстового файла...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            body = saved_path.read_text(encoding="utf-8", errors="replace")
            if not body.strip():
                raise Exception("Текстовый файл пуст")
            raw_script = _txt_to_raw_transcript_markdown(body)
            progress_logger.info(f"✅ Задача {short_id}: Текстовый файл прочитан ({len(body)} символов)")
        else:
            progress_logger.info(f"🎵 Задача {short_id}: Преобразование аудиоформата (15%)...")
            tasks[task_id].update({
                "progress": 15,
                "message": "Преобразование аудиоформата...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path = await video_processor.normalize_local_media_to_m4a(saved_path, TEMP_DIR)
            progress_logger.info(f"✅ Задача {short_id}: Аудио преобразовано ({os.path.getsize(audio_path) / (1024*1024):.1f} МБ)")

            tasks[task_id].update({
                "progress": 35,
                "message": "Аудио подготовлено, начало транскрипции...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            progress_logger.info(f"🎙️ Задача {short_id}: Транскрипция аудио (Whisper) (40%)...")
            tasks[task_id].update({
                "progress": 40,
                "message": "Транскрипция аудио (Whisper)...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            trans_lang = None if transcription_language == "auto" else transcription_language
            logger.info(f"🎙️ Задача {short_id}: Язык транскрипции: {trans_lang if trans_lang else 'автоопределение'}")
            logger.info(f"📄 Задача {short_id}: Формат транскрипции: {'простой' if simple_format else 'Markdown'}")
            
            raw_script = await _run_worker_transcription(
                audio_path, 
                language=trans_lang, 
                simple_format=simple_format
            )
            progress_logger.info(f"✅ Задача {short_id}: Транскрипция завершена")

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=source_ref,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            dedup_url=None,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
            simple_format=simple_format,
        )

    except Exception as e:
        logger.error(f"❌ Задача {short_id}: Ошибка: {str(e)}")
        if task_id in active_tasks:
            del active_tasks[task_id]
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"Ошибка обработки: {str(e)}",
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return tasks[task_id]

@app.get("/api/task-stream/{task_id}")
async def task_stream(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    async def event_generator():
        queue = asyncio.Queue()
        
        if task_id not in sse_connections:
            sse_connections[task_id] = []
        sse_connections[task_id].append(queue)
        
        try:
            current_task = tasks.get(task_id, {})
            yield f"data: {json.dumps(current_task, ensure_ascii=False)}\n\n"
            
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"
                    
                    task_data = json.loads(data)
                    if task_data.get("status") in ["completed", "error"]:
                        break
                        
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE соединение отменено: {task_id}")
        except Exception as e:
            logger.error(f"Ошибка SSE потока: {e}")
        finally:
            if task_id in sse_connections and queue in sse_connections[task_id]:
                sse_connections[task_id].remove(queue)
                if not sse_connections[task_id]:
                    del sse_connections[task_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    try:
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Поддерживаются только .md файлы")
        
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Недопустимое имя файла")
            
        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Файл не найден")
            
        return FileResponse(
            file_path,
            filename=filename,
            media_type="text/markdown"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка скачивания файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка скачивания: {str(e)}")

@app.get("/api/download/simple/{task_id}")
async def download_simple_transcript(task_id: str):
    try:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        
        task = tasks[task_id]
        if task.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Задача еще не завершена")
        
        raw_script_file = task.get("raw_script_file")
        if not raw_script_file:
            raise HTTPException(status_code=404, detail="Файл транскрипции не найден")
        
        raw_path = TEMP_DIR / raw_script_file
        if not raw_path.exists():
            raise HTTPException(status_code=404, detail="Файл транскрипции не найден")
        
        return FileResponse(
            raw_path,
            filename=f"transcript_simple_{task.get('safe_title', 'x')}_{task.get('short_id', 'x')}.txt",
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка скачивания простой транскрипции: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка скачивания: {str(e)}")

@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            logger.info(f"Задача {task_id} отменена")
        del active_tasks[task_id]
    
    task_url = tasks[task_id].get("url")
    if task_url:
        processing_urls.discard(task_url)
    
    del tasks[task_id]
    return {"message": "Задача отменена и удалена"}

@app.get("/api/tasks/active")
async def get_active_tasks():
    active_count = len(active_tasks)
    processing_count = len(processing_urls)
    return {
        "active_tasks": active_count,
        "processing_urls": processing_count,
        "task_ids": list(active_tasks.keys())
    }

@app.get("/api/worker-status")
async def worker_status():
    """Проверка состояния воркера и памяти GPU"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "worker.py"],
            capture_output=True,
            text=True
        )
        worker_pids = result.stdout.strip().split() if result.stdout else []
        
        # ═══════════════════════════════════════════════════════════
        # ✅ УБРАЛИ: информацию о CUDA
        # ═══════════════════════════════════════════════════════════
        
        return {
            "active_workers": len(worker_pids),
            "worker_pids": worker_pids,
            "cuda_memory": {}  # Пустой объект вместо информации о CUDA
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/system-info")
async def system_info():
    info = {
        "upload_max_mb": UPLOAD_MAX_MB,
        "version": get_version(),
        "whisper_device": WHISPER_DEVICE,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
        "whisper_model": WHISPER_MODEL_SIZE,
        "worker_mode": True,
    }
    return info

@app.get("/api/config")
async def get_config():
    return {
        "upload_max_mb": UPLOAD_MAX_MB,
        "whisper_model": WHISPER_MODEL_SIZE,
        "whisper_device": WHISPER_DEVICE,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
        "worker_mode": True,
    }

@app.get("/api/version")
async def get_api_version():
    return {"version": get_version()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)