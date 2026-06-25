from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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

from video_processor import VideoProcessor
from transcriber import Transcriber
from summarizer import Summarizer
from translator import Translator

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Получение версии из файла
PROJECT_ROOT = Path(__file__).parent.parent
VERSION_FILE = PROJECT_ROOT / "version"

def get_version() -> str:
    """Получение версии из файла"""
    try:
        if VERSION_FILE.exists():
            with open(VERSION_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        pass
    return "1.0.0"  # Версия по умолчанию

app = FastAPI(title="AI Видео Транскрибатор", version="1.0.0")

# Настройка CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Получение корневой директории проекта
PROJECT_ROOT = Path(__file__).parent.parent

# Подключение статических файлов
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

# Создание директории для временных файлов
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# ── ИНИЦИАЛИЗАЦИЯ ОБРАБОТЧИКОВ С ПАРАМЕТРАМИ ИЗ ENV ──

# Чтение настроек Whisper из переменных окружения
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

logger.info(f"Инициализация Whisper: model={WHISPER_MODEL_SIZE}, device={WHISPER_DEVICE}, compute_type={WHISPER_COMPUTE_TYPE}")

# Создаем экземпляры с параметрами из ENV
video_processor = VideoProcessor()
transcriber = Transcriber(
    model_size=WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)
summarizer = Summarizer()
translator = Translator()

# Хранение состояния задач - с сохранением в файл
import threading

TASKS_FILE = TEMP_DIR / "tasks.json"
tasks_lock = threading.Lock()

def load_tasks():
    """Загрузка состояния задач"""
    try:
        if TASKS_FILE.exists():
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_tasks(tasks_data):
    """Сохранение состояния задач"""
    try:
        with tasks_lock:
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния задач: {e}")

async def broadcast_task_update(task_id: str, task_data: dict):
    """Отправка обновления состояния задачи всем подключенным SSE клиентам"""
    logger.info(f"Трансляция обновления задачи: {task_id}, статус: {task_data.get('status')}, подключений: {len(sse_connections.get(task_id, []))}")
    if task_id in sse_connections:
        connections_to_remove = []
        for queue in sse_connections[task_id]:
            try:
                await queue.put(json.dumps(task_data, ensure_ascii=False))
                logger.debug(f"Сообщение отправлено в очередь: {task_id}")
            except Exception as e:
                logger.warning(f"Ошибка отправки сообщения в очередь: {e}")
                connections_to_remove.append(queue)
        
        # Удаление разорванных соединений
        for queue in connections_to_remove:
            sse_connections[task_id].remove(queue)
        
        # Если соединений нет, очищаем список
        if not sse_connections[task_id]:
            del sse_connections[task_id]

# Загрузка состояния задач при запуске
tasks = load_tasks()
# Хранение обрабатываемых URL для предотвращения дублирования
processing_urls = set()
# Хранение активных задач для управления и отмены
active_tasks = {}
# Хранение SSE соединений для отправки обновлений в реальном времени
sse_connections = {}

# Локальная загрузка: разрешенные типы и максимальный размер (МБ), можно настроить через UPLOAD_MAX_MB
UPLOAD_ALLOWED_EXT = frozenset({".txt", ".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mkv", ".ogg", ".flac"})
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "2000"))


def _sanitize_title_for_filename(title: str) -> str:
    """Преобразование заголовка видео в безопасное имя файла."""
    if not title:
        return "untitled"
    # Оставляем только буквы, цифры, подчеркивания, дефисы и пробелы
    safe = re.sub(r"[^\w\-\s]", "", title)
    # Сжатие пробелов и замена на подчеркивания
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    # Ограничение длины
    return safe[:80] or "untitled"


def _txt_to_raw_transcript_markdown(body: str) -> str:
    """Преобразование обычного текста в Markdown, совместимый с выводом Whisper."""
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
    """Общий конвейер после получения raw_script: архивация, оптимизация, перевод, резюмирование, трансляция."""
    short_id = task_id.replace("-", "")[:6]
    safe_title = _sanitize_title_for_filename(video_title)

    try:
        # Сохраняем сырую транскрипцию
        raw_md_filename = f"raw_{safe_title}_{short_id}.md"
        raw_md_path = TEMP_DIR / raw_md_filename
        with open(raw_md_path, "w", encoding="utf-8") as f:
            f.write((raw_script or "") + f"\n\nsource: {source_ref}\n")
        tasks[task_id].update({"raw_script_file": raw_md_filename})
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
    except Exception as e:
        logger.error(f"Ошибка сохранения исходной транскрипции Markdown: {e}")

    tasks[task_id].update({
        "progress": 55,
        "message": "Оптимизация текста транскрипции...",
    })
    save_tasks(tasks)
    await broadcast_task_update(task_id, tasks[task_id])

    # Если транскрипция в простом формате, пропускаем оптимизацию
    if simple_format:
        logger.info("Транскрипция в простом формате, оптимизация пропущена")
        script = raw_script
    else:
        script = await request_summarizer.optimize_transcript(raw_script)

    script_with_title = f"# {video_title}\n\n{script}\n\nsource: {source_ref}\n"

    detected_language = transcriber.get_detected_language(raw_script)
    detected_language = (detected_language or "").strip()
    if not detected_language:
        detected_language = translator.infer_language_code(raw_script)
    detected_language = translator.normalize_lang_code(detected_language) or detected_language

    logger.info(f"Определенный язык: {detected_language}, язык резюме: {summary_language}")

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
        logger.info(f"Требуется перевод: {detected_language} -> {summary_language}")
        tasks[task_id].update({
            "progress": 70,
            "message": "Создание перевода...",
        })
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
    else:
        logger.info(
            f"Перевод не требуется: detected_language={detected_language}, summary_language={summary_language}, "
            f"need_translation={need_translation}"
        )

    tasks[task_id].update({
        "progress": 80,
        "message": "Создание резюме...",
    })
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
        "simple_format": simple_format,  # Сохраняем информацию о формате
    }

    if translation_content and translation_path:
        task_result.update({
            "translation": translation_with_title,
            "translation_path": str(translation_path),
            "translation_filename": translation_filename,
        })

    tasks[task_id].update(task_result)
    save_tasks(tasks)
    logger.info(f"Задача завершена, отправка финального состояния: {task_id}")
    await broadcast_task_update(task_id, tasks[task_id])
    logger.info(f"Финальное состояние отправлено: {task_id}")

    if dedup_url:
        processing_urls.discard(dedup_url)
    if task_id in active_tasks:
        del active_tasks[task_id]


@app.get("/")
async def read_root():
    """Возвращает главную страницу"""
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))

@app.post("/api/models")
async def list_models(
    base_url: str = Form(default=""),
    api_key:  str = Form(default=""),
):
    """Прокси: получение списка моделей от любого OpenAI-совместимого API."""
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
    effective_url = base_url.rstrip("/") or os.getenv("OPENAI_BASE_URL") or None

    if not effective_key:
        raise HTTPException(status_code=400, detail="Требуется API ключ")

    try:
        client = openai.OpenAI(api_key=effective_key, base_url=effective_url)
        resp   = await asyncio.to_thread(client.models.list)
        models = [{"id": m.id, "name": getattr(m, "name", m.id)} for m in resp.data]
        # Сортировка для удобства чтения
        models.sort(key=lambda x: x["id"])
        return {"data": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _enqueue_upload_job(
    file: UploadFile,
    summary_language: str,
    transcription_language: str,
    simple_format: bool,
    api_key: str,
    model_base_url: str,
    model_id: str,
) -> dict:
    """Сохранение загруженного файла и постановка в очередь process_upload_task, возвращает {task_id, message}."""
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
                except Exception:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл превышает лимит {UPLOAD_MAX_MB} МБ",
                )
            out_f.write(chunk)

    if total == 0:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Пустой файл")

    video_title = _sanitize_title_for_filename(Path(safe_name).stem) or "upload"
    source_label = f"upload:{safe_name}"

    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Начало обработки загруженного файла...",
        "script": None,
        "summary": None,
        "error": None,
        "url": source_label,
        "transcription_language": transcription_language,
        "simple_format": simple_format,
    }
    save_tasks(tasks)

    bg = asyncio.create_task(
        process_upload_task(
            task_id,
            dest,
            safe_name,
            video_title,
            ext,
            summary_language,
            transcription_language,
            simple_format,
            api_key,
            model_base_url,
            model_id,
        )
    )
    active_tasks[task_id] = bg

    return {"task_id": task_id, "message": "Задача создана, обработка выполняется..."}


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
    """
    Обработка ссылки на видео или локальной загрузки (если передан file и нет URL).
    Загрузка и URL используют общий путь, что удобно для обратных прокси,
    разрешающих только /api/process-video.
    """
    try:
        simple_format_bool = simple_format.lower() == "true"
        
        if file is not None and (file.filename or "").strip():
            return await _enqueue_upload_job(
                file, summary_language, transcription_language, simple_format_bool, api_key, model_base_url, model_id
            )

        stripped = (url or "").strip()
        if not stripped:
            raise HTTPException(
                status_code=400,
                detail="Укажите URL видео или загрузите файл",
            )

        url = stripped

        # Проверка, не обрабатывается ли уже этот URL
        if url in processing_urls:
            # Поиск существующей задачи
            for tid, task in tasks.items():
                if task.get("url") == url:
                    return {"task_id": tid, "message": "Это видео уже обрабатывается, пожалуйста, подождите..."}
            
        # Генерация уникального ID задачи
        task_id = str(uuid.uuid4())
        
        # Отметка URL как обрабатываемого
        processing_urls.add(url)
        
        # Инициализация состояния задачи
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
        
        # Создание и отслеживание асинхронной задачи
        task = asyncio.create_task(process_video_task(
            task_id, url, summary_language, transcription_language, simple_format_bool, api_key, model_base_url, model_id
        ))
        active_tasks[task_id] = task
        
        return {"task_id": task_id, "message": "Задача создана, обработка выполняется..."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки видео: {str(e)}")
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
    """
    Асинхронная обработка задачи видео
    """
    try:
        # ── Этап 1: Попытка получить субтитры с платформы (быстрый путь) ──
        tasks[task_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Проверка наличия субтитров..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        await asyncio.sleep(0.1)

        # Если с фронтенда переданы учетные данные API, создаем отдельный Summarizer
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(f"Использование API Key с фронтенда, base_url={effective_url}, model={model_id or 'default'}")
        else:
            request_summarizer = summarizer

        subtitle_text, sub_title, sub_lang = await video_processor.fetch_subtitles(url, TEMP_DIR)

        if subtitle_text:
            # ── Быстрый путь: есть субтитры, пропускаем загрузку аудио и Whisper ──
            video_title = sub_title
            raw_script = subtitle_text
            # Сохраняем язык в transcriber для совместимости
            transcriber.last_detected_language = sub_lang

            tasks[task_id].update({
                "progress": 40,
                "message": f"Субтитры получены ({sub_lang}), обработка текста..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
        else:
            # ── Медленный путь: нет субтитров, загрузка аудио → Whisper ──
            tasks[task_id].update({
                "progress": 15,
                "message": "Субтитры не найдены, загрузка аудио видео..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path, video_title = await video_processor.download_and_convert(
                url, TEMP_DIR, prefetched_title=sub_title or None
            )

            tasks[task_id].update({
                "progress": 35,
                "message": "Аудио загружено, подготовка к транскрипции..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            tasks[task_id].update({
                "progress": 40,
                "message": "Транскрипция аудио (Whisper)..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            # Если выбран конкретный язык (не auto), передаем его в Whisper для ускорения
            trans_lang = None if transcription_language == "auto" else transcription_language
            logger.info(f"Язык транскрипции: {trans_lang if trans_lang else 'автоопределение'}")
            logger.info(f"Формат транскрипции: {'простой' if simple_format else 'Markdown'}")
            raw_script = await transcriber.transcribe(audio_path, language=trans_lang, simple_format=simple_format)

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
        logger.error(f"Ошибка обработки задачи {task_id}: {str(e)}")
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
    """Отдельный эндпоинт для загрузки; логика аналогична /api/process-video с file."""
    simple_format_bool = simple_format.lower() == "true"
    return await _enqueue_upload_job(
        file, summary_language, transcription_language, simple_format_bool, api_key, model_base_url, model_id
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
    source_ref = f"upload:{original_name}"
    try:
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(
                f"Задача загрузки использует API Key с фронтенда, base_url={effective_url}, model={model_id or 'default'}"
            )
        else:
            request_summarizer = summarizer

        if ext_lower == ".txt":
            tasks[task_id].update({
                "progress": 20,
                "message": "Чтение текстового файла...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            body = saved_path.read_text(encoding="utf-8", errors="replace")
            if not body.strip():
                raise Exception("Текстовый файл пуст")
            transcriber.last_detected_language = None
            raw_script = _txt_to_raw_transcript_markdown(body)
        else:
            tasks[task_id].update({
                "progress": 15,
                "message": "Преобразование аудиоформата...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path = await video_processor.normalize_local_media_to_m4a(saved_path, TEMP_DIR)

            tasks[task_id].update({
                "progress": 35,
                "message": "Аудио подготовлено, начало транскрипции...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            tasks[task_id].update({
                "progress": 40,
                "message": "Транскрипция аудио (Whisper)...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            # Если выбран конкретный язык (не auto), передаем его в Whisper для ускорения
            trans_lang = None if transcription_language == "auto" else transcription_language
            logger.info(f"Язык транскрипции (загрузка): {trans_lang if trans_lang else 'автоопределение'}")
            logger.info(f"Формат транскрипции: {'простой' if simple_format else 'Markdown'}")
            raw_script = await transcriber.transcribe(audio_path, language=trans_lang, simple_format=simple_format)

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
        logger.error(f"Ошибка обработки задачи {task_id}: {str(e)}")
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
    """
    Получение статуса задачи
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return tasks[task_id]

@app.get("/api/task-stream/{task_id}")
async def task_stream(task_id: str):
    """
    SSE поток статуса задачи в реальном времени
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    async def event_generator():
        # Создание очереди для задачи
        queue = asyncio.Queue()
        
        # Добавление очереди в список соединений
        if task_id not in sse_connections:
            sse_connections[task_id] = []
        sse_connections[task_id].append(queue)
        
        try:
            # Немедленная отправка текущего состояния
            current_task = tasks.get(task_id, {})
            yield f"data: {json.dumps(current_task, ensure_ascii=False)}\n\n"
            
            # Постоянный мониторинг обновлений
            while True:
                try:
                    # Ожидание обновления с таймаутом 30 секунд (heartbeat)
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"
                    
                    # Если задача завершена или ошибка, завершаем поток
                    task_data = json.loads(data)
                    if task_data.get("status") in ["completed", "error"]:
                        break
                        
                except asyncio.TimeoutError:
                    # Отправка heartbeat для поддержания соединения
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE соединение отменено: {task_id}")
        except Exception as e:
            logger.error(f"Ошибка SSE потока: {e}")
        finally:
            # Очистка соединения
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
    """
    Скачивание файла из директории temp (упрощенная схема)
    """
    try:
        # Проверка расширения файла
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Поддерживаются только .md файлы")
        
        # Проверка имени файла (защита от path traversal)
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
    """
    Скачивание транскрипции в простом формате (без Markdown)
    """
    try:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        
        task = tasks[task_id]
        if task.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Задача еще не завершена")
        
        # Получаем сырую транскрипцию из файла
        raw_script_file = task.get("raw_script_file")
        if not raw_script_file:
            raise HTTPException(status_code=404, detail="Файл транскрипции не найден")
        
        raw_path = TEMP_DIR / raw_script_file
        if not raw_path.exists():
            raise HTTPException(status_code=404, detail="Файл транскрипции не найден")
        
        with open(raw_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Извлекаем только транскрипцию без метаданных
        lines = content.split('\n')
        transcript_lines = []
        in_transcript = False
        
        for line in lines:
            if line.startswith('## Transcription Content'):
                in_transcript = True
                continue
            if in_transcript and line.strip():
                # Убираем временные метки в формате **[HH:MM - HH:MM]**
                if line.startswith('**[') and line.endswith(']**'):
                    continue
                transcript_lines.append(line)
        
        simple_text = '\n'.join(transcript_lines).strip()
        
        # Сохраняем как .txt
        filename = f"transcript_simple_{task.get('safe_title', 'x')}_{task.get('short_id', 'x')}.txt"
        
        return FileResponse(
            raw_path,
            filename=filename,
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка скачивания простой транскрипции: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка скачивания: {str(e)}")


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """
    Отмена и удаление задачи
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    # Если задача еще выполняется, отменяем её    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            logger.info(f"Задача {task_id} отменена")
        del active_tasks[task_id]
    
    # Удаление URL из списка обрабатываемых
    task_url = tasks[task_id].get("url")
    if task_url:
        processing_urls.discard(task_url)
    
    # Удаление записи задачи
    del tasks[task_id]
    return {"message": "Задача отменена и удалена"}

@app.get("/api/tasks/active")
async def get_active_tasks():
    """
    Получение списка активных задач (для отладки)
    """
    active_count = len(active_tasks)
    processing_count = len(processing_urls)
    return {
        "active_tasks": active_count,
        "processing_urls": processing_count,
        "task_ids": list(active_tasks.keys())
    }

@app.get("/api/system-info")
async def system_info():
    """Информация о системе и доступных устройствах"""
    info = {
        "version": get_version(),
        "whisper_device": transcriber.device,
        "whisper_compute_type": transcriber.compute_type,
        "whisper_model": transcriber.model_size,
    }
    return info

@app.get("/api/version")
async def get_api_version():
    """Получение версии приложения"""
    return {"version": get_version()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)