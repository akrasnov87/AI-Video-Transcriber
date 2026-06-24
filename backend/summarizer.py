import os
import openai
import logging
from typing import Optional

from llm_sanitize import strip_llm_artifacts

logger = logging.getLogger(__name__)

class Summarizer:
    """Текстовый суммаризатор, использующий OpenAI API для создания многоязычных резюме"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        Инициализация суммаризатора.

        Приоритет: параметры > переменные окружения.
        Если указан model, он используется как для fast_model, так и для advanced_model.
        """
        effective_key = api_key or os.getenv("OPENAI_API_KEY")
        effective_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not effective_key:
            logger.warning("OPENAI_API_KEY не установлен, функция создания резюме будет недоступна")

        if effective_key:
            kwargs = {"api_key": effective_key}
            if effective_url:
                kwargs["base_url"] = effective_url
                logger.info(f"Клиент OpenAI инициализирован, base_url={effective_url}")
            else:
                logger.info("Клиент OpenAI инициализирован, используется стандартный эндпоинт")
            self.client = openai.OpenAI(**kwargs)
        else:
            self.client = None

        # Возможность указать модель на фронтенде, переопределяя жестко заданные gpt-3.5-turbo / gpt-4o
        self.fast_model     = model or "gpt-3.5-turbo"
        self.advanced_model = model or "gpt-4o"
        
        # Поддерживаемые языки
        self.language_map = {
            "en": "English",
            "zh": "中文（简体）",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ja": "日本語",
            "ko": "한국어",
            "ar": "العربية"
        }
    
    async def optimize_transcript(self, raw_transcript: str) -> str:
        """
        Оптимизация текста транскрипции: исправление опечаток, разделение на абзацы по смыслу
        Поддерживает автоматическую обработку длинных текстов по частям
        
        Args:
            raw_transcript: Исходный текст транскрипции
            
        Returns:
            Оптимизированный текст транскрипции (в формате Markdown)
        """
        try:
            if not self.client:
                logger.warning("OpenAI API недоступен, возвращена исходная транскрипция")
                return raw_transcript

            # Предобработка: удаление временных меток и метаданных
            preprocessed = self._remove_timestamps_and_meta(raw_transcript)
            detected_lang_code = self._detect_transcript_language(preprocessed)
            max_chars_per_chunk = 4000  # Максимум символов на блок

            if len(preprocessed) > max_chars_per_chunk:
                logger.info(f"Длинный текст ({len(preprocessed)} символов), включена пофрагментная обработка")
                return await self._format_long_transcript_in_chunks(preprocessed, detected_lang_code, max_chars_per_chunk)
            else:
                return await self._format_single_chunk(preprocessed, detected_lang_code)

        except Exception as e:
            logger.error(f"Ошибка оптимизации транскрипции: {str(e)}")
            logger.info("Возвращена исходная транскрипция")
            return raw_transcript

    def _estimate_tokens(self, text: str) -> int:
        """
        Улучшенный алгоритм оценки количества токенов
        Более консервативная оценка с учетом системного промпта и форматирования
        """
        # Более консервативная оценка с учетом реального расширения токенов
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in text.split() if word.isascii() and word.isalpha()])
        
        # Базовые токены
        base_tokens = chinese_chars * 1.5 + english_words * 1.3
        
        # Учет разметки Markdown, временных меток и т.д. (~30% дополнительных затрат)
        format_overhead = len(text) * 0.15
        
        # Учет системного промпта (~2000-3000 токенов)
        system_prompt_overhead = 2500
        
        total_estimated = int(base_tokens + format_overhead + system_prompt_overhead)
        
        return total_estimated

    async def _optimize_single_chunk(self, raw_transcript: str) -> str:
        """
        Оптимизация одного текстового блока
        """
        detected_lang = self._detect_transcript_language(raw_transcript)
        lang_instruction = self._get_language_instruction(detected_lang)
        
        system_prompt = f"""Вы профессиональный редактор текста. Оптимизируйте предоставленный текст транскрипции видео.

Особое внимание: это может быть интервью, диалог или выступление. Если присутствует несколько говорящих, необходимо строго сохранять исходную перспективу каждого говорящего.

Требования:
1. **Строго сохраняйте исходный язык ({lang_instruction}), НИКОГДА не переводите на другие языки**
2. **Полностью удалите все временные метки (например, [00:00 - 00:05])**
3. **Интеллектуально определяйте и объединяйте полные предложения, разбитые временными метками**, грамматически неполные фрагменты должны быть объединены с контекстом
4. Исправляйте явные опечатки и грамматические ошибки
5. Разбивайте восстановленные полные предложения на естественные абзацы по смыслу и логике
6. Разделяйте абзацы пустыми строками
7. **Строго сохраняйте исходный смысл, не добавляйте и не удаляйте фактическое содержание**
8. **НИКОГДА не меняйте личные местоимения (I/я, you/ты/вы, he/он, she/она и т.д.)**
9. **Сохраняйте исходную перспективу и контекст каждого говорящего**
10. **Определяйте структуру диалога: интервьюер использует "you", респондент использует "I/we" — НЕ ПУТАЙТЕ**
11. Убедитесь, что каждое предложение грамматически завершено, язык плавный и естественный

Стратегия обработки:
- Сначала определяйте неполные фрагменты предложений (заканчивающиеся на предлоги, союзы, прилагательные)
- Просматривайте соседние фрагменты текста для объединения в полные предложения
- Разбивайте предложения заново, обеспечивая грамматическую завершенность
- Разбивайте на абзацы по темам и логике

Требования к абзацам:
- По темам и логическому смыслу, каждый абзац содержит 1-8 связанных предложений
- Длина одного абзаца не более 400 символов
- Избегайте слишком коротких абзацев, объединяйте связанный контент
- Разбивайте, когда завершена одна мысль или точка зрения

Формат вывода:
- Только текст абзацев, без временных меток или форматирования
- Каждое предложение структурно завершено
- Каждый абзац раскрывает одну основную тему
- Абзацы разделены пустыми строками

Важно: это текст на {lang_instruction}, оптимизируйте строго на {lang_instruction}, особое внимание уделите устранению несвязности из-за разбиения временными метками! Обязательно выполняйте разумное разделение на абзацы, избегайте слишком длинных абзацев!

**Ключевое требование: это может быть диалог интервью, НИКОГДА не меняйте личные местоимения или перспективу говорящего! Интервьюер говорит "you", респондент говорит "I/we" — это должно строго сохраняться!**"""

        user_prompt = f"""Оптимизируйте следующий текст транскрипции видео на {lang_instruction} в плавный текст с абзацами:

{raw_transcript}

Основные задачи:
1. Удалить все временные метки
2. Определить и восстановить разбитые полные предложения
3. Обеспечить грамматическую завершенность и смысловую связность каждого предложения
4. Разбить на абзацы по смыслу, разделяя их пустыми строками
5. Сохранить язык {lang_instruction}

Руководство по разбиению:
- По темам и логическому смыслу, каждый абзац содержит 1-8 связанных предложений
- Длина одного абзаца не более 400 символов
- Избегайте слишком коротких абзацев, объединяйте связанный контент
- Обязательно разделяйте абзацы пустыми строками

Особое внимание уделите восстановлению предложений, разбитых временными метками, и правильному разделению на абзацы!"""

        response = self.client.chat.completions.create(
            model=self.fast_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        return strip_llm_artifacts(response.choices[0].message.content or "")

    async def _optimize_with_chunks(self, raw_transcript: str, max_tokens: int) -> str:
        """
        Пофрагментная оптимизация длинного текста
        """
        detected_lang = self._detect_transcript_language(raw_transcript)
        lang_instruction = self._get_language_instruction(detected_lang)
        
        # Разбивка исходной транскрипции по абзацам (с учетом временных меток для ориентира)
        chunks = self._split_into_chunks(raw_transcript, max_tokens)
        logger.info(f"Разбито на {len(chunks)} фрагментов для обработки")
        
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Оптимизация фрагмента {i+1}/{len(chunks)}...")
            
            system_prompt = f"""Вы профессиональный редактор текста. Выполните простую оптимизацию этого фрагмента транскрипции.

Это часть {i+1} из {len(chunks)} полного текста.

Требования простой оптимизации:
1. **Строго сохраняйте исходный язык ({lang_instruction})** — НИКОГДА не переводите
2. **Исправляйте только явные опечатки и грамматические ошибки**
3. **Незначительно улучшайте плавность предложений**, но не переписывайте кардинально
4. **Сохраняйте исходную структуру и длину**, не выполняйте сложную перегруппировку абзацев
5. **Сохраняйте исходный смысл на 100%**

Важно: это только предварительная очистка, не выполняйте сложное переписывание или реорганизацию."""

            user_prompt = f"""Выполните простую оптимизацию этого фрагмента текста на {lang_instruction} (только исправление опечаток и грамматики):

{chunk}

Выведите очищенный текст, сохраняя исходную структуру."""

            try:
                response = self.client.chat.completions.create(
                    model=self.fast_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1200,
                    temperature=0.1
                )
                
                optimized_chunk = strip_llm_artifacts(response.choices[0].message.content or "")
                optimized_chunks.append(optimized_chunk)
                
            except Exception as e:
                logger.error(f"Ошибка оптимизации фрагмента {i+1}: {e}")
                # При ошибке используем базовую очистку
                cleaned_chunk = self._basic_transcript_cleanup(chunk)
                optimized_chunks.append(cleaned_chunk)
        
        # Объединение всех оптимизированных фрагментов
        merged_text = "\n\n".join(optimized_chunks)
        
        # Финальная обработка абзацев
        logger.info("Выполняется финальная обработка абзацев...")
        final_result = await self._final_paragraph_organization(merged_text, lang_instruction)
        
        logger.info("Пофрагментная оптимизация завершена")
        return final_result

    # ===== Перенос из JS openaiService.js: разбивка/контекст/дедупликация/форматирование =====

    def _ensure_markdown_paragraphs(self, text: str) -> str:
        """Обеспечение правильного форматирования Markdown: пустые строки между абзацами, после заголовков, удаление лишних переносов."""
        if not text:
            return text
        formatted = text.replace("\r\n", "\n")
        import re
        # Добавление пустой строки после заголовков
        formatted = re.sub(r"(^#{1,6}\s+.*)\n([^\n#])", r"\1\n\n\2", formatted, flags=re.M)
        # Сжатие ≥3 переносов до 2
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)
        # Удаление пустых строк в начале и конце
        formatted = re.sub(r"^\n+", "", formatted)
        formatted = re.sub(r"\n+$", "", formatted)
        return formatted

    async def _format_single_chunk(self, chunk_text: str, transcript_language: str = 'zh') -> str:
        """Оптимизация одного блока (исправление + форматирование), соблюдая лимит 4000 токенов."""
        # Формирование промптов, аналогичных JS-версии
        if transcript_language == 'zh':
            prompt = (
                "Выполните интеллектуальную оптимизацию и форматирование следующего текста аудиотранскрипции:\n\n"
                "**Оптимизация содержания (приоритет точности):**\n"
                "1. Исправление ошибок (опечатки/омонимы/имена собственные)\n"
                "2. Умеренное улучшение грамматики, завершение неполных предложений, сохранение исходного языка и смысла\n"
                "3. Обработка устной речи: сохраняйте естественные повторы и междометия, НЕ удаляйте контент, добавляйте только необходимую пунктуацию\n"
                "4. **НИКОГДА не меняйте личные местоимения (I/я, you/ты/вы и т.д.) и перспективу говорящего**\n\n"
                "**Правила разбиения на абзацы:**\n"
                "- По темам и логическому смыслу, каждый абзац содержит 1-8 связанных предложений\n"
                "- Длина одного абзаца не более 400 символов\n"
                "- Избегайте слишком коротких абзацев, объединяйте связанный контент\n\n"
                "**Требования к формату:** абзацы в Markdown с пустыми строками между ними\n\n"
                f"Исходный текст транскрипции:\n{chunk_text}"
            )
            system_prompt = (
                "Вы профессиональный помощник по оптимизации текстов аудиотранскрипций. Исправляйте ошибки, улучшайте связность и форматирование, "
                "строго сохраняйте исходный смысл, НЕ удаляйте устную речь, повторы и детали; удаляйте только временные метки и метаданные. "
                "НИКОГДА не меняйте личные местоимения или перспективу говорящего. Это может быть интервью: интервьюер использует 'you', респондент использует 'I/we'."
            )
        else:
            prompt = (
                "Please intelligently optimize and format the following audio transcript text:\n\n"
                "Content Optimization (Accuracy First):\n"
                "1. Error Correction (typos, homophones, proper nouns)\n"
                "2. Moderate grammar improvement, complete incomplete sentences, keep original language/meaning\n"
                "3. Speech processing: keep natural fillers and repetitions, do NOT remove content; only add punctuation if needed\n"
                "4. **NEVER change pronouns (I, you, he, she, etc.) or speaker perspective**\n\n"
                "Segmentation Rules: Group 1-8 related sentences per paragraph by topic/logic; paragraph length NOT exceed 400 characters; avoid too many short paragraphs\n\n"
                "Format: Markdown paragraphs with blank lines between paragraphs\n\n"
                f"Original transcript text:\n{chunk_text}"
            )
            system_prompt = (
                "You are a professional transcript formatting assistant. Fix errors and improve fluency "
                "without changing meaning or removing any content; only timestamps/meta may be removed; keep Markdown paragraphs with blank lines. "
                "NEVER change pronouns or speaker perspective. This may be an interview: interviewer uses 'you', interviewee uses 'I/we'."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            optimized_text = strip_llm_artifacts(response.choices[0].message.content or "")
            # Удаление заголовков типа "# Transcript" / "## Transcript"
            optimized_text = self._remove_transcript_heading(optimized_text)
            enforced = self._enforce_paragraph_max_chars(optimized_text.strip(), max_chars=400)
            return self._ensure_markdown_paragraphs(enforced)
        except Exception as e:
            logger.error(f"Ошибка оптимизации одного блока: {e}")
            return self._apply_basic_formatting(chunk_text)

    def _smart_split_long_chunk(self, text: str, max_chars_per_chunk: int) -> list:
        """Безопасная разбивка очень длинного текста по границам предложений/пробелов."""
        chunks = []
        pos = 0
        while pos < len(text):
            end = min(pos + max_chars_per_chunk, len(text))
            if end < len(text):
                # Приоритет: границы предложений
                sentence_endings = ['。', '！', '？', '.', '!', '?']
                best = -1
                for ch in sentence_endings:
                    idx = text.rfind(ch, pos, end)
                    if idx > best:
                        best = idx
                if best > pos + int(max_chars_per_chunk * 0.7):
                    end = best + 1
                else:
                    # Второй приоритет: пробелы
                    space_idx = text.rfind(' ', pos, end)
                    if space_idx > pos + int(max_chars_per_chunk * 0.8):
                        end = space_idx
            chunks.append(text[pos:end].strip())
            pos = end
        return [c for c in chunks if c]

    def _find_safe_cut_point(self, text: str) -> int:
        """Поиск безопасной точки разбиения (абзац > предложение > фраза)."""
        import re
        # Абзац
        p = text.rfind("\n\n")
        if p > 0:
            return p + 2
        # Предложение
        last_sentence_end = -1
        for m in re.finditer(r"[。！？\.!?]\s*", text):
            last_sentence_end = m.end()
        if last_sentence_end > 20:
            return last_sentence_end
        # Фраза
        last_phrase_end = -1
        for m in re.finditer(r"[，；,;]\s*", text):
            last_phrase_end = m.end()
        if last_phrase_end > 20:
            return last_phrase_end
        return len(text)

    def _find_overlap_between_texts(self, text1: str, text2: str) -> str:
        """Обнаружение перекрывающегося содержимого между соседними блоками для дедупликации."""
        max_len = min(len(text1), len(text2))
        # Перебор от длинного к короткому
        for length in range(max_len, 19, -1):
            suffix = text1[-length:]
            prefix = text2[:length]
            if suffix == prefix:
                cut = self._find_safe_cut_point(prefix)
                if cut > 20:
                    return prefix[:cut]
                return suffix
        return ""

    def _apply_basic_formatting(self, text: str) -> str:
        """Базовое форматирование при сбое ИИ: разбивка по предложениям, абзацы ≤250 символов, разделение двойным переносом."""
        if not text or not text.strip():
            return text
        import re
        parts = re.split(r"([。！？\.!?]+\s*)", text)
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current += part
            else:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                    current = ""
        if current.strip():
            sentences.append(current.strip())
        paras = []
        cur = ""
        sentence_count = 0
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            sentence_count += 1
            # Улучшенная логика разбиения: учет количества предложений и длины
            should_break = False
            if len(candidate) > 400 and cur:  # Слишком длинный абзац
                should_break = True
            elif len(candidate) > 200 and sentence_count >= 3:  # Средняя длина и достаточно предложений
                should_break = True
            elif sentence_count >= 6:  # Слишком много предложений
                should_break = True
            
            if should_break:
                paras.append(cur.strip())
                cur = s
                sentence_count = 1
            else:
                cur = candidate
        if cur.strip():
            paras.append(cur.strip())
        return self._ensure_markdown_paragraphs("\n\n".join(paras))

    async def _format_long_transcript_in_chunks(self, raw_transcript: str, transcript_language: str, max_chars_per_chunk: int) -> str:
        """Интеллектуальная разбивка + контекст + дедупликация для оптимизации длинного текста (портировано из JS)."""
        import re
        # Сначала разбивка по предложениям, сборка блоков не более max_chars_per_chunk
        parts = re.split(r"([。！？\.!?]+\s*)", raw_transcript)
        sentences = []
        buf = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                buf += part
            else:
                buf += part
                if buf.strip():
                    sentences.append(buf.strip())
                    buf = ""
        if buf.strip():
            sentences.append(buf.strip())

        chunks = []
        cur = ""
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # Вторичная безопасная разбивка все еще слишком длинных блоков
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                final_chunks.extend(self._smart_split_long_chunk(c, max_chars_per_chunk))

        logger.info(f"Текст разбит на {len(final_chunks)} блоков для обработки")

        optimized = []
        for i, c in enumerate(final_chunks):
            chunk_with_context = c
            if i > 0:
                prev_tail = final_chunks[i - 1][-100:]
                marker = f"[上文续：{prev_tail}]" if transcript_language == 'zh' else f"[Context continued: {prev_tail}]"
                chunk_with_context = marker + "\n\n" + c
            try:
                oc = await self._format_single_chunk(chunk_with_context, transcript_language)
                # Удаление меток контекста
                oc = re.sub(r"^\[(上文续|Context continued)：?:?.*?\]\s*", "", oc, flags=re.S)
                optimized.append(oc)
            except Exception as e:
                logger.warning(f"Ошибка оптимизации блока {i+1}, используется базовое форматирование: {e}")
                optimized.append(self._apply_basic_formatting(c))

        # Дедупликация соседних блоков
        deduped = []
        for i, c in enumerate(optimized):
            cur_txt = c
            if i > 0 and deduped:
                prev = deduped[-1]
                overlap = self._find_overlap_between_texts(prev[-200:], cur_txt[:200])
                if overlap:
                    cur_txt = cur_txt[len(overlap):].lstrip()
                    if not cur_txt:
                        continue
            if cur_txt.strip():
                deduped.append(cur_txt)

        merged = "\n\n".join(deduped)
        merged = self._remove_transcript_heading(merged)
        enforced = self._enforce_paragraph_max_chars(merged, max_chars=400)
        return self._ensure_markdown_paragraphs(enforced)

    def _remove_timestamps_and_meta(self, text: str) -> str:
        """Удаление временных меток и явных метаданных (заголовки, язык, вероятность)."""
        lines = text.split('\n')
        kept = []
        for line in lines:
            s = line.strip()
            # Пропуск временных меток и метаданных
            if (s.startswith('**[') and s.endswith(']**')):
                continue
            if s.startswith('# '):
                continue
            if s.startswith('**检测语言:**') or s.startswith('**语言概率:**'):
                continue
            kept.append(line)
        return '\n'.join(kept)

    def _enforce_paragraph_max_chars(self, text: str, max_chars: int = 400) -> str:
        """Разбивка абзацев, чтобы каждый не превышал max_chars, при необходимости — по предложениям."""
        if not text:
            return text
        import re
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p is not None]
        new_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) <= max_chars:
                new_paragraphs.append(para)
                continue
            # Разбивка по предложениям
            parts = re.split(r"([。！？\.!?]+\s*)", para)
            sentences = []
            buf = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    buf += part
                else:
                    buf += part
                    if buf.strip():
                        sentences.append(buf.strip())
                        buf = ""
            if buf.strip():
                sentences.append(buf.strip())
            cur = ""
            for s in sentences:
                candidate = (cur + (" " if cur else "") + s).strip()
                if len(candidate) > max_chars and cur:
                    new_paragraphs.append(cur)
                    cur = s
                else:
                    cur = candidate
            if cur:
                new_paragraphs.append(cur)
        return "\n\n".join([p.strip() for p in new_paragraphs if p is not None])

    def _remove_transcript_heading(self, text: str) -> str:
        """Удаление строк-заголовков вида Transcript (любого уровня #), не затрагивая основной текст."""
        if not text:
            return text
        import re
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#{1,6}\s*transcript(\s+text)?\s*$", stripped, flags=re.I):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    def _split_into_chunks(self, text: str, max_tokens: int) -> list:
        """
        Интеллектуальная разбивка исходного текста транскрипции на блоки подходящего размера
        Стратегия: извлечение чистого текста, разбивка по предложениям и абзацам
        """
        import re
        
        # 1. Извлечение чистого текста (удаление временных меток, заголовков и т.д.)
        pure_text = self._extract_pure_text(text)
        
        # 2. Разбивка по предложениям с сохранением целостности
        sentences = self._split_into_sentences(pure_text)
        
        # 3. Сборка в блоки с учетом лимита токенов
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Проверка, можно ли добавить в текущий блок
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Текущий блок заполнен, сохраняем и начинаем новый
                chunks.append(self._join_sentences(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                # Добавление в текущий блок
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Добавление последнего блока
        if current_chunk:
            chunks.append(self._join_sentences(current_chunk))
        
        return chunks
    
    def _extract_pure_text(self, raw_transcript: str) -> str:
        """
        Извлечение чистого текста из исходной транскрипции (удаление временных меток и метаданных)
        """
        lines = raw_transcript.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Пропуск временных меток, заголовков, метаданных
            if (line.startswith('**[') and line.endswith(']**') or
                line.startswith('#') or
                line.startswith('**检测语言:**') or
                line.startswith('**语言概率:**') or
                not line):
                continue
            text_lines.append(line)
        
        return ' '.join(text_lines)
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Разбивка текста на предложения с учетом различий между языками
        """
        import re
        
        # Разделители предложений для китайского и английского языков
        sentence_endings = r'[.!?。！？;；]+'
        
        # Разбивка с сохранением разделителей
        parts = re.split(f'({sentence_endings})', text)
        
        sentences = []
        current = ""
        
        for i, part in enumerate(parts):
            if re.match(sentence_endings, part):
                # Это разделитель предложения, добавляем к текущему
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                # Это содержание предложения
                current += part
        
        # Обработка последней части без разделителя
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if s.strip()]
    

    
    def _join_sentences(self, sentences: list) -> str:
        """
        Объединение предложений в абзац
        """
        return ' '.join(sentences)

    def _basic_transcript_cleanup(self, raw_transcript: str) -> str:
        """
        Базовая очистка текста транскрипции: удаление временных меток и заголовков
        Запасной вариант при сбое оптимизации через GPT
        """
        lines = raw_transcript.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Пропуск временных меток
            if line.strip().startswith('**[') and line.strip().endswith(']**'):
                continue
            # Пропуск заголовков
            if line.strip().startswith('# ') or line.strip().startswith('## '):
                continue
            # Пропуск метаданных
            if line.strip().startswith('**检测语言:**') or line.strip().startswith('**语言概率:**'):
                continue
            # Сохранение непустых строк текста
            if line.strip():
                cleaned_lines.append(line.strip())
        
        # Объединение предложений и интеллектуальная разбивка на абзацы
        text = ' '.join(cleaned_lines)
        
        import re
        
        # Разбивка по предложениям
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(sentences):
            if sentence:
                current_paragraph.append(sentence)
                
                # Условия для разбиения абзаца:
                # 1. Каждые 3 предложения (базовое правило)
                # 2. Принудительное разбиение при смене темы
                # 3. Предотвращение слишком длинных абзацев
                topic_change_keywords = [
                    '首先', '其次', '然后', '接下来', '另外', '此外', '最后', '总之',
                    'first', 'second', 'third', 'next', 'also', 'however', 'finally',
                    '现在', '那么', '所以', '因此', '但是', '然而',
                    'now', 'so', 'therefore', 'but', 'however'
                ]
                
                should_break = False
                
                # Проверка необходимости разбиения
                if len(current_paragraph) >= 3:  # Базовое условие по длине
                    should_break = True
                elif len(current_paragraph) >= 2:  # Более короткий, но смена темы
                    for keyword in topic_change_keywords:
                        if sentence.lower().startswith(keyword.lower()):
                            should_break = True
                            break
                
                if should_break or len(current_paragraph) >= 4:  # Максимальная длина
                    paragraph_text = '. '.join(current_paragraph)
                    if not paragraph_text.endswith('.'):
                        paragraph_text += '.'
                    paragraphs.append(paragraph_text)
                    current_paragraph = []
        
        # Добавление оставшихся предложений
        if current_paragraph:
            paragraph_text = '. '.join(current_paragraph)
            if not paragraph_text.endswith('.'):
                paragraph_text += '.'
            paragraphs.append(paragraph_text)
        
        return '\n\n'.join(paragraphs)

    async def _final_paragraph_organization(self, text: str, lang_instruction: str) -> str:
        """
        Финальная обработка абзацев объединенного текста
        Использует улучшенный промпт и инженерную валидацию
        """
        try:
            # Оценка длины текста, при необходимости — разбивка на блоки
            estimated_tokens = self._estimate_tokens(text)
            if estimated_tokens > 3000:  # Для очень длинных текстов — пофрагментная обработка
                return await self._organize_long_text_paragraphs(text, lang_instruction)
            
            system_prompt = f"""Вы профессиональный эксперт по организации абзацев на {lang_instruction}. Ваша задача — перегруппировать абзацы по смыслу и логике.

🎯 **Основные принципы**:
1. **Строго сохраняйте исходный язык ({lang_instruction})** — НИКОГДА не переводите
2. **Сохраняйте всё содержание полностью**, не удаляйте и не добавляйте информацию
3. **Разбивайте по смысловой логике**: каждый абзац — одна завершенная мысль или тема
4. **Строго контролируйте длину абзацев**: не более 250 слов
5. **Сохраняйте естественную связность**: между абзацами должна быть логическая связь

📏 **Стандарты разбиения**:
- **Смысловая целостность**: каждый абзац описывает одну завершенную идею
- **Оптимальная длина**: 3-7 предложений, не более 250 слов
- **Логические границы**: разбивайте при смене темы, временного периода или точки зрения
- **Естественные паузы**: следуйте естественным паузам говорящего и логике повествования

⚠️ **Запрещено**:
- Создавать абзацы длиннее 250 слов
- Принудительно объединять несвязанное содержание
- Разрывать целостные истории или рассуждения

Формат вывода: абзацы разделены пустыми строками."""

            user_prompt = f"""Перегруппируйте абзацы следующего текста на {lang_instruction}. Строго соблюдайте смысловую и логическую структуру, убедитесь, что каждый абзац не превышает 200 слов:

{text}

Текст с перегруппированными абзацами:"""

            response = self.client.chat.completions.create(
                model=self.advanced_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.05
            )
            
            organized_text = strip_llm_artifacts(response.choices[0].message.content or "")
            
            # Инженерная проверка длины абзацев
            validated_text = self._validate_paragraph_lengths(organized_text)
            
            return validated_text
            
        except Exception as e:
            logger.error(f"Ошибка финальной обработки абзацев: {e}")
            # При ошибке — базовое разбиение
            return self._basic_paragraph_fallback(text)

    async def _organize_long_text_paragraphs(self, text: str, lang_instruction: str) -> str:
        """
        Пофрагментная обработка абзацев для очень длинных текстов
        """
        try:
            # Разбивка по существующим абзацам
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            organized_chunks = []
            
            current_chunk = []
            current_tokens = 0
            max_chunk_tokens = 2500  # Размер блока для адаптации к лимиту 4000 токенов
            
            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)
                
                if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
                    # Обработка текущего блока
                    chunk_text = '\n\n'.join(current_chunk)
                    organized_chunk = await self._organize_single_chunk(chunk_text, lang_instruction)
                    organized_chunks.append(organized_chunk)
                    
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
            
            # Обработка последнего блока
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                organized_chunk = await self._organize_single_chunk(chunk_text, lang_instruction)
                organized_chunks.append(organized_chunk)
            
            return '\n\n'.join(organized_chunks)
            
        except Exception as e:
            logger.error(f"Ошибка обработки длинного текста: {e}")
            return self._basic_paragraph_fallback(text)

    async def _organize_single_chunk(self, text: str, lang_instruction: str) -> str:
        """
        Организация абзацев в одном текстовом блоке
        """
        system_prompt = f"""You are a {lang_instruction} paragraph organization expert. Reorganize paragraphs by semantics, ensuring each paragraph does not exceed 200 words.

Core requirements:
1. Strictly maintain the original {lang_instruction} language
2. Organize by semantic logic, one theme per paragraph
3. Each paragraph must not exceed 250 words
4. Separate paragraphs with blank lines
5. Keep content complete, do not reduce information"""

        user_prompt = f"""Re-paragraph the following text in {lang_instruction}, ensuring each paragraph does not exceed 200 words:

{text}"""

        response = self.client.chat.completions.create(
            model=self.advanced_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.05
        )
        
        return strip_llm_artifacts(response.choices[0].message.content or "")

    def _validate_paragraph_lengths(self, text: str) -> str:
        """
        Проверка длины абзацев, при обнаружении слишком длинных — попытка разбиения
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        validated_paragraphs = []
        
        for para in paragraphs:
            word_count = len(para.split())
            
            if word_count > 300:  # Если абзац превышает 300 слов
                logger.warning(f"Обнаружен слишком длинный абзац ({word_count} слов), попытка разбиения")
                # Попытка разбиения по предложениям
                split_paras = self._split_long_paragraph(para)
                validated_paragraphs.extend(split_paras)
            else:
                validated_paragraphs.append(para)
        
        return '\n\n'.join(validated_paragraphs)

    def _split_long_paragraph(self, paragraph: str) -> list:
        """
        Разбиение слишком длинного абзаца
        """
        import re
        
        # Разбивка по предложениям
        sentences = re.split(r'[.!?。！？]\s+', paragraph)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        split_paragraphs = []
        current_para = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words > 200 and current_para:
                # Текущий абзац достиг лимита длины
                split_paragraphs.append(' '.join(current_para))
                current_para = [sentence]
                current_words = sentence_words
            else:
                current_para.append(sentence)
                current_words += sentence_words
        
        # Добавление последнего абзаца
        if current_para:
            split_paragraphs.append(' '.join(current_para))
        
        return split_paragraphs

    def _basic_paragraph_fallback(self, text: str) -> str:
        """
        Базовый механизм разбиения на абзацы (запасной вариант)
        При сбое GPT используется простые правила
        """
        import re
        
        # Удаление лишних переносов
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        basic_paragraphs = []
        
        for para in paragraphs:
            word_count = len(para.split())
            
            if word_count > 250:
                # Длинный абзац — разбивка по предложениям
                split_paras = self._split_long_paragraph(para)
                basic_paragraphs.extend(split_paras)
            elif word_count < 30 and basic_paragraphs:
                # Короткий абзац — объединение с предыдущим (если не превышает 200 слов)
                last_para = basic_paragraphs[-1]
                combined_words = len(last_para.split()) + word_count
                
                if combined_words <= 200:
                    basic_paragraphs[-1] = last_para + ' ' + para
                else:
                    basic_paragraphs.append(para)
            else:
                basic_paragraphs.append(para)
        
        return '\n\n'.join(basic_paragraphs)

    async def summarize(self, transcript: str, target_language: str = "zh", video_title: str = None) -> str:
        """
        Создание резюме транскрипции видео
        
        Args:
            transcript: Текст транскрипции
            target_language: Код целевого языка
            
        Returns:
            Текст резюме (в формате Markdown)
        """
        try:
            if not self.client:
                logger.warning("OpenAI API недоступен, создание запасного резюме")
                return self._generate_fallback_summary(transcript, target_language, video_title)
            
            # Оценка длины текста транскрипции, определение необходимости пофрагментной обработки
            estimated_tokens = self._estimate_tokens(transcript)
            max_summarize_tokens = 4000  # Повышенный лимит для использования одноблочной обработки (лучшее качество)
            
            if estimated_tokens <= max_summarize_tokens:
                # Короткий текст — прямое резюме
                return await self._summarize_single_text(transcript, target_language, video_title)
            else:
                # Длинный текст — пофрагментное резюме
                logger.info(f"Длинный текст ({estimated_tokens} токенов), включена пофрагментная обработка")
                return await self._summarize_with_chunks(transcript, target_language, video_title, max_summarize_tokens)
            
        except Exception as e:
            logger.error(f"Ошибка создания резюме: {str(e)}")
            return self._generate_fallback_summary(transcript, target_language, video_title)

    async def _summarize_single_text(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        Создание резюме для одного текстового блока
        """
        # Получение названия целевого языка
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        # Промпт на английском (подходит для всех целевых языков)
        system_prompt = f"""You are an expert editor. Write a concise EXECUTIVE SUMMARY in {language_name} of the following material.

Hard rules:
- Length: about 180–450 words in {language_name} (use the lower end if the source is short). Never reproduce long verbatim quotes or extended sentence-by-sentence rewrites of the transcript.
- Content: main thesis, 3–7 key takeaways, important conclusions, and critical facts or numbers only. Tight prose; short bullet lists are OK for takeaways.
- Do NOT restate the full transcript, do NOT add preamble ("Here is…"), and do NOT add closings such as offers to revise or "let me know if…" / 客套尾注.
- Markdown: optional `## Key takeaways` then paragraphs; avoid decorative filler headings.

Output ONLY the summary body in {language_name}."""

        user_prompt = f"""Summarize the following content in {language_name}. Follow the system rules strictly (brief executive summary, no meta-commentary):

{transcript}"""

        logger.info(f"Создание резюме на {language_name}...")
        
        # Вызов OpenAI API
        response = self.client.chat.completions.create(
            model=self.advanced_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2200,
            temperature=0.25
        )
        
        summary = strip_llm_artifacts(response.choices[0].message.content or "")

        return self._format_summary_with_meta(summary, target_language, video_title)

    async def _summarize_with_chunks(self, transcript: str, target_language: str, video_title: str, max_tokens: int) -> str:
        """
        Пофрагментное создание резюме для длинного текста
        """
        language_name = self.language_map.get(target_language, "中文（简体）")

        # Использование стратегии из JS: интеллектуальная разбивка по символам (абзац > предложение)
        chunks = self._smart_chunk_text(transcript, max_chars_per_chunk=4000)
        logger.info(f"Разбито на {len(chunks)} блоков для резюмирования")
        
        chunk_summaries = []
        
        # Создание локального резюме для каждого блока
        for i, chunk in enumerate(chunks):
            logger.info(f"Резюмирование блока {i+1}/{len(chunks)}...")
            
            system_prompt = f"""You are a summarization expert. Write a brief section summary in {language_name}.

This is part {i+1} of {len(chunks)} of the full transcript.

Rules:
- About 80–160 words in {language_name}; bullets OK for key points.
- Do not echo the transcript verbatim; capture only new information in this segment.
- No preamble or meta-closings."""

            user_prompt = f"""[Part {i+1}/{len(chunks)}] Summarize in {language_name} (80–160 words, tight prose):

{chunk}

Output content only, no headings like "Summary:"."""

            try:
                response = self.client.chat.completions.create(
                    model=self.advanced_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.25
                )
                
                chunk_summary = strip_llm_artifacts(response.choices[0].message.content or "")
                chunk_summaries.append(chunk_summary)
                
            except Exception as e:
                logger.error(f"Ошибка резюмирования блока {i+1}: {e}")
                # При ошибке — краткое изложение
                simple_summary = f"Содержание части {i+1}: " + chunk[:200] + "..."
                chunk_summaries.append(simple_summary)
        
        # Объединение локальных резюме (с номерами)
        combined_summaries = "\n\n".join([f"[Part {idx+1}]\n" + s for idx, s in enumerate(chunk_summaries)])

        logger.info("Интеграция финального резюме...")
        if len(chunk_summaries) > 10:
            final_summary = await self._integrate_hierarchical_summaries(chunk_summaries, target_language)
        else:
            final_summary = await self._integrate_chunk_summaries(combined_summaries, target_language)

        return self._format_summary_with_meta(final_summary, target_language, video_title)

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 3500) -> list:
        """Интеллектуальная разбивка текста (сначала абзацы, затем предложения) по лимиту символов."""
        chunks = []
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        cur = ""
        for p in paragraphs:
            candidate = (cur + "\n\n" + p).strip() if cur else p
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = p
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # Вторичная разбивка слишком длинных блоков по предложениям
        import re
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                sentences = [s.strip() for s in re.split(r"[。！？\.!?]+", c) if s.strip()]
                scur = ""
                for s in sentences:
                    candidate = (scur + '。' + s).strip() if scur else s
                    if len(candidate) > max_chars_per_chunk and scur:
                        final_chunks.append(scur.strip())
                        scur = s
                    else:
                        scur = candidate
                if scur.strip():
                    final_chunks.append(scur.strip())
        return final_chunks

    async def _integrate_hierarchical_summaries(
        self, chunk_summaries: list, target_language: str
    ) -> str:
        """Много частичных резюме: свертка через тот же интегратор, что и для <=10."""
        combined = "\n\n".join(
            f"[Part {idx + 1}]\n{s}" for idx, s in enumerate(chunk_summaries)
        )
        return await self._integrate_chunk_summaries(combined, target_language)

    async def _integrate_chunk_summaries(self, combined_summaries: str, target_language: str) -> str:
        """
        Интеграция частичных резюме в единое связное резюме
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        try:
            system_prompt = f"""You integrate partial summaries into ONE concise executive summary in {language_name}.

Rules:
- Total length about 280–650 words in {language_name}; remove duplication, do not expand into a transcript-length rewrite.
- Markdown: paragraphs separated by blank lines; optional `## Key takeaways` only if it adds clarity.
- No preamble, no meta-closings (e.g. offers to revise or "let me know")."""

            user_prompt = f"""Merge the following partial summaries into one executive summary in {language_name}:

{combined_summaries}"""

            response = self.client.chat.completions.create(
                model=self.advanced_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2200,
                temperature=0.25
            )

            return strip_llm_artifacts(response.choices[0].message.content or "")
        except Exception as e:
            logger.error(f"Ошибка интеграции резюме: {e}")
            # При ошибке — прямое объединение
            return combined_summaries

    def _format_summary_with_meta(self, summary: str, target_language: str, video_title: str = None) -> str:
        """
        Добавление заголовка и метаданных к резюме
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        meta_labels = self._get_summary_labels(target_language)
        
        # Заголовок видео, если доступен
        if video_title:
            prefix = f"# {video_title}\n\n"
        else:
            prefix = ""
        return prefix + summary

    def _generate_fallback_summary(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        Создание запасного резюме (при недоступности OpenAI API)
        
        Args:
            transcript: Текст транскрипции
            video_title: Заголовок видео
            target_language: Код целевого языка
            
        Returns:
            Текст запасного резюме
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        # Простая обработка текста, извлечение ключевой информации
        lines = transcript.split('\n')
        content_lines = [line for line in lines if line.strip() and not line.startswith('#') and not line.startswith('**')]
        
        # Подсчет длины
        total_chars = sum(len(line) for line in content_lines)
        
        # Многоязычные метки
        meta_labels = self._get_summary_labels(target_language)
        fallback_labels = self._get_fallback_labels(target_language)
        
        title = video_title if video_title else "Summary"
        
        summary = f"""# {title}

**{meta_labels['language_label']}:** {language_name}
**{fallback_labels['notice']}:** {fallback_labels['api_unavailable']}



## {fallback_labels['overview_title']}

**{fallback_labels['content_length']}:** {fallback_labels['about']} {total_chars} {fallback_labels['characters']}
**{fallback_labels['paragraph_count']}:** {len(content_lines)} {fallback_labels['paragraphs']}

## {fallback_labels['main_content']}

{fallback_labels['content_description']}

{fallback_labels['suggestions_intro']}

1. {fallback_labels['suggestion_1']}
2. {fallback_labels['suggestion_2']}
3. {fallback_labels['suggestion_3']}

## {fallback_labels['recommendations']}

- {fallback_labels['recommendation_1']}
- {fallback_labels['recommendation_2']}


<br/>

<p style="color: #888; font-style: italic; text-align: center; margin-top: 16px;"><em>{fallback_labels['fallback_disclaimer']}</em></p>"""
        
        return summary
    
    def _get_current_time(self) -> str:
        """Получение текущей даты и времени"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_supported_languages(self) -> dict:
        """
        Получение списка поддерживаемых языков
        
        Returns:
            Словарь с кодами языков и их названиями
        """
        return self.language_map.copy()
    
    def _detect_transcript_language(self, transcript: str) -> str:
        """
        Определение основного языка текста транскрипции
        
        Args:
            transcript: Текст транскрипции
            
        Returns:
            Код определенного языка
        """
        # Простая логика: поиск языковой метки в транскрипции
        if "**检测语言:**" in transcript:
            # Извлечение языка из метаданных Whisper
            lines = transcript.split('\n')
            for line in lines:
                if "**检测语言:**" in line:
                    # Извлечение кода языка, например: "**检测语言:** en"
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # Если метка не найдена, использование простого определения по символам
        total_chars = len(transcript)
        if total_chars == 0:
            return "en"  # По умолчанию английский
            
        # Подсчет китайских символов
        chinese_chars = sum(1 for char in transcript if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / total_chars
        
        # Подсчет английских букв
        english_chars = sum(1 for char in transcript if char.isascii() and char.isalpha())
        english_ratio = english_chars / total_chars
        
        # Определение по соотношению
        if chinese_ratio > 0.3:
            return "zh"
        elif english_ratio > 0.3:
            return "en"
        else:
            return "en"  # По умолчанию английский
    
    def _get_language_instruction(self, lang_code: str) -> str:
        """
        Получение названия языка для инструкций на основе кода
        
        Args:
            lang_code: Код языка
            
        Returns:
            Название языка
        """
        language_instructions = {
            "en": "English",
            "zh": "中文",
            "ja": "日本語",
            "ko": "한국어",
            "es": "Español",
            "fr": "Français",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ar": "العربية"
        }
        return language_instructions.get(lang_code, "English")
    

    def _get_summary_labels(self, lang_code: str) -> dict:
        """
        Получение многоязычных меток для страницы резюме
        
        Args:
            lang_code: Код языка
            
        Returns:
            Словарь с метками
        """
        labels = {
            "en": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "zh": {
                "language_label": "摘要语言",
                "disclaimer": "本摘要由AI自动生成，仅供参考"
            },
            "ja": {
                "language_label": "要約言語",
                "disclaimer": "この要約はAIによって自動生成されており、参考用です"
            },
            "ko": {
                "language_label": "요약 언어",
                "disclaimer": "이 요약은 AI에 의해 자동 생성되었으며 참고용입니다"
            },
            "es": {
                "language_label": "Idioma del Resumen",
                "disclaimer": "Este resumen es generado automáticamente por IA, solo para referencia"
            },
            "fr": {
                "language_label": "Langue du Résumé",
                "disclaimer": "Ce résumé est généré automatiquement par IA, à titre de référence uniquement"
            },
            "de": {
                "language_label": "Zusammenfassungssprache",
                "disclaimer": "Diese Zusammenfassung wird automatisch von KI generiert, nur zur Referenz"
            },
            "it": {
                "language_label": "Lingua del Riassunto",
                "disclaimer": "Questo riassunto è generato automaticamente dall'IA, solo per riferimento"
            },
            "pt": {
                "language_label": "Idioma do Resumo",
                "disclaimer": "Este resumo é gerado automaticamente por IA, apenas para referência"
            },
            "ru": {
                "language_label": "Язык резюме",
                "disclaimer": "Это резюме автоматически генерируется ИИ, только для справки"
            },
            "ar": {
                "language_label": "لغة الملخص",
                "disclaimer": "هذا الملخص تم إنشاؤه تلقائياً بواسطة الذكاء الاصطناعي، للمرجع فقط"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def _get_fallback_labels(self, lang_code: str) -> dict:
        """
        Получение многоязычных меток для запасного резюме
        
        Args:
            lang_code: Код языка
            
        Returns:
            Словарь с метками
        """
        labels = {
            "en": {
                "notice": "Notice",
                "api_unavailable": "OpenAI API is unavailable, this is a simplified summary",
                "overview_title": "Transcript Overview",
                "content_length": "Content Length",
                "about": "About",
                "characters": "characters",
                "paragraph_count": "Paragraph Count",
                "paragraphs": "paragraphs",
                "main_content": "Main Content",
                "content_description": "The transcript contains complete video speech content. Since AI summary cannot be generated currently, we recommend:",
                "suggestions_intro": "For detailed information, we suggest you:",
                "suggestion_1": "Review the complete transcript text for detailed information",
                "suggestion_2": "Focus on important paragraphs marked with timestamps",
                "suggestion_3": "Manually extract key points and takeaways",
                "recommendations": "Recommendations",
                "recommendation_1": "Configure OpenAI API key for better summary functionality",
                "recommendation_2": "Or use other AI services for text summarization",
                "fallback_disclaimer": "This is an automatically generated fallback summary"
            },
            "zh": {
                "notice": "注意",
                "api_unavailable": "由于OpenAI API不可用，这是一个简化的摘要",
                "overview_title": "转录概览",
                "content_length": "内容长度",
                "about": "约",
                "characters": "字符",
                "paragraph_count": "段落数量",
                "paragraphs": "段",
                "main_content": "主要内容",
                "content_description": "转录文本包含了完整的视频语音内容。由于当前无法生成智能摘要，建议您：",
                "suggestions_intro": "为获取详细信息，建议您：",
                "suggestion_1": "查看完整的转录文本以获取详细信息",
                "suggestion_2": "关注时间戳标记的重要段落",
                "suggestion_3": "手动提取关键观点和要点",
                "recommendations": "建议",
                "recommendation_1": "配置OpenAI API密钥以获得更好的摘要功能",
                "recommendation_2": "或者使用其他AI服务进行文本总结",
                "fallback_disclaimer": "本摘要为自动生成的备用版本"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def is_available(self) -> bool:
        """
        Проверка доступности сервиса резюмирования
        
        Returns:
            True если OpenAI API настроен, иначе False
        """
        return self.client is not None