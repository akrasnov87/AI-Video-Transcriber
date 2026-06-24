import logging
import os
import re
from typing import Optional

from openai import OpenAI

from llm_sanitize import strip_llm_artifacts

logger = logging.getLogger(__name__)


class Translator:
    """Переводчик текста; поддерживает API Key / Base URL из переменных окружения или переданных в запросе (аналогично Summarizer)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.client = None
        self._translation_model = model or os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o")

        self.language_map = {
            "zh": "中文（简体）",
            "zh-tw": "中文（繁体）",
            "en": "English",
            "ja": "日本語",
            "ko": "한국어",
            "fr": "Français",
            "de": "Deutsch",
            "es": "Español",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ar": "العربية",
            "hi": "हिन्दी",
        }

        eff_key = (api_key.strip() if isinstance(api_key, str) and api_key.strip() else None) or os.getenv(
            "OPENAI_API_KEY"
        )
        if isinstance(api_key, str) and api_key.strip():
            eff_base = (base_url or "").strip().rstrip("/") or os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )
        else:
            eff_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not eff_key:
            logger.warning("Действительный OpenAI API Key не установлен, перевод будет недоступен")
            return

        try:
            self.client = OpenAI(api_key=eff_key, base_url=eff_base)
            logger.info("Клиент OpenAI Translator успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации клиента OpenAI: {e}")
            self.client = None
    
    def _detect_source_language(self, text: str) -> str:
        """Определение языка исходного текста"""
        # Простая логика определения языка
        if "**检测语言:**" in text:
            lines = text.split('\n')
            for line in lines:
                if "**检测语言:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # Простое определение на основе статистики символов
        total_chars = len(text)
        if total_chars == 0:
            return "en"
        
        # Подсчет китайских символов
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        chinese_ratio = chinese_chars / total_chars
        
        # Подсчет японских символов
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        japanese_ratio = japanese_chars / total_chars
        
        # Подсчет корейских символов
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
        korean_ratio = korean_chars / total_chars
        
        if chinese_ratio > 0.1:
            return "zh"
        elif japanese_ratio > 0.05:
            return "ja"
        elif korean_ratio > 0.05:
            return "ko"
        else:
            return "en"

    def _normalize_lang_code(self, code: str) -> str:
        if not code:
            return ""
        c = str(code).lower().strip()
        if c.startswith("zh"):
            return "zh"
        if len(c) >= 2 and c[:2] in self.language_map:
            return c[:2]
        return c

    def normalize_lang_code(self, code: Optional[str]) -> str:
        """Единый метод нормализации кода языка, согласованный с should_translate."""
        return self._normalize_lang_code(code or "")

    def infer_language_code(self, text: str) -> str:
        """Определение кода языка из текста (в стиле ISO), используется при отсутствии метаданных транскрипции."""
        return self._detect_source_language(text or "")

    def should_translate(self, source_language: str, target_language: str) -> bool:
        """Определение необходимости перевода"""
        if not source_language or not target_language:
            return False

        source_lang = self._normalize_lang_code(source_language)
        target_lang = self._normalize_lang_code(target_language)

        if source_lang == target_lang:
            return False

        chinese_variants = ["zh", "zh-cn", "zh-hans", "chinese"]
        if source_lang in chinese_variants and target_lang in chinese_variants:
            return False

        return True

    def languages_differ_for_translation(self, source_code: Optional[str], summary_lang: Optional[str]) -> bool:
        """Возвращает True, если язык саммари (выбранный пользователем) отличается от исходного языка.
        Используется для определения необходимости генерации/отображения перевода."""
        s = self.normalize_lang_code(source_code or "")
        t = self.normalize_lang_code(summary_lang or "")
        return bool(s and t and self.should_translate(s, t))

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 4000) -> list:
        """Интеллектуальная разбивка текста на фрагменты для перевода"""
        chunks = []

        # Сначала разбиваем по абзацам
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        current_chunk = ""

        for paragraph in paragraphs:
            # Если текущий абзац + существующий блок превышает лимит
            if len(current_chunk) + len(paragraph) + 2 > max_chars_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Добавляем последний блок
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Если какой-то блок все еще слишком длинный, разбиваем дальше по предложениям
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars_per_chunk:
                final_chunks.append(chunk)
            else:
                # Разбивка по предложениям
                sentences = re.split(r'[.!?。！？]\s+', chunk)
                current_sub_chunk = ""

                for sentence in sentences:
                    if len(current_sub_chunk) + len(sentence) + 2 > max_chars_per_chunk and current_sub_chunk:
                        final_chunks.append(current_sub_chunk.strip())
                        current_sub_chunk = sentence
                    else:
                        if current_sub_chunk:
                            current_sub_chunk += ". " + sentence
                        else:
                            current_sub_chunk = sentence

                if current_sub_chunk.strip():
                    final_chunks.append(current_sub_chunk.strip())

        return final_chunks

    async def translate_text(self, text: str, target_language: str, source_language: Optional[str] = None) -> str:
        """
        Перевод текста на целевой язык
        
        Args:
            text: Текст для перевода
            target_language: Код целевого языка
            source_language: Код исходного языка (опционально, определяется автоматически)
            
        Returns:
            Переведенный текст
        """
        try:
            if not self.client:
                logger.warning("OpenAI API недоступен, перевод невозможен")
                return text
            
            # Определение исходного языка
            if not source_language:
                source_language = self._detect_source_language(text)
            
            # Если исходный и целевой языки совпадают, возвращаем оригинал
            src_n = self._normalize_lang_code(source_language or "")
            tgt_n = self._normalize_lang_code(target_language)
            if src_n and tgt_n and src_n == tgt_n:
                return text
            
            source_lang_name = self.language_map.get(src_n, self.language_map.get(source_language, source_language))
            target_lang_name = self.language_map.get(tgt_n, self.language_map.get(target_language, target_language))
            
            logger.info(f"Начало перевода: {source_lang_name} -> {target_lang_name}")
            
            # Оценка длины текста для определения необходимости разбивки на фрагменты
            if len(text) > 3000:
                logger.info(f"Длинный текст ({len(text)} символов), включена пофрагментная обработка")
                return await self._translate_with_chunks(text, target_lang_name, source_lang_name)
            else:
                return await self._translate_single_text(text, target_lang_name, source_lang_name)
                
        except Exception as e:
            logger.error(f"Ошибка перевода: {str(e)}")
            return text
    
    async def _translate_single_text(self, text: str, target_lang_name: str, source_lang_name: str) -> str:
        """Перевод одного текстового фрагмента"""
        system_prompt = f"""Вы профессиональный переводчик. Переведите текст с {source_lang_name} на {target_lang_name}.

Требования к переводу:
- Сохраняйте форматирование и структуру оригинала (включая разделение на абзацы, заголовки)
- Точная передача смысла, естественный и плавный язык
- Сохранение точности профессиональных терминов
- Не добавляйте пояснений или примечаний
- Если встречается разметка Markdown, сохраняйте её
- Выводите только текст перевода: без вступлений, заключений, вежливых фраз, не пишите «При необходимости скорректируйте» и т.п."""

        user_prompt = f"""Переведите следующий текст с {source_lang_name} на {target_lang_name}:

{text}

Верните только результат перевода, без каких-либо пояснений."""

        try:
            response = self.client.chat.completions.create(
                model=self._translation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )

            return strip_llm_artifacts(response.choices[0].message.content or "")
        except Exception as e:
            logger.error(f"Ошибка перевода одного фрагмента: {e}")
            return text
    
    async def _translate_with_chunks(self, text: str, target_lang_name: str, source_lang_name: str) -> str:
        """Пофрагментный перевод длинного текста"""
        chunks = self._smart_chunk_text(text, max_chars_per_chunk=4000)
        logger.info(f"Разбито на {len(chunks)} фрагментов для перевода")
        
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Перевод фрагмента {i+1}/{len(chunks)}...")
            
            system_prompt = f"""Вы профессиональный переводчик. Переведите текст с {source_lang_name} на {target_lang_name}.

Это часть {i+1} из {len(chunks)} полного документа.

Требования к переводу:
- Сохраняйте форматирование и структуру оригинала
- Точная передача смысла, естественный и плавный язык
- Сохранение точности профессиональных терминов
- Не добавляйте пояснений или примечаний
- Обеспечьте связность с предыдущими и следующими частями
- Выводите только текст перевода, без заключений или посторонних фраз."""

            user_prompt = f"""Переведите следующий текст с {source_lang_name} на {target_lang_name}:

{chunk}

Верните только результат перевода."""

            try:
                response = self.client.chat.completions.create(
                    model=self._translation_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )

                translated_chunk = response.choices[0].message.content or ""
                translated_chunks.append(strip_llm_artifacts(translated_chunk))
            except Exception as e:
                logger.error(f"Ошибка перевода фрагмента {i+1}: {e}")
                # В случае ошибки сохраняем оригинал
                translated_chunks.append(chunk)
        
        # Объединение результатов перевода
        return strip_llm_artifacts("\n\n".join(translated_chunks))