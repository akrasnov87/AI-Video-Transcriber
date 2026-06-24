"""Удаление типовых мета-фраз и заключений LLM из вывода модели."""
from __future__ import annotations

import re
from typing import Optional

# Завершающие блоки (английский + китайский), часто добавляемые вопреки инструкциям
_PATTERNS = [
    re.compile(r"(?is)\n{1,2}let me know[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}feel free to[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}if you (?:need|have|would like)[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}(?:happy to|please let me know|don't hesitate)[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}(?:hope this helps|thanks for reading)[\s\S]{0,400}\Z"),
    re.compile(r"(?is)\n{1,2}(?:请告诉|如有需要|如需|欢迎反馈|希望对你|以上(?:内容)?)[\s\S]{0,800}\Z"),
]


def strip_llm_artifacts(text: Optional[str]) -> str:
    """
    Удаление артефактов LLM из текста.
    
    Очищает вывод модели от типовых мета-фраз, заключений и
    служебных предложений, которые не являются частью основного контента.
    
    Args:
        text: Исходный текст от модели
        
    Returns:
        Очищенный текст без артефактов
    """
    if not text or not isinstance(text, str):
        return (text or "").strip()
    t = text.strip()
    # Многократное применение паттернов для удаления вложенных артефактов
    for _ in range(6):
        before = t
        for pat in _PATTERNS:
            t = pat.sub("", t).strip()
        if t == before:
            break
    # Дополнительная проверка последних строк
    lines = t.split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        low = last.lower()
        # Если последняя строка содержит типовые фразы — удаляем
        if len(last) < 200 and any(
            x in low
            for x in (
                "let me know",
                "further adjustments",
                "feel free",
                "hope this helps",
                "请告诉我",
                "如需调整",
                "欢迎反馈",
            )
        ):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()