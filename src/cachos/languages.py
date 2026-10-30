"""Language codes and text language detection, built on proven libraries.

* Normalization: ``langcodes`` (BCP-47 / ISO 639-1/2/3 / English names).
* Detection: ``lingua`` – the most accurate open-source detector on short
  text (75 languages; see its published benchmarks vs CLD3, fastText, langid).

Internally cachos speaks ISO 639-1 (``"en"``, ``"fr"``, ``"zh"``), falling back
to ISO 639-3 for languages without a 2-letter code.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

log = logging.getLogger("cachos.languages")

# Cases where engines and langcodes disagree, or where the BCP-47 region matters.
_SPECIAL = {"zh-hk": "yue", "zh-yue": "yue", "yue-hk": "yue", "jw": "jw", "auto": "auto", "*": "*"}


@lru_cache(maxsize=4096)
def normalize_language(lang: Optional[str]) -> Optional[str]:
    """``"en-US"`` / ``"English"`` / ``"eng"`` / ``"pt_BR"`` -> ``"en"`` / ``"en"`` / ``"en"`` / ``"pt"``.

    Unknown values pass through lowercased.
    """
    if not lang:
        return None
    raw = lang.strip().lower().replace("_", "-")
    if raw in _SPECIAL:
        return _SPECIAL[raw]
    import langcodes

    try:
        tag = langcodes.Language.get(raw)
    except Exception:
        try:
            tag = langcodes.find(raw)  # English names: "Spanish", "Farsi"
        except LookupError:
            return raw
    if not tag.language:
        return raw
    code = tag.language
    return {"iw": "he", "in": "id", "nb": "no", "ji": "yi"}.get(code, code)


def to_iso3(lang: str) -> str:
    """ISO 639-3 code (MMS models are keyed by it)."""
    import langcodes

    code = normalize_language(lang) or lang
    try:
        return langcodes.Language.get(code).to_alpha3()
    except Exception:
        return code


def language_name(lang: str) -> str:
    import langcodes

    code = normalize_language(lang) or lang
    try:
        return langcodes.Language.get(code).display_name()
    except Exception:
        return code


@lru_cache(maxsize=1)
def _detector():
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        log.warning("lingua not installed: text language detection disabled "
                    "(`pip install cachos[langdetect]`); pass language= explicitly")
        return None
    return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


def detect_language(text: str, candidates: Optional[list[str]] = None) -> Optional[str]:
    """ISO 639-1 code of ``text``'s language, or ``None`` if unknown.

    ``candidates`` restricts the answer (e.g. to the languages your voices support),
    which makes short-text detection much more reliable.
    """
    text = (text or "").strip()
    detector = _detector()
    if not text or detector is None:
        return None
    allowed = {normalize_language(c) for c in candidates} if candidates else None
    for v in detector.compute_language_confidence_values(text):
        code = v.language.iso_code_639_1.name.lower()
        if allowed is None or code in allowed:
            return code
    return None
