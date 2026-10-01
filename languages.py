"""Language options shared by the CLI and isolated WhisperX workers."""
from __future__ import annotations

import unicodedata

# WhisperX's default Romanian aligner; keep automatic and explicit Romanian equal.
ROMANIAN_ALIGNMENT_MODEL = "gigant/romanian-wav2vec2"


def language_code(value: str) -> str | None:
    value = unicodedata.normalize("NFC", value.strip()).casefold()
    if value == "auto":
        return None
    if value in {"ro", "ron", "rum", "romanian", "romana", "română", "ro-ro", "ro_ro", "ro-md", "ro_md"}:
        return "ro"
    return value
