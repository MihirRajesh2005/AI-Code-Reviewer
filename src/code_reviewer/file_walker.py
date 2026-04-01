import os
from typing import Iterator, Optional
from code_reviewer.language_detector import supported_extensions, validate_language, EXTENSION_MAP

SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", ".tox", ".mypy_cache"}


def walk_supported_files(directory: str, language_filter: Optional[str] = None) -> Iterator[str]:
    exts = supported_extensions()

    if language_filter:
        canonical = validate_language(language_filter)
        exts = {ext for ext, lang in EXTENSION_MAP.items() if lang == canonical}

    for root, dirs, files in os.walk(directory):
        # Prune skipped directories in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in sorted(files):
            _, ext = os.path.splitext(filename)
            if ext.lower() in exts:
                yield os.path.join(root, filename)
