import os

EXTENSION_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JavaScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".c": "C",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".sh": "Shell",
    ".bash": "Shell",
    ".html": "HTML",
    ".css": "CSS",
    ".sql": "SQL",
    ".r": "R",
    ".scala": "Scala",
    ".lua": "Lua",
    ".dart": "Dart",
}

# Canonical names for user-supplied --lang values
_CANONICAL = {name.lower(): name for name in EXTENSION_MAP.values()}
_CANONICAL.update({
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "js": "JavaScript",
    "ts": "TypeScript",
    "golang": "Go",
    "bash": "Shell",
    "sh": "Shell",
})


def detect_language(file_path: str) -> str:
    """Returns language name from file extension. Defaults to 'Unknown'."""
    _, ext = os.path.splitext(file_path)
    return EXTENSION_MAP.get(ext.lower(), "Unknown")


def validate_language(lang: str) -> str:
    """Normalizes a user-supplied --lang value to a canonical name.
    Returns the input unchanged (title-cased) if not recognized."""
    return _CANONICAL.get(lang.lower(), lang.title())


def supported_extensions() -> set:
    """Returns the set of all supported file extensions."""
    return set(EXTENSION_MAP.keys())
