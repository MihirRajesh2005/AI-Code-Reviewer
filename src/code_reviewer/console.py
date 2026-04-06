from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "severity.critical": "bold red",
    "severity.major": "bold yellow",
    "severity.minor": "bold cyan",
    "status": "dim italic",
    "success": "bold green",
    "error": "bold red",
    "info": "bold blue",
    "heading": "bold underline",
})

console = Console(theme=custom_theme)

SEVERITY_STYLE = {
    "CRITICAL": "severity.critical",
    "MAJOR": "severity.major",
    "MINOR": "severity.minor",
}
