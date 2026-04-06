import importlib.resources
import json
from code_reviewer.console import console
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_and_format_poml(file: str, **kwargs) -> str:
    prompt_path = importlib.resources.files('code_reviewer.prompts').joinpath(f"{file}.poml")
    try:
        with open(prompt_path, "r") as f:
            template = f.read()

        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value) if value is not None else "")

        return template

    except FileNotFoundError:
        console.print(f"ERROR: Prompt file not found at {prompt_path}", style="error")
        return ""


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    description: str
    severity: str   # "critical" | "major" | "minor"
    location: str
    suggestion: str

class FindingList(BaseModel):
    findings: List[Finding]


def _findings_to_text(findings: List[Finding]) -> str:
    """Serialize findings to a readable string for use in downstream prompts."""
    if not findings:
        return "None found."
    lines = []
    for i, f in enumerate(findings, 1):
        lines.append(
            f"{i}. [{f.severity.upper()}] {f.description} "
            f"(at: {f.location}) — Suggestion: {f.suggestion}"
        )
    return "\n".join(lines)


def _parse_structured(llm: BaseChatModel, prompt: str) -> List[Finding]:
    """Invoke LLM with structured output, falling back to JSON parse on failure."""
    try:
        structured_llm = llm.with_structured_output(FindingList)
        result: FindingList = structured_llm.invoke(prompt)
        return result.findings
    except Exception:
        # Fallback: try parsing raw JSON from plain invocation
        try:
            raw = llm.invoke(prompt).content
            data = json.loads(raw)
            return [Finding(**item) for item in data.get("findings", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def summariser_agent(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    console.print("Summariser agent working...", style="status")
    code = state["code_to_review"]
    language = state.get("language", "Python")
    prompt = load_and_format_poml("summariser", code_to_review=code, language=language)
    if not prompt:
        return {}
    response = llm.invoke(prompt)
    console.print("Summariser agent finished.", style="status")
    return {"summary": response.content}


def error_detector_agent(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    console.print("Error detection in progress...", style="status")
    code = state["code_to_review"]
    language = state.get("language", "Python")
    custom_rules = state.get("custom_rules", "")
    summary = state.get("summary", "No summary was provided")
    prompt = load_and_format_poml(
        "error_detector",
        code_to_review=code,
        language=language,
        summary=summary,
        custom_rules=custom_rules,
    )
    if not prompt:
        return {}
    findings = _parse_structured(llm, prompt)
    has_critical = any(f.severity == "critical" for f in findings)
    console.print("Error detection finished.", style="status")
    return {
        "errors": _findings_to_text(findings),
        "error_findings": [f.model_dump() for f in findings],
        "has_critical": has_critical,
    }


def bug_detector_agent(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    console.print("Bug identification in progress...", style="status")
    code = state["code_to_review"]
    language = state.get("language", "Python")
    summary = state.get("summary", "")
    errors = state.get("errors", "No errors were found")
    prompt = load_and_format_poml(
        "bug_detector",
        code_to_review=code,
        language=language,
        summary=summary,
        errors=errors,
    )
    if not prompt:
        return {}
    findings = _parse_structured(llm, prompt)
    has_critical = state.get("has_critical", False) or any(f.severity == "critical" for f in findings)
    console.print("Bug identification finished.", style="status")
    return {
        "bugs": _findings_to_text(findings),
        "bug_findings": [f.model_dump() for f in findings],
        "has_critical": has_critical,
    }


def improvements_agent(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    console.print("Generating improvement suggestions...", style="status")
    code = state["code_to_review"]
    language = state.get("language", "Python")
    custom_rules = state.get("custom_rules", "")
    summary = state.get("summary", "")
    errors = state.get("errors", "")
    bugs = state.get("bugs", "No bugs were identified")
    prompt = load_and_format_poml(
        "improvements",
        code_to_review=code,
        language=language,
        summary=summary,
        errors=errors,
        bugs=bugs,
        custom_rules=custom_rules,
    )
    if not prompt:
        return {}
    findings = _parse_structured(llm, prompt)
    console.print("Suggestions generated.", style="status")
    return {
        "improvements": _findings_to_text(findings),
        "improvement_findings": [f.model_dump() for f in findings],
    }
