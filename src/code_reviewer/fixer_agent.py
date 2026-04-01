import json
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from typing import Dict, Any, List

from code_reviewer.reviewer_agents import load_and_format_poml

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


class PatchedFile(BaseModel):
    patched_code: str
    changes_made: List[str]


def fixer_agent(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    code = state["code_to_review"]
    language = state.get("language", "Python")
    custom_rules = state.get("custom_rules", "")
    min_severity = state.get("min_severity", "minor")
    threshold = SEVERITY_ORDER.get(min_severity, 2)

    all_findings = state.get("error_findings", []) + state.get("bug_findings", [])
    filtered = [f for f in all_findings if SEVERITY_ORDER.get(f.get("severity", "minor"), 2) <= threshold]

    if not filtered:
        print("Fixer: no findings to fix at the selected severity level.")
        return {"patched_code": code, "changes_made": []}

    issues_lines = []
    for i, f in enumerate(filtered, 1):
        issues_lines.append(
            f"{i}. [{f.get('severity','').upper()}] {f.get('description','')} "
            f"(at: {f.get('location','')}) — Suggestion: {f.get('suggestion','')}"
        )
    issues_to_fix = "\n".join(issues_lines)

    print(f"Fixer agent working on {len(filtered)} issue(s)...")
    prompt = load_and_format_poml(
        "fixer",
        code_to_review=code,
        language=language,
        issues_to_fix=issues_to_fix,
        custom_rules=custom_rules,
    )
    if not prompt:
        return {}

    try:
        structured_llm = llm.with_structured_output(PatchedFile)
        result: PatchedFile = structured_llm.invoke(prompt)
        patched_code = result.patched_code
        changes_made = result.changes_made
    except Exception:
        try:
            raw = llm.invoke(prompt).content
            data = json.loads(raw)
            patched_code = data.get("patched_code", code)
            changes_made = data.get("changes_made", [])
        except Exception:
            print("Fixer: failed to parse response, returning original code.")
            return {"patched_code": code, "changes_made": []}

    print("Fixer agent finished.")
    return {"patched_code": patched_code, "changes_made": changes_made}
