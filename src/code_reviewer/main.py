import argparse
import sys
from code_reviewer.language_detector import detect_language, validate_language
from code_reviewer.llm_selector import get_llm
from code_reviewer.reviewer_agents import (
    bug_detector_agent,
    error_detector_agent,
    improvements_agent,
    summariser_agent,
)
from code_reviewer.utils import get_api_key, get_model, get_provider, ConfigurationError
from functools import partial
from langgraph.graph import END, StateGraph
from typing import TypedDict, List, Dict, Any

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


class code_reviewer_state(TypedDict):
    # Input
    code_to_review: str
    language: str
    custom_rules: str
    min_severity: str
    # Agent outputs — narrative strings (used for prompt chaining)
    summary: str
    errors: str
    bugs: str
    improvements: str
    # Agent outputs — structured findings (used for reports, fix, severity filtering)
    error_findings: List[Dict[str, Any]]
    bug_findings: List[Dict[str, Any]]
    improvement_findings: List[Dict[str, Any]]
    # Derived
    has_critical: bool
    # Fixer outputs (Phase 6)
    patched_code: str
    changes_made: List[str]


def _filter_findings(findings: List[Dict], min_severity: str) -> List[Dict]:
    threshold = SEVERITY_ORDER.get(min_severity, 2)
    return [f for f in findings if SEVERITY_ORDER.get(f.get("severity", "minor"), 2) <= threshold]


def _print_findings(label: str, findings: List[Dict], min_severity: str) -> None:
    filtered = _filter_findings(findings, min_severity)
    print(f"\n{label}:")
    if not filtered:
        print("  None found.")
        return
    for f in filtered:
        sev = f.get("severity", "minor").upper()
        loc = f.get("location", "")
        desc = f.get("description", "")
        sug = f.get("suggestion", "")
        print(f"  [{sev}] {desc}")
        if loc:
            print(f"         Location : {loc}")
        if sug:
            print(f"         Suggestion: {sug}")


def _build_workflow(code_reviewer_model, use_fixer: bool):
    workflow = StateGraph(code_reviewer_state)

    workflow.add_node("summariser", partial(summariser_agent, llm=code_reviewer_model))
    workflow.add_node("error_detector", partial(error_detector_agent, llm=code_reviewer_model))
    workflow.add_node("bug_detector", partial(bug_detector_agent, llm=code_reviewer_model))
    workflow.add_node("improvements", partial(improvements_agent, llm=code_reviewer_model))

    workflow.set_entry_point("summariser")
    workflow.add_edge("summariser", "error_detector")
    workflow.add_edge("error_detector", "bug_detector")
    workflow.add_edge("bug_detector", "improvements")

    if use_fixer:
        from code_reviewer.fixer_agent import fixer_agent
        workflow.add_node("fixer", partial(fixer_agent, llm=code_reviewer_model))
        workflow.add_edge("improvements", "fixer")
        workflow.add_edge("fixer", END)
    else:
        workflow.add_edge("improvements", END)

    return workflow.compile()


def _review_file(file_path: str, code_reviewer_model, args) -> Dict[str, Any]:
    """Run the full review pipeline on a single file. Returns final_state."""
    with open(file_path, "r", encoding="utf-8") as f:
        code_to_review = f.read()
    print(f"Successfully read {len(code_to_review.splitlines())} lines of code.")

    language = validate_language(args.lang) if args.lang else detect_language(file_path)
    print(f"Language: {language}")

    custom_rules = ""
    if args.rules:
        with open(args.rules, "r", encoding="utf-8") as f:
            custom_rules = f.read()

    app = _build_workflow(code_reviewer_model, use_fixer=args.fix)

    initial_state: Dict[str, Any] = {
        "code_to_review": code_to_review,
        "language": language,
        "custom_rules": custom_rules,
        "min_severity": args.min_severity,
        "error_findings": [],
        "bug_findings": [],
        "improvement_findings": [],
        "has_critical": False,
        "patched_code": "",
        "changes_made": [],
    }

    print("Starting code review pipeline...")
    final_state = app.invoke(initial_state)
    print("Pipeline executed.\n")
    return final_state


def run():
    parser = argparse.ArgumentParser(
        description="AI Code Reviewing Agent",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Input — file or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("file", nargs="?", help="Source file to review")
    input_group.add_argument("--dir", metavar="DIRECTORY",
                             help="Analyze all supported source files in a directory recursively")

    # Model config
    parser.add_argument("--provider", help="LLM provider (e.g. openai, google-genai, ollama)")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Base URL for Ollama server (default: http://localhost:11434)")
    parser.add_argument("--base-url", default=None,
                        help="Custom base URL for OpenAI-compatible providers (e.g. OpenRouter)")

    # Language
    parser.add_argument("--lang", default=None,
                        help="Programming language of the input file (auto-detected from extension if omitted)")

    # Severity
    parser.add_argument("--min-severity", choices=["critical", "major", "minor"], default="minor",
                        help="Only display findings at or above this severity level (default: minor)")

    # Custom rules
    parser.add_argument("--rules", metavar="FILE",
                        help="Path to a Markdown file containing custom coding standards to enforce")

    # Auto-fix
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to auto-fix identified issues")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --fix: show unified diff without writing the file")

    args = parser.parse_args()

    if not args.dir and args.file is None:
        parser.error("one of the arguments file/--dir is required")

    # Validate: --dry-run only makes sense with --fix
    if args.dry_run and not args.fix:
        parser.error("--dry-run requires --fix")

    try:
        provider = get_provider(args.provider)
        model = get_model(args.model)
        api_key = get_api_key(provider=provider)

        print(f"Using provider {provider} and model {model}.")

        if args.rules:
            print(f"Loading custom rules from {args.rules}...")

        print(f"Initialising model {model}...")
        code_reviewer_model = get_llm(
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            ollama_base_url=args.ollama_url,
            openai_base_url=args.base_url,
        )
        print("Model initialised successfully.")

        # ---------------------------------------------------------------
        # Multi-file mode
        # ---------------------------------------------------------------
        if args.dir:
            from code_reviewer.file_walker import walk_supported_files

            files = list(walk_supported_files(args.dir, language_filter=args.lang))
            if not files:
                print(f"No supported source files found in {args.dir}")
                return

            print(f"Found {len(files)} file(s) to review in {args.dir}\n")

            aggregate = {
                "files_reviewed": 0,
                "total_critical": 0,
                "total_major": 0,
                "total_minor": 0,
                "per_file": [],
            }
            any_critical = False

            for file_path in files:
                print(f"\n{'='*60}")
                print(f"Reviewing: {file_path}")
                print(f"{'='*60}")
                final_state = _review_file(file_path, code_reviewer_model, args)

                all_findings = (
                    final_state.get("error_findings", [])
                    + final_state.get("bug_findings", [])
                    + final_state.get("improvement_findings", [])
                )
                counts = {"critical": 0, "major": 0, "minor": 0}
                for f in all_findings:
                    sev = f.get("severity", "minor")
                    counts[sev] = counts.get(sev, 0) + 1

                aggregate["files_reviewed"] += 1
                aggregate["total_critical"] += counts["critical"]
                aggregate["total_major"] += counts["major"]
                aggregate["total_minor"] += counts["minor"]
                aggregate["per_file"].append({"path": file_path, "state": final_state, "counts": counts})

                if final_state.get("has_critical"):
                    any_critical = True

                if args.fix and final_state.get("patched_code"):
                    patched = final_state["patched_code"]
                    original = final_state.get("code_to_review", "")
                    if patched != original:
                        if args.dry_run:
                            import difflib
                            diff = list(difflib.unified_diff(
                                original.splitlines(keepends=True),
                                patched.splitlines(keepends=True),
                                fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
                            ))
                            print("".join(diff) if diff else "No changes.")
                        else:
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(patched)
                            print(f"Patched: {file_path}")

                _print_findings("Errors", final_state.get("error_findings", []), args.min_severity)
                _print_findings("Bugs", final_state.get("bug_findings", []), args.min_severity)
                _print_findings("Improvements", final_state.get("improvement_findings", []), args.min_severity)

            print(f"\n{'='*60}")
            print(f"AGGREGATE SUMMARY — {aggregate['files_reviewed']} files reviewed")
            print(f"  Critical: {aggregate['total_critical']}  Major: {aggregate['total_major']}  Minor: {aggregate['total_minor']}")
            print(f"{'='*60}")

            if any_critical:
                sys.exit(1)
            return

        # ---------------------------------------------------------------
        # Single-file mode
        # ---------------------------------------------------------------
        print(f"Starting review for {args.file}...")
        final_state = _review_file(args.file, code_reviewer_model, args)

        print("\nSummary:")
        print(final_state.get("summary", "No summary generated"))

        _print_findings("Errors", final_state.get("error_findings", []), args.min_severity)
        _print_findings("Bugs", final_state.get("bug_findings", []), args.min_severity)
        _print_findings("Improvements", final_state.get("improvement_findings", []), args.min_severity)

        # Auto-fix handling
        if args.fix and final_state.get("patched_code"):
            import difflib
            with open(args.file, "r", encoding="utf-8") as f:
                original = f.read()
            original_lines = original.splitlines(keepends=True)
            patched_lines = final_state["patched_code"].splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                original_lines, patched_lines,
                fromfile=f"a/{args.file}", tofile=f"b/{args.file}"
            ))
            if args.dry_run:
                print("\n--- Proposed patch ---")
                print("".join(diff) if diff else "No changes.")
            else:
                with open(args.file, "w", encoding="utf-8") as f:
                    f.write(final_state["patched_code"])
                print(f"\nFile patched: {args.file}")
                if final_state.get("changes_made"):
                    print("Changes made:")
                    for change in final_state["changes_made"]:
                        print(f"  - {change}")

        if final_state.get("has_critical"):
            sys.exit(1)

    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please correct the arguments and try again.")
        return
    except Exception as e:
        print(f"\nAn unexpected error has occurred: {e}")
        raise


if __name__ == "__main__":
    run()
