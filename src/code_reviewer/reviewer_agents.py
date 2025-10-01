import importlib.resources
from langchain_core.language_models import BaseChatModel
from typing import Dict, Any

def load_and_format_poml(file: str, **kwargs) -> str:
    prompt_path = importlib.resources.files('code_reviewer.prompts').joinpath(f"{file}.poml")
    try:
        with open (prompt_path, "r") as f:
            template = f.read()
        
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        
        return template
    
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at {prompt_path}")
        return ""


def summariser_agent(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, str]:
    print("Summariser agent working...")
    code  = state["code_to_review"]
    prompt = load_and_format_poml("summariser", code_to_review=code)
    if not prompt:
        return {}
    response = llm.invoke(prompt)
    print("Summariser agent finished.")
    return {"summary":response.content}

def error_detector_agent(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, str]:
    print("Error detection in progress...")
    code = state["code_to_review"]
    summary = state.get("summary", "No summary was provided")
    prompt = load_and_format_poml("error_detector", code_to_review=code, summary=summary)
    if not prompt:
        return {}
    response = llm.invoke(prompt)
    print("Error detection finished.")
    return {"errors":response.content}

def bug_detector_agent(state: Dict[str,Any], llm: BaseChatModel) -> Dict[str, str]:
    print("Bug identification in progress...")
    code = state["code_to_review"]
    summary = state.get("summary", "")
    errors = state.get("errors","No errors were found")
    prompt = load_and_format_poml("bug_detector", code_to_reviewer=code, summary=summary, errors=errors)
    if not prompt:
        return {}
    response = llm.invoke(prompt)
    print("Bug identification finished.")
    return {"bugs":response.content}

def improvements_agent(state: Dict[str,Any], llm: BaseChatModel) -> Dict[str, str]:
    print("Generating improvement suggestions...")
    code = state["code_to_review"]
    summary = state.get("summary", )
    errors = state.get("errors","")
    bugs = state.get("bugs", "No bugs were identified")
    prompt = load_and_format_poml("improvements", code_to_reviewer=code, summary=summary, errors=errors, bugs=bugs)
    if not prompt:
        return {}
    response = llm.invoke(prompt)
    print("Suggestions generated.")
    return {"improvements": response.content}