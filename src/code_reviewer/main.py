import argparse
from code_reviewer.llm_selector import get_llm
from code_reviewer.reviewer_agents import bug_detector_agent, error_detector_agent, improvements_agent, summariser_agent
from code_reviewer.utils import get_api_key, get_model, get_provider, ConfigurationError
from functools import partial
from langgraph.graph import END, StateGraph
from typing import TypedDict


class code_reviewer_state(TypedDict):
    code_to_review: str
    summary: str
    errors: str
    bugs: str
    improvements: str

def run():
    parser = argparse.ArgumentParser(
        description="AI Code Reviewing Agent",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("file", help="The python file for the model to review")
    parser.add_argument("--provider",
                        help="Your choice of provider."
                        " Please ensure that you have the associated integration package installed")
    parser.add_argument("--model", help="Pick a model to use.")
    args = parser.parse_args()

    try:
        provider = get_provider(args.provider)
        model = get_model(args.model)
        api_key = get_api_key(provider=provider)

        print(f"Using provider {provider} and model {model}.")
        print(f"Starting review for {args.file}...")
        

        print(f"Reading file {args.file}...")
        with open (args.file, "r", encoding="utf-8") as f:
            code_to_review = f.read()
            print(f"Successfully read {len(code_to_review.splitlines())} lines of code.")
        
        print(f"Initialising code reviewer model {model}...")
        code_reviewer_model = get_llm(llm_provider=provider, llm_model=model, llm_api_key=api_key)
        print(f"Model initialised successfully.")


        workflow = StateGraph(code_reviewer_state)

        workflow.add_node("summariser", partial(summariser_agent, llm=code_reviewer_model))
        workflow.add_node("error_detector", partial(error_detector_agent, llm=code_reviewer_model))
        workflow.add_node("bug_detector", partial(bug_detector_agent, llm=code_reviewer_model))
        workflow.add_node("improvements", partial(improvements_agent, llm=code_reviewer_model))

        workflow.set_entry_point("summariser")
        workflow.add_edge("summariser", "error_detector")
        workflow.add_edge("error_detector", "bug_detector")
        workflow.add_edge("bug_detector", "improvements")
        workflow.add_edge("improvements", END)

        app = workflow.compile()
        initial_state = {"code_to_review":code_to_review}
        print("Starting code review pipeline...")
        final_state = app.invoke(initial_state)
        print("Pipeline executed.\n")

        print("Summary:")
        print(final_state.get("summary","No summary generated"))
        print("\nErrors:")
        print(final_state.get("errors","No errors detected."))
        print("\nBugs:")
        print(final_state.get("bugs","No bugs identified"))
        print("\nImprovements:")
        print(final_state.get("improvements","No improvements suggested."))

    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please correct the arguments and try again.")
        return
    except Exception as e:
        print(f"\nAn unexpected error has occured: {e}")


if __name__ == "__main__":
    run()