from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

def get_llm(llm_provider: str, llm_model: str, llm_api_key: str | None) -> BaseChatModel:
            kwargs = {}
            if llm_provider == "ollama":
                pass
            elif llm_provider == "gemini":
                kwargs["google_api_key"] = llm_api_key
            else:
                kwargs["api_key"] = llm_api_key
            
            try:
                llm = init_chat_model(
                    model_provider=llm_provider,
                    model=llm_model,
                    **kwargs
                )
                return llm
            except ImportError:
                print(f"\\nERROR: The integration package for '{llm_provider}' is not installed.")
                package_name = f"langchain-{llm_provider}"
                if llm_provider == "gemini":
                    package_name = "langchain-google-genai"
                print(f"Please install it using: pip install {package_name}")
                return None
            except Exception as e:
                print(f"\\nERROR: An error occurred while initializing the model: {e}")
                return None