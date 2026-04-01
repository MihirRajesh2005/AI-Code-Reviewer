from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

def get_llm(llm_provider: str, llm_model: str, llm_api_key: str | None,
            ollama_base_url: str = "http://localhost:11434",
            openai_base_url: str | None = None) -> BaseChatModel:
            kwargs = {}
            if llm_provider == "ollama":
                kwargs["base_url"] = ollama_base_url
            elif llm_provider == "gemini":
                kwargs["google_api_key"] = llm_api_key
            else:
                kwargs["api_key"] = llm_api_key
                if llm_provider == "openai" and openai_base_url:
                    kwargs["base_url"] = openai_base_url
            
            try:
                llm = init_chat_model(
                    model_provider=llm_provider,
                    model=llm_model,
                    **kwargs
                )
                return llm
            except ImportError:
                print(f"\nERROR: The integration package for '{llm_provider}' is not installed.")
                package_name = f"langchain-{llm_provider}"
                if llm_provider == "gemini":
                    package_name = "langchain-google-genai"
                print(f"Please install it using: pip install {package_name}")
                return None
            except Exception as e:
                print(f"\nERROR: An error occurred while initializing the model: {e}")
                return None