import os

class ConfigurationError(Exception):
    pass

def get_provider(provider_arg: str | None):
    if not provider_arg:
        raise ConfigurationError(
            "The --provider argument is required. Choose a provider supported by langchain and ensure that you have the associated libraries installed."
        )
    return provider_arg

def get_model(model_arg: str | None) -> str:
    if not model_arg:
        raise ConfigurationError("The --model argument is required.")
    return model_arg

def get_api_key(provider: str) -> str | None:
    
    if provider == "ollama":
        return None

    api_key = os.getenv("code_reviewer_api_key")
    if not api_key:
        raise ConfigurationError(
            f"API key for '{provider}' not found. Please set the 'code_reviewer_api_key' environment variable."
        )
    return api_key