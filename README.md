# AI Code Reviewer
An automated code review and analysis tool that detects bugs, errors, and suggests improvements — across multiple programming languages.

---
# Getting Started

## Installation
```bash
pip install git+https://github.com/MihirRajesh2005/AI-Code-Reviewer.git
```
Or install from a local clone:
```bash
pip install .
```

---
## Additional Requirements
You need an API key from an inference provider supported by LangChain, or a locally-running model through Ollama.
You will also need to install the LangChain integration package for your chosen provider:

| Provider | Package |
|----------|---------|
| OpenAI | `pip install langchain-openai` |
| Google Gemini | `pip install langchain-google-genai` |
| Ollama (local) | `pip install langchain-ollama` |
| Others | `pip install langchain-<provider>` |

For Ollama, no API key is needed. Use `--ollama-url` to point at a non-default server address.

---
## Supported Languages
Language is automatically detected from the file extension. You can override this with `--lang`.

Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, C#, Ruby, PHP, Swift, Kotlin, Shell, HTML, CSS, SQL, R, Scala, Lua, Dart

---
## Usage

Set your API key as an environment variable (not required for Ollama):
```bash
# Linux / macOS
export code_reviewer_api_key="your api key here"

# Windows
$env:code_reviewer_api_key="your api key here"
```

**Review a single file** (language auto-detected from extension):
```bash
code-reviewer yourfile.ts --provider openai --model gpt-4o
code-reviewer yourfile.py --provider google-genai --model gemini-2.0-flash
code-reviewer yourfile.go --provider ollama --model llama3.2
```

**Use an OpenAI-compatible provider (e.g. OpenRouter):**
```bash
code-reviewer yourfile.py --provider openai --model anthropic/claude-sonnet-4-5 --base-url https://openrouter.ai/api/v1
```

**Review all supported files in a directory:**
```bash
code-reviewer --dir ./my-project --provider openai --model gpt-4o
```

---
## All Options

| Flag | Description |
|------|-------------|
| `--provider` | LLM provider name (e.g. `openai`, `google-genai`, `ollama`) |
| `--model` | Model name to use |
| `--ollama-url` | Ollama server base URL (default: `http://localhost:11434`) |
| `--base-url` | Custom base URL for OpenAI-compatible APIs (e.g. OpenRouter). Use with `--provider openai`. |
| `--lang LANGUAGE` | Override language detection (e.g. `--lang Go`) |
| `--min-severity` | Only show findings at or above this level: `critical`, `major`, or `minor` (default: `minor`) |
| `--rules FILE` | Path to a Markdown file containing custom coding standards to enforce |
| `--dir DIRECTORY` | Review all supported source files in a directory recursively |
| `--fix` | Attempt to auto-fix identified issues and overwrite the original file |
| `--dry-run` | With `--fix`: show the proposed patch diff without writing to disk |

---
## CI / Exit Codes
The tool exits with code `1` if any `critical` severity finding is detected, making it easy to fail a CI pipeline on serious issues:
```bash
code-reviewer src/main.rs --provider openai --model gpt-4o --min-severity critical
```
