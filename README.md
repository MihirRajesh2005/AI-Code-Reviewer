# AI Code Reviewer
An automated code review and analysis tool that checks for bugs, errors, and gives you suggestions on how to improve your code.

---
# Getting Started
This tool works only on **Python** files. It has not been tested on or designed for any other programming languages.

---
## Installation
You can either run the following command:
```bash
pip install git+https://github.com/MihirRajesh2005/AI-Code-Reviewer.git
```
Or you can download pyproject.toml and the src files, save them to a directory, and run this in that directory's location in command line:
```bash
pip install .
```
---
## Additional Requirements
You will need an api key from an inference provider supported by langchain, or a locally-running model through Ollama.
You will also need to install the Langchain package associated with said provider for the tool to work.

---
## Usage
This tool runs in any Command-Line Interface(CLI).
To use it, first navigate to the directory of the file you intend to run it on.
```bash
cd your_project_directory
```
If you're using an online inference provider, you will then need to set the api key as an environment variable in the following format:
```bash
# For Linux/MacOS:
export code_reviewer_api_key="your api key here"

# For Windows:
$env:code_reviewer_api_key="your api key here"
```
You can then run the tool in the following format:
```bash
code-reviewer yourfile.py --provider yourprovidername --model yourmodelname
```
Your api key (if applicable) will be applied automatically.
