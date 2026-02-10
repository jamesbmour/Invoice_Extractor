# AGENTS.md

## Project overview


## Preferences and dependencies

1. Use Python 3.12 or later
2. Install dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Load environment variables from a `.env` file for API keys and configurations.

4. Use Ollama as the local LLM backend.

```Python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

llm = ChatOllama(model='llama3.2')
embedding = OllamaEmbeddings(model='nomic-embed-text')

llm = ChatOpenAI(model="gpt-4.1-nano", max_tokens=500)
```

## Project structure

## Key files and their purposes

## Build and test commands

## Code style guidelines
