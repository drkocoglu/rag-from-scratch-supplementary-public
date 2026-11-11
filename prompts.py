# Public-release review note: locally authored project support module; verify that no secrets, private URLs, or third-party prompts are embedded.

"""
Goals:
- Keep a clean and organized prompt management system for the project.
- Use LangSmith to track prompt versions privately and compare performance over time.

Usage:
- Run this file to push your prompts to LangSmith (Prompts section).
- No prompt edits = no new versions (commits), Prompt edits = new versions (commits).

Pulling a pushed prompt from LangSmith:
- Once pushed to LangSmith other files can reference the prompt by name and version.
- Useful for sharing prompts across multiple projects, files, and with other developers.

Prompts can be shared with other developers but:
    - Same workspace: multiple developers in one workspace = $/developer (check Langsmith pricing).
    - Workarounds:
        - Make your prompts public in LangSmith (Hub) and share the prompt name/version with other developers.
        - if public is off limits, collaborate on a private github repo, share the prompts.py, and push to your private workspace.

Pull a specific version of a pushed prompt from LangSmith:
    - hub.pull("rag-from-scratch:<version>") or Client().pull_prompt("rag-from-scratch:<version>").
    - use only the name "rag-from-scratch" to pull the latest version of the prompt.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Changes to the prompt creates new immutable versions in LangSmith. If the prompt is unchanged, no new version is created.
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context} \n"
    "Answer:"
)

# Could be used for multi-query retrieval, etc. but not used yet.
# MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(...)
# STEP_BACK_PROMPT = ChatPromptTemplate.from_template(...)

# Run this file to push the prompt to LangSmith PROMPTS. If the prompt already exists, no new version is created.
if __name__ == "__main__":
    from langsmith import Client
    from langsmith.utils import LangSmithConflictError

    try:
        url = Client().push_prompt("rag-from-scratch", object=RAG_PROMPT)
        print(f"Prompt pushed successfully. URL: {url}")
    except LangSmithConflictError:
        print("Prompt already exists in LangSmith. No action taken.")
