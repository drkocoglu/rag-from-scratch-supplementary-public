# Public-release review note: locally authored project support module; verify that no secrets, private URLs, or third-party prompts are embedded.

"""
Model configuration for the RAG course & functions to call the llm and embeddings.
Configuration allows both commercial and local models to be used but, required an API key for commercial models.
Swap models via environment variables in ".env" rather than editing this file.

"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Load environment variables from the .env file.
load_dotenv()

# Default model configuration. Override via environment variables in the .env file.
PROVIDER = os.getenv("PROVIDER", "ollama")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen3.5:9b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
DEVICE = os.getenv("DEVICE", "cuda")


# LLM and embeddings factory functions.
def get_llm(
    temperature: float = 0.0,
    keep_alive: str | int | None = None,
    reasoning: bool | None = False,
    max_retries: int | None = 0,
    num_ctx: int = NUM_CTX,
    streaming: bool = True,
) -> BaseChatModel:
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4.1-mini", temperature=temperature)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=LOCAL_MODEL,
        temperature=temperature,
        num_ctx=num_ctx,
        reasoning=reasoning,
        keep_alive=keep_alive,
        max_retries=max_retries,
        disable_streaming=not streaming,
    )


# Embeddings factory function
def get_embeddings() -> Embeddings:
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
    )


# Smoke test to verify that the models are working correctly.
if __name__ == "__main__":
    print(f"Provider: {PROVIDER}")
    print(f"LLM: {LOCAL_MODEL}   Embeddings: {EMBED_MODEL}   Device: {DEVICE}")

    # keep_alive=0: unload llm from vram immediately - don't let LLM linger in VRAM after smoke testing it
    llm = get_llm(keep_alive=0)
    reply = llm.invoke("Reply with exactly: OK")
    print(f"LLM reply: {reply.content!r}")

    emb = get_embeddings()
    vec = emb.embed_query("smoke test")
    print(f"Embedding dim: {len(vec)}   first value: {vec[0]:.4f}")

    print("Smoke test passed.")
