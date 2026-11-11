# Local RAG & Agentic Chunking Tutorials — Qwen + Ollama

Educational Python examples for **retrieval-augmented generation (RAG), LangGraph workflows, and text/agentic chunking**. These scripts adapt the linked tutorials to run primarily with **local Qwen 3.5 9B through Ollama**, with additional prompts, explanations, structured outputs, debugging, and experiments.

Each script illustrates a technique rather than forming part of one production-ready application. The examples are adaptations, **not original implementations of the underlying methods**.

## Included Python files

### Text splitting and agentic chunking

| File                                    | What it builds and what changed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| :-------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chunking_strategies.py`                | Explores **five chunking approaches**: character-based, recursive, document-specific (including Markdown, code, and PDF content), semantic splitting using embedding-distance changes, and LLM-guided agentic chunking. Adapted to local Qwen/embeddings with additional examples, comments, and diagnostics.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `agentic_chunker.py`                    | **Baseline, one-pass agentic chunker.** An LLM processes extracted propositions individually, assigns each to an existing semantic chunk or creates a new one, and generates/updates chunk titles and summaries. Adapted for local inference and supplemented with explanatory notes.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `agentic_chunker_modified.py`           | **Modified, two-pass agentic chunker.** First, the LLM compares each proposition with existing propositions and returns a structured `chunk_id` (or `null` for a new chunk). Then a second LLM review inspects each sufficiently large chunk, detects mixed subtopics or unrelated examples, and **splits overly broad chunks**, preserving the original proposition text and regenerating metadata. Adds stricter grouping instructions, output validation, token diagnostics, and a reasoning-enabled review stage. **Limitations:** It cannot merge chunks incorrectly separated in pass 1; chunks with fewer than three propositions are not reviewed; and improved grouping or retrieval quality is not guaranteed. |
| `agentic_chunking_modified_tutorial.py` | **End-to-end demonstration of the modified chunker**: split an essay into paragraphs → extract self-contained propositions using Qwen → perform first-pass grouping → run second-pass review/splitting → inspect the resulting chunks and logs. Shows the modified architecture in practice; it is not a separate retrieval pipeline.                                                                                                                                                                                                                                                                                                                                                                                    |

**Baseline vs. modified chunker:** The baseline makes incremental assignments and stops. The modified version makes more specific, proposition-aware assignments **and then revisits the resulting groups** to split chunks that mix distinct discussions. This seeks more focused chunks while keeping causal chains and details of the same example together. Its decisions remain model- and prompt-dependent.

### LangGraph RAG workflows

| File                              | Architecture and goal                                                                                                                                                                                                                                                                                                                                                       |
| :-------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `langgraph_crag_local.py`         | **Corrective RAG (CRAG):** Chroma retrieval → document relevance grading → **generate** if evidence is adequate, or **rewrite the query → Tavily web search → generate** when retrieval is weak. Demonstrates a corrective fallback rather than always trusting vector search.                                                                                              |
| `langgraph_self_rag_local.py`     | **Self-RAG-style evaluation loop:** retrieve from Chroma → grade document relevance → generate → check whether the answer is **grounded in the retrieved documents** and **answers the question**. Can regenerate or rewrite/retrieve again when checks fail. The grading is LLM-based, not a guarantee of factual accuracy.                                                |
| `langgraph_adaptive_rag_local.py` | **Source-routing Adaptive RAG:** an LLM first routes the question to either **Chroma vector retrieval** or **Tavily web search**. The vector path filters documents and can rewrite/retrieve again; both paths generate an answer and check grounding/usefulness. This example adapts *source routing*, not the research paper's full complexity-based strategy.            |
| `langgraph_agentic_rag_local.py`  | **Tool-using Agentic RAG:** an LLM agent decides **whether to call a Chroma-backed retrieval tool** instead of being forced to retrieve every time. LangGraph executes the tool call, grades the retrieved content, and can **rewrite the question and return to the agent** before answer generation. Tavily is not part of this script's primary retrieval-tool decision. |

The four workflows were separated into standalone lessons and adapted for local Qwen/Ollama. Added tutorial comments, logging, intermediate decision traces, and debugging aids make the routes and grading behavior easier to inspect.

### Shared support

| File         | Purpose                                                                                                                                                             |
| :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `config.py`  | Centralizes `get_llm()` and `get_embeddings()` for Ollama/Qwen and local Hugging Face embeddings, with an optional hosted-model configuration.                      |
| `prompts.py` | Defines a reusable RAG answer prompt and **publishes it to your own LangSmith workspace**. Useful because some workflows pull `rag-from-scratch` through LangSmith. |

## Original tutorials and learning resources

**Text splitting and agentic chunking**

- [5 Levels of Text Splitting — YouTube tutorial](https://www.youtube.com/watch?v=8OJC21T2SL4)
- [Text splitting tutorial and explanations](https://community.fullstackretrieval.com/document-loaders/text-splitting)
- [Full Stack Retrieval — GitHub source and examples](https://github.com/FullStackRetrieval-com/RetrievalTutorials)
- [Original text-splitting tutorial files](https://github.com/FullStackRetrieval-com/RetrievalTutorials/tree/main/tutorials/LevelsOfTextSplitting)

**RAG and LangGraph**

- [Lance Martin's RAG From Scratch — full YouTube course](https://www.youtube.com/watch?v=sVcwVQRHIc8)
- [RAG From Scratch — YouTube playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
- [Lance Martin's RAG From Scratch — GitHub notebooks](https://github.com/langchain-ai/rag-from-scratch) (learning reference)
- [LangGraph RAG example notebooks — GitHub](https://github.com/langchain-ai/langgraph/tree/main/examples/rag) (upstream source for the four workflow adaptations)
- [Agentic RAG with LangGraph — article](https://www.langchain.com/blog/agentic-rag-with-langgraph)

## Tools, libraries, and accounts

| Tool / service                                                                                                                                                  | Used for                                                                                            | Account / key required?                                                                                          |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| [Python](https://www.python.org/) and [uv](https://docs.astral.sh/uv/)                                                                                          | Running scripts and managing dependencies                                                           | No account                                                                                                       |
| [Ollama](https://ollama.com/download) + `qwen3.5:9b`                                                                                                            | Local chat model, classification, summarization, agent decisions                                    | **No key or account for local models - install Ollama app first**                                                |
| [Hugging Face](https://huggingface.co/) + `BAAI/bge-small-en-v1.5`                                                                                              | Local text embeddings/model downloads                                                               | Public models usually need **no token**; `HF_TOKEN` is optional for authenticated, rate-limited, or gated access |
| [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), [Chroma](https://github.com/chroma-core/chroma) | Prompt chains, graph control flow/tool calling, vector database and retrieval                       | Open-source Python libraries; **no service account needed**                                                      |
| [LangSmith](https://smith.langchain.com/)                                                                                                                       | Saving/pulling a named prompt; optional tracing                                                     | **Account + `LANGSMITH_API_KEY`** for remote prompt operations                                                   |
| [Tavily](https://tavily.com/)                                                                                                                                   | Web search in CRAG and Adaptive RAG                                                                 | **Account + `TAVILY_API_KEY`** for those workflows                                                               |
| LlamaIndex, Unstructured, Pydantic, scikit-learn, NumPy, Matplotlib, Pillow and PDF tools                                                                       | Alternative text splitting, PDF extraction, structured responses, semantic distances and inspection | Install the packages needed by the selected script; some PDF features need additional system dependencies        |

**Free access (checked November 2026):** [Tavily's free Researcher plan](https://www.tavily.com/pricing) currently provides a limited monthly API-credit allowance without requiring a payment card. [LangSmith's Developer plan](https://www.langchain.com/pricing) currently has a free tier with a limited monthly tracing allowance; prompt operations require an account/key. These are generally sufficient for modest tutorial use, **but this repository does not control their availability, limits, terms, or pricing**. Other optional hosted providers may charge. Check each provider's current terms before signing up or running code.

## Quick start

1. Install Python, `uv`, and Ollama. From the project root, use the committed dependency files (`pyproject.toml` and `uv.lock`), if provided:
   ```powershell
   ollama pull qwen3.5:9b
   uv sync --locked
   Copy-Item .env.example .env

   ```
2. Open your private `.env` and set the local model/embedding settings. Use `DEVICE=cpu` if CUDA embeddings are unavailable. Add **only** the optional keys required for the example you plan to run. Keep `LANGSMITH_TRACING=false` unless you intentionally want tracing.
3. If a script retrieves the `rag-from-scratch` prompt from LangSmith, sign in, create an API key, and run **once** to publish the prompt into your own workspace:
   ```powershell
   uv run python prompts.py

   ```
4. Test the configuration and run one example at a time:
   ```powershell
   uv run python config.py
   uv run python langgraph_agentic_rag_local.py
   # Or: uv run python agentic_chunking_modified_tutorial.py

   ```

Some scripts download webpages, run many LLM calls, or need document-specific dependencies. Running every file is neither necessary nor recommended. `uv sync --locked` assumes a matching `pyproject.toml` and `uv.lock` are included in your published checkout.

## Download the example documents

**Download these files from the [Full Stack Retrieval GitHub repository](https://github.com/FullStackRetrieval-com/RetrievalTutorials)** and save them at the paths below. The third-party text, PDFs, and figures are **not distributed in this repository**.

| Local path expected by scripts                    | Source                                                                                                                                                                                                |
| :------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/PGEssays/mit.txt`                           | [Download/view `mit.txt`](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/data/PGEssays/mit.txt)                                                                               |
| `data/PGEssays/superlinear.txt`                   | [Download/view `superlinear.txt`](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/data/PGEssays/superlinear.txt)                                                               |
| `data/static/SalesforceFinancial.pdf`             | [`SalesforceFinancial.pdf`](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/static/SalesforceFinancial.pdf)                                    |
| `data/static/VisualInstruction.pdf`               | [`VisualInstruction.pdf`](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/static/VisualInstruction.pdf)                                        |
| `data/static/pdfImages/figure-15-6.jpg` (if used) | [Reference `pdfImages` directory](https://github.com/FullStackRetrieval-com/RetrievalTutorials/tree/main/tutorials/LevelsOfTextSplitting/static/pdfImages), or generate the figure via PDF extraction |

**PDF path note:** In the current `chunking_strategies.py`, extraction writes images to `static/pdfImages/` while the later example reads from `data/static/pdfImages/`. Make the paths consistent, and check the actual generated filename before loading an image.

## Security, billing, and licenses

- **Protect credentials.** Publish `.env.example` with empty placeholders, not `.env`. Add `.env` and other local secret files to `.gitignore`, and check your Git history and notebooks before making the repository public. A leaked API key can let others use your quotas, access services, or incur charges on your account. Revoke/rotate exposed keys promptly.
- **Avoid unexpected charges.** If you intend to use only free services, **do not add a credit or debit card or enable paid billing wherever a no-card free plan is available**. This substantially reduces billing risk but is **not an absolute guarantee**. Review limits, usage, and provider changes yourself; free tiers can be reduced, removed, or exhausted. Local Ollama inference itself does not have per-token cloud-API fees.
- **Account responsibility.** Users are responsible for their own accounts, API-key security, billing settings, and any charges incurred through third-party services. The repository author does not control these services or their pricing.
- **Attribution and licenses.** See the applicable upstream MIT notices in `LICENSES/` and preserve required copyright/license text for redistributed code. Links to external tutorials or data do not make those materials part of this repository or establish permission to redistribute them.

## Educational Use and Disclaimer

This repository contains educational examples adapted from the original tutorials referenced above, with additional modifications, explanations, and experiments.

Users are encouraged to consult the original tutorials and official documentation of third-party libraries for installation instructions, updates, compatibility requirements, and known limitations.

The code is provided **"AS IS"** for educational purposes. Users are encouraged to review the code, understand the tools and services involved, and configure their environments according to their own needs. The author cannot guarantee the functionality, compatibility, or availability of third-party libraries and services and is not responsible for issues arising from their use.
