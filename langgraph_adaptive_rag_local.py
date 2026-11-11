# Adaptive RAG: extracted and locally adapted educational example.
# Original LangGraph example: https://github.com/langchain-ai/langgraph/tree/main/examples/rag
# Origin: langgraph/examples/rag/langgraph_adaptive_rag.ipynb (MIT)
# LangGraph license: MIT, Copyright (c) 2024 LangChain, Inc.
# Modifications: local Qwen/Ollama, explanations, debugging and standalone imports.
# See LICENSES/LANGGRAPH_MIT.txt and LICENSE_AUDIT.md.
# This code may require external downloads, Tavily and/or LangSmith credentials.

# Shared imports supplied to make the extracted tutorial section standalone.
from dotenv import load_dotenv

load_dotenv()

import bs4
import tiktoken
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langsmith import Client
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

from config import LOCAL_MODEL, get_embeddings, get_llm

"""
# ----------------------------------------- Adaptive RAG -----------------------------------------
"""

print("\n\n" + "*" * 20 + " ADAPTIVE RAG RESULTS " + "*" * 20 + "\n\n")

"""
Adaptive RAG is a RAG approach that analyzes a user's question and dynamically
selects an appropriate retrieval or answering strategy instead of using the
same retrieval process for every question.

Unlike traditional RAG, which generally follows a fixed retrieval-generation
pipeline, Adaptive RAG can route different questions through different workflows.

    * Query Analysis:
    Examines the question to determine which retrieval or answering strategy
    is appropriate.

    * Dynamic Routing:
    Selects an information source or retrieval strategy based on the question.

    * Retrieval and Generation:
    Obtains relevant information through the selected strategy and uses it
    to generate an answer.

    * Evaluation and Correction (optional):
    Evaluates retrieved information and generated answers, potentially
    triggering additional retrieval, query rewriting, or regeneration.

        NOTE: Evaluation and correction are not mandatory components of every
        Adaptive RAG implementation. Lance's tutorial combines:

        - Adaptive RAG: LLM-based routing between Chroma and Tavily.
        - CRAG: Document relevance grading, filtering, query rewriting,
        and corrective retrieval.
        - Self-RAG: Groundedness checking, answer usefulness grading,
        and feedback-driven regeneration or retrieval.

        These are simplified tutorial implementations, not complete
        reproductions of the original research architectures.


ORIGINAL ADAPTIVE-RAG VS. LANCE'S SIMPLIFIED IMPLEMENTATION:

The original Adaptive-RAG paper, "Adaptive-RAG: Learning to Adapt
Retrieval-Augmented Large Language Models through Question Complexity"
(Jeong et al., 2024), focuses on choosing a retrieval strategy based
on the complexity of the question.

*** Original Adaptive-RAG:

  A classifier predicts the question's complexity and selects among
  three strategies:

  1. No Retrieval:
     The LLM answers directly without retrieving external documents.
     
     - Appropriate when the question can be answered without additional
     information.

  2. Single-Step Retrieval:
     Retrieve relevant information once, then generate an answer.

     - Appropriate for questions that require external information but
     can be answered using a relatively straightforward retrieval step.

  3. Iterative Retrieval:
     Perform multiple retrieval and reasoning steps to answer a
     more complex question.

     - Appropriate for questions that require multiple pieces of
     information or intermediate reasoning.

  NOTE: The main objective is to balance answer quality and computational
  efficiency by avoiding unnecessary retrieval while providing more
  sophisticated retrieval for difficult questions.


*** Lance's Simplified Adaptive RAG:

  Lance's implementation demonstrates query routing using LangGraph
  rather than reproducing the original paper's trained complexity classifier.

  The workflow selects between two information sources:

  1. Vectorstore (Chroma):
     Used for questions related to the indexed documents.

  2. Web Search (Tavily):
     Used for questions outside the vectorstore's subject areas.

  The tutorial combines this routing mechanism with:

  - Document relevance grading.
  - Question rewriting.
  - Hallucination / groundedness grading.
  - Answer usefulness grading.
  - Conditional regeneration and retrieval.

  Therefore, this implementation demonstrates source-based Adaptive RAG
  with self-correcting behavior, rather than the original paper's
  complexity-based selection among no retrieval, single-step retrieval,
  and iterative retrieval.

- Below is Table 1 - explicitly shows the differences between Lance's Self-RAG and Adaptive-RAG vs. Original Adaptive RAG

*** Table 1: Self-RAG vs. Lance's Adaptive RAG vs. Original Adaptive RAG
| Feature                  | Lance's Self-RAG              | Lance's Adaptive RAG            | Original Adaptive-RAG                  |
|--------------------------|-------------------------------|---------------------------------|----------------------------------------|
| Main objective           | Evaluate and improve answers  | Select an information source    | Match retrieval strategy to question   |
| Initial decision         | No initial source router      | LLM selects Chroma or Tavily    | Classifier predicts query complexity   |
| Retrieval strategy       | Starts with Chroma            | Chroma or Tavily                | No, single-step, or iterative retrieval|
| No-retrieval option      | No                            | No                              | Yes                                    |
| Document relevance       | LLM grades retrieved chunks   | LLM grades Chroma chunks        | Not a required component               |
| Document filtering       | Removes irrelevant chunks     | Removes irrelevant Chroma chunks| Not a required component               |
| Hallucination grading    | Yes                           | Yes                             | Not a required component               |
| Answer usefulness grading| Yes                           | Yes                             | Not a required component               |
| Query rewriting          | Yes                           | Yes                             | Not a required shared mechanism        |
| Generation retry         | Yes, if answer unsupported    | Yes, if answer unsupported      | Not a required shared mechanism        |
| Iterative retrieval      | Through corrective loops      | Through corrective loops        | Explicit strategy for complex queries  |
| Key distinguishing idea  | Self-evaluate and correct     | Route to an information source  | Adapt retrieval effort to complexity   |


** IMPORTANT DISTINCTIONS: 

NOTE: Below is the detailed explanation of Lance's Self-RAG vs. Lance's Adaptive RAG vs. Original Adaptive RAG (Table 1).

1. Lance's Self-RAG and Adaptive RAG share most of their grading,
   query rewriting, and regeneration mechanisms.

2. The primary addition in Lance's Adaptive RAG is an initial LLM router
   that chooses between Chroma and Tavily.

3. The original Adaptive-RAG paper uses a question-complexity classifier
   to select among three strategies:
   - No retrieval for simpler questions.
   - Single-step retrieval for moderately complex questions.
   - Iterative retrieval for more complex questions.

4. The original Adaptive-RAG paper does not require the particular
   relevance, hallucination, or usefulness graders used in Lance's
   tutorial. These mechanisms can be added independently.

5. The iterative retrieval in the original paper is selected based on
   question complexity. Lance's corrective loops are instead triggered
   by retrieval or generation evaluation results.

6. Lance's Adaptive RAG is therefore best understood as source-based
   routing combined with CRAG-style retrieval correction and
   Self-RAG-inspired answer evaluation, rather than a reproduction
   of the original complexity-based Adaptive-RAG architecture.

NOTE:
These comparisons refer to Lance's tutorial implementations.
They should not be interpreted as a complete comparison of the
original Self-RAG and Adaptive-RAG research papers.


** HOW LANCE'S ADAPTIVE RAG WORKS:

1. DOCUMENT PREPARATION:

   Three Lilian Weng blog posts are loaded using WebBaseLoader.

   They cover:
   - LLM-powered autonomous agents.
   - Prompt engineering.
   - Adversarial attacks.

   The documents are split into chunks of 500 tokens with no overlap.

   Embeddings are generated and stored in Chroma, which provides the
   vectorstore retriever.

   The vectorstore is the system's indexed knowledge source.


2. QUERY ROUTING:

   The user submits a question.

   The question_router uses an LLM with structured output to select
   one of two datasource values:

   - "vectorstore"
   - "web_search"

   The routing decision is based on the question's subject and the
   description of the vectorstore's contents.

   Questions about agents, prompt engineering, or adversarial attacks
   are intended to use Chroma.

   Other questions are intended to use Tavily.

   Example:

   "What are the types of agent memory?"
       -> Intended route: vectorstore

   "Who will the Bears draft first?"
       -> Intended route: web_search

   NOTE: The routing decision is made by the LLM and may be incorrect.
   The router does not explicitly measure question complexity.


3. VECTORSTORE PATH:

   If the router selects "vectorstore":

   A. Retrieve relevant document chunks from Chroma.

   B. Grade each retrieved chunk using an LLM relevance grader.

      The intended decision is:

      - "yes": The chunk is relevant to the question.
      - "no": The chunk is irrelevant to the question.

   C. Retain relevant chunks and discard irrelevant chunks.

   D. Evaluate the filtered document list.

      If at least one relevant chunk remains:
          -> Generate an answer.

      If no relevant chunks remain:
          -> Rewrite the question.
          -> Retrieve from Chroma again.
          -> Grade the newly retrieved chunks.

   This introduces a corrective retrieval loop.

*** IMPORTANT NOTE:

In this implementation, unsuccessful vectorstore retrieval triggers
question rewriting followed by another vectorstore retrieval. This
process repeats until a relevant document is found or the graph
reaches its execution limit.

The initial route (Chroma or Tavily) is selected by the LLM BEFORE
retrieving or evaluating any documents.

In this tutorial, the router's system prompt informs the LLM that Chroma 
contains documents about agents, prompt engineering, and adversarial attacks.
Therefore, questions outside these topics should be routed directly
to Tavily.

However, this routing decision is based on the topic description,
NOT on the actual documents stored in Chroma. The LLM may still
choose the wrong route, even with a well-designed system prompt.

In real-world applications, relying exclusively on initial topic
routing without a fallback can be problematic. A more robust
implementation could switch to Tavily after a specified number
of unsuccessful Chroma retrieval attempts.

Lance's tutorial does not implement this fallback.

In short:
In this tutorial, the LLM does NOT automatically switch from Chroma to Tavily 
when Chroma retrieval fails, even after multiple attempts.

NOTE: An initially Tavily-routed question can subsequently enter
the Chroma retrieval path if its generated answer fails the
answer usefulness check.


4. WEB SEARCH PATH:

   If the router selects "web_search":

   A. Tavily searches the web using the user's question.

   B. The search tool requests up to three search results (k=3).

   C. The retrieved result contents are combined into one text string.

   D. The combined text is wrapped in a LangChain Document object.

   E. The Document is passed directly to the generation step.

   IMPORTANT:
   This path does not initially perform document relevance grading,
   query rewriting, or reranking of the Tavily results.

   The retrieved web content is passed directly to the generator.

   The current implementation also combines all Tavily results into
   one Document instead of preserving separate documents with their
   individual source metadata.


5. ANSWER GENERATION:

   Both retrieval paths eventually reach the generate() node.

   The RAG generation chain consists of:

       Prompt -> LLM -> String Output Parser

   The prompt is inherited from the earlier Self-RAG section and
   instructs the LLM to:

   - Answer using the retrieved context.
   - Avoid inventing an answer when the evidence is insufficient.
   - Use a maximum of three sentences.
   - Keep the answer concise.

   The current implementation uses a local Ollama model.

   Its context window has been adjusted to accommodate larger
   retrieved inputs.

   NOTE:
   The vectorstore and web search paths supply documents in different
   representations:

   - Vectorstore: a list of Document objects.
   - Web search: a single Document containing combined search results.

   The current generation chain receives those objects directly.


6. GENERATION EVALUATION:

   After generating an answer, the workflow uses two separate graders.

   A. Hallucination / Groundedness Grader:

      Checks whether the generated answer is supported by the
      retrieved information.

      - "yes": Answer is considered supported by the documents.
      - "no": Answer is considered unsupported.

      If the answer is unsupported:
          -> Return to generate().
          -> Generate another answer using the same retrieved documents.

      NOTE: The current implementation does not automatically acquire
      new evidence when the hallucination grader rejects an answer.


   B. Answer Grader:

      Executed only after the hallucination grader accepts the answer.

      Checks whether the generated answer resolves the user's question.

      - "yes": Answer addresses the question.
      - "no": Answer does not adequately address the question.

      If the answer addresses the question:
          -> End the workflow.

      Otherwise:
          -> Rewrite the question.
          -> Retrieve new documents from Chroma.
          -> Continue the retrieval and generation workflow.

   IMPORTANT:
   The two graders evaluate different properties.

   Groundedness asks:
       "Is this answer supported by the retrieved evidence?"

   Answer usefulness asks:
       "Does this answer actually resolve the user's question?"

   An answer can be grounded but fail to answer the question.
   Similarly, an answer can address the question while containing
   unsupported factual claims.

   NOTE:
   The graders are LLM-based evaluations, not guarantees of factual
   correctness.


LANGGRAPH WORKFLOW:

The main execution paths are:

A. WEB SEARCH ROUTE:

   START
      -> Route Question
      -> Tavily Web Search
      -> Generate
      -> Check Groundedness
      -> Check Answer Usefulness
      -> END (if accepted)


B. VECTORSTORE ROUTE:

   START
      -> Route Question
      -> Chroma Retrieval
      -> Grade Documents
      -> Generate
      -> Check Groundedness
      -> Check Answer Usefulness
      -> END (if accepted)


C. RETRIEVAL CORRECTION:

   Grade Documents
      -> No Relevant Documents
      -> Rewrite Question
      -> Chroma Retrieval
      -> Grade Documents (repeat)


D. UNSUPPORTED GENERATION:

   Generate
      -> Hallucination Grader = "no"
      -> Generate Again (same evidence)


E. ANSWER DOES NOT RESOLVE QUESTION:

   Generate
      -> Hallucination Grader = "yes"
      -> Answer Grader = "no"
      -> Rewrite Question
      -> Chroma Retrieval
      -> Grade Documents
      -> Generate Again

   NOTE:
   This last path always returns to Chroma retrieval, even if the
   initial question was routed to web search.


HOW ADAPTIVE RAG RELATES TO CRAG AND SELF-RAG:

The three approaches emphasize different decisions.

* Adaptive RAG:

  "Which retrieval or answering strategy should be used?"

  Main concept: Select a strategy based on the question.

  Lance's implementation uses an LLM router to select Chroma or Tavily.


* CRAG (Corrective RAG):

  "Is the retrieved information sufficiently relevant or reliable?"

  Main concept: Evaluate retrieved information and correct inadequate
  retrieval before generation.

  Lance's Adaptive RAG incorporates document relevance grading
  and question rewriting for unsuccessful vectorstore retrieval.


* Self-RAG:

  "Is the generated answer supported by evidence, and is it useful?"

  Main concept: Self-reflection and evaluation of retrieval or
  generation quality.

  Lance's Adaptive RAG incorporates hallucination and answer graders
  similar to those demonstrated in the earlier Self-RAG tutorial.


These concepts are complementary.

A single RAG system can combine:
  - Adaptive routing.
  - Corrective retrieval.
  - Generation self-evaluation.

However, combining these components does not automatically reproduce
the complete architectures proposed in their respective research papers.


LIMITATIONS OF THE CURRENT IMPLEMENTATION:

1. Source-Based Rather Than Complexity-Based Routing:

   The router chooses between Chroma and Tavily based on subject matter.
   It does not select among no-retrieval, single-step, and iterative RAG
   using a trained question-complexity classifier.

2. No Direct Answer Route:

   Every question is routed to an external information source.
   The workflow cannot deliberately skip retrieval.

3. Limited Retrieval Correction:

   Failed Chroma retrieval triggers question rewriting and another
   Chroma search, not automatic fallback to web search.

4. No Web Result Refinement:

   Tavily result contents are combined and passed to generation without
   relevance filtering, chunk-level reranking, or context compression.

5. Potentially Large Context:

   Passing all retrieved web content can increase token usage, memory
   requirements, and generation latency.

   A larger context window increases capacity but does not necessarily
   improve relevance or answer quality.

6. Potential Infinite Loops:

   If retrieval continually fails, the graph may repeatedly rewrite
   the question and search Chroma.

   If groundedness grading continually fails, the graph may repeatedly
   regenerate answers from the same evidence.

   The graph relies on execution limits or external safeguards rather
   than explicit retry counters in its state.

7. LLM Grading Is Imperfect:

   Retrieval relevance, groundedness, and answer usefulness are
   estimated by LLMs. Incorrect grades can cause wrong routing
   decisions, unnecessary retries, or acceptance of weak answers.

8. Limited Source Traceability:

   Combining Tavily results into one Document without preserving
   their URLs and metadata makes source-level attribution more difficult.


RELATIONSHIP TO AGENTIC RAG:

Lance's Adaptive RAG demonstrates limited agentic behavior through
LLM-driven routing and evaluation.

The LLM decides which predefined source to use and whether certain
quality checks pass.

However, the available actions and graph transitions are already
specified by the developer.

The workflow cannot independently invent new tools, design arbitrary
retrieval plans, or dynamically choose any action outside the
configured LangGraph transitions.

It is therefore more accurately described as an LLM-assisted,
conditionally routed RAG workflow than a highly autonomous agent.


SUMMARY:

Adaptive RAG dynamically chooses an appropriate retrieval or answering
strategy based on the user's question.

The original Adaptive-RAG paper emphasizes adapting retrieval complexity:
no retrieval, single-step retrieval, or iterative retrieval.

Lance's simplified implementation instead routes questions between
Chroma and Tavily and incorporates corrective retrieval and
generation self-evaluation.

Its main components are:

  Query Routing -> Retrieval -> Document Grading (Chroma path)
                -> Generation -> Groundedness Grading
                -> Answer Usefulness Grading -> END

The key distinction is that Adaptive RAG determines WHICH retrieval
strategy to use, while CRAG focuses on correcting retrieved evidence
and Self-RAG focuses on evaluating retrieval and generated answers.

These techniques can be combined to build more flexible, reliable,
and efficient retrieval-augmented question-answering systems.
"""

# Set embeddings
embd = get_embeddings()

# Docs to index (3 Lilian Weng blogpost urls)
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load the blogposts
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split the blogposts into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Add the text splits to vectorstore (Chroma)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)

retriever = vectorstore.as_retriever()

### Router - route the query to the appropriate source (Chroma or Tavily)

from typing import Literal


# Data model - Pydantic class
# Restricts the output to the Literal["vectorstore", "web_search"] --> either a vectorstore or web_search
# Gives the description in the Field "Given a user question choose to route it to web search or a vectorstore."
# to the LLM to let the LLM decide which route to take.
# LangChain converts the Pydantic definition into a JSON schema resembling:
# {
#   "properties": {
#     "datasource": {
#       "type": "string",
#       "enum": ["vectorstore", "web_search"],
#       "description": "Given a user question choose to route it to web search or a vectorstore."
#     }
#   },
#   "required": ["datasource"]
# }
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call (temperature=0) reduces hallucination (creativity) aspect significantly but, does not compeletely eliminate it
llm = get_llm(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt - removed for now due to LLM not following instructions carefully
# (few shot prompting - examples provided for the llm in the below prompt instead)

# system = """You are an expert at routing a user question to a vectorstore or web search.
# The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
# Use the vectorstore for questions on these topics. Otherwise, use web-search."""

# Added later (YK) --> start
# Few shot prompting - provides multiple examples for LLM to be better at following instructions
# Explicitly prioritize the vectorstore for topics covered by indexed documents.
system = """You are a query router. Your task is to classify the user's question
and select exactly one datasource: "vectorstore" or "web_search".

The vectorstore contains documents about:
1. LLM-powered autonomous agents, including agent memory and planning.
2. Prompt engineering, including prompting techniques for LLMs.
3. Adversarial attacks against language models.

Routing rules:
- If the question concerns ANY of these topics, select "vectorstore".
- Prefer "vectorstore" whenever the topic is covered by the indexed documents.
- Do not select "web_search" merely because the web could also answer the question.
- Select "web_search" when the question falls outside these topics.

Examples:
Question: How does agent memory work?
Datasource: vectorstore

Question: What are common LLM prompting techniques?
Datasource: vectorstore

Question: How do adversarial attacks affect language models?
Datasource: vectorstore

Question: Who was selected first in the 2024 NFL draft?
Datasource: web_search

Return only the selected datasource using the required structured output."""
# Added later (YK) --> end

# Router - route to vectorstore (Chroma) or web_search (Tavily)
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "Who will the Bears draft first in the NFL draft?"}
    )
)
print(question_router.invoke({"question": "What are the types of agent memory?"}))


### Retrieval Grader


# Data model
# Grade the retrieved documents - check for relevance to the question
# Used mainly in the vectorstore (Chroma) path
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = get_llm(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt for grading the retrieved documents from vectorstore (Chroma)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# System prompt gets priotorized by the LLM (instruction following)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)

# A small test for the retriever
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### Generate

# Prompt - write three sentence concise answers and avoid long, detailed answers
client = Client()
prompt = client.pull_prompt("rag-from-scratch")

# LLM
llm = get_llm(temperature=0)

# Added later (YK) --> start

# Increase the Ollama context window for Adaptive RAG generation.
# Test whether the long Tavily context was exceeding the previous limit.
llm.num_ctx = 16384

print("GENERATION MODEL CONTEXT WINDOW:", llm.num_ctx)
# Added later (YK) --> end


# Post-processing - \n\n for separating the docs - easier visibility/separation between docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Allows Python to execute external system/terminal commands.
# Used here to run "ollama ps" and inspect the loaded model,
# GPU/CPU usage, and context window size.
import subprocess

# Tracks LLM token usage (input, output, and total tokens) during model calls.
# Used to monitor token consumption in the Adaptive RAG generation step.
from langchain_core.callbacks import UsageMetadataCallbackHandler

# Chain - Generation
rag_chain = prompt | llm | StrOutputParser()

# Run - testing the generation
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

### Hallucination Grader


# Removed due to llm (this particular local LLM - others may not do this)
# giving numerical scores instead of a binary yes or no (YK)
# It would indefinetely loop - trying to get a yes or no...

# Data model
# class GradeHallucinations(BaseModel):
#     """Binary score for hallucination present in generation answer."""

#     binary_score: str = Field(
#         description="Answer is grounded in the facts, 'yes' or 'no'"
#     )

from typing import Literal


# Literal gives a binary choice to the LLM to output - Literal["yes", "no"]
# Field gives the helpful description on when to say yes or no
class GradeHallucinations(BaseModel):
    """Binary score for whether the answer is grounded in retrieved facts."""

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "Return exactly 'yes' if the generated answer is fully supported "
            "by the retrieved documents; otherwise return 'no'. "
            "Do not include reasoning, JSON text, or numeric scores."
        )
    )


# LLM with function call
llm = get_llm(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt - Checking Hallucination (facts vs made up answers)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader


# Data model - checking if the generated answer addresses the question asked by the user
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = get_llm(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})

### Question Re-writer

# LLM
llm = get_llm(temperature=0)

# Prompt - if retrieval fails to produce a relevant document, question-rewriter rewrites the question for better retrieval
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

### Search - Tavily Web Search

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

from typing_extensions import TypedDict


# TypedDict lets you define the expected structure of a Python dictionary, including its keys and the types of their values.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: list[str]


# retriever function to call
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


# generator function to call
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Added later (YK) --> start
    # Diagnose the context and actual prompt sent to the LLM.

    print("\n--- ADAPTIVE GENERATION DEBUG ---")
    print("DOCUMENTS TYPE:", type(documents))

    # Measure the retrieved evidence size.
    if isinstance(documents, list):
        context_text = "\n\n".join(d.page_content for d in documents)
    else:
        context_text = documents.page_content

    print("EVIDENCE LENGTH (characters):", len(context_text))

    # Inspect the inherited LangSmith prompt.
    print("\nPROMPT TEMPLATE:")
    print(prompt)

    # Format the exact inputs without calling the LLM.
    formatted_prompt = prompt.invoke({"context": documents, "question": question})

    prompt_text = formatted_prompt.to_string()

    print("\nFORMATTED PROMPT LENGTH:", len(prompt_text))
    print("\nFORMATTED PROMPT START:")
    print(prompt_text[:1200])
    print("\nFORMATTED PROMPT END:")
    print(prompt_text[-1200:])

    # Added later (YK) --> end

    # RAG generation - removed to test the token usage below using usage_cb (YK)
    # generation = rag_chain.invoke({"context": documents, "question": question})

    # RAG generation

    # Added later (YK) --> start
    # Track the actual input and output token usage for Adaptive RAG generation.
    usage_cb = UsageMetadataCallbackHandler()

    generation = rag_chain.invoke(
        {"context": documents, "question": question},
        config={"callbacks": [usage_cb]},
    )

    print("\n--- ADAPTIVE GENERATION TOKEN USAGE ---")
    print(usage_cb.usage_metadata)
    # Added later (YK) --> end

    # Added later (YK) --> start

    # Inspect Ollama's context immediately after graph generation,
    # before the hallucination and answer graders execute.
    print("\n--- OLLAMA CONTEXT AFTER ADAPTIVE GENERATION ---")

    # subprocess.run() - Executes an external command and waits for it to finish.
    # check= False - Prevents Python from automatically raising an exception if the command exits with a nonzero status.
    # capture_output=True - Captures the command's standard output (stdout) and error output (stderr) instead of displaying
    # them directly in the terminal.
    # text=True - Converts the captured output into Python strings rather than bytes.
    result = subprocess.run(
        ["ollama", "ps"], check=False, capture_output=True, text=True
    )
    print(result.stdout)
    # Added later (YK) --> end

    return {"documents": documents, "question": question, "generation": generation}


# document relevance grader function
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


# transform query function
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# web search function
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


### Edges/Nodes ###


# route question to vectorstore or web_search
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


# route - generate the answer or transform (re-write) the query
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# Hallucination and question addressability grader
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score  # hallucination grade

    # # Added later (YK) --> start

    # # Debug why the hallucination grader rejects the answer.
    # print("\n--- HALLUCINATION DEBUG ---")
    # print("QUESTION:", question)
    # print("DOCUMENTS TYPE:", type(documents))
    # print("RETRIEVED EVIDENCE:", repr(documents)[:5000])
    # print("GENERATED ANSWER:", generation)
    # print("GRADER RESULT:", grade)

    # # Diagnose the hallucination grader's output.
    # print("GRADE TYPE:", type(grade))
    # print("GRADE VALUE:", repr(grade))

    # # Stop if the grader returns something other than "yes" or "no".
    # if grade not in ("yes", "no"):
    #     raise ValueError(f"Unexpected hallucination grade: {grade!r}")

    # # Added later (YK) --> end

    # Check hallucination and question addressing (2 grade == yes but, 1st is for hallucination and 2nd is for question addressing)
    if grade == "yes":  # hallucination grade = yes
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")

        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score  # question addressing grade

        if grade == "yes":  # question addressing grade = yes
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# Stitching the workflow with Langraph
from langgraph.graph import END, START, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes (not in order necessarily) - stitching later
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
# route the question to vectorstore (Chroma) or web_search (Tavily) at the beginning
# the strings are functions defined previously (example: grade_documents is a function defined previously)
# conditional edges are used to route the flow to the correct nodes
workflow.add_conditional_edges(
    START,
    route_question,  # route question returns either (web_search = web_search or vectorstore = retrieve)
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)

# unconditional edges - execute the flow immediately (example: web_search --> to --> generate immediately)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")

# if grade_documents is executed, the next node to be executed is decide_to_generate which is a router (transform query or directly generate)
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
# transform query leads to retrieval again --> this can repeat (loop)
workflow.add_edge("transform_query", "retrieve")
# if the answer is generated, grade_generation_v_documents_and_question routes to the appropriate nodes
# based on the answer grade (hallucination and question addressing)
# transform_query leads to a loop (goes back to retrieve the document again by improving the question further)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile the Langraph workflow
app = workflow.compile()

# Added later (YK) --> start
# Test Chroma retrieval and document grading independently of the router.
test_state = retrieve(
    {"question": "What is prompt engineering for large language models?"}
)

graded_state = grade_documents(test_state)

print("RETRIEVED DOCUMENTS:", len(test_state["documents"]))
print("RELEVANT DOCUMENTS:", len(graded_state["documents"]))
# Added later (YK) --> end

# pretty print (pprint) - displays the longer text on multiple lines in a structured (prettier) way
from pprint import pprint

# Run
# temporary test - removing this (YK)
inputs = {
    "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
}

# temporary test - this should not do a web search as this information exists in the blogpost.
# inputs = {"question": "What is prompt engineering for large language models?"}

for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])

# Run
inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
