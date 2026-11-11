# CRAG: extracted and locally adapted educational example.
# Original LangGraph example: https://github.com/langchain-ai/langgraph/tree/main/examples/rag
# Origin: langgraph/examples/rag/langgraph_crag.ipynb (MIT)
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langsmith import Client
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

from config import LOCAL_MODEL, get_embeddings, get_llm

"""
# ----------------------------------------- CRAG -----------------------------------------
"""

print("\n\n" + "*" * 20 + " CRAG RESULTS " + "*" * 20 + "\n\n")

"""
Corrective RAG (CRAG) is a RAG approach that evaluates the quality of retrieved
information and takes corrective actions when the initial retrieval is
insufficient or irrelevant.

Unlike traditional RAG, which typically retrieves documents and passes them
directly to an LLM, CRAG evaluates the retrieved information before generation.

* CRAG uses a retrieval evaluator to determine whether retrieved information
  is sufficiently relevant to the user's question.

* Based on the evaluation, it can retain relevant information, discard
  irrelevant information, or retrieve additional information (e.g., web search).

* The goal is to improve the quality of the evidence provided to the LLM,
  reducing the likelihood of generating answers from irrelevant information.


ORIGINAL CRAG VS. LANCE'S SIMPLIFIED IMPLEMENTATION:

The original CRAG research includes more sophisticated retrieval evaluation
and a Knowledge Refinement mechanism.

* Retrieval Evaluation:
  The original approach distinguishes between three retrieval-quality cases:
  Correct, Incorrect, and Ambiguous, using confidence thresholds.

  These classifications determine whether to use retrieved information,
  perform corrective retrieval, or combine available information with
  additional evidence.

* Knowledge Refinement (Decompose-then-Recompose):
  Instead of evaluating only entire documents, CRAG can remove irrelevant
  information from within retrieved documents.

  1. Decompose: Split documents into smaller information units (knowledge strips).
  2. Filter: Evaluate the strips and discard irrelevant information.
  3. Recompose: Combine the relevant information into a cleaner context.

  Knowledge Refinement occurs after retrieval and evaluation, but before
  answer generation. It can be applied to retrieved documents and information
  obtained through external search.

* Lance's CRAG tutorial is a simplified implementation:

  1. Chroma retrieves documents relevant to the user's question.
  2. An LLM grades each retrieved document chunk as relevant or irrelevant.
  3. Relevant chunks are retained, and irrelevant chunks are discarded.
  4. If at least one chunk is graded irrelevant, the workflow triggers web search.
  5. An LLM rewrites the question to improve the search query.
  6. Tavily searches the web and retrieves additional information.
  7. The retained documents and web results are passed to the LLM for generation.

  NOTE: Lance's implementation does not include the original three-way
  retrieval-quality evaluation or knowledge-strip refinement.

  It also does not automatically reevaluate the retrieved web results or
  continue searching until sufficient evidence has been obtained.

  Therefore, Lance's tutorial demonstrates the fundamental corrective
  workflow rather than the complete original CRAG method.


HOW CRAG RELATES TO ACTIVE RAG:

Active RAG broadly refers to RAG systems that dynamically decide when,
how, or whether to retrieve additional information, rather than always
following a fixed retrieval process.

CRAG can be considered an Active/Dynamic RAG approach because it evaluates
retrieval quality and conditionally retrieves additional information.

However, Active RAG is not a formally standardized category with fixed
subtypes. In research, Active Retrieval can also refer more specifically
to retrieving information during answer generation.

Other related approaches include:

* Self-RAG:
  Uses self-reflection mechanisms to determine whether retrieval is needed,
  evaluate retrieved evidence, and assess whether generated answers are
  supported by the available information.

* Adaptive RAG:
  Selects the retrieval strategy based on the question's complexity or needs.
  For example, it might answer directly, retrieve from a database, or use
  a multi-step retrieval process.

* FLARE (Forward-Looking Active Retrieval):
  Performs additional retrieval during answer generation when the model
  predicts that it may need more information to continue reliably.

* IRCoT (Interleaved Retrieval with Chain-of-Thought):
  Alternates between reasoning and retrieval, using intermediate reasoning
  steps to determine what information should be retrieved next.

* Agentic RAG:
  Uses an AI agent to dynamically select information sources, tools, and
  retrieval actions based on the task, potentially across multiple steps.


HOW CRAG RELATES TO AGENTIC RAG:

An important distinction is that Active RAG and Agentic RAG are related,
but are not necessarily interchangeable.

Active RAG emphasizes dynamically controlling retrieval.

Agentic RAG emphasizes delegating decisions about actions, tools, and
information gathering to an AI agent.

Whether a system is considered agentic depends on the degree and nature
of its autonomy. There is no universally agreed strict boundary.

Consider three general architectures:

1. Traditional RAG (Non-Agentic):

   * The developer defines a fixed sequence of operations.
   * Documents are retrieved and passed to the LLM for answer generation.
   * The system does not independently decide whether retrieval is needed
     or whether to perform additional actions.

   Example:
   Question -> Chroma -> Retrieved documents -> LLM -> Answer


2. Conditional/Active RAG (Limited Agentic Behavior):

   * The system can make certain decisions during execution.
   * An LLM may evaluate documents and determine whether corrective action
     is necessary.
   * However, available actions and execution paths are predefined.

   Lance's CRAG implementation is an example:

   * The LLM evaluates document relevance and rewrites search queries.
   * LangGraph follows predefined conditional routes.
   * The developer has already specified when Tavily should be called.
   * The system cannot independently choose an entirely different tool
     or retrieval strategy unless that behavior is implemented.

   This demonstrates limited agentic behavior, but it is more accurately
   described as an LLM-assisted conditional workflow than a highly
   autonomous agent.


3. More Autonomous Agentic RAG:

   * An LLM-driven agent can decide which tools to use, which information
     sources to consult, and what actions to take next.
   * The agent can examine intermediate results and adapt its plan.
   * The system can perform multiple reasoning and tool-use steps.

   For example, an agent with access to Chroma, Tavily, SQL, and Python
   might decide:

   - "I already have enough information. No retrieval is necessary."
   - "Chroma does not contain the answer. I should search the web."
   - "This question requires a calculation. I should use Python."
   - "The search results are insufficient. I should try another source."
   - "I have enough evidence now. I can generate the final answer."

   These decisions are made dynamically by the agent, within the tools,
   permissions, and constraints established by the developer.


IMPORTANT DISTINCTIONS:

* Conditional routing alone does not make a system a fully autonomous agent.

* Using an LLM for document grading or query rewriting does not necessarily
  make the overall workflow highly agentic.

* A predefined LangGraph workflow can contain agentic components, but the
  overall level of autonomy depends on the decisions those components can make.

* Conversely, an AI agent can still be agentic while operating within
  predefined boundaries and using a restricted set of tools.

* CRAG, Self-RAG, Adaptive RAG, FLARE, and IRCoT demonstrate different forms
  of dynamic retrieval, but do not necessarily require fully autonomous agents.

* Agentic RAG generally grants an agent greater responsibility for deciding
  which actions to perform and how to adapt based on intermediate results.


SUMMARY:

CRAG improves traditional RAG by evaluating retrieved information and
performing corrective actions when retrieval quality is insufficient.

Lance's CRAG implementation demonstrates this through document grading,
conditional routing, query rewriting, and Tavily web search.

The original CRAG includes more detailed retrieval-quality evaluation and
Knowledge Refinement, which filters irrelevant information within documents.

CRAG can be considered an Active/Dynamic RAG approach and can demonstrate
limited agentic behavior. However, implementing CRAG does not automatically
create a highly autonomous AI agent.

The defining distinction is not simply WHETHER decisions are made, but
WHICH decisions are delegated to the LLM, WHICH actions are available,
and HOW freely the system can adapt its behavior based on the task.
"""

# Documents for Lance Martin's (from Langchain) simplified CRAG example
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from the specified URLs using WebBaseLoader and flatten the list of documents.
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into smaller chunks for better retrieval and embedding.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Add the chunks to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=get_embeddings(),
)

retriever = vectorstore.as_retriever()

# Import Basemodel and Field from pydantic to define a structured data model for grading documents.
from pydantic import BaseModel, Field


# Data model (Pydantic) for grading documents, with a binary score indicating relevance.
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call - GradeDocuments is the function called by the LLM to produce structured output
llm = get_llm(temperature=0)

# configures the llm to produce structured output based on the GradeDocuments model, allowing it to return
# a binary score for document relevance
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt for grading documents, instructing the LLM to assess the relevance of a retrieved document to a user question.
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# system has higher priority for the llm than the human prompt, so it will be used to guide the LLM's behavior in grading documents.
# if system is not explicitly specified, the llm treats all of the contents of the prompt as a human prompt and may not follow the grading
# instructions as intended.
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# chain for document relevance grader during retrieval.
retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### Generate

# Generation Prompt - not used for grading or retrieval, but for generating answers based on the retrieved documents.
# The prompt pulled from LangSmith:
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}
# Answer:
client = Client()
prompt = client.pull_prompt("rag-from-scratch")

# LLM
llm = get_llm(temperature=0)


# Post-processing for separating the page content of each document in the list of retrieved documents,
# joining them with two newlines for better readability.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Generation Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

### Question Re-writer

# LLM
llm = get_llm(temperature=0)

# Question Rewriting Prompt
# The original prompt was too vague, causing the LLM to generate lengthy explanations,
# multiple suggested queries, and Markdown instead of a single web search query.
# In one case, the LLM incorrectly changed "AlphaCodium" to "AlphaCode", altering
# the user's intended subject. The resulting query also exceeded Tavily's recommended
# length limit, causing an HTTP 400 (Bad Request) error.
#
# The modified prompt explicitly restricts the LLM to one short, plain-text query,
# preserves technical terms and proper nouns, and prevents unnecessary explanations.

# Original Question Rewriting Prompt for Tavily Web Search:
# system = """You a question re-writer that converts an input question to a better version that is optimized \n
#      for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

# Modified Question Rewriting Prompt for Tavily Web Search:
system = """You are a question re-writer that optimizes user questions for web search.

Rules:
- Return exactly ONE concise search query.
- Output only the rewritten query, without explanations or Markdown.
- Do not provide multiple options.
- Keep the query under 150 characters.
- Preserve technical terms, paper names, and proper nouns exactly as written.
- Never assume an unfamiliar term is a typo.
- Do not answer the question.
"""
# system has higher priority for the llm than the human prompt, so it will be used to guide the LLM's behavior in rewriting questions.
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# question rewriting chain for generating a better search query based on the user's original question.
question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

### Tavily Web Search

from langchain_community.tools.tavily_search import TavilySearchResults

# k=3 return the top 3 search results from Tavily- a web search tool that can be used to retrieve additional information
# when the initial retrieval is insufficient or irrelevant.
web_search_tool = TavilySearchResults(k=3)


from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: list[str]


from langchain_core.documents import Document


# retriever function that retrieves documents based on the user's question and adds them to the graph state.
# retriever chain used in this function is written above in the code for tutorial purposes --> not clean for production code.
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


# generator function that lets LLM generate the answer based on the user's question and adds them to the graph state.
# rag_chain used in this function is written above in the code for tutorial purposes --> not clean for production code.
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

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# grader function to grade documents based on their relevance to the user's question and adds them to the graph state.
# retrieval_grader used in this function is written above in the code for tutorial purposes --> not clean for production code.
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

    # Initialize filtered_docs as an empty list to store relevant documents and assign a "No" flag for web_search.
    # web_search is changed to the "Yes" flag if web_search becomes necessary later.
    filtered_docs = []
    web_search = "No"

    # Score each doc
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        # binary_score = "yes" - for relevant documents
        # binary_scor = "no" - for irrelevant documents
        grade = score.binary_score
        # if document is relevant, append the document to filtered documents
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # if document is irrelevant, change web_search = "yes" and then continue (go the next document)
        # continue has no real benefit in this case as there is nothing else below it to ignore but,
        # we will stick to it due to it being part of the tutorial
        # This means that if at least one irrelevant document is found, a web search is performed
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# Re-write the questions (transform the query) for web_search
# question_rewriter used in this function is written above in the code for tutorial purposes --> not clean for production code.
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


# Web search function for searching the web with Tavily
# web_search_tool used in this function is written above in the code for tutorial purposes --> not clean for production code.
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
    documents = state["documents"]

    # Web search using Tavily(k=3) where k=top 3 search results and the re-written question (translated query)
    docs = web_search_tool.invoke({"query": question})
    # Checking for empty tavily search results, etc. which will throw an error during the run
    # checking: type of docs, response in the doc, repr shows all text including new line and other characters,
    # shows [:2000] --> first 2000 characters of the document, lenght of the question
    print("TAVILY RESPONSE TYPE:", type(docs))  # added later (YK)
    print("TAVILY RESPONSE:", repr(docs)[:2000])  # added later (YK)
    print("TAVILY SEARCH QUERY:", repr(question))  # added later (YK)
    print("QUERY LENGTH:", len(question))  # added later (YK)

    # separate the documents using new line to be able to tell the different documents apart from each other
    web_results = "\n".join([d["content"] for d in docs])
    # Converts the web search results (documents) into Langchain document object so that it is easier to add to the chroma documents
    web_results = Document(page_content=web_results)
    # append the web search results back to the documents already retrieved
    documents.append(web_results)

    return {"documents": documents, "question": question}


### Edges


# Helps the llm decide generate the answer or not based on relevance of the documents
# If 1 irrelevant document is found --> perform web search with Tavily --> Do not generate an answer until the web search is complete
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
    web_search = state["web_search"]
    state["documents"]

    # If web_search = "yes", transform the query (question) for web_search with Tavily
    if web_search == "Yes":
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


# Flow engineering --> Stitching together the nodes in the CRAG graph
from langgraph.graph import END, START, StateGraph

# State that can be referred to by other nodes later if needed
workflow = StateGraph(GraphState)

# Define the nodes (not the stitching of the graph yet - just the nodes defined separately)
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
# After the grade_documents node finishes, LangGraph calls decide_to_generate(state) and uses its return value to choose the next node.
# "transform_query": "transform_query" means if decide_to_generate returns "trasform_query" then go to the node/function named transform_query,
# if it returns "generate" then go to the node/function named "generate".
# In short, either transform the query and do a web search and do not generate the answer before completing the web search
# for a more appropriate answer or generate the answer directly
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
# if it goes to transform query, then the rest follows --> web_search --> generate
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile the workflow
app = workflow.compile()

# pprint() can organize nested dictionaries and lists across multiple lines, making them easier to inspect.
# pprint() simply prints the node name and a separator as the graph executes.
from pprint import pprint

# Test Run # 1
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

# Test Run # 2
inputs = {"question": "How does the AlphaCodium paper work?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
