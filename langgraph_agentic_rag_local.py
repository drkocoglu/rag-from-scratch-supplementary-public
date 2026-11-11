# Agentic RAG: extracted and locally adapted educational example.
# Original LangGraph example: https://github.com/langchain-ai/langgraph/tree/main/examples/rag
# Origin: langgraph/examples/rag/langgraph_agentic_rag.ipynb (MIT)
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
# ----------------------------------------- Agentic RAG -----------------------------------------
"""

print("\n\n" + "*" * 20 + " AGENTIC RAG RESULTS " + "*" * 20 + "\n\n")

"""
WHAT IS AGENTIC RAG?

Agentic RAG gives an LLM the ability to decide whether it needs to retrieve
external information before answering a question.

Unlike traditional RAG workflows, where retrieval is predefined, an agent
can choose to call a retrieval tool or answer directly without retrieval.

The key concept is TOOL CALLING:
Instead of always executing retrieval, the LLM decides whether to use
the available retrieval tool based on the user's question and the tool's
description.


HOW IS AGENTIC RAG DIFFERENT FROM THE PREVIOUS RAG IMPLEMENTATIONS?

1. CRAG (Corrective RAG):
   - Retrieves documents and evaluates their relevance.
   - If retrieval is insufficient, performs corrective actions,
     including web search.
   - Main focus: Correct poor retrieval results.

2. Self-RAG (Lance's tutorial):
   - Retrieves documents and evaluates relevance.
   - Checks whether generated answers are grounded and useful.
   - Can retry generation or rewrite the question.
   - Main focus: Evaluate and improve generated answers.

3. Adaptive RAG (Lance's tutorial):
   - Uses an LLM router to initially select Chroma or Tavily.
   - Retrieval then follows the selected path.
   - Also includes document and answer evaluation.
   - Main focus: Select an appropriate information source.

4. Agentic RAG (THIS IMPLEMENTATION):
   - Gives the LLM access to a retriever as a TOOL.
   - The LLM can decide whether to call the tool or answer directly.
   - If retrieved documents are irrelevant, the question is rewritten,
     and the agent is called again.
   - Main focus: Let the LLM control when to use retrieval.


IMPORTANT DISTINCTION: ADAPTIVE RAG VS. AGENTIC RAG

Adaptive RAG:
   - The LLM makes an initial decision between Chroma and Tavily.
   - Retrieval is executed according to that routing decision.
   - Later corrective loops do not rerun the initial router.

Agentic RAG:
   - The LLM is given tools it can choose to call.
   - It can answer without retrieving anything.
   - After question rewriting, the agent is invoked again and can
     make another tool-calling decision.

NOTE:
In this tutorial, the agent has access to only ONE tool:
the Chroma retriever. It does NOT have a Tavily search tool.

Therefore, Agentic RAG does not necessarily mean multiple tools
or multiple agents. Even one LLM with a retrieval tool can implement
this agentic behavior.


HOW THIS AGENTIC RAG WORKS:

1. START -> agent
   - The LLM receives the user's question.
   - It decides whether to call the retrieval tool.

2. If the LLM does NOT call a tool:
   - The graph ends with the agent's direct response.

3. If the LLM calls the retrieval tool:
   - ToolNode executes the Chroma retriever.
   - Retrieved documents are returned as a ToolMessage.

4. grade_documents:
   - An LLM checks whether the retrieved content is relevant.
   - Relevant -> generate an answer using retrieved evidence.
   - Irrelevant -> rewrite the question.

5. rewrite -> agent:
   - The question is rewritten.
   - The agent runs again and can make another tool-calling decision.

6. generate -> END:
   - The LLM generates an answer using the retrieved content.
   - The graph ends.


NEW CONCEPTS INTRODUCED:

1. create_retriever_tool():
   Converts the Chroma retriever into a tool that the agent can call.
   The tool's name and description help the LLM decide when to use it.

2. bind_tools():
   Makes the available tools known to the LLM.
   The LLM can request a tool call instead of directly answering.

3. ToolNode:
   Executes tool calls requested by the LLM and returns their results.

4. tools_condition:
   Checks whether the agent requested a tool call.
   If yes -> retrieve.
   Otherwise -> END.

5. AgentState + add_messages:
   Stores the conversation history, including user messages,
   LLM responses, tool calls, and tool results.
   add_messages merges new messages into the existing history.

6. Streaming:
   - LLM streaming allows generated output to be received incrementally.
   - graph.stream(inputs) in this tutorial streams graph node updates.
   - These are different: node-update streaming does not automatically
     display individual generated tokens.


IMPORTANT LIMITATIONS:

- The agent can only retrieve from Chroma because no other tool is provided.
- The relevance grader evaluates retrieved documents, not generated answers.
- Unlike Self-RAG and Adaptive RAG, this graph does not separately check
  hallucinations or answer usefulness after generation.
- There is no explicit retry limit for repeated rewrite/retrieval cycles.


KEY TAKEAWAY:

The main difference is WHO controls retrieval.

CRAG        -> Retrieval is performed, then corrected if necessary.
Self-RAG    -> Retrieved evidence and generated answers are evaluated.
Adaptive RAG-> An LLM initially chooses the information source.
Agentic RAG -> The LLM decides whether to call a retrieval tool.

Agentic RAG introduces tool-calling decisions into the RAG workflow.
However, not every RAG system with an LLM router is necessarily agentic,
and not every agentic RAG system requires multiple tools or agents.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=get_embeddings(),
)
retriever = vectorstore.as_retriever()


from langchain_core.tools import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


### Edges
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = get_llm(temperature=0, streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = get_llm(temperature=0, streaming=True)
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = get_llm(temperature=0, streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    client = Client()
    prompt = client.pull_prompt("rag-from-scratch")

    # LLM
    llm = get_llm(temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# pretty print this to see...
print("*" * 20 + "Prompt[local_prompt_langsmith_saved]" + "*" * 20 + "\n")
# Removed broken print(prompt): the prompt exists only inside generate().
# Set up and run prompts.py first if the LangSmith prompt is not registered.

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()

# Visualize the compiled LangGraph workflow.
try:
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="agent_graph.png")
    print("Graph saved to agent_graph.png")

except (ImportError, OSError, RuntimeError) as e:
    print(f"Graph visualization failed: {e}")

import pprint

from langchain_core.messages import AIMessage, ToolMessage

inputs = {
    "messages": [
        ("user", "What does Lilian Weng say about the types of agent memory?"),
    ]
}

# Removed to improve readability of the inputs-outputs (YK)
# for output in graph.stream(inputs):
#     for key, value in output.items():
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint("---")
#         pprint.pprint(value, indent=2, width=80, depth=None)
#     pprint.pprint("\n---\n")

# Added later (YK) --> start
# Display the Agentic RAG workflow step by step.
# Show agent decisions, tool inputs/outputs, and generated answers.
# Avoid printing excessive metadata and complete retrieved documents.

print("\n" + "=" * 70)
print("USER QUESTION:", inputs["messages"][0][1])
print("=" * 70)

for output in graph.stream(inputs):
    for node, value in output.items():
        print(f"\n{'=' * 20} NODE: {node.upper()} {'=' * 20}")

        for message in value.get("messages", []):
            # AGENT: Did the LLM decide to call a tool?
            if node == "agent" and isinstance(message, AIMessage):
                if message.tool_calls:
                    print("AGENT DECISION: Call retrieval tool")

                    for tool_call in message.tool_calls:
                        print("TOOL NAME:", tool_call["name"])
                        print("TOOL INPUT:", tool_call["args"])

                else:
                    print("AGENT DECISION: Answer directly")
                    print("AGENT ANSWER:", message.content)

            # RETRIEVE: What information did the tool return?
            elif node == "retrieve" and isinstance(message, ToolMessage):
                content = str(message.content)
                print("TOOL EXECUTED:", message.name)
                print("RETRIEVED CONTENT LENGTH:", len(content), "characters")

                # Show only a preview of the retrieved evidence.
                clean_content = " ".join(content.split())
                print("RETRIEVED CONTENT PREVIEW:")
                print(clean_content[:500])

            # REWRITE: What improved question was generated?
            elif node == "rewrite":
                print("REWRITTEN QUESTION:", message.content)

            # GENERATE: Display the generated answer.
            elif node == "generate":
                answer = message.content if hasattr(message, "content") else message
                print("FINAL ANSWER:", answer)

        print("-" * 60)

# Added later (YK) --> end
