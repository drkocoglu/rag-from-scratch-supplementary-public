# Adapted from Greg Kamradt, Full Stack Retrieval, "Text Splitting (Chunking)".
# Original tutorial (including Level 5) states MIT license on the website:
# https://community.fullstackretrieval.com/document-loaders/text-splitting
# Source: https://github.com/FullStackRetrieval-com/RetrievalTutorials
# Changes: local Qwen/Ollama adaptation, explanatory notes and/or two-pass review.
# Preserve attribution and the accompanying LICENSES/GREG_TUTORIAL_MIT.txt.

# PUBLIC CANDIDATE: only the original five-level / Level 5 portion is included.
# Bonus MultiVector, hierarchical, and graph-extraction demonstrations have
# been separated into REVIEW_ONLY pending exact provenance verification.

"""----------------------------------- LEVEL 5 - AGENTIC CHUNKING -----------------------------------"""

"""
PURPOSE:
Use an LLM to group related propositions into semantically coherent
chunks based on their meaning rather than fixed text lengths.

HOW IT WORKS:
1. Extract individual propositions from the original document.
2. First pass (initial grouping):
   - Examine each proposition using a local Qwen LLM.
   - Add it to an existing chunk if it discusses a related idea.
   - Otherwise, create a new chunk with its own ID, title, and summary.
   - Update the summary and title when new propositions are added.
3. Second pass (semantic review - added in this adaptation):
   - Review chunks containing at least three propositions.
   - Identify distinct discussions that should be separated.
   - Split overly broad chunks into more focused groups.
   - Regenerate summaries and titles for the resulting chunks.
   - Validate that no propositions are lost or duplicated.
4. Return the final chunks with their propositions and metadata.

WHY USE IT?
Related information may appear in different parts of a document.
Agentic chunking can group these scattered ideas into coherent
discussions, potentially making them easier to organize and retrieve.

LIMITATIONS:
- Multiple LLM calls increase computational cost.
- Chunking quality depends on proposition extraction and LLM decisions.
- The second pass can split chunks but cannot merge separate chunks.
- Original paragraph locations are not preserved in this implementation.
- Better semantic grouping does not guarantee better RAG retrieval.

NOTE:
This implementation is adapted from Greg Kamradt's Agentic Chunking
tutorial using a local Qwen model instead of OpenAI.

The second-pass semantic review and its validation logic are
additional modifications made for this adaptation.

This section creates semantic chunks but does not implement
vector indexing or retrieval.
"""

# Import libraries, load environment variables, and set up logging to a file named "agentic_chunking_full.log".
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agentic_chunker_modified import AgenticChunker
from config import get_embeddings, get_llm


# Class to display long output in terminal and save it to a log file named "agentic_chunking_full.log".
# NOTE: Terminal output is saved to a log file for debugging because lengthy output may be truncated by the terminal.
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# Create a log file and redirect stdout and stderr to both the terminal and the log file.
log_file = open(
    "agentic_chunking_modified_full.log", "w", encoding="utf-8", buffering=1
)

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# load environment variables from .env file
load_dotenv()

# ===================================================== BEGINNING OF THE SCRIPT =====================================================


# OPTIONAL - Pull the prompt from LangSmith if registered in the US server - requires an US API key with Langsmith in the .env file.
# NOTE: "wfh/proposal-indexing" is a public prompt registered in the US server of LangSmith for this tutorial.

# Purpose: Demonstrate that a public prompt can be pulled from LangSmith in this manner - did not use this approach from the tutorial.
# NOTE: Always review external public prompts before using them in your own code as they may contain malicious instructions.

# NOTE: The system prompt in this tutorial has been modified significantly and no longer matches the public prompt.
# Replacing the modified system prompt with this public prompt (from the original tutiorial) may lead to different results.
# ------------------------------------------- BEGINNING OF OPTIONAL - PUBLIC PROMPT FROM LANGSMITH -----------------------------------

# from langsmith import Client

# client = Client()
# obj = client.pull_prompt(
#     "wfh/proposal-indexing",
#     dangerously_pull_public_prompt=True,
# )
# ------------------------------------------- END OF OPTIONAL - PUBLIC PROMPT FROM LANGSMITH ------------------------------------------


# Purpose: Create propositions from the content of a text document (e.g., an essay) using a local LLM.
# DEFINITION: Propositions are self-contained statements that express individual
# ideas, claims, facts, or relationships extracted from a text.
# They break complex information into smaller, more manageable units that can
# be grouped semantically, used for knowledge graph construction, or processed
# independently in tasks such as retrieval-augmented generation (RAG).

obj = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Extract clear, accurate, concise, and independently
            understandable propositions from the provided content.

            1. UNDERSTAND THE CONTENT

            Identify the main ideas, claims, and arguments expressed
            in the provided content.

            Understand how statements relate to one another,
            including whether they introduce an idea, explain it,
            provide evidence, illustrate it, or draw a conclusion.

            Use these relationships when extracting propositions.

            2. EXTRACT PROPOSITIONS

            Express each distinct fact, claim, or idea as a clear,
            self-contained proposition.

            - Preserve the original meaning and relevant context.

            - Resolve ambiguous pronouns and references by
            identifying the specific entities or concepts
            they refer to.

            - Preserve necessary conditions, causal relationships,
            and consequences.

            - Keep closely related information together when
            separating it would obscure its meaning.

            - Separate genuinely distinct facts when each can
            remain independently understandable.

            3. INTERPRET FIGURATIVE LANGUAGE

            When encountering metaphors, idioms, analogies,
            or figurative expressions:

            - Determine their intended meaning using the surrounding
            context and the document's broader argument.

            - Identify the specific idea or argument they illustrate,
            explain, compare, or support.

            - Explain the underlying relationship, mechanism,
            or comparison communicated by the expression,
            using only information supported by the content.

            - Translate figurative expressions into clear, literal
            propositions that preserve their intended meaning
            and connection to the relevant idea.

            - If the figurative expression explains or reinforces
            an existing claim, integrate its interpretation
            into that claim when no distinct information is lost.

            - Do not retain unexplained figurative expressions
            as standalone propositions.

            - Do not assume every figurative expression supports
            the main argument or invent unsupported interpretations.

            EXAMPLE: INTERPRETING A METAPHOR

            Example input text:
            '''
            The neighborhood cleanup began with five volunteers.
            Their visible progress encouraged more residents to join.
            As the group grew, it attracted even more volunteers.
            Organizers described this as a snowball effect.
            '''

            Expected JSON output:
            [
                "The neighborhood cleanup experienced a self-reinforcing growth pattern: visible progress encouraged more residents to volunteer, and the growing group attracted additional participants."
            ]

            END OF EXAMPLE

            4. CONSOLIDATE RELATED IDEAS AND REDUCE REDUNDANCY

            - Identify statements that repeat, overlap, or express
            different parts of the same underlying idea.

            - Combine such statements into one concise,
            independently understandable proposition whenever
            their complete meaning can be preserved.

            - When a general statement is followed by a more
            specific explanation of the same idea, integrate
            the explanation rather than producing repetitive
            propositions.

            - When multiple examples illustrate the same principle,
            consolidate them if their distinct details do not
            add meaningful information. Otherwise, preserve
            their differences.

            - Preserve meaningful qualifications, intentions,
            motivations, contrasts, exceptions, and conditions.

            - When combining related statements, incorporate
            these details into the resulting proposition
            rather than silently omitting them.

            - Keep genuinely distinct claims separate when
            combining them would remove meaningful information.

            - Produce the smallest set of propositions necessary
            to preserve all distinct information without
            changing its meaning.

            EXAMPLE: PRESERVING INTENTIONS AND QUALIFICATIONS

            Example input text:
            '''
            The museum staff recommended a new ticketing app.
            Their intention was to shorten entry lines.
            However, some visitors found the app confusing.
            '''

            Expected JSON output:
            [
                "Museum staff recommended a new ticketing app to shorten entry lines, although some visitors found the app confusing."
            ]

            END OF EXAMPLE

            5. MAINTAIN FAITHFULNESS

            - Use only information supported by the content.

            - Do not invent facts, interpretations, causal
            relationships, or connections.

            - Preserve relevant names, dates, quantities,
            and other specific details when provided.

            - Preserve the author's intended claims without
            treating opinions as established facts.

            - Each proposition must be understandable without
            access to the original document.

            6. OUTPUT FORMAT

            Extract propositions from the actual text supplied
            in the user message, not from the examples above.

            Return only a valid JSON array of strings.

            Each string must contain one independently
            understandable proposition.

            Do not include explanations, additional fields,
            or Markdown formatting.
            """,
        ),
        ("human", "Decompose the following:\n{input}"),
    ]
)

# llm for creating the propositions from the given text.
llm = get_llm(temperature=0, num_ctx=16384, reasoning=False)

# chain to create propositions
runnable = obj | llm


# Function to extract propositions from a given text
def get_propositions(text):
    # chain to create propositions
    runnable_output = runnable.invoke({"input": text}).content

    # Parse JSON directly instead of using a second LLM.
    propositions = JsonOutputParser().parse(runnable_output)

    # Ensure the result is a list of strings.
    if not isinstance(propositions, list) or not all(
        isinstance(p, str) for p in propositions
    ):
        raise ValueError("Expected a JSON list of strings.")

    return propositions


# Load the superlinear essay
with open("data/PGEssays/superlinear.txt") as file:
    essay = file.read()
# split the essay into paragraphs using double new lines
paragraphs = essay.split("\n\n")
# Display the number of paragraphs extracted from the essay
print("\nNumber of paragraphs:\n")
print(len(paragraphs))

# Empty list to save the propositions
essay_propositions = []

# extract propositions from the first 5 paragraphs of the superlinear essay
for i, para in enumerate(paragraphs[:5]):
    propositions = get_propositions(para)

    # add propositions to the essay_propositions list --> ["proposition1", "proposition2",...]
    essay_propositions.extend(propositions)
    print(f"Done with {i}")

# Display the number of propositions and the first 10 propositions
print(f"You have {len(essay_propositions)} propositions")
print("\nFirst 10 propositions:\n")
print(essay_propositions[:10])

# Uses the AgenticChunker class (custom script - agenticchunker_original.py) to separate the propositions into different groups
# based on their contextual relationship and similarity

# AgenticChunker instance
ac = AgenticChunker()
# Control whether the agentic chunker updates an existing chunk's title and summary whenever a new proposition is added to it (true = update).
ac.generate_new_metadata_ind = True

# 1st pass for grouping the propositions
ac.add_propositions(essay_propositions)
# 2nd pass for splitting larger chunks in case the first pass is not sufficient enough
ac.review_chunks()
# print the chunks for review as they appear during the process
ac.pretty_print_chunks()
# get the chunks as list of strings
chunks = ac.get_chunks(get_type="list_of_strings")
# print the full chunks
print(chunks)


