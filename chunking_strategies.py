# Adapted from Greg Kamradt, Full Stack Retrieval, "Text Splitting (Chunking)".
# Original tutorial (including Level 5) states MIT license on the website:
# https://community.fullstackretrieval.com/document-loaders/text-splitting
# Source: https://github.com/FullStackRetrieval-com/RetrievalTutorials
# Changes: local Qwen/Ollama adaptation, explanatory notes and/or two-pass review.
# Preserve attribution and the accompanying LICENSES/GREG_TUTORIAL_MIT.txt.

# PUBLIC CANDIDATE: only the original five-level / Level 5 portion is included.
# Bonus MultiVector, hierarchical, and graph-extraction demonstrations have
# been separated into REVIEW_ONLY pending exact provenance verification.

"""----------------------------------- 5 LEVELS OF TEXT  (CHUNKING) -----------------------------------"""

# Load environment variables from the .env file.
from dotenv import load_dotenv

load_dotenv()

from config import LOCAL_MODEL, get_embeddings, get_llm

"""----------------------------------- LEVEL 1 - CHARACTER SPLITTING -----------------------------------"""

# Example text for the chunking exercise
text = "This is the text I would like to chunk up. It is the example text for this exercise"

# Create a list that will hold your chunks
chunks = []

# Chunk size
chunk_size = 35  # Characters

# Manual character splitter example
# Run through the a range with the length of your text and iterate every chunk_size you want
for i in range(0, len(text), chunk_size):
    chunk = text[i : i + chunk_size]
    chunks.append(chunk)


print("\n----- Custom (No Library/Framework) Naive Iterative Chunking -----\n")
print(chunks)

# Langchain character splitter
from langchain_text_splitters import CharacterTextSplitter

# character spplitting with white space counted as a character (strip_whitespace=False)
text_splitter = CharacterTextSplitter(
    chunk_size=35, chunk_overlap=0, separator="", strip_whitespace=False
)

print("\n----- Using CharacterTextSplitter from LangChain -----\n")


# Create a document that contains the example text (already chunked using the specs given in the character splitter)
docs = text_splitter.create_documents([text])
print("\nchunk_size=35, chunk_overlap=0, separator=, strip_whitespace=False\n")
print(docs)

# Repeat the previous step using a chunk overlap of 4 characters
text_splitter = CharacterTextSplitter(chunk_size=35, chunk_overlap=4, separator="")
docs = text_splitter.create_documents([text])
print("\nchunk_size=35, chunk_overlap=4, separator=\n")
print(docs)

# Repeat the previous step using a separator = "ch"
text_splitter = CharacterTextSplitter(chunk_size=35, chunk_overlap=0, separator="ch")
docs = text_splitter.create_documents([text])
print("\nchunk_size=35, chunk_overlap=0, separator='ch'\n")
print(docs)

# Sentence splitter from llama Index
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Sentence splitter specs
splitter = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=15,
)

# Load the mit.txt inside data/PGEssays folder for chunking
documents = SimpleDirectoryReader(input_files=["data/PGEssays/mit.txt"]).load_data()

# In LlamaIndex, a Node is essentially a chunk of text wrapped in an object containing additional information,
# such as metadata, a unique ID, and relationships to other nodes.
nodes = splitter.get_nodes_from_documents(documents)

# Display the first node
print("\n----- Using SentenceSplitter from llamaIndex -----\n")
print("\nThe first node: \n")
print("\nnode contains info: metadata, a unique ID, and relationship to other nodes\n")
print(nodes[0])

"""----------------------------------- LEVEL 2 - RECURSIVE CHARACTER TEXT SPLITTING -----------------------------------"""

# Recursive text splitter - assumes text will be after {\n, \n\n, " ",...}
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Example text for chunking (original replacement for a third-party essay excerpt).
text = """
A local weather station records temperature every fifteen minutes.
The observations are saved with timestamps and checked for missing readings.

An engineer uses daily summaries to monitor changes in the station's behavior.
If a sensor stops responding, the monitoring system marks the gap instead of
pretending that the missing values were observed.

At the end of each week, the team compares trends across sensors and writes
a short report. The report separates confirmed measurements from hypotheses
so that readers can see the difference between evidence and interpretation.
"""

print("\n----- Using RecursiveTextSplitter from Langchain -----\n")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=65, chunk_overlap=0)
docs = text_splitter.create_documents([text])
print("\nchunk_size=65, chunk_overlap=0\n")
print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=0)
docs = text_splitter.create_documents([text])
print("\nchunk_size=450, chunk_overlap=0\n")
print(docs)

"""----------------------------------- LEVEL 3 - DOCUMENT SPECIFIC SPLITTING -----------------------------------"""

""" ### Markdown ### """

# Markdown splitter from Langchain
from langchain_text_splitters import MarkdownTextSplitter

splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)

# Example markdown text for chunking
markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""

docs = splitter.create_documents([markdown_text])
print("\nMarkdown document created using Langchain/Markdown text splitter:\n")
print(docs)

""" ### Python Code ### """

# Python text splitter
from langchain_text_splitters import PythonCodeTextSplitter

# Example python code/text
python_text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

for i in range(10):
    print (i)
"""

# Python text splitter
python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)
# Create docs with python text splitter
docs = python_splitter.create_documents([python_text])
print("\nPython document created using Langchain/Python text splitter:\n")
print(docs)

""" ### Javascipt Code ### """

# Language tells LangChain which programming language your code is written in,
# so it can choose appropriate separators when splitting the code into chunks.
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Javascript text example
javascript_text = """
// Function is called, the return value will end up in x
let x = myFunction(4, 3);

function myFunction(a, b) {
// Function returns the product of a and b
  return a * b;
}
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=65, chunk_overlap=0
)

docs = js_splitter.create_documents([javascript_text])
print("\nJavaScript document created using Langchain/RecursiveCharacterTextSplitter:\n")
print(docs)

""" ### PDF WITH TABLES ### """

import os

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

# Path to the pdf file
filename = "data/static/SalesforceFinancial.pdf"

# Extracts the elements from the PDF
elements = partition_pdf(
    filename=filename,
    # Unstructured Helpers
    strategy="hi_res",
    infer_table_structure=True,
    model_name="yolox",
)


# Print the table
print("\nElements of the table printed\n")
print(elements)
# Print the table as html
print("\nElements of the table printed as html\n")
print(elements[-4].metadata.text_as_html)


from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# Visual Instruction PDF example path
filepath = "data/static/VisualInstruction.pdf"

# Get elements
raw_pdf_elements = partition_pdf(
    filename=filepath,
    # Using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path="static/pdfImages/",
)

import base64
import io
import os

from langchain_core.messages import HumanMessage
from PIL import Image

llm = get_llm(temperature=0)


# Function to convert image to base64
def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode("utf-8")


# base64 image
image_str = image_to_base64("data/static/pdfImages/figure-15-6.jpg")

chat = get_llm(temperature=0)

# Human message
msg = chat.invoke(
    [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please give a summary of the image provided. Be descriptive",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_str}"},
                },
            ]
        )
    ]
)

# Printing LLM message
print("\nDescription of the image\n")
print(msg.content)

"""----------------------------------- LEVEL 4 - SEMANTIC CHUNKING -----------------------------------"""

# load the essay
with open("data/PGEssays/mit.txt", encoding="utf-8") as file:
    essay = file.read()

# Regex
import re

# Splitting the essay on '.', '?', and '!' - splitting on sentences instead of characters or words (assuming sentences end with [".", "?", "!"])
single_sentences_list = re.split(r"(?<=[.?!])\s+", essay)
print(f"{len(single_sentences_list)} senteneces were found")

# List of sentences and their indices from the essay (mit.txt)
# i is the index (sentence number in order)
# x is the sentence itself
sentences = [{"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)]
print("\n4th sentence in the essay after splitting\n")
print(sentences[:3])


"""
Function for combining sentences

The function behaves like a sliding window and combines sentences in a given text and saves it as a dictionary later these combined 
sentences/chunks can be used for similarity search during retrieval and possibly give better context to the llm for answering the questions

NOTE: An example and further explanations are given below to quickly see the function's output without running the code
"""


# =============================================================================
# EXAMPLE: Combining sentences with neighboring context (buffer_size=1)
# =============================================================================
#
# Suppose we have the following sentences:
#
# sentences = [
#     {"sentence": "I like apples."},                       # Index 0
#     {"sentence": "They are delicious."},                   # Index 1
#     {"sentence": "Python is a programming language."},    # Index 2
#     {"sentence": "It is widely used in AI."}               # Index 3
# ]
#
# BUFFER SIZE:
#   buffer_size=0 -> Current sentence only
#   buffer_size=1 -> Previous + current + next sentence
#   buffer_size=2 -> Two previous + current + two next sentences
#
# HOW buffer_size=1 WORKS:
#
# For each sentence (index i), the function combines:
#   1. One previous sentence (i-1), if available
#   2. The current sentence (i)
#   3. One next sentence (i+1), if available
#
# Example for i=1:
#
#   Previous (index 0): "I like apples."
#   Current  (index 1): "They are delicious."
#   Next     (index 2): "Python is a programming language."
#
#   Combined sentence:
#   "I like apples. They are delicious. Python is a programming language."
#
# RESULTS FOR ALL INDICES (buffer_size=1):
#
# i=0: "I like apples. They are delicious."
#
# i=1: "I like apples. They are delicious. Python is a programming language."
#
# i=2: "They are delicious. Python is a programming language. It is widely used in AI."
#
# i=3: "Python is a programming language. It is widely used in AI."
#
# NOTE:
# - First sentence has no previous neighbor.
# - Last sentence has no next neighbor.
# - The combined text is stored in sentences[i]["combined_sentence"].
# - The original "sentence" value is preserved.
#
# PURPOSE IN SEMANTIC CHUNKING:
# Instead of embedding isolated sentences, we embed each sentence with
# surrounding context to improve the representation of its meaning.
# Later, cosine distances between neighboring embeddings can help
# identify topic changes and determine semantic chunk boundaries.
#
# IMPORTANT: This function prepares contextualized sentences.
# It does NOT perform similarity calculations or create final chunks.
# =============================================================================
def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict (goes through the entire text until it reaches the end)
    # All the other loops are within this loop (for i...) so, the loops within (for j...)
    # will keep creating combined sentences until the sentences finish
    for i in range(len(sentences)):
        # Create an empty string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            # if j < 0 (e.g., j = -1), it could take the last sentence and combine it with the first sentence (we don't want this)
            # j < 0 happens if i-buffer_size < 0 which happens exactly at i=0 and buffer_size = 1
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


# combining sentences --> semantical chunking to provide more context to the llm during retrieval
sentences = combine_sentences(sentences)

print("\n3rd combined sentences (semantic chunks) within the mit.txt document\n")
print(sentences[:3])

# Using a local embeddings model to embed the combined sentences for similarity search/retrieval
embedding_model = get_embeddings()
embeddings = embedding_model.embed_documents(
    [x["combined_sentence"] for x in sentences]
)

# Adding the embedded combined sentences to the sentence dictionary
for i, sentence in enumerate(sentences):
    sentence["combined_sentence_embedding"] = embeddings[i]

# importing cosine similarity search for retrieval
from sklearn.metrics.pairwise import cosine_similarity


# Function to claculate cosine distances between current (i) and next (i+1) combined sentences
def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity between current and next combined sentences
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert cosine similary to cosine distance to figure out the relationship between chunks (combination of sentences)
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


# Acquire the combined sentences and the cosine distances between current (i) and next (i+1) combined sentences
distances, sentences = calculate_cosine_distances(sentences)

# The section below uses the calculated cosine distances and shows the first 3 calculated cosine distances
# between the combined sentences (1-->2-->3-->4)

# =================== INTERPRETATION OF COSINE DISTANCES (buffer_size=1) ===================
#
# This section displays the first 3 cosine distances between consecutive
# combined sentences (1st-2nd, 2nd-3rd, and 3rd-4th).
#
# HOW IT WORKS:
# - Cosine distance = 1 - cosine similarity.
# - Each distance compares the embeddings of combined sentences at i and i+1.
# - With buffer_size=1, neighboring combined sentences share overlapping text,
#   which generally increases their similarity (reduces their distance).
#
# RESULTS:
#
# 1st vs 2nd: Distance = 0.1803 (Similarity = 0.8197)
# - Largest difference among the three comparisons.
# - The second combined sentence introduces additional context about MIT,
#   graduation, and career choices.
#
# 2nd vs 3rd: Distance = 0.0278 (Similarity = 0.9722)
# - Highest similarity, partly due to shared context.
# - Both discuss startups and career choices.
#
# 3rd vs 4th: Distance = 0.0659 (Similarity = 0.9341)
# - The topic remains startups but shifts slightly toward their potential
#   popularity as an alternative to traditional career paths.
#
# PURPOSE IN SEMANTIC CHUNKING:
# - We look for unusually large cosine distances relative to the overall
#   distance distribution, rather than relying on fixed absolute values.
# - Large distances may indicate meaningful topic changes.
# - A percentile-based threshold (e.g., 95th percentile) identifies
#   candidate boundaries where the document can be split into chunks.
#
# CONCLUSION:
# - These three comparisons show relatively strong semantic continuity.
# - None can be classified as a chunk boundary without comparing its
#   distance against the threshold calculated from the full document.
# ==========================================================================================

print("-" * 70)
print(
    "\nFirst 3 calculated cosine distances between combined sentences (1-->2-->3-->4):\n"
)
print(distances[:3])

print("-" * 70)
print("\n1st and 2nd combined sentences in mit.txt and their cosine distance:\n")
print(
    "\n1st combined sentences:"
    + sentences[0]["combined_sentence"]
    + "\n\n2nd combined sentences:\n"
    + sentences[1]["combined_sentence"],
    "\n\ncosine distance = ",
    distances[0],
)
print("-" * 70)

print("-" * 70)
print("\n2nd and 3rd combined sentences in mit.txt and their cosine distance:\n")
print(
    "\n2nd combined sentences:"
    + sentences[1]["combined_sentence"]
    + "\n\n3rd combined sentences:\n"
    + sentences[2]["combined_sentence"],
    "\n\ncosine distance = ",
    distances[1],
)
print("-" * 70)

print("-" * 70)
print("\n3rd and 4th combined sentences in mit.txt and their cosine distance:\n")
print(
    "\n3rd combined sentences:"
    + sentences[2]["combined_sentence"]
    + "\n\n4th combined sentences:\n"
    + sentences[3]["combined_sentence"],
    "\n\ncosine distance = ",
    distances[2],
)
print("-" * 70)

import matplotlib.pyplot as plt
import numpy as np

# Plot the calculated cosine distances
plt.plot(distances)

# Assuming the maximum distance is 0.2, the upper bound is selected to be 0.2
y_upper_bound = 0.2
# plotting limits for y and x; x is the number of distances calculated
plt.ylim(0, y_upper_bound)
plt.xlim(0, len(distances))

# We need to get the distance threshold that we'll consider as outliers
# We'll use numpy.percentile() to achieve this goal.
# anything outside 95% percentile are outliers (5% are outliers) --> consider 5% as deviating from the semantic meaning of the chunk
# the 95th percentile is not necessarily a hard cut-off for the outliers --> it can be changed

# NOTE: The 95th-percentile method is a heuristic for semantic chunking
# and may not reliably identify meaningful topic boundaries in all cases.
#
# - It identifies unusually large cosine distances (top ~5%), which may
#   indicate significant changes in topic between neighboring sentences.
#
# - For example, an unrelated sentence within a paragraph may create
#   large semantic distances, making it a candidate for separation.
#
# - However, overlapping context (buffer_size=1) can reduce these
#   differences, and isolating a sentence requires boundaries on both sides.
#
# - The method can also create unnecessary splits in otherwise coherent
#   text because the percentile threshold is relative to the document.
breakpoint_percentile_threshold = 95
breakpoint_distance_threshold = np.percentile(
    distances, breakpoint_percentile_threshold
)

# If you want more chunks, lower the percentile cutoff (which will increase number of outliers and create more chunks)
plt.axhline(y=breakpoint_distance_threshold, color="r", linestyle="-")
# Then we'll see how many distances are actually above this one
num_distances_above_theshold = len(
    [x for x in distances if x > breakpoint_distance_threshold]
)  # The amount of distances above your threshold

# Display the estimated number of chunks on the plot.
# x: Position text near the left side of the graph.
# y: Position text near the bottom of the graph.
# s: Show the number of detected boundaries + 1,
#    since N boundaries produce N+1 chunks.
plt.text(
    x=(len(distances) * 0.01),
    y=y_upper_bound / 50,
    s=f"{num_distances_above_theshold + 1} Chunks",
)

# Then we'll get the index of the distances that are above the 95 percentile threshold.
# This will tell us where we should split our text
indices_above_thresh = [
    i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
]

# PURPOSE:
# Shade regions of the cosine distance graph between selected x-axis indices.
# These regions visually represent the semantic chunks.
colors = ["b", "g", "r", "c", "m", "y", "k"]
# EXAMPLE:
# indices_above_thresh = [3, 7]
# len(distances) = 11
#
# enumerate(indices_above_thresh) produces:
#   i=0, breakpoint_index=3
#   i=1, breakpoint_index=7
#
# CURRENT CODE BEHAVIOR:
#   i=0: start_index=0, end_index=3  --> shades x=0 to x=3
#   i=1: start_index=3, end_index=11 --> shades x=3 to x=11
#
# NOTE:
# - The first region always starts at 0 (if i=0, start_index = 0).
# - Subsequent regions start at the previous breakpoint (start_index = indices_above_thresh[i - 1]).
# - For non-final iterations, the region ends at the current breakpoint (end_index = breakpoint_index).
# - In the final iteration, the current code extends to len(distances). (end_index = len(distances)
# - The additional shading block below also shades x=7 to x=11, (last_breakpoint = indices_above_thresh[-1])
#   so the original code produces overlapping regions.
for i, breakpoint_index in enumerate(indices_above_thresh):
    # Determine the starting index
    if i == 0:
        # starting index = 0 at i=0 (always)
        start_index = 0
    else:
        start_index = indices_above_thresh[i - 1]

    # Determine the ending index
    if i < len(indices_above_thresh) - 1:
        end_index = breakpoint_index
    else:
        end_index = len(distances)

    # Shade the x-axis interval between start_index and end_index.
    plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
    # Place the chunk label at the midpoint of the shaded region.
    plt.text(
        x=np.average([start_index, end_index]),
        y=breakpoint_distance_threshold + (y_upper_bound) / 20,
        s=f"Chunk #{i}",
        horizontalalignment="center",
        rotation="vertical",
    )

# Additional step to shade from the last breakpoint to the end of the dataset
if indices_above_thresh:
    last_breakpoint = indices_above_thresh[-1]  # get the last_breakpoint
    # if last_breakpoint index is within the total number of distance indices
    # Shade the final chunk, from the last detected breakpoint
    # to the end of the cosine distance array.
    #
    # Example:
    # indices_above_thresh = [3, 7], len(distances) = 11
    # last_breakpoint = 7
    # plt.axvspan(7, 11) shades the final region (Chunk #2).
    #
    # facecolor: Select the next color, cycling through the color list.
    # alpha=0.25: Make the shaded region 25% opaque.
    if last_breakpoint < len(distances):
        plt.axvspan(
            last_breakpoint,
            len(distances),
            facecolor=colors[len(indices_above_thresh) % len(colors)],
            alpha=0.25,
        )
        # Put the Chunk# text at the specified x and y positions
        plt.text(
            x=np.average([last_breakpoint, len(distances)]),
            y=breakpoint_distance_threshold + (y_upper_bound) / 20,
            s=f"Chunk #{i + 1}",
            rotation="vertical",
        )
# Title and x & y labels of the plot
plt.title("PG Essay Chunks Based On Embedding Breakpoints")
plt.xlabel("Index of sentences in essay (Combined Sentence Position)")
plt.ylabel("Cosine distance between sequential combined sentences")
plt.show()

# Initialize the start index
start_index = 0

# Create a list to hold the grouped sentences
chunks = []

# Iterate through the breakpoints to slice the sentences.
# This section operates on the original sentences from the mit.txt essay.
# NOTE: THIS IS WHERE THE ACTUAL SEMANTIC CHUNKING TAKES PLACE.
# PURPOSE:
# - Previous steps used overlapping combined sentences (buffer_size=1)
#   to generate embeddings and calculate cosine distances.
# - Unusually large cosine distances (above the 95th-percentile threshold)
#   were identified as potential semantic chunk boundaries to split upon.
# - Since combined sentences contain overlapping text, we use the
#   original sentences to construct the final chunks without duplication.
# - Each chunk contains the original sentences up to and including
#   the detected breakpoint. The remaining sentences are handled afterward.
for index in indices_above_thresh:
    # The end index is the current breakpoint
    end_index = index

    # Slice the sentence_dicts from the current start index to the end index
    group = sentences[start_index : end_index + 1]
    combined_text = " ".join([d["sentence"] for d in group])
    chunks.append(combined_text)

    # Update the start index for the next group
    start_index = index + 1

# The last group, if any sentences remain, continue the semantic chunking process
# start_index is previously updated to be the last index at the calculated breakpoint (start_index = index + 1)
# sentences[start_index:] = any sentence after start_index all the way to the end of the sentences (mit.txt)
# sentences are separated using " " while being joined together and appended to the chunks list
if start_index < len(sentences):
    combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
    chunks.append(combined_text)

# Print the first and last 200 cahracters of the 2 saved chunks
# grouped_sentences now contains the chunked sentences
# strip() removes unwanted whitespace from the beginning and the end of the combined sentences
print("\nSemantic Chunking Results:\n")
for i, chunk in enumerate(chunks[:2]):
    # 200 characters of the chunk (sentence)
    buffer = 200

    print("\n" + "=" * 70 + f" Chunk #{i} " + "=" * 70)
    # chunk[:buffer] = first 200 characters of the chunk with whitespace from the beginning and the end stripped
    print("\nFirst 200 characters of the chunk (not full sentences)\n")
    print(chunk[:buffer].strip())
    print("...")
    # chunk[:buffer] = last 200 characters of the chunk with whitespace from the beginning and the end stripped
    print("\nLast 200 characters of the chunk (not full sentences)\n")
    print(chunk[-buffer:].strip())
    print("\n")

    print("\nFULL CHUNK - FOR DETAILED ANALYSIS\n")
    print(chunk[:].strip())

"""----------------------------------- LEVEL 5 - AGENTIC CHUNKING -----------------------------------"""

# Replaced the deprecated create_extraction_chain_pydantic() with
# LangChain's modern structured-output interface:
# llm.with_structured_output(..., method="json_schema")
#
# Reason: TypeError: Client.chat() got an unexpected keyword argument 'functions'
#
# - The original tutorial used OpenAI, whose older API supported
#   function calling through the 'functions' parameter.
# - The legacy LangChain extraction chain relies on that API.
# - Our local Qwen model runs through Ollama, whose Client.chat()
#   does not support the same 'functions' parameter.
# - We therefore use Ollama's supported JSON schema mode for
#   structured extraction instead. (YK)

# PURPOSE:
# Extract a structured chunk ID from the LLM's decision using a
# Pydantic schema. The agentic chunker uses this ID to determine
# which existing chunk should receive a new proposition.
# If no suitable chunk exists, the LLM can return None.
#
# Previously handled by create_extraction_chain_pydantic().
# Now handled by llm.with_structured_output(..., method="json_schema").
# from langchain_classic.chains import (
#     create_extraction_chain,
#     create_extraction_chain_pydantic,
# )

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

# Replaced the LangChain Hub prompt with a locally defined ChatPromptTemplate.
#
# REASON:
# - hub.pull("wfh/proposal-indexing") retrieves a prompt from LangSmith.
# - LangSmith uses region-specific endpoints (e.g., US and EU).
# - Pulling a prompt may fail if the configured endpoint does not match
#   the region where the prompt is hosted.
# - Defining the prompt locally avoids this external dependency and
#   allows the tutorial to run without retrieving prompts online. (YK)
#
# Original tutorial:
# from langchain import hub
# obj = hub.pull("wfh/proposal-indexing")


# Tutorial prompt was ignored and a custom system instructions were made for the LLM to follow during chunking
obj = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are an expert at converting text into clear, simple,
    self-contained propositions for retrieval and indexing.

    Follow these rules:

    1. Split compound sentences into simple propositions.
       Each proposition should express one distinct fact.

    2. Make every proposition understandable independently,
       without requiring surrounding sentences.

    3. Replace pronouns such as "he", "she", "they", and "it"
       with the actual entities they refer to when identifiable.

    4. Preserve important details, including dates, names,
       quantities, descriptions, and relationships.

    5. Preserve the original meaning and wording whenever possible.

    6. Do not invent facts or introduce unsupported information.

    7. Avoid duplicate propositions.
    """,
        ),
        ("human", "{input}"),
    ]
)


from langsmith import Client

client = Client()
prompt = client.pull_prompt(
    "wfh/proposal-indexing",
    dangerously_pull_public_prompt=True,
)

prompt.pretty_print()

# The llm with temperature = 0 (less likely to hallucinate)
llm = get_llm(temperature=0)

# use it in a runnable
runnable = obj | llm

# Removed (YK)
# Pydantic data class
# class Sentences(BaseModel):
#     sentences: list[str]


class Sentences(BaseModel):
    sentences: list[str] = Field(
        description="List of self-contained factual propositions"
    )


# Extraction - modified (YK)
# extraction_chain = create_extraction_chain_pydantic(
#     pydantic_schema=Sentences,
#     llm=llm.with_structured_output(Sentences, method="json_schema"),
# )

# the modified extraction chain --> output = structured JSON (YK)
extraction_chain = obj | llm.with_structured_output(Sentences, method="json_schema")


# def get_propositions(text):
#     runnable_output = runnable.invoke({"input": text}).content

#     propositions = extraction_chain.run(runnable_output)[0].sentences
#     return propositions


def get_propositions(text):
    result = extraction_chain.invoke({"input": text})
    return result.sentences


# Load the superlinear.txt essay
with open("data/PGEssays/superlinear.txt") as file:
    essay = file.read()

# Split on "\n\n" (double new lines = paragraphs)
paragraphs = essay.split("\n\n")
print("\nNumber of paragraphs in superlinear.txt\n")
print(len(paragraphs))

essay_propositions = []

# enumerate until the 5th paragraph
for i, para in enumerate(paragraphs[:5]):
    propositions = get_propositions(para)

    essay_propositions.extend(propositions)
    print(f"Done with {i}")


print(f"You have {len(essay_propositions)} propositions")
print(essay_propositions[:10])

# import the agentic chunker script (custom written - modified (YK))
from agentic_chunker import AgenticChunker

ac = AgenticChunker()

print(ac.add_propositions(essay_propositions))
print(ac.pretty_print_chunks())
chunks = ac.get_chunks(get_type="list_of_strings")
print(chunks)


