# Adapted from Greg Kamradt, Full Stack Retrieval, "Text Splitting (Chunking)".
# Original tutorial (including Level 5) states MIT license on the website:
# https://community.fullstackretrieval.com/document-loaders/text-splitting
# Source: https://github.com/FullStackRetrieval-com/RetrievalTutorials
# Changes: local Qwen/Ollama adaptation, explanatory notes and/or two-pass review.
# Preserve attribution and the accompanying LICENSES/GREG_TUTORIAL_MIT.txt.

import uuid

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from config import get_llm

load_dotenv()


"""
================================================================
AGENTIC CHUNKER - COMPLETE EXECUTION FLOW
================================================================

PURPOSE:
Group related propositions into semantically coherent chunks
using a two-pass LLM-based approach.

Each chunk stores:
  - chunk_id
  - propositions (list of strings)
  - title
  - summary
  - chunk_index


STEP 1: INITIALIZE
------------------
ac = AgenticChunker()
      |
      v
__init__()
      |
      |-- self.chunks = {}
      |      Starts with an empty dictionary.
      |
      |-- self.llm = ...
      |      Initializes the LLM used for chunk decisions,
      |      summaries, and titles.
      |
      |-- self.generate_new_metadata_ind
             Controls whether existing chunk titles and
             summaries are regenerated when propositions
             are added during the first pass.


STEP 2: FIRST PASS - INITIAL GROUPING
-------------------------------------
ac.add_propositions(propositions)
      |
      v
add_propositions()
      |
      |-- Loops through each proposition.
      |
      |-- Calls add_proposition(proposition).
                  |
                  v
            add_proposition()
                  |
                  |-- Are there existing chunks?
                  |
                  |-- NO --> _create_new_chunk()
                  |                 |
                  |                 |-- Generate a chunk ID.
                  |                 |
                  |                 |-- _get_new_chunk_summary()
                  |                 |     Generate summary.
                  |                 |
                  |                 |-- _get_new_chunk_title()
                  |                 |     Generate title.
                  |                 |
                  |                 |-- Save new chunk.
                  |
                  |-- YES --> _find_relevant_chunk()
                                  |
                                  v
                           get_chunk_outline()
                                  |
                                  |-- Collect existing chunk
                                  |   IDs and full proposition lists.
                                  |
                                  v
                           STRUCTURED LLM CALL
                                  |
                                  |-- Compare the proposition
                                  |   against existing chunks.
                                  |
                                  |-- Use a Pydantic schema:
                                  |
                                  |   ChunkDecision(BaseModel)
                                  |       chunk_id: str | None
                                  |
                                  |-- Return a structured
                                  |   chunk_id directly.
                                  |
                                  |-- None indicates that
                                  |   no existing chunk fits.
                                  |
                                  v
                           VALIDATE CHUNK ID
                                  |
                                  |-- Valid existing ID?
                                  |      |
                                  |      YES --> Return ID
                                  |      |
                                  |      No --> Is chunk_id None? 
                                                        |
                                                        YES --> Return None
                                                        |
                                                        NO --> Raise ValueError (invalid Chunk ID)
                                    
                                  |
                                  v
                         BACK TO add_proposition()
                                  |
                   +--------------+--------------+
                   |                             |
                ID found                     No ID found
                   |                             |
                   v                             v
         add_proposition_to_chunk()      _create_new_chunk()
                   |                             |
                   |-- Append proposition.       |-- Generate
                   |                             |   ID, summary,
                   |-- If metadata updates       |   and title.
                   |   are enabled:              |
                   |                             |-- Save chunk.
                   |   _update_chunk_summary()   |
                   |   _update_chunk_title()     |
                   |                             |
                   +--------------+--------------+
                                  |
                                  v
                       PROCESS NEXT PROPOSITION

                 Repeat until all are processed.


STEP 3: SECOND PASS - REVIEW AND SPLITTING
------------------------------------------
ac.review_chunks()
      |
      v
review_chunks()
      |
      |-- Copy existing chunks for review.
      |
      |-- Review each original chunk separately.
                  |
                  v
           COUNT PROPOSITIONS
                  |
                  |-- Fewer than 3?
                  |       |
                  |       YES --> Keep chunk unchanged.
                  |       |
                  |       NO  --> Continue.
                  |
                  v
           PREPARE NUMBERED PROPOSITIONS
                  |
                  |-- Assign zero-based indices.
                  |
                  |-- Example:
                  |      0: Proposition A
                  |      1: Proposition B
                  |      2: Proposition C
                  |
                  v
           STRUCTURED LLM REVIEW
                  |
                  |-- Use a copy of the LLM with
                  |   reasoning enabled.
                  |
                  |-- Review all propositions.
                  |
                  |-- Identify distinct semantic
                  |   groups, examples, subtopics,
                  |   and causal chains.
                  |
                  |-- Return:
                  |
                  |   ChunkReview(BaseModel)
                  |       groups: list[list[int]]
                  |
                  |-- Example:
                  |       [[0, 1], [2, 3, 4]]
                  |
                  v
           VALIDATE RESPONSE
                  |
                  |-- Parsing successful?
                  |
                  |-- Groups nonempty?
                  |
                  |-- Every proposition index
                  |   appears exactly once?
                  |
                  |-- No invalid indices?
                  |
                  +---------------+---------------+
                  |                               |
               INVALID                          VALID
                  |                               |
                  v                               v
          Keep original chunk             CHECK GROUP COUNT
                                                  |
                                     +------------+------------+
                                     |                         |
                                  1 group                  2+ groups
                                     |                         |
                                     v                         v
                            Keep original chunk       SPLIT CHUNK
                                                               |
                                                               |-- Order groups
                                                               |   by their first
                                                               |   original index.
                                                               |
                                                               |-- Reconstruct
                                                               |   proposition lists
                                                               |   without rewriting.
                                                               |
                                                               |-- Retain original
                                                               |   chunk ID for
                                                               |   the first group.
                                                               |
                                                               |-- Generate new IDs
                                                               |   for other groups.
                                                               |
                                                               |-- Generate new
                                                               |   summaries and
                                                               |   titles for all
                                                               |   resulting groups.
                                                               |
                                                               v
                                                        SAVE NEW CHUNKS
                  |
                  v
           AFTER ALL CHUNKS ARE REVIEWED
                  |
                  |-- Replace self.chunks with
                  |   the updated chunk dictionary.
                  |
                  |-- Rebuild chunk_index.
                  |
                  v
           SECOND PASS COMPLETE


IMPORTANT SECOND-PASS LIMITATIONS:
----------------------------------
- The extracted propositions are split into chunks, without referencing the original text (same as the original tutorial).
- The second pass can SPLIT chunks but cannot MERGE them.
- It cannot repair incorrect proposition extraction.
- Chunks containing fewer than 3 propositions are skipped.
- Invalid or unsuccessful reviews preserve the original chunk.
- Original proposition text is preserved during splitting.
- Newly split chunks receive regenerated summaries and titles.
- Unchanged chunks retain their existing metadata.


STEP 4: INSPECT RESULTS
-----------------------
ac.pretty_print_chunks()
      |
      |-- Prints each chunk's ID, summary,
      |   and individual propositions.

ac.pretty_print_chunk_outline()
      |
      |-- Calls get_chunk_outline().
      |-- Prints IDs and propositions.

ac.get_chunks(get_type="dict")
      |
      |-- Returns the full nested dictionary.

ac.get_chunks(get_type="list_of_strings")
      |
      |-- Joins propositions within each chunk.
      |-- Returns one string per chunk.


================================================================
OVERALL PIPELINE
================================================================

Text document
      |
      v
Proposition extraction (separate LLM-based extraction)
      |
      v
List of propositions
      |
      v
FIRST PASS:
Incremental LLM-based grouping
      |
      v
Initial chunks with titles and summaries
      |
      v
SECOND PASS:
LLM-based semantic review and splitting
      |
      v
Final chunks with updated metadata
      |
      v
Inspection / downstream retrieval


NOTE:
The second pass improves chunk granularity but does not
guarantee semantic correctness. Retrieval quality must be
evaluated separately.

================================================================
"""


# 1st pass - ChunkDecision class that outputs the ID number of a chunk as a decision for finding relevant chunks for the given proposition
class ChunkDecision(BaseModel):
    # chunk id to be output in strings
    chunk_id: str | None = None


# 2nd pass - Class to review the created chunks in the 1st pass - uses an llm to review the grouped chunks
class ChunkReview(BaseModel):
    groups: list[list[int]]


# An agentic chunking class that groups propositions based on their semantic and contextual relationships.
class AgenticChunker:
    # initialize chunk dictionary, id limit, control for logs, summary, title updates and llm specs
    def __init__(self):
        # empty dict for saving chunks
        self.chunks = {}
        # Use the first 5 hexadecimal characters of a UUID as the chunk ID.
        self.id_truncate_limit = 5
        # Whether or not to update/refine summaries and titles new propositions are added to the existing chunks
        self.generate_new_metadata_ind = True
        # log print = True
        self.print_logging = True
        # llm instance
        self.llm = get_llm(temperature=0, num_ctx=16384, reasoning=False)

    # NOTE: Main instance method for the initial grouping of propositions.
    # Calls other methods to find or create an appropriate chunk for each proposition.
    # Usage: ac.add_propositions(my_propositions)
    # Input --> my_propositions = [proposition1, proposition2, ...]
    # Result --> self.chunks --> contains the grouped propositions.
    def add_propositions(self, propositions):
        for proposition in propositions:
            # NOTE: add_proposition() is singular (no "s"), unlike add_propositions().
            # this function finds an appropriate chunk or creates a new one.
            self.add_proposition(proposition)

    # this function finds an appropriate chunk or creates a new one for the given proposition.
    def add_proposition(self, proposition):
        # If self.print_logging is True (default), print the proposition.
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")
        # If no chunks exist yet, create the first chunk without searching.
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            # create new chunk for the given proposition
            self._create_new_chunk(proposition)
            # returns "None" to exit the method early because the proposition has already
            # been added to a new chunk. No further processing is needed.
            return

        # Existing chunks are available; find a suitable chunk for the new proposition.
        chunk_id = self._find_relevant_chunk(proposition)
        # If a chunk was found then add the proposition to it
        if chunk_id:
            if self.print_logging:
                print(
                    f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}"
                )
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        # Append the proposition to the selected chunk's list.
        self.chunks[chunk_id]["propositions"].append(proposition)

        # If enabled, regenerate the summary and title using the updated chunk.
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]["summary"] = self._update_chunk_summary(
                self.chunks[chunk_id]
            )
            self.chunks[chunk_id]["title"] = self._update_chunk_title(
                self.chunks[chunk_id]
            )

    # function to update chunk summary when a new proposition is added to a chunk
    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        # instructions to the LLM for creating an updated 1 sentence chunk summaries by passing the new chunks with added
        # propositions to the LLM
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.
                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".
                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                (
                    "user",
                    "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}",
                ),
            ]
        )

        # chain for creating an updated new chunk summary
        runnable = PROMPT | self.llm
        new_chunk_summary = runnable.invoke(
            {
                "proposition": "\n".join(chunk["propositions"]),
                "current_summary": chunk["summary"],
            }
        ).content
        return new_chunk_summary

    # function to update chunk title when a new proposition is added to a chunk
    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        # Ask the LLM to update the short title using the propositions, summary, and previous title.
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.
                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".
                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                (
                    "user",
                    "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}",
                ),
            ]
        )

        # chain for creating an updated new chunk title
        runnable = PROMPT | self.llm
        updated_chunk_title = runnable.invoke(
            {
                "proposition": "\n".join(chunk["propositions"]),
                "current_summary": chunk["summary"],
                "current_title": chunk["title"],
            }
        ).content

        return updated_chunk_title

    # function to create a 1 sentence chunk summary for a given chunk
    def _get_new_chunk_summary(self, proposition):
        # instructions to the LLM for creating 1 sentence chunk summaries by passing the propositions to the LLM
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.
                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".
                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                (
                    "user",
                    "Determine the summary of the new chunk that this proposition will go into:\n{proposition}",
                ),
            ]
        )

        # chain for chunk summary generation
        runnable = PROMPT | self.llm
        new_chunk_summary = runnable.invoke({"proposition": proposition}).content
        return new_chunk_summary

    # function to create a new chunk title using the chunk summaries created by the LLM
    def _get_new_chunk_title(self, summary):

        # Instructions to the LLM to create chunk titles using the chunk summaries that the llm already created
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.
                    A good chunk title is brief but encompasses what the chunk is about.

                    You will be given a summary of a chunk which needs a title.

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".
                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                (
                    "user",
                    "Determine the title of the chunk that this summary belongs to:\n{summary}",
                ),
            ]
        )

        # chain for creating a new chunk title
        runnable = PROMPT | self.llm
        new_chunk_title = runnable.invoke({"summary": summary}).content

        return new_chunk_title

    # function to create a new chunk if no chunks exist and organize the chunk with a chunk ID, title, summary, and index
    def _create_new_chunk(self, proposition):
        # avoiding long ids by truncating the uuid to the id_truncate_limit (5) which is set in the constructor
        # Generate a short, unused chunk ID.
        new_chunk_id = str(uuid.uuid4())[: self.id_truncate_limit]
        while new_chunk_id in self.chunks:
            new_chunk_id = str(uuid.uuid4())[: self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)
        self.chunks[new_chunk_id] = {
            "chunk_id": new_chunk_id,
            "propositions": [proposition],
            "title": new_chunk_title,
            "summary": new_chunk_summary,
            "chunk_index": len(self.chunks),
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    # Build a string containing each chunk's ID and propositions.
    # Used by the first-pass LLM to compare new propositions with existing chunks.
    # Returns an empty string if no chunks exist.
    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        # initlize chunk outline
        chunk_outline = ""
        # get chunk ID and propositions (after chunking)
        for chunk_id, chunk in self.chunks.items():
            # NOTE: No longer includes chunk summaries and titles:
            # the llm instead uses the full propositions which leads to more accurate chunking in the 1st pass.
            single_chunk_string = (
                f"Chunk ID: {chunk['chunk_id']}\n"
                "Chunk Propositions:\n"
                + "\n".join(f"- {p}" for p in chunk["propositions"])
                + "\n\n"
            )
            # add strings that represent your chunks to chunk outline (after chunking)
            chunk_outline += single_chunk_string

        return chunk_outline

    # function to find a relevant existing chunk for a given proposition
    def _find_relevant_chunk(self, proposition):
        # get the current chunk outline to pass to the LLM for determining if a proposition should belong to any of the existing chunks
        current_chunk_outline = self.get_chunk_outline()

        # instructions to the LLM for determining if a proposition should belong to any of the existing chunks
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are organizing propositions into chunks based on their meaning.

                        Your task is to determine whether a new proposition belongs to
                        an existing chunk or requires a new one.

                        Follow these steps:

                        1. Identify the specific idea, argument, event,
                        or example expressed by the new proposition.

                        2. Examine the actual propositions within each
                        existing chunk to identify what they discuss.

                        3. Compare their underlying meanings and relationships,
                        not just their wording or general subject.

                        4. Decide whether the new proposition belongs to the
                        same specific idea, example, event, or causal chain
                        as an existing chunk. Sharing a broader argument
                        is not sufficient.

                        - Keep propositions that describe the same specific
                            example, event, or causal chain together.

                        - Causes, explanations, outcomes, and consequences
                            of the same situation belong together, even when
                            they discuss different immediate subjects.

                        - When a proposition introduces a new concrete
                            example, create a new chunk rather than attaching
                            it to an existing general discussion, unless
                            that chunk already covers the same example.

                        - Distinct examples or arguments should form
                            separate chunks when they can be understood
                            independently.

                        - Do not assign a proposition to an existing chunk
                            solely because both discuss the same broad topic.

                        - Recognize complementary factual information that
                            describes different aspects of the same subject,
                            entity, event, or shared context.

                        - Group related factual details such as dates, times,
                        locations, identities, and attributes when they
                        collectively describe one coherent subject or event.

                        - Treat separate components of the same date or time,
                        such as the month, year, day, or hour, as related
                        information rather than distinct topics.

                        - Group factual details with an existing chunk when they
                        describe the same specific entity, event, example,
                        or causal relationship.

                        - When a proposition introduces a new, self-contained
                        example illustrating a broader argument, create a
                        separate chunk for that example, even if it supports
                        the general argument. Keep subsequent details,
                        explanations, and consequences of that example together.

                        - Do not group unrelated facts merely because they
                        share a category such as dates, locations, or people.

                        - Create separate chunks when facts concern genuinely
                        different subjects, events, or discussions that do
                        not share a meaningful contextual relationship.

                        Rules:
                        - Similar words do not necessarily mean the same topic.
                        - Sharing a broad theme is not enough to belong together.
                        - Different wording can express the same idea.
                        - Different subtopics should form separate chunks.
                        - Never invent a chunk ID.
                        - If no suitable chunk exists, set chunk_id to null.
                        - When selecting an existing chunk, copy its chunk ID
                            exactly as shown in the current chunk outline.
                        - Never modify, abbreviate, or guess a chunk ID.
                        - When a proposition introduces a distinct belief, assumption,
                            or viewpoint, consider whether it deserves its own discussion
                            rather than merging it into a broader explanation.
                        - Quotations, paraphrases, and rebuttals expressing or evaluating
                            that same viewpoint should stay together, not form a separate
                            discussion merely because they are quotations.

                        Five examples are provided below.

                        Example 1 (same specific discussion):

                        New proposition:
                        "The train arrived thirty minutes late because of a signal failure."

                        Existing chunk:
                            Chunk ID: abc12
                            Chunk Propositions:
                            - "A signaling malfunction disrupted train services on the northern route."
                            - "Technicians were dispatched to repair the faulty signaling equipment."

                        Output:
                        {{"chunk_id": "abc12"}}


                        Example 2 (unrelated discussions):

                        New proposition:
                        "The museum acquired three medieval manuscripts."

                        Existing chunk:
                            Chunk ID: def34
                            Chunk Propositions:
                            - "Strong ocean currents contribute to coastal erosion."
                            - "Coastal erosion threatens buildings near the shoreline."

                        Output:
                        {{"chunk_id": null}}


                        Example 3 (same broad theme, distinct subtopics):

                        New proposition:
                        "A violin requires periodic string replacement."

                        Existing chunk:
                            Chunk ID: xyz90
                            Chunk Propositions:
                            - "Violin bows require regular cleaning to remove rosin buildup."
                            - "Damaged bow hair should be replaced to maintain playing quality."

                        Output:
                        {{"chunk_id": null}}

                        Example 4 (complementary factual information):

                        New proposition:
                        "The exhibition is scheduled for September 2031."

                        Existing chunk:
                            Chunk ID: ghi56
                            Chunk Propositions:
                            - "The exhibition will take place at the city museum."
                            - "The exhibition will feature historical photographs."

                        Explanation: These propositions provide complementary information
                        about the same exhibition, including its location,
                        content, and scheduled date.

                        Output:
                        {{"chunk_id": "ghi56"}}

                        Example 5 (concrete example vs. general principle):

                        New proposition:
                        "A delivery van was delayed when a bridge closed
                        during flooding."

                        Existing chunk:
                        Chunk ID: jkl78
                        Chunk Propositions:
                        - "Heavy rainfall can disrupt transportation networks."
                        - "Flooding can increase travel times."

                        Explanation:
                        The new proposition describes a distinct, concrete
                        incident. Although it illustrates the general principle,
                        it should start its own chunk.

                        Output: {{"chunk_id": null}}

                        Return only the structured result containing chunk_id.
                    """,
                ),
                (
                    "user",
                    "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--",
                ),
                (
                    "user",
                    "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}",
                ),
            ]
        )

        # The LLM compares the new proposition with existing chunks and returns
        # the ID of the most relevant chunk, or null if no suitable chunk exists.
        # Structured output enforces the expected JSON response format.
        # NOTE: include_raw=True returns a dictionary containing the raw LLM response,
        # the parsed ChunkDecision object, and any parsing error.
        structured_llm = self.llm.with_structured_output(
            ChunkDecision,
            method="json_schema",
            include_raw=True,
        )
        # chain for finding relevant chunks given the proposition
        # output: chunk id in json format
        runnable = PROMPT | structured_llm

        # Run the LLM to select an appropriate chunk for the given proposition.
        # NOTE: Direct comparison with an llm allows finding relevant chunks for the given proposition
        # proposition --> current proposition
        # chunk_outline --> all current chunk IDs + propositions within the chunks (not including the current proposition)
        response = runnable.invoke(
            {"proposition": proposition, "current_chunk_outline": current_chunk_outline}
        )

        # raw response + number input-output tokens (including reasoning if enabled)
        raw = response["raw"]
        prompt_tokens = raw.response_metadata.get("prompt_eval_count")
        output_tokens = raw.response_metadata.get("eval_count")

        # Printing input-output tokens and context limit to keep track of context window and token generation constraints
        print(f"Input tokens: {prompt_tokens}")
        print(f"Output tokens: {output_tokens}")
        print("Configured context limit: 16384")
        result = response["parsed"]

        print("\nQwen reasoning:")
        print(raw.additional_kwargs.get("reasoning_content", "No reasoning returned"))

        if response["parsing_error"]:
            raise ValueError(response["parsing_error"])

        # Raise an error if the LLM returned no parsed decision object.
        if result is None:
            raise ValueError("Qwen returned no parsed chunk decision.")

        if self.print_logging:
            print("Selected chunk ID:", result.chunk_id)
            print("Structured LLM response:", result)

        chunk_found = result.chunk_id

        # No suitable existing chunk was selected.
        if chunk_found is None:
            return None

        # An invented or mistyped ID is an error, not a new-chunk decision.
        if chunk_found not in self.chunks:
            raise ValueError(
                f"Invalid chunk ID: {chunk_found!r}. "
                f"Valid chunk IDs: {list(self.chunks.keys())}"
            )

        return chunk_found

    # Second pass: review chunks created during the first pass.
    # Uses a reasoning-enabled LLM to split chunks containing distinct discussions.
    # Preserves the original propositions and never merges separate chunks.
    # Usage: ac.review_chunks() after ac.add_propositions().
    def review_chunks(self):
        """Second pass: split chunks containing distinct discussions."""

        # Snapshot the original chunks and store reviewed chunks separately.
        original_chunks = list(self.chunks.items())
        updated_chunks = {}

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are reviewing an existing semantic chunk to determine
            whether its propositions should remain together or be
            divided into multiple coherent chunks.

            Examine ALL numbered propositions before making a decision.

            Grouping rules:
            - Identify the main discussion and any distinct subtopics,
              examples, scenarios, or causal chains.

            - Keep propositions describing the same specific idea,
              example, scenario, or causal chain together, even when
              they are not consecutive.

            - When several propositions develop a distinct, self-contained
              example or causal chain, consider separating them from the
              broader discussion they illustrate.

            - Propositions can share an overarching topic while still
              belonging to different semantic groups.

            - Do not split propositions merely because they mention
              different words, domains, or applications of the same idea.

            - Avoid unnecessary splitting. A single group is appropriate
              when all propositions form one coherent discussion.

            Output requirements:
            - Determine the groups from the actual propositions provided.
            - Use only the original zero-based proposition indices.
            - Every index must appear exactly once across all groups.
            - Do not invent, omit, or duplicate indices.
            - Do not rewrite the propositions.
            - Return groups as lists of proposition indices.
            """,
                ),
                ("human", "Review these numbered propositions:\n{propositions}"),
            ]
        )

        # Enable reasoning for the second pass without changing the first-pass LLM.
        review_llm = self.llm.model_copy(update={"reasoning": True})

        chain = prompt | review_llm.with_structured_output(
            ChunkReview,
            method="json_schema",
            include_raw=True,
        )

        for chunk_id, chunk in original_chunks:
            propositions = chunk["propositions"]

            # Small chunks do not need this splitting review (personal decision that could lead to less accurate results).
            if len(propositions) < 3:
                updated_chunks[chunk_id] = chunk
                continue

            # Assign zero-based indices so Qwen can return proposition groups as lists of integers.
            numbered = "\n".join(f"{i}: {p}" for i, p in enumerate(propositions))

            try:
                # Ask Qwen to group related propositions by their indices.
                response = chain.invoke({"propositions": numbered})

                # Check for structured-output parsing errors.
                if response["parsing_error"] is not None:
                    raise ValueError(response["parsing_error"])

                result = response["parsed"]

                if result is None:
                    raise ValueError("No parsed response returned.")

                # Example: [[0, 1, 2], [3, 4, 5]]
                groups = result.groups
                print("Second-pass Qwen groups:", groups)

                # Flatten the groups to verify that every original index
                # appears exactly once (no missing or duplicate propositions).
                indices = [i for group in groups for i in group]

                if (
                    not groups
                    or any(not group for group in groups)
                    or sorted(indices) != list(range(len(propositions)))
                ):
                    print(f"Invalid review for chunk {chunk_id}; keeping original.")
                    updated_chunks[chunk_id] = chunk
                    continue

            except Exception as e:
                # Preserve the original chunk if the LLM or validation fails.
                print(f"Review failed for chunk {chunk_id}: {e}")
                updated_chunks[chunk_id] = chunk
                continue

            # If Qwen returns only one group, keep the chunk unchanged.
            if len(groups) == 1:
                updated_chunks[chunk_id] = chunk
                continue

            # Sort groups by the earliest original proposition index.
            groups = sorted(groups, key=min)

            print(f"Second pass: splitting chunk {chunk_id} into {len(groups)}.")

            for group_number, group in enumerate(groups):
                # Restore the original proposition order within each group.
                new_propositions = [propositions[i] for i in sorted(group)]

                # Preserve the original ID for the first group.
                if group_number == 0:
                    new_id = chunk_id
                else:
                    new_id = str(uuid.uuid4())[: self.id_truncate_limit]
                    while new_id in self.chunks or new_id in updated_chunks:
                        new_id = str(uuid.uuid4())[: self.id_truncate_limit]

                # Generate metadata from the complete revised group.
                summary = self._get_new_chunk_summary(" ".join(new_propositions))
                title = self._get_new_chunk_title(summary)

                updated_chunks[new_id] = {
                    "chunk_id": new_id,
                    "propositions": new_propositions,
                    "summary": summary,
                    "title": title,
                    "chunk_index": 0,
                }

        # Replace the original grouping with the reviewed grouping.
        self.chunks = updated_chunks

        # Reassign sequential chunk indices after reviewing and splitting.
        for index, chunk in enumerate(self.chunks.values()):
            chunk["chunk_index"] = index

        print(
            f"Second pass complete: {len(original_chunks)} -> {len(self.chunks)} chunks"
        )

    # function to get chunks in dictionary or list format
    def get_chunks(self, get_type="dict"):
        """
        Return chunks in the requested format:
        - "dict": the original nested dictionary.
        - "list_of_strings": one string per chunk, joining its propositions.
        """
        if get_type == "dict":
            return self.chunks
        if get_type == "list_of_strings":
            chunks = []
            # chunk_id is the key and chunk is the value within chunks dictionary
            # the value in this case is another dictionary due to self.chunks being a dictionary within a dictionary = {key: {key: value}}
            for chunk_id, chunk in self.chunks.items():
                # mutliple related propositions that were grouped are now separated using a white space and appended to the chunks list
                # Example:
                # self.chunks = {
                #     "abc12": {
                #         "chunk_id": "abc12",
                #         "propositions": [
                #             "Greg likes pizza.",
                #             "Greg likes hamburgers."
                #         ],
                #         "title": "Food Greg Likes",
                #         "summary": "Information about Greg's food preferences.",
                #         "chunk_index": 0
                #     }
                # }
                #
                # NOTE:
                # chunk_id --> key, chunk --> value (key,value pair in a dictionary).
                # "abc12" is the outer dictionary key (chunk_id).
                # Its value (chunk) is another dictionary containing the chunk's information (chunk_id, propositions, title, etc.)
                #
                # Joining its propositions with " ".join([x for x in chunk["propositions"]]) produces:
                # "Greg likes pizza. Greg likes hamburgers." --> " " in between each proposition when joining.
                chunks.append(" ".join([x for x in chunk["propositions"]]))
            return chunks

    # function to print chunks in an organized manner
    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            # print each proposition for the specific chunk
            for prop in chunk["propositions"]:
                print(f"    -{prop}")
            print("\n\n")

    # Print the chunk outline containing IDs and full propositions.
    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())


# Testing Agentic Chunker class before using it in the tutorial --> Run this script to see the results and go through the code as you desire
if __name__ == "__main__":
    # instance of AgenticChunker class
    ac = AgenticChunker()
    ## Comment and uncomment the propositions to your hearts content to test AgenticChunker's accuracy and capabilities with the local llm
    # Original, self-contained test propositions replace third-party essay paraphrases.
    propositions = [
        'The current month is October.',
        'The current year is 2023.',
        'The weather station records a new temperature every fifteen minutes.',
        'Each measurement is stored with a timestamp.',
        'The monitoring dashboard displays daily temperature averages.',
        'Missing sensor readings are marked as unavailable.',
        'A faulty sensor can produce unrealistic values.',
        'The maintenance team checks sensors every Monday.',
        'The garden requires watering during extended dry weather.',
        'The garden uses a moisture sensor to reduce water consumption.',
        'The gardener checks soil moisture before watering.',
        'Pump performance is measured using pressure and flow rate.',
        'Pump flow rate may decrease when a filter becomes clogged.',
        'The pump technician inspects filters when pressure increases.',
        'Maintenance records identify which component was replaced.',
        'The library stores books by topic.',
        'The library catalog assigns unique identifiers to books.',
        'The library staff checks returned books for damage.',
    ]

    # 1st pass - to create the initial chunks (more coarse in general)
    ac.add_propositions(propositions)
    # 2nd pass -  to review and further splitting the finalized chunks when necessary (more granular in general)
    ac.review_chunks()
    # Print the chunks and the chunk outlines to track the process
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    # get chunks in the "list of strings" format (dictionary is also available as a format)
    print(ac.get_chunks(get_type="list_of_strings"))
