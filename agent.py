"""
CrewAI agents for the myRelief policy assistant.

Research Agent uses two tools (classify_query, retrieve_policy_chunks)
in sequence. Synthesis and Suggestion agents do bounded language
generation on the Research Agent's output.
"""

# imports
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import tool

from retrieve import load_index, retrieve

load_dotenv()

# For local models via Ollama: leave USE_GROQ unset/false.
# For this one-time validation test (proving the architecture works
# with a more capable model, not just the local 3B model): set
# USE_GROQ=true in .env and provide GROQ_API_KEY. See DECISIONS_LOG.md.
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

if USE_GROQ:
    OLLAMA_MODEL = "groq/llama-3.3-70b-versatile"
    llm = LLM(model=OLLAMA_MODEL, timeout=500)
else:
    OLLAMA_MODEL = f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')}"
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, timeout=500)

_index = None


@tool("classify_query")
def classify_query(query: str) -> str:
    """
    Determine which policy tier (basic, standard, premium, or none) and
    document type (policy, sop, or none) the question relates to.
    Use this FIRST, before retrieving any content.
    """
    prompt = f"""Question: "{query}"

Does this question specifically mention a policy tier -- "Basic", "Standard", or "Premium"?
Does it ask about claim procedures/process (which would be doc_type "sop") or about coverage/policy terms (doc_type "policy")?

Answer in exactly this format, nothing else:
tier: <basic/standard/premium/none>
doc_type: <policy/sop/none>"""

    response = llm.call(prompt)
    return response.strip()


@tool("retrieve_policy_chunks")
def retrieve_policy_chunks(query: str, tier: str = "", doc_type: str = "") -> str:
    """
    Retrieve and rerank relevant policy/SOP content for the query.
    Use tier and doc_type values from classify_query's output. Call
    this AFTER classify_query.
    """
    global _index
    if _index is None:
        _index = load_index()

    tier_filter = tier if tier in ("basic", "standard", "premium") else None
    doc_type_filter = doc_type if doc_type in ("policy", "sop") else None

    reranked, ok = retrieve(query, _index, tier=tier_filter, doc_type=doc_type_filter)

    if not reranked:
        return "No relevant policy content found for this query."

    formatted = []
    for i, node in enumerate(reranked):
        meta = node.metadata
        formatted.append(
            f"[Source {i+1}: {meta.get('source_file')}, "
            f"tier={meta.get('tier')}, doc_type={meta.get('doc_type')}]\n"
            f"{node.get_content()}"
        )
    return "\n\n".join(formatted)


research_agent = Agent(
    role="Policy Research Agent",
    goal="Retrieve all relevant policy/SOP chunks (with citations) needed to fully answer the user's question",
    backstory=(
        "You are an insurance policy research specialist. Your job has "
        "four steps: "
        "1) Read the question carefully and mentally break it into its "
        "distinct parts -- e.g. if it compares tiers or asks about "
        "both coverage and claims process, treat each part as a "
        "separate thing you need an answer for, and list each part to "
        "yourself before doing any retrieval. "
        "2) Call classify_query to determine the relevant tier and "
        "document type for the question (or for each part, if it has "
        "multiple parts). "
        "3) Call retrieve_policy_chunks once per part, using a distinct, "
        "specific query and tier/doc_type for each part -- for example, "
        "if comparing Basic and Premium room rent, that is TWO separate "
        "calls: one with tier='basic' and a room-rent-focused query, and "
        "one with tier='premium' and a room-rent-focused query. Do not "
        "try to answer multiple parts with a single retrieve_policy_chunks "
        "call. "
        "4) After each call, check: did this cover the part it was meant "
        "to answer? If a part is still missing, unrelated, or thin, call "
        "retrieve_policy_chunks AGAIN for that specific part -- rephrase "
        "the query with different keywords, or change/drop the tier or "
        "doc_type filter, and try again. Keep retrieving, part by part, "
        "until every distinct part of the original question has "
        "sufficient, relevant chunks. Only then pass everything to the "
        "next agent for synthesis -- do not write the final answer "
        "yourself."
    ),
    tools=[classify_query, retrieve_policy_chunks],
    llm=llm,
    verbose=True,
)

synthesis_agent = Agent(
    role="Policy Synthesis Agent",
    goal="Write a clear, accurate, cited answer using only the retrieved content",
    backstory=(
        "You write final answers to insurance policy questions using ONLY "
        "the retrieved source content provided to you. You always cite "
        "which source/tier each claim comes from. You never invent "
        "information not present in the retrieved content. If the "
        "retrieved content doesn't fully answer the question, say so "
        "explicitly rather than guessing."
    ),
    llm=llm,
    verbose=True,
)

suggestion_agent = Agent(
    role="Related Policy Info Agent",
    goal="Flag related policy/claims details the user didn't ask about but should know",
    backstory=(
        "You review the retrieved policy content and the synthesized "
        "answer, then flag genuinely relevant related details the user "
        "didn't ask about -- e.g. waiting periods, sub-limits, or claim "
        "process steps that apply to their situation. You ONLY reference "
        "information present in the retrieved content. You NEVER give "
        "medical advice, treatment guidance, or caregiving suggestions -- "
        "that is out of scope for an insurance policy assistant. If "
        "there is nothing genuinely relevant to add, respond with the "
        "exact sentence: 'No additional related details to flag.' -- "
        "never respond with just the single word 'None' or leave your "
        "answer blank."
    ),
    llm=llm,
    verbose=True,
)


def answer_policy_question(query: str) -> dict:
    """Run the full 3-agent pipeline for one user question.

    Returns the Synthesis Agent's answer combined with the Suggestion
    Agent's notes. crew.kickoff()'s overall result only reflects the
    LAST task in a sequential crew (Suggestion), so the Synthesis
    Agent's answer is read explicitly from its own task output rather
    than relying on the crew-level result.
    """
    research_task = Task(
        description=(
            f"Research this user question about myRelief policies: "
            f"'{query}'. First classify the query, then retrieve "
            f"relevant content using that classification."
        ),
        expected_output="Retrieved policy content relevant to the question, with source citations.",
        agent=research_agent,
    )

    synthesis_task = Task(
        description=(
            f"Using the research findings, write a clear, cited answer "
            f"to: '{query}'."
        ),
        expected_output="A grounded, cited answer to the user's question.",
        agent=synthesis_agent,
        context=[research_task],
    )

    suggestion_task = Task(
        description=(
            "Review the research findings and the synthesized answer. "
            "Flag any genuinely relevant related policy/claims details "
            "the user should know but didn't ask about."
        ),
        expected_output="A short list of related relevant details, or a note that there are none.",
        agent=suggestion_agent,
        context=[research_task, synthesis_task],
    )

    crew = Crew(
        agents=[research_agent, synthesis_agent, suggestion_agent],
        tasks=[research_task, synthesis_task, suggestion_task],
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()

    synthesis_answer = synthesis_task.output.raw if synthesis_task.output else ""
    suggestion_note = suggestion_task.output.raw if suggestion_task.output else ""

    final_answer = synthesis_answer.strip()
    note = suggestion_note.strip()
    if note and note.lower() not in ("none", "no additional related details to flag.", ""):
        final_answer += f"\n\n---\nYou may also want to know:\n{note}"

    return {"answer": final_answer}