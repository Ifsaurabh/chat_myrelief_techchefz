# Arize Phoenix - tracing + prompt registry.
# Wrapped defensively: a Phoenix import/init failure should not block
# the core RAG pipeline (ingestion, retrieval, agents) from running.
# See DECISIONS_LOG.md for the specific compatibility issue hit.

try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.crewai import CrewAIInstrumentor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    _PHOENIX_AVAILABLE = True
except Exception as e:
    print(f"[Phoenix unavailable, observability disabled: {e}]")
    _PHOENIX_AVAILABLE = False

PHOENIX_PROJECT_NAME = "myrelief-policy-assistant"

_session = None


# Launch the local Phoenix server + UI (http://localhost:6006)
def start_phoenix():
    global _session
    if not _PHOENIX_AVAILABLE:
        return None
    if _session is None:
        _session = px.launch_app()
        print(f"Phoenix UI running at: {_session.url}")
    return _session


# Wire up auto-instrumentation for LlamaIndex + CrewAI calls
def init_tracing():
    if not _PHOENIX_AVAILABLE:
        return
    tracer_provider = register(project_name=PHOENIX_PROJECT_NAME)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    print("Phoenix tracing initialized for LlamaIndex + CrewAI")


# Prompt registry - source of truth for this project's prompts
PROMPTS = {
    "contextual_chunk_enrichment": """<document>
{doc_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>
Give a short, succinct context (1-2 sentences) to situate this chunk
within the overall document, for the purpose of improving search
retrieval of the chunk. Answer only with the succinct context, nothing else.""",

    "classify_query": """Question: "{query}"

Does this question specifically mention a policy tier -- "Basic", "Standard", or "Premium"?
Does it ask about claim procedures/process (which would be doc_type "sop") or about coverage/policy terms (doc_type "policy")?

Answer in exactly this format, nothing else:
tier: <basic/standard/premium/none>
doc_type: <policy/sop/none>""",

    "followup_rewrite": """Conversation so far:
{history_text}

New user message: "{query}"

If the new message depends on the earlier conversation to make sense (e.g. "what about X", "and that one"), rewrite it as a complete, self-contained question. If it's already self-contained, repeat it unchanged.

Answer with ONLY the rewritten question, nothing else.""",
}


# register prompts with Phoenix, if available
def register_prompts():
    if not _PHOENIX_AVAILABLE:
        print("[Phoenix unavailable, skipping prompt registration - using local PROMPTS dict as fallback]")
        return
    try:
        client = px.Client()
        for name, template in PROMPTS.items():
            try:
                client.prompts.create(
                    name=name,
                    prompt_template=template,
                    prompt_template_format="mustache",
                )
                print(f"Registered prompt: {name}")
            except Exception as e:
                print(f"[Prompt '{name}' already registered or registration skipped: {e}]")
    except Exception as e:
        print(f"[Phoenix client unavailable, skipping prompt registration entirely: {e}]")


def get_prompt(name: str) -> str:
    if not _PHOENIX_AVAILABLE:
        return PROMPTS.get(name, "")
    try:
        client = px.Client()
        prompt = client.prompts.get(name=name)
        return prompt.template
    except Exception as e:
        print(f"[Phoenix prompt fetch failed for '{name}', using local fallback: {e}]")
        return PROMPTS.get(name, "")
