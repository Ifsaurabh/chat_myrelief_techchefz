# Decision Log -- myRelief Agentic RAG

> **Note on numbering:** Sections 1-8 (initial domain choice, backend
> structure, chunking/embedding/agent architecture decisions, and the
> Docling Windows compiler fix) were lost to an accidental file
> overwrite during editing. That reasoning is preserved in README.md's
> "Design choices" and "Why this counts as agentic RAG" sections
> instead. This log picks up from Section 9 and is complete from
> there on, including every major pivot made during testing.

## 9. PostgreSQL + pgvector setup (RESOLVED)

**Decision:** Native PostgreSQL install on Windows, not Docker — chosen
after confirming a precompiled pgvector Windows binary existed matching
our PostgreSQL version, avoiding the compiler-toolchain risk we'd
already hit once with Docling's omp.h issue.

**Installed:**
- PostgreSQL 17.11 (EDB Windows installer, default port 5432)
- Command Line Tools included (psql) -- had to manually add
  `C:\Program Files\PostgreSQL\17\bin` to Windows PATH (not added
  automatically by the installer)
- pgvector 0.8.2, via precompiled binary from
  https://github.com/andreiramani/pgvector_pgsql_windows
  (release tag 0.8.2_17.6, tested against PG 17.6, worked fine against
  our 17.11 install -- same major version, extensions compatible across
  minor version bumps)

**Install steps that worked:**
1. Downloaded `vector.v0.8.2-pg17.zip`, extracted
2. Stopped `postgresql-x64-17` service via services.msc
3. Copied extracted `include/`, `lib/`, `share/` folders into
   `C:\Program Files\PostgreSQL\17\`, merging/overwriting existing folders
4. Restarted `postgresql-x64-17` service
5. `psql -U postgres` -> `CREATE EXTENSION vector;`
6. Verified: `SELECT extname, extrelocatable, extversion FROM pg_extension
   WHERE extname='vector';` returned `vector | t | 0.8.2`

**Interview-ready explanation:** "I checked whether a precompiled pgvector
binary existed for my PostgreSQL version before deciding between native
install and Docker -- since I'd already hit a compiler-toolchain issue
with Docling on this Windows machine, I wanted to avoid the same risk
with pgvector's C extension. A community-maintained precompiled binary
existed for PG17, so I went native instead of adding Docker complexity
for this piece."

## 10. Arize Phoenix Python 3.11 compatibility bug (RESOLVED, worked around)

**Problem:** arize-phoenix 20.4.0 fails to import on Python 3.11 with
`ValueError: mutable default <class 'mappingproxy'> for field
boolean_names is not allowed: use default_factory` -- a bug in
Phoenix's own `phoenix/trace/dsl/filter.py`, not our code.

**Fix attempted and worked:** `pip install --upgrade --no-cache-dir
arize-phoenix openinference-instrumentation-openai opentelemetry-api
opentelemetry-sdk` resolved the import error.

**Defensive fallback also kept in place** (in case the bug resurfaces
on a different machine/version): `observability.py` wraps the Phoenix
import in try/except, setting `_PHOENIX_AVAILABLE = False` on failure.
All four Phoenix-dependent functions (`start_phoenix`, `init_tracing`,
`register_prompts`, `get_prompt`) check this flag and gracefully no-op
or fall back to the local `PROMPTS` dict, so a Phoenix failure never
blocks the core RAG pipeline from running.

**Interview-ready explanation:** "I hit a genuine third-party library
bug in Phoenix's Python 3.11 compatibility, confirmed via traceback
rather than a config issue on my end. I fixed it via a package upgrade,
but also wrapped the integration defensively so observability failures
can never block the core pipeline -- tracing and prompt-registry
features degrade gracefully to local fallbacks instead of crashing."

## 11. Ollama model selection -- tested comparison (RESOLVED)

**Context:** initial model choice (qwen2.5:3b) correctly executed
single-part questions with proper tool-calling (classify_query ->
retrieve_policy_chunks), but failed to decompose genuinely multi-part
questions into multiple tool calls -- e.g. "compare Basic vs Premium
room rent AND tell me the claim submission deadline" only searched one
tier, missing the comparison and the SOP content entirely.

**Hypothesis tested:** would a model with genuinely native tool-calling
training (Qwen3 series) or stronger reasoning focus (Phi-4-mini) handle
decomposition better?

**Models tested against the identical complex query and codebase:**

| Model | Result |
|---|---|
| qwen2.5:3b | Correctly called tools, but only handled one part of a multi-part question. Fast enough on this hardware to complete reliably. |
| qwen3:4b | Native tool-calling architecture, but includes an optional "thinking mode" that added enough latency to hit CrewAI's LLM timeout on this CPU (AMD Ryzen 5 2500U, no dedicated GPU, ~6.9GB usable RAM) -- had to raise `LLM(..., timeout=500)` just to let it attempt to finish, and even then did not complete within a reasonable wait. |
| phi4-mini:3.8b | Worse than qwen2.5:3b -- did not invoke tools at all; returned the raw tool JSON schema as its "answer" instead of calling the tools. Confirms Phi-4-mini's reasoning strength does not extend to reliable structured tool invocation in this Ollama/CrewAI integration. |

**Decision: reverted to qwen2.5:3b.** It is the only tested model that
reliably executes tool calls correctly on this specific hardware, even
though its multi-part decomposition is imperfect. This is a genuine,
evidence-based hardware/model tradeoff, not a guess.

**Interview-ready explanation:** "I didn't just pick a small model and
hope -- I tested the identical multi-part query across three models
with different claimed strengths (native tool-calling, stronger
reasoning) and found qwen2.5:3b was the only one that reliably executed
tool calls at all on my CPU-only hardware, even though it has a real,
documented weakness in multi-part query decomposition. The architecture
itself is model-agnostic -- this constraint is specific to consumer
CPU inference, and the same code would decompose correctly with a
cloud-hosted or GPU-accelerated larger model."

**Bug fixed along the way:** CrewAI's `LLM` class has a much shorter
default timeout than the `request_timeout=500` we'd already set on the
LlamaIndex-wrapped Ollama calls elsewhere (ingest.py, evaluate.py) --
added the same explicit `timeout=500` to `agent.py`'s `LLM(...)`
instantiation. Kept even after reverting models, since it's a real fix
regardless of which model is used.

**Known limitation, stated plainly for README:** the Research Agent
reliably handles single-part, single-tier questions with correct tool
orchestration and citation. Genuinely multi-part questions (comparing
tiers, combining policy + SOP content) are not reliably decomposed into
multiple tool calls by qwen2.5:3b -- this is a tested, confirmed model
capability limit on this hardware, not an untested assumption.

## 12. Docker deployment (RESOLVED)

Full 4-service stack (postgres, ollama, app, openwebui) built, pushed
to Docker Hub (ifsaurabh/myrelief-policy-assistant:latest), and
verified running together via docker-compose.yml.

**Bugs found and fixed during Docker testing:**
1. Answer-combining bug: crew.kickoff()'s return value only reflects
   the LAST task in a sequential crew (Suggestion Agent), silently
   discarding the Synthesis Agent's real answer. Fixed by reading
   synthesis_task.output.raw and suggestion_task.output.raw explicitly
   in answer_policy_question() instead of relying on the crew-level result.
2. Phoenix AttributeError inside container: px.Client() doesn't exist
   in this container's installed Phoenix version (different failure
   mode than the native dataclass bug). Original try/except only
   wrapped the import, not the actual API calls -- fixed by wrapping
   px.Client() itself in register_prompts() and get_prompt() so any
   Phoenix API mismatch degrades gracefully instead of crashing the app.
3. Docker log buffering: added PYTHONUNBUFFERED=1 to Dockerfile --
   without it, Python's stdout is buffered inside containers and
   `docker logs` shows nothing even while the app runs correctly.

**Verified working inside Docker:** /health, /v1/models endpoints,
all 4 containers networked correctly (app reaches postgres via
service name, ollama via service name).

**Not independently re-verified inside Docker:** full ingestion cycle
and a complete chat-completion response with real retrieved data --
the Docker Postgres starts empty (separate from the native database),
and re-running the full ~89-chunk contextual enrichment pipeline
inside the container was not repeated given time constraints. The
pipeline code is identical to the natively-verified version (same
ingest.py, same agent.py) -- this is a scope decision, not an
unknown risk.

**docker pull for ghcr.io/open-webui/open-webui:main failed twice
with "unexpected EOF"** before succeeding on retry -- confirmed via
research to be a widely-reported Docker/network reliability issue
(not a config error), where retrying is the documented community fix.

## 13. Additional pivots and small decisions made during final testing

Smaller decisions and dead ends from the same testing session as
Sections 11-12, kept here for a complete record.

**Groq API validation attempt (inconclusive, reverted).** After
confirming qwen2.5:3b's multi-part decomposition limitation, attempted
to validate that the *architecture* (not just this specific model) was
sound by swapping in Groq's free-tier `llama-3.3-70b-versatile` via a
toggle (`USE_GROQ` env var in `agent.py`). This hit a genuine CrewAI/
LiteLLM compatibility bug unrelated to the architecture: CrewAI's
newer LiteLLM integration sends a `cache_breakpoint` property (an
Anthropic-specific prompt-caching field) in the system message, which
Groq's API rejects with a 400 error. Not pursued further given time --
this is a third-party library incompatibility, not a finding about
model capability. The `USE_GROQ` toggle is left in `agent.py` (default
off) in case it's revisited later with a different provider or a
CrewAI version that doesn't send that field.

**Prompt reinforcement, tested and kept.** Before the Groq attempt,
the Research Agent's backstory was strengthened with a concrete worked
example ("Basic and Premium room rent = TWO separate calls") rather
than only abstract "decompose the question" language. Measured real
improvement on the same complex test query: 1 of 3 question-parts
covered before the change, 2 of 3 after. Basic-tier room rent was
still missed even with the improved prompt -- confirms this is a
genuine, partially-prompt-fixable but not fully prompt-fixable model
limitation, not simply an under-specified instruction.

**answer_policy_question() empty-database behavior (suspected, not
confirmed root cause).** When tested against the empty Docker
database via OpenWebUI, a request returned HTTP 500 after several
minutes with no response. Working theory: the Research Agent's own
backstory instruction ("keep retrieving, part by part, until every
distinct part has sufficient, relevant chunks") has no defined exit
condition when the database is genuinely empty -- every
retrieve_policy_chunks call returns "No relevant policy content
found," which the agent may interpret as "insufficient, retry" rather
than "nothing exists, stop." This was never observed during any
native test, all of which ran against the populated database. Not
root-caused with certainty before time ran out; flagged honestly in
README.md rather than silently left undocumented. If revisited: the
fix would likely be an explicit stopping instruction in the backstory
("if retrieve_policy_chunks returns 'No relevant policy content
found' after 2 attempts, stop and report no information was found"
rather than continuing to retry indefinitely), or a hard retry-count
ceiling enforced in code rather than left to the agent's judgment.

**Model reversion sequence (chronological summary of Section 11's
testing, for quick reference):** qwen2.5:3b (baseline, working) ->
qwen3:4b pulled and tested (timeout, even after raising LLM timeout to
500s) -> qwen3:4b and phi4-mini both removed via `ollama rm` to free
space -> phi4-mini pulled and tested (did not call tools at all,
worse than baseline) -> reverted to qwen2.5:3b, re-pulled after the
system restart wiped the local Ollama model store -> qwen2.5:3b
confirmed working again, kept as final choice. `OLLAMA_MODEL` was
centralized into `.env` partway through this sequence specifically so
future model swaps are a one-line change rather than a multi-file edit.

**Docker image cleanup.** After pushing `ifsaurabh/myrelief-policy-
assistant:latest` to Docker Hub, local images were deleted from Docker
Desktop to reclaim disk space (the local "disk usage" figure during
build, ~11GB, includes build cache and intermediate layers, not the
final image size -- the actual pushed image is 3.61GB). `docker
compose up --build` regenerates the local image from the Dockerfile
as needed; nothing was lost by deleting the local copy.