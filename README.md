# myRelief Policy Assistant — Agentic RAG System

An agentic RAG chatbot for insurance policy questions, built entirely
on open-source, locally-hosted tools.

## What this is

A chatbot that answers questions about myRelief's health insurance
policies (Basic/Standard/Premium tiers) and claims process, using
retrieval-augmented generation with a genuinely agentic research step.

## Architecture

```
PDFs (4 policy/SOP docs)
    -> Docling (parse to markdown, preserving structure)
    -> MarkdownNodeParser (chunk along heading boundaries)
    -> Contextual enrichment (Anthropic-style, LLM-generated blurb per chunk)
    -> all-MiniLM-L6-v2 (embed)
    -> PostgreSQL + pgvector (store, hybrid search enabled)

User query (via OpenWebUI)
    -> FastAPI (/v1/chat/completions, OpenAI-compatible)
    -> Conversation memory (Postgres-backed, rewrites ambiguous follow-ups)
    -> CrewAI Research Agent
         -> classify_query tool (LLM determines tier/doc_type)
         -> retrieve_policy_chunks tool (hybrid search + rerank + retry-if-weak)
    -> CrewAI Synthesis Agent (writes cited answer from retrieved chunks)
    -> CrewAI Suggestion Agent (flags related grounded info, no medical advice)
    -> Answer returned to OpenWebUI
```

Full flow diagram and design reasoning: see `DECISIONS_LOG.md`.

## Tech stack (assignment baseline requirements)

| Requirement | Tool used |
|---|---|
| Document processing | Docling |
| RAG methodology | LlamaIndex + PGVector/PostgreSQL |
| Contextual Agentic RAG | Anthropic-style contextual chunk enrichment + embeddings + reranking |
| Conversation memory | Postgres-backed conversation buffer, LLM-driven follow-up rewriting |
| Citation handling | Source file, tier, and doc_type metadata attached to every chunk, surfaced in answers |
| Model hosting | Ollama, qwen2.5:3b |
| Agentic orchestration | CrewAI (3 agents) |
| Observability | Arize Phoenix (tracing + prompt registry) |
| Evaluation | RAGAS |
| Chatbot interface | OpenWebUI (Docker), connected via OpenAI-compatible API |

## Design choices

- **Chunking:** MarkdownNodeParser, splitting along heading boundaries
  rather than fixed character counts -- Docling's markdown output
  preserves clause/section structure, so chunks stay complete logical
  units instead of arbitrary fragments.
- **Embedding model:** all-MiniLM-L6-v2 -- free, local, 384-dim,
  proven in prior projects.
- **LLM on Ollama:** qwen2.5:3b -- sized for local CPU-only inference
  on an 8GB RAM machine. See "Known limitations" below for the real
  performance tradeoff this creates.
- **Retrieval:** hybrid search (vector + Postgres full-text) via
  pgvector's native `hybrid_search=True`, rather than a separate
  Python-side BM25 retriever -- fewer moving parts, same benefit
  (catches exact terms like tier names and clause numbers that pure
  vector search can under-weight).
- **Reranking:** CrossEncoder (ms-marco-MiniLM-L-6-v2), narrowing top-5
  hybrid results.
- **Agent architecture:** 3 CrewAI agents. Only the Research Agent
  performs genuine agentic control-flow (multi-tool orchestration,
  retry-on-weak-results); Synthesis and Suggestion agents perform
  bounded language generation on the Research Agent's output. This
  distinction is deliberate -- see "Why this counts as agentic RAG"
  below.
- **Backend structure:** flat Python files (`ingest.py`, `retrieve.py`,
  `rerank.py`, `agent.py`, `app.py`, `observability.py`,
  `evaluate.py`) rather than a nested package structure -- faster to
  navigate and debug under a tight deadline, consistent with prior
  project patterns.
- **Deployment:** built and tested locally first, Dockerized last --
  same reasoning: prove logic works before adding containerization
  complexity.


## Known limitations (deliberate scope decisions)

- **qwen2.5:3b tool-calling reliability:** the Research Agent's
  two-tool orchestration (classify -> retrieve) is genuinely agentic
  architecture, but small local models are measurably worse at
  multi-step tool-calling than larger models. This design was chosen
  deliberately over a safer deterministic-orchestration alternative,
  accepting this risk, because it's the architecturally correct
  pattern -- the same code would perform more reliably with a larger
  model, with no changes needed.
- **CPU-only inference is slow.** Contextual chunk enrichment (89
  chunks, one LLM call each) takes a long time on this hardware.
  Production would use a GPU-hosted or larger model.
- **First Docker run starts with an empty database.** The
  docker-compose stack provisions fresh Postgres and Ollama containers
  -- it does not migrate data from the native dev setup used during
  development. Run `ingest.py` against the containerized Postgres (or
  restore a dump) before first use.
- **HNSW indexing not implemented.** At this corpus size (~90 chunks),
  a flat vector scan is fast enough; HNSW would be added at
  production scale (thousands+ vectors).
- **RAGAS reference answers are best-effort**, cross-checked against
  the source PDFs' known parameter structure but not independently
  verified line-by-line against every clause.
- **Docker deployment verified for API layer and container networking**
  (health check, model listing, all 4 services running and connected)
  but the full ingest-and-chat cycle was not independently re-run
  inside the container given time constraints -- the pipeline code
  is identical to the natively-tested version.
- **`docker pull` for OpenWebUI's image failed twice with "unexpected
  EOF" before succeeding on retry** -- a widely-reported Docker/network
  reliability issue, not a configuration error (see DECISIONS_LOG.md
  Section 12). If this happens, simply retry the `docker compose up`
  command; per community reports this typically succeeds within 2-4
  attempts.
- **A 500 error was encountered once via OpenWebUI** when querying
  against an empty (pre-ingestion) Docker database, after the Research
  Agent ran for several minutes with no response. Root-caused: none
  of the three CrewAI agents had explicit `max_iter`/`max_execution_time`
  bounds set, so a query with nothing retrievable could run for an
  extended (though not literally infinite) number of tool-calling
  rounds before returning. **Fixed** -- `research_agent` now has
  `max_iter=6` and `max_execution_time=300` (5-minute hard ceiling),
  `synthesis_agent`/`suggestion_agent` each have `max_execution_time=180`,
  and the Research Agent's backstory now explicitly caps retries per
  question-part at two attempts before reporting "not found" instead
  of continuing to retry. See DECISIONS_LOG.md Section 13.

## Project structure

```
chat_myrelief_techchefz/
├── ingest.py           # one-time: Docling -> chunk -> contextual enrich -> embed -> store
├── retrieve.py          # hybrid retrieval + rerank + retry (imported, not run standalone)
├── rerank.py             # CrossEncoder reranking
├── agent.py              # CrewAI agents (Research, Synthesis, Suggestion)
├── app.py                 # FastAPI server, OpenAI-compatible endpoint, conversation memory
├── observability.py        # Arize Phoenix tracing + prompt registry
├── evaluate.py               # RAGAS evaluation script (standalone, manual use only)
├── setup.sql                  # one-time DB/extension creation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-observability.txt  # arize-phoenix + openinference, installed separately (see comment inside)
├── data/raw_pdfs/               # source policy/SOP PDFs
└── DECISIONS_LOG.md              # full running log of design decisions and debugging history
```

## Reproducibility -- full step-by-step (Docker, recommended)

This is the exact sequence a reviewer should follow, from nothing to
a working chatbot. Every step is listed -- nothing is assumed.

**Prerequisites:** Docker Desktop installed and running. Nothing else
-- no local Python, Postgres, or Ollama needed; Docker provides all of it.

**Step 1 -- clone the repository:**
```bash
git clone https://github.com/Ifsaurabh/chat_myrelief_techchefz.git
cd chat_myrelief_techchefz
```

**Step 2 -- create your environment file:**
```bash
cp .env.example .env
```
Open `.env` and set `DB_PASSWORD` and `WEBUI_SECRET_KEY` to values of
your choice (`DB_PASSWORD` is used only for the local Postgres
container; `WEBUI_SECRET_KEY` signs OpenWebUI's session cookies and is
required -- `docker compose up` will refuse to start without it).
Leave the other values as-is unless you want a different model or
port, or want to require an API key on the chat endpoint (`API_KEY`).

**Step 3 -- build and start all four containers:**
```bash
docker compose up --build -d
```
This pulls `pgvector/pgvector:pg16`, `ollama/ollama:latest`, and
`ghcr.io/open-webui/open-webui:main`, and builds the `app` image from
the included `Dockerfile`. First run takes several minutes.

*Note on the prebuilt image:* an earlier build of the `app` image was
pushed to Docker Hub at
[`ifsaurabh/myrelief-policy-assistant`](https://hub.docker.com/r/ifsaurabh/myrelief-policy-assistant)
(see DECISIONS_LOG.md Section 12). It predates the fixes in this
audit (connection handling, auth, pinned dependencies, the pickle ->
JSON checkpoint change) and has not been rebuilt/re-pushed since, so
`docker-compose.yml` intentionally always builds `app` from the local
`Dockerfile` (`build: .`) rather than pulling that tag -- use
`docker compose up --build` as shown above, not `docker pull
ifsaurabh/myrelief-policy-assistant`.

A GitHub Actions workflow (`.github/workflows/docker-publish.yml`)
now rebuilds and pushes that tag automatically on every push to
`main`, so it stays current going forward. It reads Docker Hub
credentials from two repo secrets (Settings -> Secrets and variables
-> Actions -> New repository secret):

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | `ifsaurabh` |
| `DOCKERHUB_TOKEN` | A Docker Hub access token (Account Settings -> Security -> New Access Token, Read & Write scope) -- not your account password |

Once those two secrets are set, you can also trigger a rebuild on
demand from the repo's Actions tab -> "Build and push Docker image" ->
Run workflow, without waiting for a push to `main`.

*If the OpenWebUI image pull fails with "unexpected EOF":* this is a
known, widely-reported Docker/network issue (see DECISIONS_LOG.md
Section 12), not a problem with this project. Simply re-run the same
command -- it typically succeeds within 2-4 attempts.

**Step 4 -- confirm all four containers are running:**
```bash
docker ps
```
You should see `policy-postgres`, `policy-ollama`, `policy-app`, and
`policy-openwebui`, all with status "Up".

**Step 5 -- pull the LLM into the Ollama container (one-time, ~2GB download):**
```bash
docker compose exec ollama ollama pull qwen2.5:3b
```

**Step 6 -- run ingestion against the containerized database (one-time):**
```bash
docker compose exec app python ingest.py
```
This parses the 4 source PDFs (Docling), chunks them, generates an
Anthropic-style contextual blurb per chunk via the LLM (89 chunks --
this is the slow step, expect it to take a genuinely long time on
CPU-only hardware), embeds, and stores everything in Postgres. Progress
prints as it goes (`[12/89] context added ...`); let it run to completion.

**Step 7 -- verify the backend API directly:**
```bash
curl http://localhost:8000/health
```
Expected: `{"status":"ok"}`

```bash
curl http://localhost:8000/v1/models
```
Expected: a JSON object listing `myrelief-policy-assistant`.

**Step 8 -- open the chatbot in your browser:**
```
http://localhost:3000
```
- First visit: create a local admin account (OpenWebUI's own
  first-run setup, stored only in your local `openwebui_data` volume)
- From the model picker, select **myrelief-policy-assistant**
- Ask a question, e.g. *"What is the room rent capping under the
  Basic plan?"*
- **Expect a genuinely long wait for a response** -- this deployment
  runs entirely on local CPU inference by design (see "Known
  limitations"); a response taking one to several minutes is expected
  behavior on typical consumer hardware, not a hang.

**Step 9 (optional) -- view Phoenix traces:**
```
http://localhost:6006
```
Available once `app` has started, if Phoenix loaded successfully in
your environment (see DECISIONS_LOG.md Section 10 for a known
version-compatibility caveat with graceful fallback).

**Step 10 (optional) -- run the RAGAS evaluation:**
```bash
docker compose exec app python evaluate.py
```

**To stop everything cleanly:**
```bash
docker compose down
```
Containers stop; your data (Postgres volume, pulled Ollama model,
OpenWebUI account) persists for next time. Add `-v` to also delete
that persisted data and start completely fresh.


## Running it (local development, no Docker)

Used during development; requires PostgreSQL 15+ with pgvector and
Ollama installed natively. See `DECISIONS_LOG.md` for the exact
Windows setup steps used (including a Windows-specific PyTorch/Docling
compiler workaround, and a precompiled pgvector binary route that
avoided needing a C++ toolchain).

```bash
cp .env.example .env   # fill in your local Postgres password
pip install -r requirements.txt
pip install -r requirements-observability.txt   # separate step, see comment in requirements.txt
psql -U postgres -f setup.sql
ollama pull qwen2.5:3b
python ingest.py
uvicorn app:app --reload
python evaluate.py     # optional, standalone
```