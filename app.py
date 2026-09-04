# FastAPI backend, OpenAI-compatible /v1/chat/completions endpoint,
# so OpenWebUI can connect directly. Wraps agent.py's pipeline, with
# simple Postgres-backed conversation memory.

import os
import uuid
from contextlib import contextmanager
from datetime import datetime

import psycopg2
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from agent import answer_policy_question, llm
from observability import start_phoenix, init_tracing, register_prompts, get_prompt

load_dotenv()
start_phoenix()
init_tracing()
register_prompts()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Optional bearer-token auth - if API_KEY is unset, the API stays open
# (matches the local/dev deployment this project targets).
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="myRelief Policy Assistant")


def require_api_key(authorization: str | None = Header(default=None)):
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@contextmanager
def get_conn():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT,
    )
    try:
        yield conn
    finally:
        conn.close()


# Create the conversation memory table if it doesn't exist yet
def init_memory_table():
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
        finally:
            cur.close()


init_memory_table()


# Fetch the last N turns for a conversation
def get_recent_history(conversation_id: str, limit: int = 6) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT role, message FROM conversation_memory "
                "WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s",
                (conversation_id, limit),
            )
            rows = cur.fetchall()
        finally:
            cur.close()
    return [{"role": r, "message": m} for r, m in reversed(rows)]


# Save one turn (user or assistant message) to memory
def save_turn(conversation_id: str, role: str, message: str):
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO conversation_memory (conversation_id, role, message) VALUES (%s, %s, %s)",
                (conversation_id, role, message),
            )
            conn.commit()
        finally:
            cur.close()


# If the new query depends on earlier context, rewrite it self-contained
def rewrite_if_followup(query: str, history: list[dict]) -> str:
    if not history:
        return query

    history_text = "\n".join(f"{h['role']}: {h['message']}" for h in history)
    prompt_template = get_prompt("followup_rewrite")
    prompt = prompt_template.format(history_text=history_text, query=query)

    rewritten = llm.call(prompt).strip()
    return rewritten if rewritten else query


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "myrelief-policy-assistant"
    messages: list[ChatMessage]
    conversation_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


# Model list endpoint - OpenWebUI queries this to populate its model
# picker before allowing chat; without it the connection setup fails.
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "myrelief-policy-assistant",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "myrelief",
        }],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    query = req.messages[-1].content
    conversation_id = req.conversation_id or str(uuid.uuid4())

    history = get_recent_history(conversation_id)
    self_contained_query = rewrite_if_followup(query, history)

    try:
        result = answer_policy_question(self_contained_query)
    except Exception as e:
        print(f"[chat_completions] pipeline failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to generate an answer, please try again") from e

    answer = result["answer"]

    save_turn(conversation_id, "user", query)
    save_turn(conversation_id, "assistant", answer)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
    }
