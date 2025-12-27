import os
import logging
import time
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from middleware import check_policy, log_telemetry

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

app = FastAPI(title="Secure RAG Gateway")

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://retriever:8001")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://host.docker.internal:11434/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "secure-rag-llama3")

# ------------------------------------------------------------------
# Template loading
# ------------------------------------------------------------------

TEMPLATES = {}
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def load_templates():
    """
    Load system prompt templates from disk.
    """
    try:
        os.makedirs(TEMPLATE_DIR, exist_ok=True)

        def read_template(name, default):
            path = os.path.join(TEMPLATE_DIR, f"{name}.txt")
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read()
            return default

        TEMPLATES["vanilla"] = read_template(
            "vanilla", "Context:\n{context_text}"
        )
        TEMPLATES["skeptical"] = read_template(
            "skeptical", "Context:\n{context_text}"
        )

        logger.info("System prompt templates loaded")

    except Exception as exc:
        logger.error(f"Failed to load templates: {exc}")
        TEMPLATES["vanilla"] = "Context:\n{context_text}"


@app.on_event("startup")
async def startup_event():
    load_templates()

# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    search_query: Optional[str] = None
    topology: str = "sequential"
    profile: str = "P1"
    seed: int = 42
    documents: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    response: str
    model: str
    context: List[Dict[str, Any]] = []

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def fetch_documents(request: ChatRequest) -> List[Dict[str, Any]]:
    """
    Resolve documents based on topology.
    """
    if request.topology in {"pi", "direct_pi"}:
        logger.info(
            f"Retriever bypassed ({request.topology}). "
            f"Using provided documents."
        )
        return request.documents or []

    search_term = request.search_query or request.query

    try:
        response = requests.post(
            f"{RETRIEVER_URL}/search",
            json={
                "query": search_term,
                "k": 1,
                "profile": request.profile,
            },
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("documents", [])

    except Exception as exc:
        logger.error(f"Retriever request failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Retriever unavailable",
        )


def build_llm_messages(
    request: ChatRequest,
    retrieved_docs: List[Dict[str, Any]],
):
    """
    Build system and user messages based on topology.
    """
    if request.topology in {"pi", "direct_pi"}:
        system_content = TEMPLATES["vanilla"].format(
            context_text="[Provided in user message]"
        )

        context_text = "\n\n".join(
            doc.get("content", "") for doc in retrieved_docs
        )

        user_content = (
            f"Context:\n{context_text}\n\n"
            f"Task: {request.query}"
        )

        return system_content, user_content

    context_text = "\n\n".join(
        f"[Document {doc.get('id', 'unknown')}]: "
        f"{doc.get('content', '')}"
        for doc in retrieved_docs
    )

    template_key = "skeptical" if request.profile == "P2" else "vanilla"

    system_content = TEMPLATES.get(
        template_key,
        TEMPLATES["vanilla"],
    ).format(context_text=context_text)

    return system_content, request.query

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat_handler(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    start_time = time.time()
    logger.info(
        f"Received query (topology={request.topology}): {request.query}"
    )

    retrieved_docs = fetch_documents(request)

    # Policy enforcement
    check_policy(request.query, retrieved_docs)

    system_content, user_content = build_llm_messages(
        request, retrieved_docs
    )

    llm_payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 0.0,
        "seed": request.seed,
    }

    try:
        llm_response = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            json=llm_payload,
            timeout=90,
        )
        llm_response.raise_for_status()
        generated_text = (
            llm_response.json()["choices"][0]["message"]["content"]
        )

    except Exception as exc:
        logger.error(f"LLM request failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="LLM unavailable",
        )

    latency = time.time() - start_time
    background_tasks.add_task(
        log_telemetry,
        {
            "timestamp": time.time(),
            "latency": latency,
            "profile": request.profile,
            "status": "success",
            "topology": request.topology,
        },
    )

    return ChatResponse(
        response=generated_text,
        model=LLM_MODEL_NAME,
        context=retrieved_docs,
    )
