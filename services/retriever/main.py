import os
import logging
import asyncio
import psycopg2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from rankers.dense import DenseRanker
from rankers.sparse import SparseRanker
from rankers.fuser import RRFMerger

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retriever")
app = FastAPI(title="Hybrid Retriever Service")

# ------------------------------------------------------------------
# Database configuration
# ------------------------------------------------------------------

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "vector_db"),
    "database": os.getenv("POSTGRES_DB", "ragdb"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

# ------------------------------------------------------------------
# Component initialization
# ------------------------------------------------------------------

dense_ranker = DenseRanker(DB_CONFIG)
sparse_ranker = SparseRanker(DB_CONFIG)
merger = RRFMerger()

# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    profile: Optional[str] = "P1"

# ------------------------------------------------------------------
# Lifecycle events
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    # Start a non-blocking sparse index build on startup
    asyncio.create_task(sparse_ranker.build_index_background())

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/refresh")
async def refresh_index(background_tasks: BackgroundTasks):
    """
    Triggers a background rebuild of the sparse index.
    Intended to be called after document ingestion.
    """
    logger.info("Received index refresh request.")
    background_tasks.add_task(sparse_ranker.build_index_background)
    return {"status": "refresh_scheduled"}

@app.post("/search")
async def search(request: SearchRequest):
    logger.info(f"Hybrid search request received: '{request.query}'")

    try:
        # Fetch more candidates than requested to improve fusion quality
        candidate_k = request.k * 2

        dense_hits = dense_ranker.search(request.query, k=candidate_k)
        sparse_hits = sparse_ranker.search(request.query, k=candidate_k)

        logger.info(
            f"Retrieved candidates | Dense: {len(dense_hits)}, "
            f"Sparse: {len(sparse_hits)}"
        )

        # Fuse dense and sparse results
        merged_results = merger.merge(
            dense_hits,
            sparse_hits,
            limit=request.k,
        )

        # Fetch full document content for the ranked results
        final_docs = fetch_documents(merged_results)

        return {"documents": final_docs}

    except Exception as exc:
        logger.error(f"Search failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def fetch_documents(ranked_results):
    """
    Fetches document content and metadata for ranked document IDs.
    Preserves the ranking order.
    """
    if not ranked_results:
        return []

    doc_ids = [result["id"] for result in ranked_results]

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = "SELECT id, content, metadata FROM documents WHERE id = ANY(%s)"
    cur.execute(query, (doc_ids,))
    rows = cur.fetchall()

    doc_map = {
        row[0]: {
            "content": row[1],
            "metadata": row[2],
        }
        for row in rows
    }

    final_output = []
    for result in ranked_results:
        doc_data = doc_map.get(result["id"])
        if doc_data:
            final_output.append(
                {
                    "id": result["id"],
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "score": result["score"],
                    "source_scores": result["source_scores"],
                }
            )

    cur.close()
    conn.close()
    return final_output
