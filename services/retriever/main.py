import os
import logging
import json
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
DB_HOST = os.getenv("DB_HOST", "vector_db")
DB_NAME = os.getenv("POSTGRES_DB", "ragdb")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")

# Load the baked model
logger.info("Loading embedding model...")
model = SentenceTransformer('./model_data', device='cpu')
logger.info("Model loaded.")

class SearchRequest(BaseModel):
    query: str
    k: int = 3
    profile: Optional[str] = "P1"  # For future tenant isolation tests

class SearchResult(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    documents: List[SearchResult]

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    logger.info(f"Searching for: {request.query}")
    
    conn = get_db_connection()
    register_vector(conn)
    cur = conn.cursor()
    
    try:
        # 1. Vectorize the Query
        query_vector = model.encode(request.query).tolist()
        
        # 2. Execute Semantic Search (Cosine Distance)
        # The <=> operator in pgvector is Cosine Distance
        cur.execute("""
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) as similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vector, query_vector, request.k))
        
        results = []
        for row in cur.fetchall():
            results.append(SearchResult(
                id=row[0],
                content=row[1],
                metadata=row[2] if row[2] else {},
                score=float(row[3])
            ))
            
        logger.info(f"Found {len(results)} matches.")
        return {"documents": results}
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)