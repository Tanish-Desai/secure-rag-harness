import logging
import psycopg2
import asyncio
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Ensure required NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger("retriever.sparse")


class SparseRanker:
    def __init__(self, db_config):
        self.db_config = db_config
        self.bm25 = None
        self.doc_ids = []
        self.is_ready = False
        self.is_building = False

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def _build_index_sync(self):
        """
        Blocking method that fetches documents and builds the BM25 index.
        Intended to be executed in a background thread.
        """
        logger.info("Starting BM25 index build...")
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("SELECT id, content FROM documents")
            rows = cur.fetchall()

            self.doc_ids = [row[0] for row in rows]
            corpus = [row[1] for row in rows]

            tokenized_corpus = [word_tokenize(text.lower()) for text in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

            self.is_ready = True
            logger.info(
                f"Sparse index built successfully with {len(self.doc_ids)} documents."
            )

            cur.close()
            conn.close()

        except Exception as exc:
            logger.error(f"Failed to build sparse index: {exc}")
            self.is_ready = False

        finally:
            self.is_building = False

    async def build_index_background(self):
        """
        Runs the index build in a background thread.
        Prevents concurrent rebuilds.
        """
        if self.is_building:
            logger.warning("Sparse index build already in progress.")
            return

        self.is_building = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._build_index_sync)

    def search(self, query: str, k: int = 20) -> list:
        """
        Performs keyword search using BM25.

        Returns a list of dictionaries with keys:
        - id: document identifier
        - score: relevance score
        """
        if not self.is_ready:
            logger.warning("Sparse index is not ready. Returning empty results.")
            return []

        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        doc_scores = zip(self.doc_ids, scores)
        top_docs = sorted(
            doc_scores,
            key=lambda item: item[1],
            reverse=True,
        )[:k]

        return [
            {"id": doc_id, "score": float(score)}
            for doc_id, score in top_docs
            if score > 0
        ]
