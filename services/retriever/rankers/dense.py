import logging
import psycopg2
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("retriever.dense")


class DenseRanker:
    def __init__(self, db_config, model_path="./model_data"):
        self.db_config = db_config

        logger.info("Loading dense ranking model...")
        self.model = SentenceTransformer(model_path, device="cpu")
        logger.info("Dense ranker initialized.")

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def search(self, query: str, k: int = 20) -> list:
        """
        Performs semantic search using pgvector.

        Returns a list of dictionaries with keys:
        - id: document identifier
        - score: similarity score
        """
        try:
            embedding = self.model.encode(query).tolist()

            conn = self._get_connection()
            cur = conn.cursor()

            # pgvector cosine distance returns a distance value,
            # so similarity is computed as (1 - distance)
            cur.execute(
                """
                SELECT id, 1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, k),
            )

            results = [
                {"id": row[0], "score": float(row[1])}
                for row in cur.fetchall()
            ]

            cur.close()
            conn.close()
            return results

        except Exception as exc:
            logger.error(f"Dense search failed: {exc}")
            return []
