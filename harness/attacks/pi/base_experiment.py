from abc import ABC, abstractmethod
import requests
import time


class BaseExperiment(ABC):
    def __init__(self, config):
        self.config = config
        self.ingest_host = "http://localhost:8004"
        self.retriever_host = "http://localhost:8001"
        self.gateway_host = "http://localhost:8000"

    @abstractmethod
    def run(self):
        """
        Executes the full experiment lifecycle.
        """
        pass

    def reset_and_ingest(self, documents):
        """
        Resets the vector database and ingests a new set of documents.
        """

        # Reset the vector database
        reset_url = f"{self.ingest_host}/reset"
        try:
            response = requests.post(reset_url, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            print(f"Critical error: failed to reset database. Error: {exc}")
            raise

        # Ingest documents in batches
        ingest_url = f"{self.ingest_host}/ingest"
        batch_size = 50

        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            payload = {
                "documents": [
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                    }
                    for doc in batch
                ]
            }

            try:
                response = requests.post(
                    ingest_url,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
            except Exception as exc:
                print(f"Ingestion failed for batch starting at index {start}. Error: {exc}")
                raise

        # Notify the retriever to refresh its index so new documents are visible
        try:
            requests.post(
                f"{self.retriever_host}/refresh",
                timeout=5,
            )
            # Allow time for the background index build to complete
            time.sleep(0.5)
        except Exception as exc:
            # Index refresh failure should not abort the experiment
            print(f"Warning: failed to refresh retriever index: {exc}")

        print("Ingestion and indexing completed successfully.")
