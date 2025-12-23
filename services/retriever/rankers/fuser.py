class RRFMerger:
    def __init__(self, k_constant=60):
        self.k = k_constant

    def merge(self, dense_results, sparse_results, limit=10):
        """
        Merges dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).

        RRF score:
        score(d) = sum(1 / (k + rank(d))) across all result lists
        """
        scores = {}
        metadata = {}

        # Process dense retrieval results
        for rank, item in enumerate(dense_results):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (self.k + rank + 1))

            if doc_id not in metadata:
                metadata[doc_id] = {
                    "dense_rank": rank + 1,
                    "sparse_rank": None,
                }
            else:
                metadata[doc_id]["dense_rank"] = rank + 1

        # Process sparse retrieval results
        for rank, item in enumerate(sparse_results):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (self.k + rank + 1))

            if doc_id not in metadata:
                metadata[doc_id] = {
                    "dense_rank": None,
                    "sparse_rank": rank + 1,
                }
            else:
                metadata[doc_id]["sparse_rank"] = rank + 1

        # Sort documents by final RRF score
        sorted_doc_ids = sorted(
            scores.keys(),
            key=lambda doc_id: scores[doc_id],
            reverse=True,
        )[:limit]

        return [
            {
                "id": doc_id,
                "score": scores[doc_id],
                "source_scores": metadata[doc_id],
            }
            for doc_id in sorted_doc_ids
        ]
