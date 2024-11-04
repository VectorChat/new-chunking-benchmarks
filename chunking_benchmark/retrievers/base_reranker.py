import logging
import os
from abc import ABC, abstractmethod

logging.basicConfig(level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper()))
logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    def __init__(self, name: str = "base-reranker"):
        self.name = name

    @abstractmethod
    def _rerank(self, res, query, top_n=3, model="rerank-english-v3.0", **kwargs):
        """The actual reranking logic."""
        raise NotImplementedError

    def rerank(self, res, query, top_n=3, model="rerank-english-v3.0"):
        """Rerank the retrieved objects from pinecone database"""
        if "matches" not in res or not res["matches"]:
            logger.error(f"res: {res}")
            logger.error(f"query: {query}")
            logger.error("Response does not contain any matches.")
            return []
        docs = {x["metadata"]["text"]: i for i, x in enumerate(res["matches"])}
        docs_text = [doc for doc in docs.keys()]
        try:
            logger.info(f"Reranking {len(docs)} documents")
            rerank_docs = self._rerank(
                res=docs_text,
                query=query,
                top_n=top_n,
                model=model,
                max_chunks_per_doc=1,
                return_documents=True,
            ).get("results", [])
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return ""
        reranked_ids = [docs[doc.get("document", {}).get("text", "")] for doc in rerank_docs]
        logger.info("Reranked doc ids: %s", reranked_ids)
        return [doc.get("document", {}).get("text", "") for doc in rerank_docs]

    def get_name(self) -> str:
        return self.name
