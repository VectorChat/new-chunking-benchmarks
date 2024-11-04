import cohere
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper()))
logger = logging.getLogger(__name__)


class CohereReranker:
    def __init__(self):
        load_dotenv(override=True)
        self.client = cohere.Client(os.getenv("COHERE_API_KEY"))

    def rerank(self, res, query, top_n=3, model="rerank-english-v3.0"):
        if "matches" not in res or not res["matches"]:
            logger.error(f"res: {res}")
            logger.error(f"query: {query}")
            logger.error("Response does not contain any matches.")
            return []
        docs = {x["metadata"]["text"]: i for i, x in enumerate(res["matches"])}
        docs_text = [doc for doc in docs.keys()]
        try:
            logger.info(f"Reranking {len(docs)} documents")
            rerank_docs = self.client.rerank(
                query=query,
                documents=[doc for doc in docs.keys()],
                top_n=top_n,
                model=model,
                max_chunks_per_doc=1,
                return_documents=True,
            ).results
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return ""

        reranked_ids = [docs[doc.document.text] for doc in rerank_docs]
        logger.info("Reranked doc ids: %s", reranked_ids)

        return [doc.document.text for doc in rerank_docs]

    def _rerank(self, docs, query, top_n=3, model="rerank-english-v3.0"):
        try:
            res = self.client.rerank(
                query=query,
                documents=docs,
                top_n=top_n,
                model=model,
                max_chunks_per_doc=1,
                return_documents=True,
            ).results
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return ""
        return res


if __name__ == "__main__":
    reranker = CohereReranker()
    query = "What is the capital of the United States?"
    docs = [
        "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
    ]
    reranked = reranker._rerank(docs=docs, query=query)
    print(f"reranked: {reranked}")
