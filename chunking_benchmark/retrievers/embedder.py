from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv


class BaseEmbedder(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    def embed_chunk(self, text):
        if not self.embeddings:
            return None
        return self.embeddings.embed_query(text)

    def embed_chunk_s(self, text_s):
        if not self.embeddings:
            return None
        print(f"Embedding {len(text_s)} documents via {self.embeddings.__class__.__name__}")
        res = self.embeddings.embed_documents(text_s)
        print([len(x) for x in res])
        return res


class HFEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
        **kwargs,
    ):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
        except Exception as e:
            self.embeddings = None
            print(f"Error loading HuggingFaceEmbeddings: {e}")


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model="text-embedding-3-small", **kwargs):
        load_dotenv(override=True)
        try:
            from langchain_openai import OpenAIEmbeddings

            self.embeddings = OpenAIEmbeddings(model=model)
        except Exception as e:
            self.embeddings = None
            print(f"Error loading OpenAIEmbeddings: {e}")
