from langchain_core.pydantic_v1 import BaseModel, Field
from enum import Enum
from typing import Literal


class ChunkingMethod(Enum):
    UNSTRUCTURED = "unstructured"
    AI21 = "ai21"
    CUSTOM = "custom"
    SUBNET_API = "subnet-api"


class VectorDB(Enum):
    PINECONE = "pinecone"
    ASTRA = "astra"


class EmbeddingModel(Enum):
    HF_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
    OPENAI_LARGE = "text-embedding-3-small"


class FictionAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D", "E", "F", "G", "H"] = Field(
        description="The correct answer to the multiple choice question"
    )
    confidence: float = Field(description="Confidence in the answer choice", ge=0, le=1)
    reasoning: str = Field(description="One sentence reasoning for the answer choice")
