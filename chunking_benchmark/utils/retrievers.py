from typing import Any, Dict, List
import json


def format_vectors(
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    exclude_text_from_metadata: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    if exclude_text_from_metadata:
        metadatas = [{k: v for k, v in metadata.items() if k != "text"} for metadata in metadatas]
    return [
        {
            "id": id,
            "values": emb,
            "metadata": metadata,
        }
        for emb, id, metadata in zip(embeddings, ids, metadatas)
    ]


def read_chunk_from_json(file_path: str, num_chunks=None) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data[:num_chunks]
