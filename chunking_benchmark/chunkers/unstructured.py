import argparse
import asyncio
import os
from tabulate import tabulate

import unstructured_client
from unstructured_client.models import operations, shared

from chunking_benchmark.chunkers.base import BaseChunkRunner
from chunking_benchmark.utils.chunks import get_chunk_stats, save_chunk_size_distribution
from dotenv import load_dotenv

load_dotenv(override=True)

client = unstructured_client.UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)

def unstructured_io_chunker(filename: str, max_chunk_size_chars: int, soft_min: int | None = None, similarity_threshold: float | None = None) -> list[str]:
  with open(filename, "r") as f:
    content = f.read()
    
  print(f"running unstructured_io_chunker with {filename}, args: max_chunk_size_chars={max_chunk_size_chars}, soft_min={soft_min}, similarity_threshold={similarity_threshold}")

  # maybe use `max_characters` and/or `combine_under_n_chars` for chunk_size? https://github.com/Unstructured-IO/unstructured-python-client/blob/main/src/unstructured_client/models/shared/partition_parameters.py#L64
  req = operations.PartitionRequest(
      partition_parameters=shared.PartitionParameters(
          files=shared.Files(
              content=content,
              file_name=filename,
          ),
          strategy=shared.Strategy.FAST,
          chunking_strategy=shared.ChunkingStrategy.BY_SIMILARITY,
          languages=['eng'],
          combine_under_n_chars=soft_min,
          new_after_n_chars=max_chunk_size_chars,
          max_characters=max_chunk_size_chars,
          similarity_threshold=similarity_threshold,
      ),
      )

  try:
    res = client.general.partition(request=req)
    elements = res.elements
    return [element['text'] for element in elements]
  except Exception as e:
    print(e)

class UnstructuredChunkerRunner(BaseChunkRunner):
    def __init__(self, chunk_size: int, chunk_size_char_multiplier: int, text_file: str, soft_min: int | None = None, similarity_threshold: float | None = None):
        super().__init__("unstructured_chunker", -1)
        self.chunk_size = chunk_size
        self.chunk_size_char_multiplier = chunk_size_char_multiplier
        self.soft_min = soft_min
        self.similarity_threshold = similarity_threshold
        self.text_file = text_file        
    
    async def run(self, text: str) -> tuple[list[str], dict]:
        event_loop = asyncio.get_event_loop()
        chunks = await event_loop.run_in_executor(None, unstructured_io_chunker, self.text_file, self.chunk_size * self.chunk_size_char_multiplier, self.soft_min, self.similarity_threshold)
        return chunks, {}  