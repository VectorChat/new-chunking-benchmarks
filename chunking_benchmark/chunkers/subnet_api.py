import os
import requests
import asyncio
from chunking_benchmark.chunkers.base import BaseChunkRunner
import logging
import time

from dotenv import load_dotenv

load_dotenv(override=True)


class SubnetAPIChunker(BaseChunkRunner):
    def __init__(
        self,
        max_chunk_size_chars: int,
        max_num_chunks: int=None,
        chunker_name: str="subnet_api",
        embedding_model: str = "text-embedding-3-small",
        **kwargs,
    ):
        super().__init__(chunker_name, max_chunk_size_chars)
        self.max_chunk_size_chars = max_chunk_size_chars
        self.max_num_chunks = max_num_chunks
        self.embedding_model = embedding_model

    async def run(self, text: str):
        event_loop = asyncio.get_event_loop()
        return await event_loop.run_in_executor(None, self.subnet_api_chunk_text, text)

    def subnet_api_chunk_text(self, text: str):

        URL = f"{os.getenv("SUBNET_API_URL")}{os.getenv("SUBNET_API_ENDPOINT")}"
        headers = {"Content-Type": "application/json"}
        data = {
            "text": text,
            "method": os.getenv("SUBNET_API_METHOD"),
            "max_chunk_size_chars": self.max_chunk_size_chars,
            "max_num_chunks": self.max_num_chunks,
            "embedding_model": self.embedding_model,
        }
        start_time = time.time()
        try:
            response = requests.post(URL, headers=headers, json=data)
            response.raise_for_status()
            chunks = response.json()["chunks"]
            response_running_time = response.json().get("time_taken", None)
        except Exception as e:
            print(f"Error in subnet api request: {e}")
            logging.exception(e)
            return [], {}
        running_time = time.time() - start_time
        logging.info(f"Server running time: {response_running_time}s")
        logging.info(f"Client running time: {running_time}s")
        return chunks, {"client_running_time": running_time, "server_running_time": response_running_time}


if __name__ == "__main__":
    with open("test.txt", "r") as f:
        text = f.read()
    chunker = SubnetAPIChunker(1000, 10000)
    chunks, metadata = chunker.subnet_api_chunk_text(text)
