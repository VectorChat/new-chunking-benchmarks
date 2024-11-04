import argparse
import asyncio
import os
from typing import Any, Coroutine
from ai21 import AI21Client
import nltk

from chunking_benchmark.chunkers.base import BaseChunkRunner

from dotenv import load_dotenv

load_dotenv(override=True)

client = AI21Client(api_key=os.getenv("AI21_API_KEY"))

from langchain_ai21 import AI21SemanticTextSplitter

class AI21ChunkRunner(BaseChunkRunner):
    def __init__(self, chunk_size: int, min_chunk_size: int = 0):
        super().__init__("ai21", chunk_size)
        self.min_chunk_size = min_chunk_size
        
    async def run(self, text: str) -> Coroutine[Any, Any, tuple[list[str], dict]]:
        event_loop = asyncio.get_event_loop()
        
        return await event_loop.run_in_executor(None, self.ai21_chunk_text, text)    

    def ai21_chunk_text(self, text: str):
        semantic_text_splitter = AI21SemanticTextSplitter(chunk_size=self.min_chunk_size)
        
        text_groups = []
        
        api_call_count = 0
        
        # api is limited to 100k chars per request
        if len(text) > 100_000:
            # batch by sentences
            sentences = nltk.sent_tokenize(text)
            sentence_lengths = [len(sentence) for sentence in sentences]
            
            cur_text = ""
            
            for i, sentence in enumerate(sentences):
                if len(cur_text) + sentence_lengths[i] > 100_000:
                    text_groups.append(cur_text)
                    cur_text = ""
                
                cur_text += sentence + " "
                
            
            if cur_text != "":
                text_groups.append(cur_text)
        else:
            text_groups.append(text)        
                
        print(f"Split into {len(text_groups)} text groups")
        print([len(text_group) for text_group in text_groups])
        
        all_chunks = []
                            
        for text_group in text_groups:
            chunks = semantic_text_splitter.split_text(text_group)
            api_call_count += 1
            all_chunks.extend(chunks)
            
        return all_chunks, {"api_call_count": api_call_count}
    
parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=400)
parser.add_argument("--min_chunk_size", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    
    text = "This is a sample text. It contains multiple sentences. We will use this to test our chunking algorithm. The algorithm should split this text into chunks based on the specified size and overlap."    

    runner = AI21ChunkRunner(chunk_size=args.chunk_size)
    
    chunks, metadata = runner.ai21_chunk_text(text)

    print(metadata)

    print("Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")
        print(f"Length: {len(chunk)}")
        print()