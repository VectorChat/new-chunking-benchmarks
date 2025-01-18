import argparse
import asyncio
import os
import json
import nltk
from typing import Any, Coroutine
from ai21 import AI21Client
from ai21.models.chat import UserMessage


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

    def ai21_segment_via_prompt(self, text: str, avg_size_in_word: int = 125, model : str="jamba-1.5-large", **kwargs):
        print(f"Splitting text of length {len(text)} into segments of avg size {avg_size_in_word}")
        prompt = (
            f"Split the text below into segments by semantic. Extract text only with no additional characters or explanations. Segment length should be around {avg_size_in_word} words.\n\n"
            f"Format as a list of segments where each segment as a JSON object with 'segmentText' and 'segmentType':\n\n"
            f'"""\n{text}\n"""\n\nSegments:\n'
        )
        segments = []
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=prompt)],
                top_p=0.01,
            )
            segments = json.loads(response.choices[0].message.content)
            print(
                f"Extracted {len(segments)} segments, avg {sum([len(_['segmentText']) for _ in segments])/len(segments):.0f} tokens"
            )
            print(f'Integrity: {[seg["segmentText"] in text for seg in segments]}')

        except Exception as e:
            print(e)
        return [_['segmentText'] for _ in segments]

    def ai21_chunk_text(self, text: str):
        semantic_text_splitter = AI21SemanticTextSplitter(chunk_size=self.min_chunk_size)
        
        text_groups = []
        
        api_call_count = 0
        
        # api is limited to 10k chars per request
        if len(text) > 10_000:
            # batch by sentences
            sentences = nltk.sent_tokenize(text)
            sentence_lengths = [len(sentence) for sentence in sentences]
            
            cur_text = ""
            
            for i, sentence in enumerate(sentences):
                if len(cur_text) + sentence_lengths[i] > 10_000:
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
            # chunks = semantic_text_splitter.split_text(text_group)
            chunks = self.ai21_segment_via_prompt(text_group, avg_size_in_word=self.chunk_size)
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