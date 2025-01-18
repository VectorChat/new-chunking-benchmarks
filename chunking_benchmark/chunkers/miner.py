import asyncio
import json
from math import ceil
import bittensor as bt

from typing import Any, Coroutine
from typing import Optional, List
from chunking_benchmark.chunkers.base import BaseChunkRunner

class chunkSynapse(bt.Synapse):
    """
    A simple chunking synapse protocol representation which uses bt.Synapse as its base.
    This protocol enables communication between the miner and the validator.

    Attributes:
    - document: str containing plaintext to be split by the miner.
    - chunk_size: int containing the soft max characters per chunk.
    - time_soft_max: float containing the maximum time the miner can take before being penalized.
    - chunks: List[str] containing chunks of text from document.
    - miner_signature: str containing the miner's signature of a json object containing document, chunk_size, and chunks.
    """

    name: str = 'chunkSynapse'
    
    # Required request input, filled by sending dendrite caller. It is a base64 encoded string.
    document: str
    chunk_size: int = None
    chunk_qty: int = None
    time_soft_max: float = None

    # Optional request output, filled by recieving axon.
    chunks: Optional[List[str]] = None
    miner_signature: Optional[str] = None

    def deserialize(self) -> List[str]:
        return self.chunks

# Global metagraph for netuid 40
metagraph: bt.metagraph | None = None

def get_metagraph():
    global metagraph
    if not metagraph:
        metagraph = bt.metagraph(netuid=40)
    return metagraph

class MinerChunkRunner(BaseChunkRunner):
    def __init__(self, uid: int, chunk_size: int, vali_wallet: str, vali_hotkey: str, timeout = 60, chunk_size_char_multiplier = 4, custom_name: str = None):
        super().__init__(custom_name or f"miner_{uid}", chunk_size)
        self.uid = uid
        self.axon = get_metagraph().axons[uid]
        self.vali_wallet = bt.wallet(vali_wallet, vali_hotkey)
        self.dendrite = bt.dendrite(self.vali_wallet)
        self.timeout = timeout
        self.chunk_size_char_multiplier = chunk_size_char_multiplier  
        self.custom_name = custom_name      
    
    def get_name(self) -> str:        
        return self.name
    
    async def run(self, text: str) -> Coroutine[Any, Any, tuple[list[str], dict]]:        
        
        chunk_qty = ceil((len(text) / self.chunk_size * self.chunk_size_char_multiplier) * 1.5)
        
        chunk_size_chars = ceil(self.chunk_size * self.chunk_size_char_multiplier)
        
        # Create the synapse
        synapse = chunkSynapse(
            document=text,
            chunk_size=chunk_size_chars,
            chunk_qty=chunk_qty,
            time_soft_max=self.timeout * 0.75,
            timeout=self.timeout
        )
        
        print(f"Querying miner {self.uid} with chunk size {chunk_size_chars} and chunk qty {chunk_qty}, timeout {self.timeout}")
        
        # Use run_in_executor to run the synchronous query in a separate thread
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.dendrite.query(
                axons=[self.axon],
                synapse=synapse,
                deserialize=False,
                timeout=synapse.timeout
            )
        )
        
        
                            
        print(f"Received response from miner {self.uid}")
        
        if not response or not response[0].chunks:
            print(f"No response or empty chunks from miner {self.uid}")
            return [], {}
        
        raw_chunks = response[0].chunks
        
        return raw_chunks, {}