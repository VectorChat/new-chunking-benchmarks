from abc import ABC, abstractmethod
from typing import Any, Coroutine

class BaseChunkRunner(ABC):
    def __init__(self, name: str, chunk_size: int):
        self.name = name
        self.chunk_size = chunk_size
        
    @abstractmethod
    async def run(self, text: str) -> Coroutine[Any, Any, tuple[list[str], dict]]:
        """
        Returns a list of chunks and a dictionary of metadata
        """
        raise NotImplementedError
    
    def get_name(self) -> str:        
        return self.name