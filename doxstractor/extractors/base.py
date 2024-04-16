from ..models import BaseModel
from typing import List


class BaseExtractor:
    def __init__(
        self,
        name: str,  # Has to be unique within graph as it identifies extractor
        query: str,
        model: BaseModel,
        max_chunk_size: float = 10_000,
        first_chunk_only: bool = False,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.model = model
        self.query = query
        self.name = name
        self.first_chunk_only = first_chunk_only

    def _chunk_text(self, doc_text: str) -> List[str]:
        chunks = doc_text.split("\n")

        merged_chunks = [""]
        for chunk in chunks:
            prev_chunk = merged_chunks[-1]
            if (len(prev_chunk) + len(chunk)) < (self.max_chunk_size - 1):
                merged_chunks[-1] = prev_chunk + "\n" + chunk
            else:
                merged_chunks.append(chunk)

        # In some cases it can be useful to only provide the opening of the document
        # E.g. when classifying the document
        if self.first_chunk_only:
            return [merged_chunks[0]]

        return merged_chunks

    def extract(self, doc_text: str):
        raise NotImplementedError
