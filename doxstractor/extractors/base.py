from ..models import BaseModel
from typing import List


class BaseExtractor:
    def __init__(
        self,
        name: str,  # Has to be unique within graph as it identifies extractor
        query: str,
        model: BaseModel,
        max_chunk_size: float = 10_000,
    ) -> None:
        """Create a new extractor

        Args:
            name (str): A unique name. This identifies the extractor within a graph and provides the attribute name.
            model (BaseModel): The natural language model which is used to extract text.
            max_chunk_size (float, optional): Maximum size to chunk data into.. Defaults to 10_000.
        """
        # TODO: This maybe should be a model attribute.
        self.max_chunk_size = max_chunk_size
        self.model = model
        self.query = query
        self.name = name

    def _chunk_text(self, doc_text: str) -> List[str]:
        """Splits a document by newlines. Then generates chunks which are shorter than
        max_chunk_size.

        Args:
            doc_text (str): The full document to chunk.

        Returns:
            List[str]: A list of chunks
        """
        chunks = doc_text.split("\n")

        merged_chunks = [""]
        for chunk in chunks:
            prev_chunk = merged_chunks[-1]
            if (len(prev_chunk) + len(chunk)) < (self.max_chunk_size - 1):
                merged_chunks[-1] = prev_chunk + "\n" + chunk
            else:
                merged_chunks.append(chunk)

        return merged_chunks

    def extract(self, doc_text: str):
        raise NotImplementedError
