from ..utils import most_common
from ..models import BaseModel
from .base import BaseExtractor
from typing import List, Optional
import re


class CategoryExtractor(BaseExtractor):
    def __init__(
        self,
        name: str,
        query: str,
        categories: List[str],
        model: BaseModel,
        max_chunk_size: float = 10_000,
        first_chunk_only: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            query=query,
            max_chunk_size=max_chunk_size,
            model=model,
            first_chunk_only=first_chunk_only,
        )
        self.categories = categories

    def extract(self, doc_text: str) -> str:
        merged_chunks = self._chunk_text(doc_text)
        categories_str = "The possible categories are " + ", ".join(
            [f'"{w}"' for w in self.categories]
        )
        snip_messages = []

        # Classifier models don't have actual queries, you just provide the possible categories.
        if self.model.model_type() == "classifier":
            query = self.categories
        else:
            query = self.query

        for snippet in merged_chunks:

            message = self.model.complete(
                query=query,
                context=snippet,
                task_description=f"Valid categories are: {categories_str} \n Use the information below:",
                system_prompt='Your job is to provide a categorial answer based on provided text. Answer only with the category, and no other text. If there is no relevant information in the text provided, respond with "NA". Do not make things up.',
            )
            snip_messages.append(message)

        valid_answers = [
            m for m in snip_messages if (m != "NA") and (m in self.categories)
        ]

        if len(valid_answers) > 0:
            consensus = most_common(valid_answers)
        else:
            consensus = "NA"

        return consensus
