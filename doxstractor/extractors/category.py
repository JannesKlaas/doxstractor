from ..utils import most_common
from ..models import BaseModel
from .base import BaseExtractor
from typing import List

import numpy as np


TASK_DESCRIPTION = (
    "Valid categories are: {categories_str} \n Use the information below:"
)
SYSTEM_PROMPT = 'Your job is to provide a categorial answer based on provided text. Answer only with the category, and no other text. If there is no relevant information in the text provided, respond with "NA". Do not make things up.'


class CategoryExtractor(BaseExtractor):
    def __init__(
        self,
        name: str,
        query: str,
        categories: List[str],
        model: BaseModel,
        max_chunk_size: float = 10_000,
    ) -> None:
        """Create a new extractor

        Args:
            name (str): A unique name. This identifies the extractor within a graph and provides the attribute name.
            categories (list): Allowed categories.
            model (BaseModel): The natural language model which is used to extract text.
            max_chunk_size (float, optional): Maximum size to chunk data into.. Defaults to 10_000.
        """
        super().__init__(
            name=name,
            query=query,
            max_chunk_size=max_chunk_size,
            model=model,
        )
        self.categories = categories

    def extract(self, doc_text: str) -> str:
        """Extracts a category from a document, similar to zero shot classification.

        Args:
            doc_text (str): The document from which to extract

        Returns:
            str: The extracted category. Guaranteed to be one of the categories provided on init or 'NA'
        """

        merged_chunks = self._chunk_text(doc_text)
        categories_str = "The possible categories are " + ", ".join(
            [f'"{w}"' for w in self.categories]
        )

        # Classifier models don't have actual queries, you just provide the possible categories.
        if self.model.model_description()["type"] == "classifier":
            query = self.categories
        else:
            query = self.query

        if self.model.model_description()["scores"]:
            results_with_scores = self.model.batch_complete_with_scores(
                query=query,
                context=merged_chunks,
                task_description=TASK_DESCRIPTION.format(categories_str=categories_str),
                system_prompt=SYSTEM_PROMPT,
            )
            filtered = [
                r for r in results_with_scores if (r["answer"] in self.categories)
            ]
            all_scores = [r["score"] for r in filtered]
            idx = np.argmax(all_scores)
            consensus = results_with_scores[idx]["answer"]
            return consensus
        else:
            results = self.model.batch_complete(
                query=query,
                context=merged_chunks,
                task_description=TASK_DESCRIPTION.format(categories_str=categories_str),
                system_prompt=SYSTEM_PROMPT,
            )
            valid_answers = [
                m for m in results if (m != "NA") and (m in self.categories)
            ]

            if len(valid_answers) > 0:
                consensus = most_common(valid_answers)
            else:
                consensus = "NA"

            return consensus
