from ..utils import most_common
from ..models import BaseModel
from .base import BaseExtractor
from typing import List, Optional
import re
import numpy as np

# TODO: Need to find a way to pull the task description which depends on the categories up here
SYSTEM_PROMPT = 'Your job is to provide a categorial answer based on provided text. Answer only with the category, and no other text. If there is no relevant information in the text provided, respond with "NA". Do not make things up.'


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
        TASK_DESCRIPTION = (
            f"Valid categories are: {categories_str} \n Use the information below:"
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
                task_description=TASK_DESCRIPTION,
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
                task_description=TASK_DESCRIPTION,
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
