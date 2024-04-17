from ..utils import parseNumber, most_common
from ..models import BaseModel
from .base import BaseExtractor
from typing import List, Optional
import re
import numpy as np

TASK_DESCRIPTION = "Use the information given below."

SYSTEM_PROMPT = 'Answer only with the relevant text snippet you have found below, and no other text. Do not explain your answer or provide any context. If there is no relevant information in the text provided, respond with "NA". Do not make things up.'


class TextExtractor(BaseExtractor):

    def extract(self, doc_text: str) -> str:
        merged_chunks = self._chunk_text(doc_text)

        if self.model.model_description()["scores"]:
            results_with_scores = self.model.batch_complete_with_scores(
                query=self.query,
                context=merged_chunks,
                task_description=TASK_DESCRIPTION,
                system_prompt=SYSTEM_PROMPT,
            )
            all_scores = [r["score"] for r in results_with_scores]
            idx = np.argmax(all_scores)
            consensus = results_with_scores[idx]["answer"]
            return consensus
        else:
            results = self.model.batch_complete(
                query=self.query,
                context=merged_chunks,
                task_description=TASK_DESCRIPTION,
                system_prompt=SYSTEM_PROMPT,
            )
            valid_answers = []
            valid_indices = []
            for idx, answer in enumerate(results):
                if answer != "NA":
                    valid_answers.append(answer)
                    valid_indices.append(idx)

            if len(valid_answers) > 0:
                consensus = most_common(valid_answers)
            else:
                consensus = "NA"
            return consensus
