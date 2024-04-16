from ..utils import parseNumber, most_common
from ..models import BaseModel
from .base import BaseExtractor
from typing import List, Optional
import re


class TextExtractor(BaseExtractor):

    def extract(self, doc_text: str) -> str:
        merged_chunks = self._chunk_text(doc_text)
        snip_messages = []
        for snippet in merged_chunks:

            message = self.model.complete(
                query=self.query,
                context=snippet,
                task_description="Use the information given below.",
                system_prompt='Answer only with the relevant text snippet you have found below, and no other text. Do not explain your answer or provide any context. If there is no relevant information in the text provided, respond with "NA". Do not make things up.',
            )
            snip_messages.append(message)

        valid_answers = []
        valid_indices = []
        for idx, answer in enumerate(snip_messages):
            if answer != "NA":
                valid_answers.append(answer)
                valid_indices.append(idx)

        if len(valid_answers) > 0:
            consensus = most_common(valid_answers)
        else:
            consensus = "NA"
        return consensus
