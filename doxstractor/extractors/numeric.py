from ..utils import parseNumber, most_common
from ..models import BaseModel
from .base import BaseExtractor
from typing import List, Optional
import re


class NumericExtractor(BaseExtractor):

    def extract(self, doc_text: str) -> float:

        merged_chunks = self._chunk_text(doc_text)
        snip_messages = []
        for snippet in merged_chunks:

            message = self.model.complete(
                query=self.query,
                context=snippet,
                task_description="Use the information given below.",
                system_prompt='Your job is to extract a numerical value from a document. Respond with a single number. Do not explain your answer. Do not provide context. If there is no relevant information in the text provided, respond with "NA". Do not make things up.',
            )

            snip_messages.append(message)

        valid_answers = [m for m in snip_messages if m != "NA"]

        numerized = []
        for answer in valid_answers:

            num = "".join(
                re.findall(
                    "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", answer
                )
            )
            if num != "":
                numerized.append(num)

        if len(numerized) > 0:
            consensus = most_common(numerized)
        else:
            consensus = "NA"

        return consensus
