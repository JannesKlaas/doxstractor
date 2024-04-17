from ..utils import most_common

from .base import BaseExtractor

import re
import numpy as np

TASK_DESCRIPTION = "Use the information given below."
SYSTEM_PROMPT = 'Your job is to extract a numerical value from a document. Respond with a single number. Do not explain your answer. Do not provide context. If there is no relevant information in the text provided, respond with "NA". Do not make things up.'


class NumericExtractor(BaseExtractor):

    def _numerize(self, answer: str):
        num = "".join(
            re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", answer)
        )
        return num

    def extract(self, doc_text: str) -> float:
        """Extracts a number from a document.

        Args:
            doc_text (str): The document text from which to extract.

        Returns:
            float: The extracted number.
        """

        merged_chunks = self._chunk_text(doc_text)

        if self.model.model_description()["scores"]:
            results_with_scores = self.model.batch_complete_with_scores(
                query=self.query,
                context=merged_chunks,
                task_description=TASK_DESCRIPTION,
                system_prompt=SYSTEM_PROMPT,
            )
            filtered = [
                r for r in results_with_scores if (self._numerize(r["answer"]) != "")
            ]
            all_scores = [r["score"] for r in filtered]
            idx = np.argmax(all_scores)
            consensus = self._numerize(results_with_scores[idx]["answer"])
            return consensus
        else:
            results = self.model.batch_complete(
                query=self.query,
                context=merged_chunks,
                task_description=TASK_DESCRIPTION,
                system_prompt=SYSTEM_PROMPT,
            )
            valid_answers = [m for m in results if m != "NA"]

            numerized = []
            for answer in valid_answers:

                num = self._numerize(answer)
                if num != "":
                    numerized.append(num)

            if len(numerized) > 0:
                consensus = most_common(numerized)
            else:
                consensus = "NA"

            return consensus
