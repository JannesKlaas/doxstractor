from .base import BaseModel
from transformers import pipeline
from typing import Dict, Optional, List
import numpy as np


class TransformerClassifierModel(BaseModel):
    def __init__(
        self, model: str, temperature: float = 0, max_tokens: int = 1000
    ) -> None:
        super().__init__(model, temperature, max_tokens)

        self.classifier = pipeline("zero-shot-classification", model=model)

    def model_description(self):
        return {"type": "classifier", "scores": True}

    def complete(
        self,
        query: List[str],
        context: str,
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        res = self.classifier(context, query)
        best_answer_idx = np.argmax(res["scores"])
        return res["labels"][best_answer_idx]

    def batch_complete_with_scores(
        self,
        query: str,
        context: List[str],
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        # TODO: Implement this
        return super().batch_complete_with_scores(
            query, context, task_description, system_prompt
        )
