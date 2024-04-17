from .base import BaseModel
from transformers import pipeline
from typing import Optional, List, Dict
import torch
import numpy as np


class TransformersQAModel(BaseModel):

    def __init__(
        self,
        model: str,
        device: int = 1,
        na_threshold: float = 0.5,
        temperature: float = 0,
        max_tokens: int = 1000,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "question-answering", model=model, tokenizer=model, device=device
        )
        self.na_threshold = na_threshold

    def model_description(self):
        return {"type": "text", "scores": True}

    def complete(
        self,
        query: str,
        context: str,
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):

        input = {"question": query, "context": context}
        res = self.pipeline(input)

        if res["score"] >= self.na_threshold:
            return res["answer"]
        else:
            return "NA"

    def batch_complete_with_scores(
        self,
        query: str,
        context: List[str],
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        all_inputs = []
        for c in context:
            all_inputs.append({{"question": query, "context": c}})

        all_results = self.pipeline(all_inputs)
        return all_results
