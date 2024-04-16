from .base import BaseModel
from transformers import pipeline
from typing import Optional


class TransformersQAModel(BaseModel):

    def __init__(
        self,
        model: str,
        na_threshold: float = 0.5,
        temperature: float = 0,
        max_tokens: int = 1000,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        self.pipeline = pipeline("question-answering", model=model, tokenizer=model)
        self.na_threshold = na_threshold

    def model_type(self):
        return "text"

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
