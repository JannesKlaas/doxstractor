from .base import BaseModel
from transformers import pipeline
from typing import Optional, List, Dict
import torch


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
    ) -> str:
        """Extracts a snippet of text to answer the query.

        Args:
            query (str): The query specifying what we want to extract.
            context (List[str]): The text from which to extract.
            task_description (Optional[str], optional): Not used, only for compatibility. Defaults to None.
            system_prompt (Optional[str], optional): Not used, only for compatibility. Defaults to None.

        Returns:
            str: The extracted response.
        """
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
        """Extracts a snippet of text to answer the query for each context provided.
        Returns a dictionary including answers and confidence scores.

        Args:
            query (str): The query specifying what we want to extract.
            context (List[str]): The text from which to extract. Model will answer query for each element of list.
            task_description (Optional[str], optional): Not used, only for compatibility. Defaults to None.
            system_prompt (Optional[str], optional): Not used, only for compatibility. Defaults to None.

        Returns:
            List[Dict]: A list of dictionaries like {'score':confidence_score, 'answer':answer_text}
        """
        all_inputs = []
        for c in context:
            all_inputs.append({"question": query, "context": c})

        all_results = self.pipeline(all_inputs)
        return all_results
