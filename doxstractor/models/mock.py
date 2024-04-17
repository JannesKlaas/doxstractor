from .base import BaseModel
from typing import Dict, List, Optional
import numpy as np


class MockModel(BaseModel):

    def complete(
        self,
        query: str,
        context: str,
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        return context.replace("\n", "")

    def model_description(self):
        return {"type": "text", "scores": False}

    def batch_complete(
        self,
        query: str,
        context: List[str],
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        results = []
        for c in context:
            results.append(self.complete(query=query, context=c))
        return results


class MockModelWithScores(BaseModel):

    def complete(
        self,
        query: str,
        context: str,
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        return context.replace("\n", "")

    def model_description(self):
        return {"type": "text", "scores": True}

    def batch_complete_with_scores(
        self,
        query: str,
        context: List[str],
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:

        results = []
        for i, c in enumerate(context):
            res = {
                "score": np.log10(i + 1),
                "answer": self.complete(query=query, context=c),
            }
            results.append(res)
        return results
