from .base import BaseModel
from typing import Optional


class MockModel(BaseModel):

    def complete(
        self,
        query: str,
        context: str,
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        return context.replace("\n", "")

    def model_type(self):
        return "text"
