from typing import Optional, List, Dict


class BaseModel:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1_000,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(
        self,
        query: str,
        context: str,
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        raise NotImplementedError

    def model_description(self):
        raise NotImplementedError

    def batch_complete(
        self,
        query: str,
        context: List[str],
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        raise NotImplementedError

    def batch_complete_with_scores(
        self,
        query: str,
        context: List[str],
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        raise NotImplementedError
