from .base import BaseModel
from typing import Dict, List, Optional
import requests
import os


class HFEndPointQAModel(BaseModel):
    def __init__(
        self,
        model: str,
        api_token: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1000,
    ) -> None:
        """Huggingface model using hosted endpoints. Requires a valid token.

        Args:
            model (str): The endpoint URL
            api_token (Optional[str], optional): Huggingface token. Defaults to HF_API_TOKEN environment variable.
            temperature (float, optional): For compatibility reasons only. Defaults to 0.
            max_tokens (int, optional): For compatibility reasons only. Defaults to 1000.

        Raises:
            ValueError: If no token is provided.
        """
        super().__init__(model, temperature, max_tokens)

        if api_token:
            self.api_token = api_token
        else:
            env_token = os.environ.get("HF_API_TOKEN")
            if env_token:
                self.api_token = env_token
            else:
                raise ValueError(
                    "The huggingface API requires a token. Either pass an API token or set the HF_API_TOKEN environment variable"
                )

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
        payload = {"inputs": input}

        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = requests.post(self.model, headers=headers, json=payload)
        return response.json()

    def batch_complete_with_scores(
        self,
        query: str,
        context: List[str],
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        all_results = []
        for c in context:
            result = self.complete(
                query=query,
                context=c,
                task_description=task_description,
                system_prompt=system_prompt,
            )
            all_results.append(result)
        return all_results
