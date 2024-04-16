import anthropic
from .base import BaseModel
from typing import Optional
import time


class AnthropicAPIModel(BaseModel):
    def __init__(
        self,
        model: Optional[str] = "claude-3-haiku-20240307",
        temperature: float = 0.0,
        max_tokens: int = 1_000,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        self.client = anthropic.Anthropic()

    def model_type(self):
        return "text"

    def complete(
        self,
        query: str,
        context: str,
        task_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        if task_description:
            user_prompt = query + "\n" + task_description + "\n" + context
        else:
            user_prompt = query + "\n" + context

        for attempt in range(10):
            try:
                message = self._query_anthropic(system_prompt, user_prompt)
                return message.content[0].text
            except anthropic.APIConnectionError as e:
                print("The anthropic API could not be reached")
                print(e.__cause__)

            except anthropic.RateLimitError as e:
                print(
                    "A 429 status code was received; sleeping 60s to reset rate limit"
                )
                time.sleep(60)

            except anthropic.APIStatusError as e:
                print("Another non-200-range status code was received")
                print(e.status_code)
                print(e.response)
            else:
                break

        return message.content[0].text

    def _query_anthropic(self, system_prompt, user_prompt):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ],
                }
            ],
        )

        return message
