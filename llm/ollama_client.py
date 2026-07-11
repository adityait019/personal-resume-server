import json
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from llm.base import BaseLLM

T = TypeVar("T", bound=BaseModel)


class OllamaClient(BaseLLM):

    def __init__(self,
                 model="qwen2.5:3b",
                 base_url="http://localhost:11434/v1"):

        self.model = model

        self.client = OpenAI(
            api_key="ollama",
            base_url=base_url
        )

    def generate(self, # type: ignore
                 system_prompt: str,
                 user_prompt: str,
                 response_model: Type[T]) -> T:

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format={
                "type": "json_object"
            }
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No response received from the LLM.")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM did not return valid JSON: {e}\nRaw content: {content!r}"
            )

        try:
            return response_model.model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"LLM JSON did not match {response_model.__name__} schema: {e}"
            )
