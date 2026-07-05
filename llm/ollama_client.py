from typing import Type

from openai import BaseModel, OpenAI

from llm.base import BaseLLM


class OllamaClient(BaseLLM):

    def __init__(self,
                 model="qwen2.5:3b",
                 base_url="http://localhost:11434/v1"):

        self.model = model

        self.client = OpenAI(
            api_key="ollama",
            base_url=base_url
        )

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 response_model: Type[BaseModel]) -> str:

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
        return content