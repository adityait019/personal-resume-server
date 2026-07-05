from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type


class BaseLLM(ABC):

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, response_model: Type[BaseModel]):
        pass