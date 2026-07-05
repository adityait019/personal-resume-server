from openai import AzureOpenAI
from pydantic import BaseModel
from typing import Type
from llm.base import BaseLLM

class AzureOpenAIClient(BaseLLM):
    
    def __init__(self,
                 azure_endpoint="https://your-resource-name.openai.azure.com/",
                 api_version="2023-06-01-preview",
                 api_key="your-api-key",
                 model="your-deployment-name",):

        self.model = model

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 response_model: Type[BaseModel]):

        response = self.client.beta.chat.completions.parse( 
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
            response_format=response_model,
 
        )
     
        content = response.choices[0].message.parsed
        if content is None:
            raise ValueError("No response received from the LLM.")
        return content