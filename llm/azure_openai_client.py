from openai import OpenAI
from pydantic import BaseModel
from typing import Type, Any
from llm.base import BaseLLM

class AzureOpenAIClient(BaseLLM):

    _REASONING_MODEL_MARKERS = ("gpt-5", "o1", "o3", "o4")

    def __init__(self,
                 azure_endpoint="https://your-resource-name.services.ai.azure.com/openai/v1/",
                 api_key="your-api-key",
                 model="your-deployment-name",
                 is_reasoning_model: bool | None = None):
        """
        azure_endpoint must end in /openai/v1/ — this targets the newer
        versionless Foundry API (services.ai.azure.com or openai.azure.com
        resources both work with this path). No api_version needed here;
        that param belongs to the older AzureOpenAI SDK class, which
        expects the classic /openai/deployments/{name}/... URL shape and
        404s against Foundry resources.
        """

        self.model = model

        # Deployment names are user-defined (e.g. "prod-parser-v2"), so they
        # don't reliably tell you the underlying model. Auto-detect from
        # common substrings, but let the caller override explicitly since
        # that's the only way to be certain.
        self._is_reasoning_model_override = is_reasoning_model

        if not azure_endpoint.rstrip("/").endswith("/v1"):
            raise ValueError(
                "azure_endpoint must end in /openai/v1/ for the Foundry API "
                f"(got: {azure_endpoint!r}). Check your AZURE_OPENAI_ENDPOINT."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=azure_endpoint,
        )

    def _is_reasoning_model(self, model_name: str) -> bool:
        if self._is_reasoning_model_override is not None:
            return self._is_reasoning_model_override
        lowered = model_name.lower()
        return any(marker in lowered for marker in self._REASONING_MODEL_MARKERS)

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 response_model: Type[BaseModel]):

        kwargs: dict[str, Any] = dict(
            model=self.model,
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

        # Reasoning models (gpt-5 family, o1, o3, ...) only support the
        # default temperature (1) — passing temperature=0 throws a 400.
        if not self._is_reasoning_model(self.model):
            kwargs["temperature"] = 0.0

        response = self.client.beta.chat.completions.parse(**kwargs)

        content = response.choices[0].message.parsed
        if content is None:
            raise ValueError("No response received from the LLM.")
        return content