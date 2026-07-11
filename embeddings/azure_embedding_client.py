from openai import OpenAI

from embeddings.base import BaseEmbeddingClient


class AzureEmbeddingClient(BaseEmbeddingClient):

    def __init__(self,
                 azure_endpoint: str,
                 api_key: str,
                 model: str = "text-embedding-3-small"):
        """
        azure_endpoint must end in /openai/v1/ — same Foundry versionless
        API as AzureOpenAIClient. No api_version needed.
        """
        self.model = model

        if not azure_endpoint.rstrip("/").endswith("/v1"):
            raise ValueError(
                "azure_endpoint must end in /openai/v1/ for the Foundry API "
                f"(got: {azure_endpoint!r})."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=azure_endpoint,
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Azure/OpenAI embeddings choke on empty strings — guard rather
        # than let a blank chunk (e.g. missing professional_summary)
        # blow up an entire batch.
        cleaned = [t if t.strip() else " " for t in texts]

        response = self.client.embeddings.create(
            model=self.model,
            input=cleaned
        )

        # API preserves input order, but sort by index defensively.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]