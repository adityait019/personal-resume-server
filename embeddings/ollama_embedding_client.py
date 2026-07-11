from openai import OpenAI

from embeddings.base import BaseEmbeddingClient


class OllamaEmbeddingClient(BaseEmbeddingClient):
    """For fully-local pipelines (paired with OllamaClient as the LLM).
    Requires an embedding-capable model pulled in Ollama, e.g.
    `ollama pull nomic-embed-text`.
    """

    def __init__(self,
                 model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434/v1"):

        self.model = model
        self.client = OpenAI(api_key="ollama", base_url=base_url)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Ollama's OpenAI-compatible endpoint doesn't reliably batch,
        # so embed one at a time rather than risk a silent mis-ordering.
        embeddings = []
        for text in texts:
            cleaned = text if text.strip() else " "
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
