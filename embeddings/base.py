from abc import ABC, abstractmethod


class BaseEmbeddingClient(ABC):
    """Common interface so the ingestion service doesn't care which
    embedding provider (Azure OpenAI, OpenAI, local sentence-transformers,
    Ollama embeddings, etc.) is behind it."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Implementations should batch the
        underlying API call where the provider supports it, rather than
        looping embed() one at a time."""
        raise NotImplementedError
