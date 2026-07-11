"""
Resume Ingestion Service — orchestrates the full pipeline:

    PDF -> raw text -> ParsedResume -> KnowledgeChunk[] -> embedded chunks -> pgvector
"""

from uuid import UUID, uuid4

from parser.pdf_parser import PDFParser
from parser.resume_parser import ResumeParser
from parser.models import ParsedResume
from chunking.chunk_dataclass import KnowledgeChunk
from chunking.resume_chunk_builder import ResumeChunkBuilder
from embeddings.base import BaseEmbeddingClient
from store.pgvector_store import PgVectorStore
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class ResumeIngestionService:

    def __init__(
        self,
        resume_parser: ResumeParser,
        embedding_client: BaseEmbeddingClient,
        store: PgVectorStore,
        chunk_builder: ResumeChunkBuilder | None = None,
    ):
        self.resume_parser = resume_parser
        self.embedding_client = embedding_client
        self.store = store
        self.chunk_builder = chunk_builder or ResumeChunkBuilder()

    def ingest_pdf(self, pdf_path: str, resume_id: UUID | None = None) -> UUID:
        """Runs the full pipeline for one resume PDF. Returns the resume_id
        chunks were stored under (generates one if not provided), so callers
        can later re-ingest/replace or query by it."""

        resume_id = resume_id or uuid4()

        raw_text = PDFParser(pdf_path).extract_text()

        parsed_resume: ParsedResume = self.resume_parser.parse(raw_text)

        chunks: list[KnowledgeChunk] = self.chunk_builder.build(parsed_resume)

        self._embed_chunks(chunks)

        # Replace any prior chunks for this resume_id so re-ingestion
        # (e.g. an updated resume upload) doesn't leave stale chunks behind.
        self.store.delete_resume(resume_id)
        self.store.upsert_chunks(resume_id, chunks)

        return resume_id

    def _embed_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        embeddable = [c for c in chunks if c.content and c.content.strip()]
        if not embeddable:
            return

        vectors = self.embedding_client.embed_batch(
            [c.content for c in embeddable]
        )
        for chunk, vector in zip(embeddable, vectors):
            chunk.embedding = vector


def _build_llm(backend: str):
    """LLM_BACKEND controls parsing only — independent of embeddings."""
    import os

    if backend == "ollama":
        from llm.ollama_client import OllamaClient
        return OllamaClient(
            model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )

    if backend == "azure":
        from llm.azure_openai_client import AzureOpenAIClient
        return AzureOpenAIClient(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        )

    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected 'azure' or 'ollama')")


def _build_embedding_client(backend: str) -> tuple[BaseEmbeddingClient, int]:
    """EMBEDDING_BACKEND controls embeddings only — independent of the LLM.
    Returns (client, embedding_dim) since the dim must match whatever
    model is actually generating the vectors, not whatever the LLM is."""
    import os

    if backend == "ollama":
        from embeddings.ollama_embedding_client import OllamaEmbeddingClient
        client = OllamaEmbeddingClient(
            model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
        dim = int(os.environ.get("EMBEDDING_DIM", "768"))
        return client, dim

    if backend == "azure":
        from embeddings.azure_embedding_client import AzureEmbeddingClient
        client = AzureEmbeddingClient(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            model=os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        )
        dim = int(os.environ.get("EMBEDDING_DIM", "1536"))
        return client, dim

    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} (expected 'azure' or 'ollama')")


def build_default_service() -> ResumeIngestionService:
    """Wires up a service from environment variables. LLM_BACKEND and
    EMBEDDING_BACKEND are read separately so you can mix providers,
    e.g. Azure for parsing + Ollama for embeddings."""
    import os

    llm_backend = os.environ.get("LLM_BACKEND", "azure")
    embedding_backend = os.environ.get("EMBEDDING_BACKEND", llm_backend)

    llm = _build_llm(llm_backend)
    embedding_client, embedding_dim = _build_embedding_client(embedding_backend)

    resume_parser = ResumeParser(llm)

    store = PgVectorStore(
        dsn=os.environ["POSTGRES_DSN"],
        embedding_dim=embedding_dim,
    )

    return ResumeIngestionService(
        resume_parser=resume_parser,
        embedding_client=embedding_client,
        store=store,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python service.py <path_to_resume.pdf>")
        sys.exit(1)

    service = build_default_service()
    resume_id = service.ingest_pdf(sys.argv[1])
    print(f"Ingested resume. resume_id={resume_id}")