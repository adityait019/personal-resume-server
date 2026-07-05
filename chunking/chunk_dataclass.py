from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):

    id: UUID = Field(default_factory=uuid4)

    chunk_type: str

    title: str

    content: str

    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None, description="Vector representation of the chunk for semantic search")