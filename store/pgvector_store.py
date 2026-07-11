import json
from uuid import UUID

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from chunking.chunk_dataclass import KnowledgeChunk


class PgVectorStore:
    """Persists KnowledgeChunks with their embeddings and supports
    similarity search, scoped per resume via resume_id."""

    def __init__(self, dsn: str, embedding_dim: int = 1536):
        """
        dsn: standard postgres connection string, e.g.
             "postgresql://user:pass@localhost:5432/resumes"
        embedding_dim: must match the embedding model's output size
             (e.g. 1536 for text-embedding-3-small, 768 for nomic-embed-text).
        """
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False

        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()

        register_vector(self.conn)
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id UUID PRIMARY KEY,
                    resume_id UUID NOT NULL,
                    chunk_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding VECTOR({self.embedding_dim}),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_resume_id
                ON knowledge_chunks (resume_id);
            """)
            # IVFFlat needs data to train lists well; fine to create empty,
            # but on a fresh table you may want to add this after the
            # first bulk load instead. Left here for convenience.
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
                ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        self.conn.commit()

    def upsert_chunks(self, resume_id: UUID, chunks: list[KnowledgeChunk]) -> None:
        """Insert or replace all chunks for a given resume_id. Chunks must
        already have `embedding` populated."""

        missing = [c.title for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"Cannot store chunks without embeddings: {missing}"
            )

        rows = [
            (
                str(chunk.id),
                str(resume_id),
                chunk.chunk_type,
                chunk.title,
                chunk.content,
                json.dumps(chunk.metadata),
                chunk.embedding,
            )
            for chunk in chunks
        ]

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO knowledge_chunks
                    (id, resume_id, chunk_type, title, content, metadata, embedding)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    chunk_type = EXCLUDED.chunk_type,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
        self.conn.commit()

    def delete_resume(self, resume_id: UUID) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_chunks WHERE resume_id = %s",
                (str(resume_id),)
            )
        self.conn.commit()

    def similarity_search(
        self,
        query_embedding: list[float],
        resume_id: UUID | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Cosine-similarity search, optionally scoped to one resume_id."""

        where_clause = "WHERE resume_id = %s" if resume_id else ""
        params: list[object] = []
        if resume_id:
            params.append(str(resume_id))
        params.append(query_embedding)
        params.append(top_k)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, chunk_type, title, content, metadata,
                       1 - (embedding <=> %s) AS similarity
                FROM knowledge_chunks
                {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "chunk_type": row[1],
                "title": row[2],
                "content": row[3],
                "metadata": row[4],
                "similarity": float(row[5]),
            }
            for row in rows
        ]

    def close(self):
        self.conn.close()
