 # Resume ingestion service
 
 ```text
                Upload Resume
                      │
                      ▼
            Resume Ingestion Service
                      │
     ┌────────────────┼─────────────────┐
     ▼                ▼                 ▼
 PDF Parser      Resume Parser      Chunk Builder
                      │                 │
                      ▼                 ▼
               Structured JSON     Knowledge Chunks
                      │                 │
                      └────────┬────────┘
                               ▼
                      Embedding Service
                               │
                               ▼
                          pgvector Store

```