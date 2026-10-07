# Multimodal Video RAG --- Technical Stack and Alternatives

## 1. Repository Baseline

The repository describes an open-source, local-first multimodal Video
RAG system using:

-   LangChain
-   LangGraph
-   vector search
-   Whisper
-   OCR
-   vision embeddings
-   a vision-language model
-   a reasoning LLM
-   Gradio
-   Docker

The current repository documentation contains an important
inconsistency:

-   README currently lists **Milvus** as the vector DB.
-   ARCHITECTURE.md describes **Qdrant** collections and Qdrant-based
    retrieval.
-   requirements.txt currently contains `qdrant-client`.

For a clean implementation, use **Qdrant as the baseline** unless there
is a strong reason to migrate to Milvus.

------------------------------------------------------------------------

# 2. Recommended Production Stack

  -----------------------------------------------------------------------------
  Layer             Recommended          Alternatives      Reason
  ----------------- -------------------- ----------------- --------------------
  Language          Python 3.11/3.12     Python 3.10       ML ecosystem

  API               FastAPI              Flask             Async/API-friendly

  UI                Gradio               Streamlit, React  Fast multimodal
                                                           prototype

  Orchestration     LangGraph            custom state      Explicit agent
                                         machine           workflow

  Video             FFmpeg + OpenCV      PyAV              Mature video tooling

  ASR               Whisper Large-v3 /   Whisper medium    Accuracy/speed
                    faster-whisper                         tradeoff

  OCR               PaddleOCR            EasyOCR,          Better
                                         Tesseract         document/slide OCR

  Text embedding    BGE-large / BGE-M3   E5, Nomic         Strong retrieval

  Vision embedding  SigLIP               CLIP              Better modern
                                                           image-text retrieval
                                                           baseline

  VLM               Llama 3.2 Vision 11B Qwen2.5-VL,       Visual reasoning
                                         Qwen2-VL          

  Reasoning LLM     Llama 3.1 8B         Qwen2.5 7B/14B    Local reasoning

  Reranker          BGE reranker         CrossEncoder      Better top-K
                                                           precision

  Vector DB         Qdrant               Milvus, Weaviate, Simple multimodal
                                         pgvector          collections

  Metadata DB       PostgreSQL           SQLite            Production
                                                           structured data

  Object storage    MinIO/S3             local filesystem  Video/frame storage

  Cache             Redis                local cache       Jobs + caching

  Jobs              Celery/RQ            asyncio           Long-running video
                                                           processing

  Containers        Docker               Podman            Deployment

  Monitoring        Prometheus/Grafana   OpenTelemetry     Production
                                                           observability
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

# 3. Model Architecture

## 3.1 Speech

### Baseline

**Whisper Large-v3**

Purpose:

``` text
video
 ↓
audio
 ↓
Whisper
 ↓
timestamped transcript
```

Use `faster-whisper` for inference when throughput matters.

Store:

``` text
segment_id
start
end
text
confidence
```

### Alternatives

  Model              Use when
  ------------------ -------------------------------
  Whisper Large-v3   maximum transcription quality
  Whisper Medium     limited VRAM
  faster-whisper     production throughput
  distil-whisper     low latency

------------------------------------------------------------------------

# 4. Text Embeddings

## Recommended

### BGE-large-en-v1.5

Good starting point for English lecture retrieval.

### Better general alternative

### BGE-M3

Prefer BGE-M3 when:

-   multilingual data exists
-   long documents matter
-   hybrid retrieval is important

### Other alternatives

``` text
BGE-large
BGE-M3
multilingual-e5-large
Nomic Embed
GTE
```

Do not select the embedding model based only on benchmark reputation.
Evaluate it on your actual lecture questions.

------------------------------------------------------------------------

# 5. Vision Encoder

## Baseline

### SigLIP SO400M

Use it for:

-   frame embeddings
-   slide embeddings
-   visual similarity

The repository README currently specifies SigLIP SO400M for frame
embeddings.

### Alternatives

  Encoder                     Best use
  --------------------------- ------------------------------------------
  SigLIP                      general image-text retrieval
  CLIP ViT-L/14               stable baseline
  OpenCLIP                    many pretrained variants
  DINOv2                      visual similarity without text alignment
  newer multimodal encoders   benchmark if available

Important distinction:

**DINOv2 is not a direct replacement for CLIP/SigLIP when the query is
text.**

DINOv2 is better suited to pure visual similarity.

------------------------------------------------------------------------

# 6. Vision-Language Model

## Baseline

### Llama 3.2 Vision 11B

Use only after retrieval.

Correct architecture:

``` text
Query
 ↓
retrieve candidate frames
 ↓
top 3–5 frames
 ↓
VLM verification
 ↓
final evidence
```

Incorrect architecture:

``` text
Query
 ↓
send hundreds of frames to VLM
```

The second design is expensive and slow.

------------------------------------------------------------------------

# 7. Alternative VLMs

### Qwen2.5-VL

Strong alternative for:

-   OCR-heavy content
-   diagrams
-   charts
-   screenshots
-   visual reasoning

### Qwen2-VL

Good alternative when hardware/software compatibility favors it.

### Llama Vision

Good local baseline if the rest of the stack already uses Meta models.

### Cloud alternatives

If privacy/cost constraints are relaxed:

``` text
Gemini
GPT vision models
Claude vision models
```

These can be substantially stronger for difficult visual reasoning, but
they introduce API cost, latency, and data-governance considerations.

------------------------------------------------------------------------

# 8. Reasoning LLM

## Baseline

### Llama 3.1 8B Instruct

Responsibilities:

-   query classification
-   retrieval planning
-   evidence synthesis
-   final answer

### Alternatives

``` text
Qwen2.5 7B
Qwen2.5 14B
Llama 3.x 8B
Mistral
larger reasoning models
```

A 14B model may improve complex multi-step reasoning but costs more
memory and latency.

------------------------------------------------------------------------

# 9. Vector Database

## Recommended baseline: Qdrant

Use separate collections or named vectors for:

``` text
transcript
frames
slides
```

Example:

``` text
transcript_collection
frame_collection
slide_collection
```

Metadata:

``` text
video_id
lecture_id
timestamp
timestamp_end
slide_id
frame_id
content_type
path
```

### Why Qdrant?

-   easy local deployment
-   metadata filtering
-   named vectors
-   straightforward Python API
-   good fit for this project size

------------------------------------------------------------------------

# 10. Vector DB Alternatives

## Milvus

Use when:

-   dataset becomes very large
-   distributed vector infrastructure is required
-   high-scale ANN workloads matter

The repository itself includes a Milvus Docker configuration, but its
architecture documentation and Python requirements currently point
toward Qdrant. Pick one and remove the other from the implementation.

## pgvector

Use when:

-   PostgreSQL is already the main database
-   dataset is moderate
-   operational simplicity is more important than specialized vector
    infrastructure

This is an attractive architecture for a smaller production system:

``` text
PostgreSQL
 ├── metadata
 ├── transcript
 ├── slides
 └── embeddings via pgvector
```

## Weaviate

Good if you want more built-in multimodal/vector features and managed
options.

------------------------------------------------------------------------

# 11. Non-Vector Database

## Recommended: PostgreSQL

The vector DB should not become your entire database.

PostgreSQL should store:

``` text
courses
lectures
videos
chapters
slides
transcript_segments
frames
OCR
processing_jobs
users
queries
evaluation_results
```

Example relationship:

``` text
course
  ↓
lecture
  ↓
video
  ├── transcript_segment
  ├── slide
  ├── frame
  └── chapter
```

------------------------------------------------------------------------

# 12. Object Storage

Do not store large frame/video binaries inside PostgreSQL or the vector
database.

Use:

### Development

``` text
data/
 ├── videos/
 ├── frames/
 ├── slides/
 ├── audio/
 └── thumbnails/
```

### Production

``` text
S3 / MinIO
```

Database stores only paths/URIs.

------------------------------------------------------------------------

# 13. Search Architecture

Do not rely on dense vector search alone.

Recommended:

``` text
                  Query
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Dense        BM25       Metadata
     Search       Search       Filter
        │           │           │
        └───────────┼───────────┘
                    ▼
              Score Fusion
                    │
                    ▼
                Reranker
                    │
                    ▼
              Top Evidence
```

This is more robust than:

``` text
query → embedding → vector DB → LLM
```

------------------------------------------------------------------------

# 14. Reranking

## Recommended

BGE reranker / CrossEncoder.

Flow:

``` text
Top 50 retrieved
       ↓
CrossEncoder
       ↓
Top 5–10
```

This should be one of the first major improvements over a basic RAG
implementation.

------------------------------------------------------------------------

# 15. Temporal Database Design

Every content unit must have:

``` text
start_time
end_time
```

Then support temporal operators:

``` text
before
after
during
near
next
previous
first
last
```

Example:

``` text
event A = 120.0–135.0
event B = 136.0–150.0
```

Query:

``` text
"What happened after A?"
```

becomes a database operation, not an LLM guessing exercise.

------------------------------------------------------------------------

# 16. Multimodal Data Schema

Recommended:

``` python
ContentUnit:
    id
    video_id
    lecture_id
    type
    start_time
    end_time
    text
    ocr_text
    media_uri
    slide_id
    chapter_id
    embedding_ref
```

Types:

``` text
transcript
slide
frame
ocr
diagram
chapter
```

Relationships:

``` text
transcript → shown_on → slide
transcript → near → frame
slide → contains → diagram
frame → belongs_to → slide
content → belongs_to → chapter
```

------------------------------------------------------------------------

# 17. Query Planner

Instead of a completely free-form agent, use structured planning.

Example:

``` json
{
  "intent": "temporal_visual",
  "modalities": [
    "transcript",
    "slide",
    "frame"
  ],
  "temporal_operator": "after",
  "needs_vlm": true,
  "top_k": 10
}
```

This is easier to test and much easier to debug.

------------------------------------------------------------------------

# 18. LangGraph

Use LangGraph for orchestration:

``` text
START
  ↓
query_parser
  ↓
retrieval_router
  ├── text_retriever
  ├── visual_retriever
  └── slide_retriever
  ↓
fusion
  ↓
reranker
  ↓
temporal_expansion
  ↓
vlm_verifier
  ↓
answer_generator
  ↓
END
```

Do not turn every operation into an autonomous agent.

Deterministic nodes should remain deterministic.

------------------------------------------------------------------------

# 19. Evaluation Stack

Create a custom benchmark.

### Retrieval

``` text
Recall@5
Recall@10
MRR
nDCG
Precision@K
```

### Generation

``` text
faithfulness
answer correctness
citation correctness
groundedness
```

### Performance

``` text
p50 latency
p95 latency
GPU memory
VLM calls/query
tokens/query
preprocessing time/hour
index size
```

------------------------------------------------------------------------

# 20. Recommended Hardware Strategy

## 8 GB VRAM

Use:

``` text
Whisper medium / faster-whisper
7B-class LLM
7B-class VLM where feasible
BGE-base/smaller embedding model
FP16/quantization
```

## 12--16 GB VRAM

Comfortable for:

``` text
Whisper large
8B LLM
7B–11B VLM
larger embedding models
```

## 24 GB+

Better for:

``` text
11B+ VLM
14B LLM
larger batch sizes
parallel embedding
reranking
```

For a single-developer project, optimize for **accuracy per GB of
VRAM**, not model size.

------------------------------------------------------------------------

# 21. Alternative Architecture A --- Simple and Practical

Best for a portfolio/MVP.

``` text
FastAPI
   │
PostgreSQL + pgvector
   │
Local filesystem
   │
Whisper
   │
BGE
   │
SigLIP
   │
Qwen/Llama Vision
   │
Llama/Qwen
```

Advantages:

-   fewer services
-   easy deployment
-   easier debugging
-   low operational complexity

This is the architecture I would choose for a first strong
implementation.

------------------------------------------------------------------------

# 22. Alternative Architecture B --- Scalable

``` text
FastAPI
   │
Redis
   │
Celery workers
   │
S3/MinIO
   │
PostgreSQL
   │
Milvus
   │
GPU model servers
```

Use when:

-   hundreds/thousands of videos
-   concurrent users
-   distributed processing
-   large vector collections

------------------------------------------------------------------------

# 23. Alternative Architecture C --- Research/Advanced

``` text
              Query
                │
         Query Planner
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
 Text RAG    Visual RAG  Temporal RAG
      │         │         │
      └─────────┼─────────┘
                ▼
        Multimodal Reranker
                │
                ▼
          Knowledge Graph
                │
                ▼
           VLM verifier
                │
                ▼
          Reasoning LLM
```

Add:

-   concept graph
-   lecture graph
-   semantic event graph
-   multimodal reranking
-   long-context synthesis

This is the most interesting version for research.

------------------------------------------------------------------------

# 24. What I Would Actually Build

## Recommended stack

``` text
Python 3.11/3.12
FastAPI
Gradio
LangGraph
FFmpeg
OpenCV
faster-whisper
PaddleOCR
BGE-M3
SigLIP
Qwen2.5-VL or Llama Vision
Qwen2.5 7B/14B or Llama 8B
Qdrant
PostgreSQL
Redis
MinIO/S3
Docker
```

### Why this stack?

It separates concerns cleanly:

``` text
PostgreSQL → application metadata
Qdrant     → vector retrieval
MinIO      → video/images
Redis      → cache/jobs
LLMs       → reasoning
VLM        → visual verification
```

------------------------------------------------------------------------

# 25. Biggest Improvements Over the Existing Baseline

## Improvement 1 --- Better retrieval

Current concept:

``` text
vector search
```

Improved:

``` text
dense + BM25 + metadata + reranking
```

## Improvement 2 --- Better temporal reasoning

Current concept:

``` text
sort timestamp
```

Improved:

``` text
explicit event intervals
+ temporal operators
+ linked content units
```

## Improvement 3 --- Better visual grounding

Current:

``` text
retrieve frame
→ VLM
```

Improved:

``` text
visual retrieval
→ slide retrieval
→ OCR retrieval
→ reranking
→ VLM verification
```

## Improvement 4 --- Better multimodal alignment

Current:

``` text
same timestamp
```

Improved:

``` text
transcript interval
↔ slide interval
↔ frame interval
↔ OCR regions
```

## Improvement 5 --- Better observability

Record:

``` text
query
plan
retrievers
candidates
scores
reranker
VLM
sources
latency
```

------------------------------------------------------------------------

# 26. Final Recommendation

Do not start by replacing every model in the repository.

Use this sequence:

``` text
Existing repository
       ↓
Make baseline work
       ↓
Create evaluation dataset
       ↓
Improve chunking
       ↓
Hybrid retrieval
       ↓
Reranking
       ↓
Temporal alignment
       ↓
Cross-modal linking
       ↓
VLM verification
       ↓
Grounded citations
       ↓
Multi-lecture RAG
       ↓
Production API
       ↓
Knowledge graph
```

The strongest portfolio/research story is not:

> "I used many multimodal models."

It is:

> "I built a multimodal retrieval system, measured its weaknesses,
> improved retrieval and temporal grounding systematically, and
> demonstrated measurable gains in Recall@K, MRR, grounded answer
> accuracy, and latency."

That is the difference between a demo and a serious AI engineering
project.
