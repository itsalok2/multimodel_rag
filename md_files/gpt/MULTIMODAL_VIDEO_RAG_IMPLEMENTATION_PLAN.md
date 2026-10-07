# Multimodal Video RAG --- Strong Implementation Plan

## 1. Objective

Build a production-quality multimodal Video RAG system that can answer
questions using:

-   spoken content from the video
-   slide/document content
-   visual frames and diagrams
-   temporal relationships
-   cross-modal relationships between speech and visuals
-   multiple lectures/videos
-   grounded citations with timestamps and thumbnails

The existing repository already contains most of the conceptual building
blocks: preprocessing, Whisper transcription, OCR, visual/text
embeddings, vector retrieval, LangGraph-based orchestration, VLM
analysis, reranking, and a Gradio UI. The goal of this plan is therefore
**not to rewrite everything blindly**. The correct approach is to
establish a working baseline first, measure it, then improve retrieval,
temporal reasoning, grounding, and production quality.

Repository: https://github.com/iisc-kc/Multimodal-Video-RAG

------------------------------------------------------------------------

## 2. Target Architecture

``` text
                         ┌─────────────────────┐
                         │      Web / API       │
                         │  Gradio / FastAPI    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Query Understanding │
                         │ intent + modality   │
                         │ + temporal parsing   │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              Text Retriever   Visual Retriever  Slide Retriever
                    │               │                │
                    └───────────────┼────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Candidate Fusion    │
                         │ + Metadata Filters  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder /     │
                         │ Multimodal Reranker │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Temporal Expansion  │
                         │ ± context window    │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ VLM Verification    │
                         │ only when required  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Grounded LLM        │
                         │ answer synthesis    │
                         └──────────┬──────────┘
                                    ▼
                    Answer + timestamps + sources
                         + thumbnails / slides
```

------------------------------------------------------------------------

# Phase 0 --- Repository Audit and Baseline

## Goal

Do not start by replacing models. First prove what currently works.

### Tasks

1.  Clone repository.
2.  Create isolated environment.
3.  Resolve dependency conflicts.
4.  Start vector database.
5.  Start Ollama/local inference.
6.  Run preprocessing on one short lecture.
7.  Build index.
8.  Run CLI query.
9.  Run Gradio UI.
10. Inspect generated frames, transcripts, OCR, metadata, and vector
    records.

### Deliverables

-   reproducible environment
-   one processed test video
-   one populated vector database
-   working query pipeline
-   baseline latency measurements
-   baseline retrieval metrics

### Baseline test queries

``` text
What is Q-learning?

What did the professor say about the reward function?

Show the slide where the neural network architecture was explained.

What was shown immediately after the professor introduced backpropagation?

What did the professor explain before the loss function?

Compare the explanation of CNNs in lecture 2 and lecture 5.
```

### Important

The repository currently has a documentation mismatch: the README
describes Milvus while ARCHITECTURE.md describes Qdrant. Pick exactly
one backend before serious development. Do not maintain two competing
implementations.

------------------------------------------------------------------------

# Phase 1 --- Build a Reliable Video Ingestion Pipeline

## Goal

Convert raw videos into a canonical multimodal representation.

### 1.1 Video metadata

Extract:

-   video_id
-   filename
-   duration
-   FPS
-   width
-   height
-   codec
-   audio sample rate
-   lecture/course metadata

Store this in a relational database.

### 1.2 Frame extraction

Do not permanently commit to a naive fixed 1 FPS strategy.

Use a two-stage approach:

``` text
Video
  │
  ├── coarse sampling: 0.5–1 FPS
  │
  └── scene/slide-change detection
          │
          └── dense sampling around changes
```

This reduces storage while preserving important visual events.

### 1.3 Audio extraction

Extract audio with FFmpeg.

Generate:

-   transcript
-   segment start/end
-   confidence if available
-   speaker information if implemented later

### 1.4 Slide detection

Detect:

-   slide transitions
-   repeated frames
-   lecturer-only frames
-   whiteboard frames
-   diagram-heavy frames

Persist slide intervals:

``` text
slide_id
lecture_id
start_time
end_time
representative_frame
```

### 1.5 OCR

Run OCR primarily on slide/whiteboard regions rather than every video
frame.

Store:

``` text
ocr_text
bounding_boxes
confidence
timestamp
slide_id
```

### Deliverable

A deterministic preprocessing command:

``` bash
python scripts/preprocess_videos.py \
    --input data/videos \
    --output data/processed
```

------------------------------------------------------------------------

# Phase 2 --- Canonical Data Model

This is critical.

Do not let each modality invent its own metadata format.

Create a unified `ContentUnit`.

``` python
ContentUnit(
    id,
    video_id,
    modality,
    start_time,
    end_time,
    parent_id,
    text,
    media_path,
    ocr_text,
    embedding_id,
    metadata
)
```

Recommended modalities:

``` text
transcript
frame
slide
ocr
diagram
chapter
```

Relationships:

``` text
video
 ├── chapter
 │    ├── transcript chunks
 │    ├── slides
 │    └── frames
 └── events
```

This becomes the backbone for temporal and cross-modal retrieval.

------------------------------------------------------------------------

# Phase 3 --- Improve Transcript Chunking

## Problem

Fixed-size text chunks are easy but are not ideal for lectures.

## Better strategy

Use semantic + temporal chunking.

``` text
Whisper segments
       │
       ▼
merge short segments
       │
       ▼
semantic boundary detection
       │
       ▼
target 20–60 second chunks
       │
       ▼
preserve start/end timestamps
```

Each chunk should contain:

-   text
-   start
-   end
-   preceding context
-   following context
-   slide_id
-   chapter_id
-   lecture_id

Avoid destroying timestamp information during chunking.

------------------------------------------------------------------------

# Phase 4 --- Build the Multimodal Index

## Text index

Embed transcript chunks and OCR text.

Recommended first baseline:

-   BGE-large-en-v1.5

Alternative:

-   BGE-M3
-   multilingual-e5-large
-   newer strong embedding model if GPU budget permits

## Visual index

The repository currently uses SigLIP SO400M according to README.

Keep it for the baseline.

Then benchmark:

-   SigLIP
-   CLIP
-   newer image-text embedding models
-   domain-specific visual encoders if the videos are technical

## Slide index

Each slide should have at least two representations:

``` text
visual_embedding
text_embedding
```

This supports:

``` text
"What does the diagram look like?"
"What slide mentions Bellman equation?"
```

## Do not force all modalities into one vector space

A transcript embedding and an image embedding do not automatically
become comparable merely because both are called "embeddings".

Use modality-specific indexes and fuse scores at retrieval time unless
the chosen model is explicitly trained for shared multimodal retrieval.

------------------------------------------------------------------------

# Phase 5 --- Metadata and Relational Database

Use a relational database for structured metadata.

Recommended:

**PostgreSQL**

Store:

``` text
videos
lectures
chapters
content_units
slides
transcript_segments
frame_metadata
ocr_regions
processing_jobs
query_logs
evaluation_results
```

Vector database stores vectors and retrieval metadata.

Object storage/filesystem stores:

``` text
videos
frames
slides
audio
thumbnails
OCR artifacts
```

This separation is much cleaner than treating the vector DB as the
entire application database.

------------------------------------------------------------------------

# Phase 6 --- Retrieval Engine v1

Implement independent retrievers.

## Text retriever

``` text
query
 → embedding
 → vector search
 → top K
```

## Visual retriever

``` text
query
 → image/text-compatible embedding
 → visual search
 → top K
```

## Slide retriever

``` text
query
 → text search
 → visual search
 → score fusion
```

## Temporal retriever

Given a result at time `t`:

``` text
[t - window, t + window]
```

retrieve:

-   transcript
-   slide
-   frames
-   OCR
-   adjacent events

------------------------------------------------------------------------

# Phase 7 --- Hybrid Retrieval

Do not trust one retrieval method.

Combine:

``` text
Dense retrieval
+
BM25 / lexical retrieval
+
metadata filtering
+
temporal filtering
```

Example:

``` text
final_score =
    0.45 * dense_score
  + 0.20 * bm25_score
  + 0.15 * visual_score
  + 0.10 * temporal_score
  + 0.10 * metadata_score
```

The weights should be learned/tuned from evaluation data rather than
treated as permanent constants.

------------------------------------------------------------------------

# Phase 8 --- Reranking

This is one of the highest-value improvements.

Pipeline:

``` text
query
   │
   ├── retrieve 30–100 candidates
   │
   ▼
reranker
   │
   ▼
top 5–10
```

For text:

-   BGE reranker
-   multilingual reranker if needed
-   strong CrossEncoder

For visual/slide candidates:

-   multimodal reranker or VLM scoring
-   use expensive VLM verification only on a small candidate set

Do not send 50 frames to an 11B VLM for every query.

------------------------------------------------------------------------

# Phase 9 --- Query Understanding

Replace a vague "agent decides everything" approach with a typed query
plan.

Example:

``` json
{
  "intent": "visual_explanation",
  "modalities": ["transcript", "slide", "frame"],
  "lecture_filter": ["lecture_03"],
  "time_range": null,
  "requires_temporal_reasoning": false,
  "requires_visual_verification": true,
  "top_k": 8
}
```

Classify:

-   factual
-   visual
-   temporal
-   comparison
-   summarization
-   navigation
-   multi-lecture

This makes the system easier to debug than free-form agent reasoning.

------------------------------------------------------------------------

# Phase 10 --- Temporal Reasoning

Implement explicit temporal operators:

``` text
before
after
during
between
first
last
next
previous
near
```

Example:

``` text
"What was shown after the reward function?"

1. retrieve reward-function event
2. locate timestamp
3. find next slide/event
4. expand ± context
5. answer using next event
```

Do not ask the LLM to infer timestamps from prose alone.

------------------------------------------------------------------------

# Phase 11 --- Cross-Modal Alignment

Create explicit links:

``` text
Transcript Chunk
       │
       ├── shown_on → Slide
       │
       ├── near → Frame
       │
       └── part_of → Chapter
```

Example:

``` text
Transcript:
"Here we update Q using the Bellman equation."

Timestamp:
00:24:35–00:24:48

Linked:
Slide 17
Frame 1478
OCR text
Diagram
```

This allows questions such as:

> "What did the professor say while this diagram was visible?"

------------------------------------------------------------------------

# Phase 12 --- VLM Verification

Use the VLM as a verifier, not the primary retrieval mechanism.

Pipeline:

``` text
Retriever
   ↓
Top 5 candidate frames/slides
   ↓
VLM
   ↓
"Which candidate actually answers the query?"
   ↓
final evidence
```

Ask the VLM structured questions:

``` json
{
  "relevant": true,
  "objects": [],
  "diagram_type": "",
  "visible_text": "",
  "explanation": ""
}
```

This reduces hallucination and makes visual reasoning auditable.

------------------------------------------------------------------------

# Phase 13 --- Grounded Answer Generation

The final LLM should receive only selected evidence.

Context format:

``` text
SOURCE 1
lecture=03
timestamp=00:24:35
type=transcript
content=...

SOURCE 2
lecture=03
timestamp=00:24:40
type=slide
ocr=...

SOURCE 3
lecture=03
timestamp=00:24:41
type=frame
vlm_analysis=...
```

Rules:

1.  answer only from retrieved evidence
2.  cite timestamps
3.  state uncertainty when evidence is insufficient
4.  never invent a slide/frame
5.  distinguish transcript evidence from visual evidence

------------------------------------------------------------------------

# Phase 14 --- Evaluation Framework

This should be implemented early, not at the end.

Create a benchmark dataset with:

``` text
question
expected_answer
lecture_id
gold_timestamp
gold_slide
required_modalities
difficulty
```

### Retrieval metrics

-   Recall@5
-   Recall@10
-   MRR
-   nDCG
-   Precision@K

### Answer metrics

-   faithfulness
-   answer correctness
-   citation correctness
-   groundedness

### System metrics

-   preprocessing time/video hour
-   index size
-   query latency
-   GPU memory
-   VLM calls/query
-   tokens/query

------------------------------------------------------------------------

# Phase 15 --- UI

Build the UI around evidence, not just chat.

Recommended layout:

``` text
┌───────────────────────────────────────────┐
│ Chat / Question                           │
├───────────────────────────────────────────┤
│ Answer                                    │
│                                           │
│ [00:24:35] [Open Video]                  │
│                                           │
├───────────────────────────────────────────┤
│ Evidence                                  │
│ [Slide thumbnail] [Frame thumbnail]       │
│                                           │
│ Transcript excerpt                        │
└───────────────────────────────────────────┘
```

Important features:

-   clickable timestamp
-   frame thumbnail
-   slide thumbnail
-   transcript excerpt
-   source confidence
-   lecture filter
-   time-range filter

------------------------------------------------------------------------

# Phase 16 --- Multi-Lecture / Course RAG

After single-video RAG is stable:

``` text
Course
 ├── Lecture 1
 ├── Lecture 2
 ├── Lecture 3
 └── Lecture N
```

Support:

``` text
"Compare how CNNs were explained in lectures 2 and 7."

"Which lecture first introduced attention?"

"Show all lectures where backpropagation was discussed."
```

------------------------------------------------------------------------

# Phase 17 --- Knowledge Graph

Add a graph only after retrieval works.

Entities:

``` text
Concept
Lecture
Slide
Speaker
Topic
Formula
Algorithm
Example
```

Edges:

``` text
INTRODUCES
EXPLAINS
FOLLOWS
PRECEDES
USES
RELATED_TO
APPEARS_IN
```

Use graph retrieval for:

-   concept progression
-   prerequisite discovery
-   lecture comparison
-   course navigation

Do not build a knowledge graph just because it sounds advanced. It is
useful only if it improves measurable retrieval/reasoning tasks.

------------------------------------------------------------------------

# Phase 18 --- Productionization

## API

Use FastAPI.

Endpoints:

``` text
POST /videos
POST /videos/{id}/process
GET  /videos/{id}/status
POST /query
GET  /videos/{id}/timeline
GET  /sources/{id}
```

## Background jobs

Use:

-   Celery + Redis, or
-   Redis Queue, or
-   a lightweight task queue initially

Pipeline:

``` text
Upload
 ↓
Job queue
 ↓
Preprocess
 ↓
Embed
 ↓
Index
 ↓
Ready
```

## Caching

Cache:

-   embeddings
-   VLM results
-   query plans
-   reranking results
-   common queries

------------------------------------------------------------------------

# Phase 19 --- Observability

Log every query as:

``` text
query_id
query
query_plan
retrievers_used
candidate_count
reranker_results
VLM_calls
final_sources
latency
answer
```

This lets you answer:

> Why did the system give this answer?

Without this, debugging RAG becomes guesswork.

------------------------------------------------------------------------

# Phase 20 --- Final Optimization

Only after accuracy is good:

-   FP16/BF16
-   quantized LLM
-   batched embedding
-   async retrieval
-   parallel modality searches
-   GPU/CPU separation
-   embedding cache
-   frame deduplication
-   approximate nearest-neighbor tuning
-   model serving optimization

------------------------------------------------------------------------

# Recommended Implementation Order

## MVP

1.  Environment
2.  Video extraction
3.  Whisper transcription
4.  slide detection
5.  OCR
6.  text + visual embeddings
7.  vector DB
8.  basic retrieval
9.  grounded LLM answer
10. timestamps

## V1

11. hybrid retrieval
12. reranking
13. temporal retrieval
14. cross-modal alignment
15. VLM verification
16. evidence UI
17. evaluation benchmark

## V2

18. multi-lecture RAG
19. query planner
20. knowledge graph
21. quiz generation
22. personalization
23. API + job queue
24. observability

------------------------------------------------------------------------

# Highest-Value Improvements

If time is limited, prioritize these:

1.  **Better chunking**
2.  **Hybrid dense + BM25 retrieval**
3.  **Cross-encoder reranking**
4.  **Explicit temporal indexing**
5.  **Transcript ↔ slide ↔ frame alignment**
6.  **VLM verification of top candidates**
7.  **Strong evaluation dataset**
8.  **Grounded citations**
9.  **Multi-lecture retrieval**
10. **Production API**

The biggest mistake would be spending weeks making the LangGraph agent
more complicated before proving retrieval quality. In multimodal RAG,
bad retrieval cannot be fixed by a smarter final prompt.

------------------------------------------------------------------------

# Definition of Done

The system should answer correctly:

### Text

"Define Q-learning."

### Visual

"Show the diagram used to explain Q-learning."

### Temporal

"What was explained immediately before Q-learning?"

### Cross-modal

"What did the professor say while this diagram was displayed?"

### Comparison

"Compare the explanation in lecture 2 and lecture 6."

### Multi-hop

"Find where the professor introduces Bellman equations, then show the
slide used immediately afterward."

### Grounding

Every answer should expose:

-   lecture
-   timestamp
-   transcript evidence
-   slide/frame evidence when applicable
