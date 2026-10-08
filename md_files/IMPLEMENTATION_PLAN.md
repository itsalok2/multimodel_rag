# 🎥 Multimodal Video RAG: Definitive Step-by-Step Implementation Plan

> **Executive Scope**: This plan provides an actionable, production-grade roadmap for building and enhancing the **Multimodal Video RAG** system based on [iisc-kc/Multimodal-Video-RAG](https://github.com/iisc-kc/Multimodal-Video-RAG.git). It unifies the repository's existing code assets, resolves underlying architectural mismatches, adopts a canonical multimodal data model, and incorporates an evidence-grounded agentic workflow.

---

## 🧭 Core Architectural Principles

1. **Baseline-First Verification**: Never swap or rewrite models blindly. Audit the existing repository code, resolve package conflicts, and run one complete test video through ingestion and query before introducing new components.
2. **Standardized Vector Database**: Resolve the repository's documentation conflict (`Milvus` in README vs. `Qdrant` in `ARCHITECTURE.md` and `docker-compose.yml`) by standardizing on **Qdrant** as the primary local vector backend (matching `requirements.txt` and `docker-compose.yml`).
3. **Canonical Data Model (`ContentUnit`)**: Enforce a unified data contract across speech transcripts, video keyframes, slide images, and OCR text rather than letting each modality invent its own schema.
4. **Retrieval-First Priority**: Complex agentic prompts cannot rescue poor retrieval. Focus first on high-recall hybrid retrieval (Dense + BM25) and Cross-Encoder reranking before tuning LLM generation.
5. **Selective VLM Verification**: Treat the Vision-Language Model (`Llama-3.2-Vision` / `Qwen2-VL`) as a surgical verification tool for top candidates, not a bulk frame scanner.
6. **Auditable Observability**: Trace every query through a typed execution plan with logged candidate counts, reranker scores, and VLM call logs.

---

## 🧠 Model Selection & Hardware Tiering (Local 6GB VRAM & Cloud Options)

To accommodate both local edge hardware constraints (e.g. 6GB GPU VRAM) and high-accuracy frontier reasoning, the architecture supports three model profiles:

### 1. Local Text & Reasoning Models (LLMs)
| Model | Size & VRAM (4-bit Q4_K_M) | Best Use Case | Performance on 6GB VRAM |
|---|---|---|---|
| **`qwen2.5:7b-instruct`** ⭐ | ~4.7 GB | **Top Open-Source Pick**: Outstanding at structured JSON output, mathematical reasoning, and prompt constraints. | Fits 100% in VRAM (~25–35 tok/s). |
| **`llama3.1:8b-instruct`** | ~4.7 GB | Repository baseline; solid instruction following and general text synthesis. | Fits 100% in VRAM (~20–30 tok/s). |
| **`qwen2.5:3b-instruct`** | ~2.0 GB | Ultra-fast lightweight reasoning; great for low-latency query planning. | Fits 100% in VRAM (> 50 tok/s). |
| **`llama3.2:3b-instruct`** | ~2.0 GB | Fast conversational edge model; minimal memory footprint. | Fits 100% in VRAM (> 50 tok/s). |

### 2. Local Vision-Language Models (VLMs for Slides/Diagrams)
| Model | Size & VRAM (4-bit Q4_K_M) | Best Use Case | Performance on 6GB VRAM |
|---|---|---|---|
| **`qwen2.5-vl:7b`** ⭐ | ~5.0 GB | **King of Open Vision**: Dynamic resolution, reads dense charts, flowcharts, and technical equations without distortion. | Fits in 6GB VRAM (~15–20 tok/s). |
| **`minicpm-v:8b` (v2.6)** | ~5.5 GB | Outperforms Llama 3.2 Vision on OCRBench & MathVista; great slide detail extraction. | Fits in 6GB VRAM (~12–18 tok/s). |
| **`llama3.2-vision:11b`** | ~7.9 GB | Baseline repo model; general scene understanding. | Spills slightly to CPU RAM (~4–8 tok/s). |
| **`moondream2`** (1.8B) | ~1.5 GB | Tiny, ultra-fast visual inspector for slide-vs-face frame filtering. | Fits easily in VRAM (> 40 tok/s). |

### 3. High-Performance Cloud Fallback (Optional, Cost-Effective)
| Model | Provider | Token Window | Cost Profile & Value |
|---|---|---|---|
| **`Gemini 2.0 Flash`** ⭐ | Google AI Studio | **1,000,000 tokens** | **Virtually instant** (< 500ms TTFT), native multimodal video/image reasoning, dirt cheap (~$0.10/M tokens with generous free tier). |
| **`Gemini 1.5 Flash-8B`** | Google AI Studio | **1,000,000 tokens** | Ultra-lightweight cloud model for batch Q&A. |
| **`Claude 3.5 Sonnet`** | Anthropic API | 200,000 tokens | Industry gold standard for complex STEM diagrams and LaTeX derivations. |

---

## 🏛️ End-to-End System Architecture

```text
                             ┌─────────────────────────────────┐
                             │    User Interface (Gradio/API)  │
                             │  • Video Timeline Player        │
                             │  • Clickable Timestamp Links    │
                             │  • Visual Evidence Cards        │
                             └───────────────┬─────────────────┘
                                             │ User Question
                                             ▼
                             ┌─────────────────────────────────┐
                             │       Typed Query Planner       │
                             │  • Intent & Modality Classifier │
                             │  • Temporal Filter Parser       │
                             └───────────────┬─────────────────┘
                                             │ Structured Plan
                     ┌───────────────────────┼────────────────────────┐
                     ▼                       ▼                        ▼
        ┌────────────────────────┐ ┌───────────────────┐ ┌────────────────────────┐
        │ Text Retriever         │ │ Visual Retriever  │ │ Slide Retriever        │
        │ • BGE Dense Search     │ │ • SigLIP Text-to- │ │ • Visual Cosine Match  │
        │ • BM25 Lexical Keyword │ │   Image Vector    │ │ • OCR Keyword Match    │
        └────────────┬───────────┘ └─────────┬─────────┘ └────────────┬───────────┘
                     │                       │                        │
                     └───────────────────────┼────────────────────────┘
                                             ▼
                             ┌─────────────────────────────────┐
                             │   Candidate Fusion & Reranking  │
                             │  • Reciprocal Rank Fusion (RRF) │
                             │  • Cross-Encoder Reranker       │
                             └───────────────┬─────────────────┘
                                             │ Top 5-8 Ranked Candidates
                                             ▼
                             ┌─────────────────────────────────┐
                             │   Temporal Window Expansion     │
                             │  • Fetch ±45s Context & Slides  │
                             │  • Cross-Modal Entity Linking   │
                             └───────────────┬─────────────────┘
                                             │
                       Is visual verification needed?
                                  ├── Yes ──► [VLM Verification: Llama 3.2 Vision]
                                  └── No  ──┐
                                            ▼
                             ┌─────────────────────────────────┐
                             │    Grounded Answer Synthesis    │
                             │  • Llama-3.1-8B-Instruct        │
                             │  • Strict Grounding & Citations │
                             └───────────────┬─────────────────┘
                                             ▼
                             Answer + Citations [MM:SS] + Thumbnails
```

---

## 📋 Phased Implementation Roadmap

| Phase | Title | Focus & Core Deliverables | Dependencies |
|---|---|---|---|
| **Phase 0** | **Baseline Audit & Conflict Resolution** | Standardize Vector DB (Qdrant), install dependencies, run 1 video end-to-end | Repo Clone |
| **Phase 1** | **Canonical Data Model (`ContentUnit`)** | Unified Pydantic schemas, SQLite relational store for video metadata | Phase 0 |
| **Phase 2** | **Robust Ingestion Pipeline** | Adaptive keyframe detector (SSIM/pHash), WhisperX transcription, PaddleOCR | Phase 1 |
| **Phase 3** | **Embeddings & Qdrant Indexing** | SigLIP visual embedder, BGE text embedder, multi-vector collections | Phase 2 |
| **Phase 4** | **Hybrid Retrieval & Cross-Encoder** | Dense vector search, BM25 sparse index, Reciprocal Rank Fusion, reranker | Phase 3 |
| **Phase 5** | **Temporal & Cross-Modal Linking** | Chronological expansion (±45s), slide-to-speech alignment | Phase 4 |
| **Phase 6** | **Typed Query Planner & Agent Orchestration** | LangGraph cyclic graph, intent routing, structured tool invocation | Phase 5 |
| **Phase 7** | **VLM Verification & Grounded Synthesis** | Llama 3.2 Vision verification tool, strict prompt template with citations | Phase 6 |
| **Phase 8** | **Evidence-Centered Web UI** | Gradio UI with HTML5 video player, timestamp seeking, base64 thumbnails | Phase 7 |
| **Phase 9** | **Evaluation & Golden Benchmark** | 50 test QA queries, RAGAS metrics (Recall, Precision, Faithfulness) | Phase 8 |
| **Phase 10** | **Multi-Lecture Search & Observability** | Multi-video search, trace logging (`query_id`, latency, candidate drops) | Phase 9 |

---

## 🛠️ Step-by-Step Phase Specifications

### Phase 0: Baseline Audit & Database Conflict Resolution

#### Goal
Verify existing functionality and resolve configuration inconsistencies before writing new logic.

#### Key Conflict in Existing Repository:
- `docker-compose.yml` specifies `image: qdrant/qdrant:latest` on port `6333`.
- `requirements.txt` specifies `qdrant-client==1.11.3`.
- `src/vector_store/` contains both `qdrant_client.py` and `milvus_client.py`.
- `src/agent/graph.py` imports `from ..vector_store.milvus_client import MultimodalVectorStore`.
- **Decision**: Standardize on **Qdrant**. Update `src/agent/graph.py` to import `qdrant_client.py`. Keep Milvus client as an optional alternative.

#### Tasks:
1. Initialize virtual environment (`Python 3.10` or `3.11`):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start Qdrant vector database via Docker:
   ```bash
   docker compose up -d
   curl http://localhost:6333/healthz  # Expected: 200 OK
   ```
3. Verify Ollama local models (choose based on VRAM profile):
   - **Recommended Fast Setup (6GB VRAM)**:
     ```bash
     ollama pull qwen2.5:7b-instruct
     ollama pull qwen2.5-vl:7b
     # Or ultra-fast lightweight:
     # ollama pull qwen2.5:3b-instruct
     ```
   - **Baseline Repo Setup**:
     ```bash
     ollama pull llama3.1:8b-instruct
     ollama pull llama3.2-vision:11b
     ```
4. Run preprocessing on a single test video (3–5 minutes):
   ```bash
   python scripts/preprocess_videos.py --video data/videos/sample.mp4
   python scripts/build_index.py
   python app/cli.py --query "What is discussed in this lecture?"
   ```

#### Acceptance Criteria:
- Baseline query produces an answer using Qdrant without import errors.
- Baseline latency recorded for transcription, embedding, and retrieval.

---

### Phase 1: Canonical Data Model (`ContentUnit`)

#### Goal
Prevent mismatched schemas across transcription, OCR, frame extraction, and retrieval.

#### Design Specification (`src/vector_store/schemas.py`):
```python
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ModalityType(str, Enum):
    TRANSCRIPT = "transcript"
    FRAME = "frame"
    SLIDE = "slide"
    OCR = "ocr"
    CHAPTER = "chapter"

class ContentUnit(BaseModel):
    """Canonical data model for all multimodal content in the pipeline."""
    id: str = Field(description="Unique UUID or hash")
    lecture_id: str = Field(description="Lecture or video identifier")
    modality: ModalityType
    start_time: float = Field(description="Start timestamp in seconds")
    end_time: float = Field(description="End timestamp in seconds")
    text: Optional[str] = Field(default=None, description="Spoken transcript or OCR text")
    media_path: Optional[str] = Field(default=None, description="Relative path to frame/slide image")
    parent_slide_id: Optional[str] = Field(default=None, description="Linked slide ID if known")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Bounding boxes, confidence, FPS")
```

#### Relational Metadata Store (`src/storage/metadata_db.py`):
- Implement lightweight SQLite database (`data/metadata.db`):
  - Table `lectures`: `lecture_id`, `filename`, `duration_seconds`, `fps`, `processed_at`.
  - Table `slides`: `slide_id`, `lecture_id`, `start_time`, `end_time`, `image_path`, `ocr_summary`.
  - Table `transcript_segments`: `segment_id`, `lecture_id`, `start_time`, `end_time`, `text`, `slide_id`.

#### Acceptance Criteria:
- All preprocessing scripts output objects conforming strictly to `ContentUnit`.
- Relational tables populated and queryable via SQL.

---

### Phase 2: Robust Multimodal Ingestion Pipeline

#### Goal
Transform video and lecture slides into high-fidelity atomic units without redundant frame bloat.

```text
Raw Video (.mp4)
  │
  ├── 1. Audio Extraction ──► WhisperX ──► Word/Sentence Timestamps [start, end, text]
  │
  └── 2. Visual Extraction
        ├── Coarse Sampling (1 FPS)
        └── Scene Transition Filter (SSIM < 0.85 / pHash delta > 12)
              │
              ├── Keyframes ──► PaddleOCR ──► Filtered Text & Bounding Boxes
              └── (Optional) PDF Slide Matcher ──► Slide-to-Video Synchronization
```

#### Tasks:
1. **Adaptive Keyframe Extraction (`src/preprocessing/slide_detector.py`)**:
   - Compare successive frames using SSIM (Structural Similarity Index) and Perceptual Hashing.
   - Ignore static frames where only the mouse moves or minor lighting changes occur.
   - For a 60-minute lecture, extract ~40–80 meaningful visual slides instead of 3,600 redundant frames.
2. **Audio Transcription (`src/preprocessing/transcription.py`)**:
   - Run `faster-whisper` or `whisperX` with `--model large-v3`.
   - Preserve precise word and sentence level timestamp intervals `[t_start, t_end]`.
3. **Semantic + Temporal Chunking (`src/preprocessing/chunking.py`)**:
   - Group short Whisper sentences into 30–60 second semantic chunks.
   - Never split a sentence across chunks.
   - Store both local chunk text and preceding/succeeding sentence context.
4. **Slide OCR Extraction (`src/preprocessing/ocr_processor.py`)**:
   - Run PaddleOCR on detected keyframes.
   - Clean low-confidence noise (< 0.6 confidence); detect bullet hierarchies.

#### Acceptance Criteria:
- Keyframes stored under `data/processed/{lecture_id}/frames/frame_{timestamp:.2f}.jpg`.
- Clean JSON export of all `ContentUnit` items generated.

---

### Phase 3: Embeddings & Qdrant Vector Indexing

#### Goal
Embed text and visual units and store them in indexed Qdrant collections.

#### Collections in Qdrant:
1. **`lecture_transcripts`**:
   - Vector: `1024-d` (BGE-large-en-v1.5 or BGE-M3).
   - Distance: `Cosine`.
   - Payload: `content_unit_id`, `lecture_id`, `start_time`, `end_time`, `text`, `parent_slide_id`.
2. **`lecture_frames`**:
   - Vector: `1152-d` (SigLIP SO400M) or `768-d` (CLIP ViT-L/14).
   - Distance: `Cosine`.
   - Payload: `content_unit_id`, `lecture_id`, `timestamp`, `image_path`, `ocr_summary`.
3. **`lecture_slides`**:
   - Multi-vector or dual-index: stores both visual frame vector and OCR text vector.
   - Payload: `slide_id`, `lecture_id`, `start_time`, `end_time`, `image_path`, `ocr_text`.

#### Indexing Implementation (`scripts/build_index.py`):
- Batch upload payloads and vectors using `qdrant_client.upload_points()`.
- Create HNSW index configuration: `m=16`, `ef_construct=100`.

#### Acceptance Criteria:
- Indexing 1 hour of lecture material completes in under 3 minutes on GPU.
- Point queries by `lecture_id` metadata execute in under 10ms.

---

### Phase 4: Hybrid Retrieval & Cross-Encoder Reranking

#### Goal
Eliminate retrieval blind spots by combining dense semantic embeddings, sparse keyword search, and Cross-Encoder scoring.

```text
Query: "What is the update rule in Bellman equation?"
  │
  ├── 1. Dense Semantic Search (BGE Text Embeddings) ──────► Top 25 Transcripts
  ├── 2. Sparse Lexical Search (BM25 on Transcripts & OCR) ─► Top 25 Keywords
  └── 3. Visual Search (SigLIP Text-to-Image Vector) ──────► Top 15 Slides
        │
        ▼
   Reciprocal Rank Fusion (RRF) ──► Merged Top 30 Candidates
        │
        ▼
   Cross-Encoder Reranker (`bge-reranker-large`)
        │
        ▼
   Top 6 High-Confidence Evidence Units (Calibrated Score 0.0 - 1.0)
```

#### Tasks:
1. **Reciprocal Rank Fusion (RRF)**:
   $$RRF\_Score(d) = \sum_{m \in Modalities} \frac{1}{60 + rank_m(d)}$$
2. **Cross-Encoder Reranking (`src/utils/reranker.py`)**:
   - Evaluate `(query, passage)` pairs using `BAAI/bge-reranker-large`.
   - Normalize logit outputs into a calibrated `0.0 - 1.0` confidence scale.
   - Filter out candidates scoring below threshold (`score < 0.45`).

#### Acceptance Criteria:
- Exact technical keywords (e.g., *"ResNet-50"*, *"Learning Rate"*, *"AdamW"*) retrieve the exact matching slide/transcript in the top 3 hits.

---

### Phase 5: Temporal Expansion & Cross-Modal Alignment

#### Goal
Provide complete conversational and visual context when an answer references an exact moment in time.

#### Tasks:
1. **Temporal Context Expansion (`src/retrieval/temporal.py`)**:
   - When a top candidate matches timestamp $t$, query the SQLite metadata store for:
     - Preceding transcript segment: $[t - 45s, t]$.
     - Matching transcript segment: $[t, t + \Delta]$.
     - Subsequent transcript segment: $[t + \Delta, t + 45s]$.
     - The active visual slide displayed throughout this time window.
2. **Cross-Modal Linking**:
   - If the user asks: *"What did the professor say while showing this slide?"*
     1. Retrieve slide candidate $\rightarrow$ obtain $[t_{start}, t_{end}]$.
     2. Query transcripts overlapping $[t_{start}, t_{end}]$.
     3. Group together as a synchronized multi-modal evidence packet.

#### Acceptance Criteria:
- Temporal queries like *"What was discussed right after introducing the loss function?"* correctly locate the next sequential event.

---

### Phase 6: Typed Query Planner & Agent Orchestration

#### Goal
Replace ambiguous "agent-free-for-all" prompting with a deterministic, typed query plan orchestrated via **LangGraph**.

#### 1. Typed Query Plan Schema:
```python
from pydantic import BaseModel
from typing import List, Optional

class QueryPlan(BaseModel):
    intent: str  # "conceptual", "visual_diagram", "temporal_progression", "slide_lookup"
    target_modalities: List[str]  # ["transcript", "frame", "slide"]
    lecture_filter: Optional[str] = None
    time_range_filter: Optional[tuple] = None
    requires_vlm_verification: bool = False
    requires_temporal_expansion: bool = False
    top_k: int = 5
```

#### 2. LangGraph State Machine (`src/agent/graph.py`):

```text
                  ┌──────────────────────┐
                  │     Start Node       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Analyze Query Plan  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Multimodal Retrieval │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Candidate Reranking  │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
  requires_vlm == True           requires_vlm == False
              ▼                             │
  ┌──────────────────────┐                  │
  │   VLM Verification   │                  │
  │ (Llama 3.2 Vision)   │                  │
  └──────────┬───────────┘                  │
              └──────────────┬──────────────┘
                             ▼
                  ┌──────────────────────┐
                  │  Temporal Expansion  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Grounded Synthesis  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │       End Node       │
                  └──────────────────────┘
```

#### Acceptance Criteria:
- Graph routes dynamically: text-only questions skip the VLM completely; diagram questions route to VLM verification.
- Output includes complete reasoning metadata for auditing.

---

### Phase 7: VLM Verification & Grounded Synthesis

#### Goal
Ensure visual questions are validated by a vision model and that answers never hallucinate ungrounded statements.

#### Tasks:
1. **VLM Verification Tool (`src/inference/vision_llm.py`)**:
   - Send only the top 2–3 visual candidates to VLM (`qwen2.5-vl:7b`, `minicpm-v:8b`, or `llama3.2-vision:11b` via Ollama; or `Gemini 2.0 Flash` via API).
   - Prompt format:
     ```text
     Examine this lecture frame and determine if it answers: "{query}".
     Return JSON:
     {
       "is_relevant": true/false,
       "diagram_type": "flowchart/neural_net/graph/code/table",
       "key_visual_elements": ["box A", "arrow to box B"],
       "visual_explanation": "..."
     }
     ```
2. **Grounded Answer Synthesis Prompt (`src/agent/prompts.py`)**:
   - Provide structured evidence blocks labeled with exact timestamps:
     ```text
     [EVIDENCE 1] Lecture: {lecture_id} | Time: {start} - {end} | Modality: Transcript
     Content: {transcript_text}

     [EVIDENCE 2] Lecture: {lecture_id} | Time: {timestamp} | Modality: Visual Slide
     VLM Interpretation: {vlm_analysis}
     OCR Text: {ocr_text}
     ```
   - Constraints:
     1. Rely strictly on provided evidence.
     2. Cite timestamps using bracket notation `[MM:SS]`.
     3. If evidence does not contain the answer, explicitly state that it was not discussed.

#### Acceptance Criteria:
- Answer faithfully cites specific lecture moments.
- Zero fabricated facts or phantom slide references.

---

### Phase 8: Evidence-Centered Interactive Web UI

#### Goal
Deliver an intuitive, evidence-first web interface where users can verify AI claims directly against video footage.

#### Tasks (`app/gradio_app.py`):
1. **Interactive Chat Layout**:
   - Left Column: Chat conversation and question input.
   - Right Column: Synchronized Evidence Viewer:
     - HTML5 Video Player initialized with the lecture video.
     - Clickable thumbnail gallery of retrieved keyframes and slides.
     - Collapsible transcript excerpts.
2. **Timestamp Deep-Linking**:
   - Clicking a timestamp citation (e.g., `[14:22]`) in the chat or on an evidence card triggers an HTML/JavaScript event:
     ```javascript
     document.getElementById('lecture-video').currentTime = 862;
     document.getElementById('lecture-video').play();
     ```
3. **Confidence Badges**:
   - Color-coded badges indicating match reliability:
     - **Green** ($\ge 0.80$): High confidence match.
     - **Yellow** ($0.60 - 0.79$): Moderate confidence match.
     - **Red** ($< 0.60$): Low confidence / speculative context.

#### Acceptance Criteria:
- Video seeking works seamlessly in browser.
- Thumbnails render efficiently via base64 or static media file endpoints.

---

### Phase 9: Evaluation Framework & Golden Benchmark

#### Goal
Establish quantitative metrics to measure retrieval and answer quality continuously.

#### Tasks:
1. **Curate Golden Test Dataset (`tests/benchmark_queries.json`)**:
   - 50 ground-truth questions across 6 categories:
     - **Factual/Text**: *"What is the reward discount factor $\gamma$ used in lecture 1?"*
     - **Visual/Diagram**: *"Show the architecture diagram of the actor-critic network."*
     - **Temporal**: *"What topic was introduced right after gradient descent?"*
     - **Cross-Modal**: *"What did the professor say while pointing to the confusion matrix?"*
     - **Comparative**: *"Compare policy iteration in Lecture 3 vs value iteration in Lecture 4."*
     - **Negative/Out-of-Scope**: Query a topic not present in the video to verify refusal.
2. **Automated Evaluation Runner (`scripts/evaluate.py`)**:
   - **Retrieval Metrics**:
     - $\text{Recall}@K$ ($K=3, 5$): Did the gold timestamp/slide appear in top $K$?
     - Mean Reciprocal Rank (MRR).
   - **Generation Metrics (RAGAS / LLM-as-a-judge)**:
     - Faithfulness (0.0 – 1.0).
     - Answer Relevancy (0.0 – 1.0).
     - Timestamp Precision ($\pm 15$ seconds tolerance).

#### Acceptance Criteria:
- $\text{Recall}@5 \ge 85\%$ on the golden benchmark dataset.
- Faithfulness score $\ge 0.90$.

---

### Phase 10: Multi-Lecture Scaling & Observability

#### Goal
Support course-scale repositories spanning dozens of lectures with full operational visibility.

#### Tasks:
1. **Multi-Lecture Query Routing**:
   - Allow cross-lecture synthesis (e.g., *"Summarize how optimization algorithms evolved from Lecture 1 to Lecture 8"*).
   - Partition Qdrant payload filters by `course_id` and `lecture_id`.
2. **Execution Trace Logging (`src/utils/logger.py`)**:
   - Log structured JSON traces for every query:
     ```json
     {
       "query_id": "req-98213",
       "query": "Show the loss curve",
       "intent": "visual_diagram",
       "retrieval_candidates_count": 30,
       "reranker_top_scores": [0.89, 0.74, 0.52],
       "vlm_invocations": 1,
       "latency_ms": {
         "retrieval": 42,
         "rerank": 115,
         "vlm": 680,
         "synthesis": 1240,
         "total": 2077
       }
     }
     ```

#### Acceptance Criteria:
- Detailed latency profile and candidate funnel recorded for every query.
- Multi-lecture queries retrieve evidence across distinct video IDs without collision.

---

## 🎯 Definition of Done (Checklist)

A feature milestone is considered **Done** only when all the following query capabilities pass automated tests:

- [ ] **Factual Text**: Answers correctly from speech transcripts with citation.
- [ ] **Visual Diagrams**: Identifies and returns the exact slide containing the diagram.
- [ ] **Temporal Progression**: Correctly identifies what preceded or followed an event.
- [ ] **Cross-Modal Grounding**: Synthesizes what was spoken during a displayed slide.
- [ ] **Comparative Synthesis**: Compares concepts across different lecture timestamps.
- [ ] **Out-of-Scope Refusal**: Gracefully declines to answer topics absent from the lecture.
- [ ] **Interactive Verification**: Clicking any citation seeks the video player to the exact second.
