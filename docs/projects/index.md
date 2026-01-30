# Projects

Three production-grade AI systems demonstrating end-to-end engineering: from research to deployment.

---

## [1. Maya: Production Voice AI](maya.md)

**Status:** Production-Ready | **Timeline:** Jan 2026

### The Challenge
Traditional voice assistants suffer from 3-4 second latency due to cascaded ASR→LLM→TTS pipelines. This breaks conversational flow and signals artificial interaction.

### The Solution
Streaming neural architecture achieving **1.35s end-to-end latency** (2.5x faster than industry baseline) through:
- Static KV-cache implementation enabling CUDA graph capture (6x speedup)
- Multi-model memory management within 24GB GPU budget (Whisper INT8 + Llama INT4 + CSM BF16)
- Constrained 6-8 word responses reducing TTS latency by 75%

### Key Metrics
- **Latency:** 1.35s P50, 1.85s P95
- **Real-Time Factor:** 0.52x (audio generation 2x faster than playback)
- **GPU Memory:** 16.2GB / 24GB
- **Voice Quality:** UTMOS 3.8+

**Stack:** Whisper large-v3-turbo, Llama 3.2 3B, CSM-1B, AWS SageMaker, CUDA, vLLM, WebRTC

[Technical Deep Dive →](maya.md)

---

## [2. Indian Art Music Separation (SOTA)](music.md)

**Status:** State-of-the-Art | **Timeline:** Dec 2025

### The Challenge
Western-trained source separation models fail on Indian classical music due to:
- Extreme harmonic density (tanpura's 20-30 overtones)
- Non-standard tuning (22 shrutis vs 12 semitones)
- Dataset heterogeneity ("mridangam" vs "drums" vs "percussion")

### The Solution
Fine-tuned Band-Split RoFormer with **+53% improvement over Demucs v4** through:
- Smart data loader with stem synonym mapping
- Consensus loss function (L1 + multi-resolution STFT + high-frequency crosstalk penalty)
- Mixed-precision training on consumer GPUs (T4/L4)

### Key Metrics
- **Vocals SDR:** 12.45 dB (vs 8.12 dB Demucs v4)
- **Drums SDR:** 11.30 dB (vs 7.50 dB)
- **Bass SDR:** 14.20 dB (vs 9.10 dB)
- **Training Time:** 8 hours (50 epochs, T4)

**Stack:** PyTorch 2.0+, Band-Split RoFormer, Mixed-Precision Training, Flash Attention

[Technical Deep Dive →](music.md)

---

## [3. Modular RAG Research Backend](rag.md)

**Status:** Production-Ready | **Timeline:** 2024

### The Challenge
LLMs hallucinate confidently when lacking domain knowledge, making them unreliable for research and fact-sensitive applications.

### The Solution
Retrieval-augmented generation grounding LLM responses in real-time arXiv paper retrieval:
- In-memory FAISS enabling sub-100ms similarity search across 10K+ documents
- Async I/O throughout (2-3x throughput improvement)
- Two-tier caching (80% cache hit rate → 80% latency reduction)

### Key Metrics
- **Accuracy Improvement:** 60% boost in LLM answer accuracy through context retrieval
- **Vector Search:** Sub-100ms similarity search
- **Document Scale:** 10,000+ indexed research papers
- **Throughput:** ~200 requests/min

**Stack:** FastAPI, FAISS, AWS Bedrock (Claude 3), sentence-transformers, arXiv API

[Technical Deep Dive →](rag.md)

---

## What These Projects Demonstrate

### Production Engineering
- Real GPU memory budgets (24GB)
- Cost-effective AWS deployments
- Actual performance numbers (not "expected" or "estimated")

### Optimization Depth
- 6x TTS speedup (CUDA graphs, static KV-cache)
- 53% quality improvement (domain-specific loss functions)
- Sub-millisecond vector search (FAISS over managed DBs)

### Full-Stack ML
- Data preprocessing → model training → optimization → deployment → monitoring
- Not just training models. Building systems that work.

### Engineering Trade-offs
- FAISS vs managed vector DBs (latency vs scalability)
- INT4 quantization (memory vs accuracy)
- Streaming vs batch processing (responsiveness vs throughput)
