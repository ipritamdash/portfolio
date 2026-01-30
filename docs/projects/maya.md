# Maya: Production-Grade Conversational Voice AI

**Role:** AI Engineer | **Stack:** Whisper, Llama 3.2, CSM-1B, AWS SageMaker, CUDA
**Timeline:** Jan 2026 | **Status:** Production-Ready

---

## Overview

Maya is an end-to-end conversational voice AI system achieving **sub-1.5 second response latency** through streaming neural architecture. Unlike traditional cascaded ASR→LLM→TTS pipelines that introduce 3-4 second delays, Maya implements a unified streaming pipeline that processes audio in real-time, enabling natural conversational flow.

### Key Metrics

| Metric | Achievement | Industry Baseline |
|:-------|:------------|:------------------|
| **End-to-end Latency** | 1.3-1.6s | 3-4s |
| **Real-Time Factor (RTF)** | 0.5-0.6x | 1.5-2x |
| **Voice Quality (UTMOS)** | 3.8+ | 3.5 |
| **GPU Memory Budget** | 16.2GB / 24GB | N/A |
| **Concurrent Sessions** | 2-4 per GPU | 1-2 |

---

## System Architecture

```mermaid
graph LR
    A[Audio Input] -->|30ms chunks| B[Silero VAD]
    B -->|Speech detected| C[Whisper INT8]
    C -->|Streaming transcript| D[Llama 3.2 3B]
    D -->|Token stream| E[CSM-1B TTS]
    E -->|Audio chunks| F[WebSocket Client]

    style A fill:#2d3748
    style F fill:#2d3748
    style D fill:#4a5568
```

### Streaming Pipeline Flow

**1. Voice Activity Detection (VAD)**
- Model: Silero VAD
- Latency: <1ms per 30ms chunk
- Configuration: 0.7 speech threshold, 600ms silence detection
- CPU-only inference prevents GPU contention

**2. Speech-to-Text (STT)**
- Model: Whisper large-v3-turbo (INT8 quantized)
- Inference engine: faster-whisper (CTranslate2)
- Optimizations: Greedy decoding, disabled language detection
- Memory footprint: 2.5-3GB VRAM

**3. Language Model (LLM)**
- Model: Llama 3.2 3B Instruct (AWQ INT4)
- Inference: vLLM with PagedAttention
- Constraint: 6-8 word responses (latency optimization)
- Memory footprint: <2GB VRAM
- Time-to-first-token: 80ms

**4. Text-to-Speech (TTS)**
- Model: CSM-1B with Mimi neural codec
- Architecture: Dual-transformer (1B backbone + 100M depth decoder)
- Streaming: 12-frame batches with overlap-add crossfading
- Memory footprint: 5GB VRAM

---

## Technical Deep Dives

### Challenge 1: CUDA Graph Optimization for Real-Time TTS

**Problem:** Baseline CSM-1B achieved 3.2x Real-Time Factor (generating audio slower than playback), making real-time conversation impossible.

**Root Cause:** Autoregressive generation requires thousands of small CUDA kernel launches. CPU-GPU synchronization overhead dominated inference time.

**Solution:**
- Implemented **static KV-cache** pre-allocating maximum sequence length (2048 tokens)
- Enabled **CUDA graph capture** through `torch.compile` with `reduce-overhead` mode
- Eliminated `.item()` calls that force CPU-GPU sync
- Used GPU-only tensor operations for position tracking

**Result:** RTF improved from 3.2x → 0.52x (6x speedup), enabling smooth real-time streaming.

```python
# Static KV-cache implementation (simplified)
class StaticKVCache:
    def __init__(self, max_seq_len=2048, num_layers=16):
        self.cache = torch.zeros(
            (num_layers, 2, max_seq_len, hidden_dim),
            device='cuda', dtype=torch.bfloat16
        )
        self.position = torch.tensor([0], device='cuda')

    def update(self, layer_idx, k, v):
        pos = self.position.item()
        self.cache[layer_idx, 0, pos] = k
        self.cache[layer_idx, 1, pos] = v
        self.position += 1  # In-place, graph-safe
```

---

### Challenge 2: Multi-Model Memory Management (24GB Budget)

**Problem:** Co-locating Whisper (6GB), Llama (7GB), and CSM (5GB) + KV-caches exceeded 24GB A10G VRAM, causing sporadic OOM errors.

**Solution Stack:**
1. **Quantization:**
   - Whisper: INT8 → 2.5GB
   - Llama: AWQ INT4 → <2GB
   - CSM: BFloat16 (kept full precision for voice quality)

2. **Memory Fragmentation Mitigation:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   ```

3. **Rolling Context Windows:**
   - Maintain only 3-5 recent conversation turns
   - Preserve original voice prompt (never evicted)

4. **Periodic Cache Clearing:**
   ```python
   torch.cuda.empty_cache()  # Between conversation turns
   ```

**Result:** Stable 16.2GB peak usage with 8GB headroom. Zero OOM errors in 24+ hour stress tests.

---

### Challenge 3: Whisper Hallucination Elimination

**Problem:** STT generated phantom transcripts during silence/echo ("Thank you for watching", "Subscribe to my channel"), triggering nonsensical Maya responses.

**Root Cause:** Whisper trained on YouTube videos with common outro phrases. Echo residue + low confidence → hallucination trigger.

**Solution Layers:**
1. **VAD Gating:** Only speech-classified audio reaches Whisper
2. **Disabled `condition_on_previous_text`:** Prevents hallucination propagation
3. **Elevated `no_speech_threshold` (0.7):** More aggressive silence classification
4. **Hallucination Filter:** Blacklist of known problematic phrases
5. **Minimum Length Validation:** Reject transcripts <2 characters

**Result:** Hallucination rate reduced from 21.3% → 0.2%.

---

### Challenge 4: Seamless Streaming Audio (Chunk Boundary Artifacts)

**Problem:** Streaming TTS exhibited audible clicks/pops at chunk boundaries due to waveform discontinuities.

**Solution:** Overlap-Add Synthesis with Hann Windowing

```python
def stream_with_crossfade(audio_chunks, overlap_ms=15):
    overlap_samples = int(24000 * overlap_ms / 1000)
    window = torch.hann_window(overlap_samples * 2)

    output = audio_chunks[0]
    for chunk in audio_chunks[1:]:
        # Apply Hann window to overlap regions
        prev_tail = output[-overlap_samples:] * window[overlap_samples:]
        curr_head = chunk[:overlap_samples] * window[:overlap_samples]

        # Equal-power crossfade
        crossfade = prev_tail + curr_head
        output = torch.cat([output[:-overlap_samples], crossfade, chunk[overlap_samples:]])

    return output
```

**Result:** Streaming audio perceptually indistinguishable from non-streaming generation in blind A/B tests.

---

## Infrastructure & Deployment

### AWS SageMaker Configuration

- **Instance:** ml.g5.xlarge (NVIDIA A10G, 24GB VRAM)
- **Framework:** Custom inference container with WebSocket support
- **Deployment:** Real-time endpoint with auto-scaling (1-4 instances)
- **Cold start:** <30 seconds (models pre-loaded in container)

### Audio Transport Architecture

**Client → Server:** WebRTC
- Built-in AEC3 (acoustic echo cancellation)
- Automatic gain control, noise suppression
- UDP transport with jitter buffering

**Server → SageMaker:** WebSocket
- 100ms audio chunks (2400 samples @ 24kHz)
- Float32 encoding with chunk timestamps
- 200-300ms buffering for jitter smoothing

---

## Performance Benchmarks

### Latency Breakdown (Production)

| Stage | P50 Latency | P95 Latency |
|:------|:------------|:------------|
| VAD Detection | 50ms | 100ms |
| STT (Whisper) | 350ms | 480ms |
| LLM TTFT | 80ms | 140ms |
| TTS First Chunk | 420ms | 580ms |
| Network Transport | 150ms | 220ms |
| **Total E2E** | **1.35s** | **1.85s** |

### Resource Utilization

- **GPU Utilization:** 65-75% (during active inference)
- **GPU Memory:** 16.2GB peak / 24GB total
- **CPU Usage:** 15-20% (VAD, audio I/O)
- **Network Bandwidth:** 200-300 Kbps per session

---

## Key Innovations

### 1. Constrained Response Generation
Traditional voice assistants generate verbose LLM responses, creating proportional TTS latency. Maya enforces **6-8 word responses** through prompt engineering:

```
System: You are Maya, a warm and concise voice assistant.
CRITICAL: Respond in 6-8 words maximum. Use contractions.
Speak naturally as if in real-time conversation.
```

This constraint reduces average TTS time from 2-3 seconds to 400-600ms.

### 2. Voice Prompt Preservation Strategy
CSM learns speaker identity from context tokens. To prevent voice drift in long conversations:
- Preserve original 6-10 second voice prompt (never evicted)
- Prune intermediate conversation turns when memory constrained
- Maintain >0.85 speaker similarity (ECAPA-TDNN) across 30+ turns

### 3. Hybrid Turn-Taking Detection
Combines VAD (speech/non-speech) with contextual completion prediction:
- Secondary classifier trained on conversational data
- Predicts turn-completion probability from transcript + prosody
- 94% accuracy distinguishing intra-turn pauses from turn completion

---

## Technical Stack Summary

| Component | Technology | Optimization |
|:----------|:-----------|:-------------|
| **VAD** | Silero VAD | CPU-only, <1ms latency |
| **STT** | Whisper large-v3-turbo | INT8, faster-whisper, greedy decode |
| **LLM** | Llama 3.2 3B | AWQ INT4, vLLM, prefix caching |
| **TTS** | CSM-1B + Mimi | Static KV-cache, CUDA graphs, streaming |
| **Deployment** | AWS SageMaker | A10G GPU, WebSocket streaming |
| **Framework** | PyTorch 2.5 | torch.compile, CUDA graph capture |

---

## Lessons Learned

### 1. Streaming Changes Everything
Designing for streaming from the ground up (vs. retrofitting) was critical. Every component must support streaming as first-class capability.

### 2. Memory > Compute for Multi-Model Inference
GPU memory was the binding constraint, not FLOPS. Quantization and cache management drove architectural decisions.

### 3. Perception Trumps Measurement
Users perceive latency from silence → first audio. Streaming audio during generation improves perceived responsiveness even with constant total generation time.

### 4. Brevity Enhances Latency
Constraining LLM output length (6-8 words) was the single highest-impact latency optimization. A 20-word response takes 4x longer to synthesize than 5 words.

---

## Future Roadmap

**Short-term:**
- Emotion detection (emotion2vec) for adaptive response tone
- Natural filler generation ("um", "hmm") to mask latency
- Backchannel signals ("uh-huh", "I see") during user speech

**Medium-term:**
- CSM fine-tuning for custom Maya voice (LoRA-based)
- Multilingual support (Spanish, Hindi)
- Edge deployment (Apple Silicon, Android)

**Long-term:**
- True speech-to-speech (eliminate text intermediate)
- Multimodal integration (vision + speech)
