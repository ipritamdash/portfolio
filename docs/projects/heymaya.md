# HeyMaya: Production LLM Platform for WhatsApp Recruitment

**Role:** ML Engineering Advisor | **Stack:** Llama 3.1 8B, QLoRA, AWS SageMaker, vLLM, Docker
**Timeline:** Oct 2025 - Present | **Status:** Production (150K+ Users)

[View Code on GitHub →](https://github.com/ipritamdash/hr-ai-project){ .md-button .md-button--primary }

---

## Overview

HeyMaya is a WhatsApp-based AI recruitment platform serving 150,000+ users with conversational HR workflows. Built on fine-tuned Llama 3.1 8B, the system handles multi-persona conversations (HR managers, job candidates, clients, freelancers) with context-aware dialogue management.

### Key Metrics

| Metric | Achievement |
|:-------|:------------|
| **Active Users** | 150,000+ |
| **Response Latency** | <2s (p95: 200ms) |
| **Hallucination Rate** | <3% |
| **Daily Conversations** | 10,000+ |
| **Model Size** | 8B parameters (4-bit quantized) |

---

## System Architecture

### Multi-Persona Conversation Engine

The platform supports four distinct user types with specialized system prompts and conversation flows:

**1. HR Manager Flow**
- Job posting creation and editing
- Candidate screening and filtering
- Interview scheduling
- Hiring decision support

**2. Job Candidate Flow**
- Profile onboarding (resume parsing)
- Job matching and recommendations
- Application tracking
- Interview preparation

**3. Client Flow**
- Hiring requirements specification
- Candidate shortlist review
- Contract management
- Billing and invoicing

**4. Freelancer Flow**
- Skill verification and portfolio review
- Project matching
- Payment milestone tracking
- Availability management

---

## Technical Implementation

### Model Fine-Tuning

**Base Model:** Llama 3.1 8B Instruct

**Training Approach:**
- **Method:** QLoRA (4-bit NF4 quantization with LoRA adapters)
- **Dataset:** 50,000+ HR conversation turns (synthesized + human-labeled)
- **Training Hardware:** AWS SageMaker ml.g5.2xlarge (A10G GPU)
- **Training Duration:** 12 hours (8 epochs)
- **LoRA Configuration:** r=16, alpha=32, dropout=0.1
- **Target Modules:** q_proj, v_proj, o_proj, gate_proj

**Training Pipeline:**
```python
# Key training configuration
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Dataset Composition:**
- Conversational HR dialogues: 35,000 turns
- Multi-turn context (3-10 exchanges): 12,000 sequences
- Error recovery scenarios: 3,000 turns
- Validation set: 5,000 turns

---

### Deployment Architecture

**Inference Stack:**
- **Framework:** vLLM with PagedAttention
- **Quantization:** AWQ 4-bit (post-training optimization)
- **Hardware:** AWS SageMaker ml.g5.xlarge (24GB GPU)
- **Batching:** Dynamic batching with 50ms timeout
- **Caching:** Prefix caching for system prompts

**Infrastructure:**
- **Container:** Custom Docker image with vLLM 0.4.2
- **Endpoint:** SageMaker real-time endpoint with autoscaling
- **Load Balancing:** 2-8 instances based on request volume
- **Cold Start:** <30 seconds (model pre-loaded in container)

**API Integration:**
- **WhatsApp Business API** for message delivery
- **PostgreSQL** for conversation state management
- **Redis** for session caching
- **AWS Lambda** for webhook handling

---

## Prompt Engineering

### Context-Aware Conversation Management

Each conversation maintains a rolling context window optimized for mobile latency:

**Context Structure:**
```
[System Prompt: 200 tokens]
[User Persona: HR Manager]
[Conversation History: Last 5 turns, ~300 tokens]
[Current User Message: ~50 tokens]
---
Total Context: ~550 tokens (target: <800 for <200ms latency)
```

**Persona-Specific System Prompts:**

```
# HR Manager Prompt (Example)
You are Maya, an AI recruitment assistant helping HR managers.

Your role:
1. Help create and refine job postings
2. Screen candidates based on requirements
3. Schedule interviews and manage pipelines
4. Provide data-driven hiring insights

Response guidelines:
- Be professional yet conversational
- Ask clarifying questions when requirements are ambiguous
- Provide structured data (candidate lists, schedules) in clean format
- Keep responses under 100 words for mobile readability
```

### Hallucination Mitigation

**Challenge:** LLMs hallucinate job details, candidate information, or company policies.

**Solution Layers:**
1. **Grounding with Database Retrieval:** All factual claims verified against PostgreSQL
2. **Confidence Scoring:** Model outputs calibrated probability; low-confidence responses trigger human handoff
3. **Template Constraints:** Critical info (job IDs, candidate names) uses structured templates
4. **Fact-Checking Layer:** Post-processing validates entity consistency

**Result:** Hallucination rate reduced from 18% (base model) to <3% (fine-tuned + guardrails)

---

## Performance Optimization

### Latency Breakdown

| Stage | P50 Latency | P95 Latency |
|:------|:------------|:------------|
| WhatsApp Message Receipt | 50ms | 100ms |
| Lambda Webhook Processing | 20ms | 40ms |
| SageMaker Inference | 120ms | 200ms |
| Database State Update | 15ms | 30ms |
| WhatsApp Message Send | 80ms | 150ms |
| **Total End-to-End** | **285ms** | **520ms** |

### Inference Optimization Strategies

**1. Prefix Caching**
- System prompts (200 tokens) cached per persona
- Saves ~40ms per request

**2. Dynamic Batching**
- Batch incoming requests with 50ms window
- Achieves 3-8x throughput increase during peak hours

**3. Model Quantization**
- AWQ 4-bit quantization post-training
- Maintains 98% of full-precision quality
- Reduces memory footprint from 16GB → 4.5GB

**4. Speculative Decoding** (Under Testing)
- Draft model: Llama 3.2 1B
- Target model: Llama 3.1 8B
- Expected 1.5-2x speedup for typical responses

---

## Monitoring & Observability

**Key Metrics Tracked:**
- **Latency:** p50, p95, p99 response times
- **Quality:** Hallucination rate (human eval on 500 samples/week)
- **Engagement:** Conversation completion rate, user satisfaction (CSAT)
- **Reliability:** Endpoint uptime, error rate, retry rate

**Alerting Thresholds:**
- Latency p95 > 500ms → Page on-call
- Hallucination rate > 5% → Trigger review
- Error rate > 2% → Scale up endpoints

**Cost Optimization:**
- Average cost per conversation: $0.003 (inference + infrastructure)
- Monthly infrastructure: ~$2,500 (2-4 SageMaker instances + Lambda + RDS)

---

## Technical Challenges & Solutions

### Challenge 1: Mobile Network Latency

**Problem:** Users on 3G/4G networks experience high variability in message delivery times.

**Solution:**
- Implemented optimistic UI updates (show "Maya is typing..." immediately)
- Added retry logic with exponential backoff for message sending
- Compressed responses to minimize payload size (<500 characters target)

**Result:** User-perceived latency reduced by 40% despite unchanged inference time.

---

### Challenge 2: Multi-Turn Context Management

**Problem:** Conversations spanning 20+ turns exceeded context window (4096 tokens), causing context truncation and coherence loss.

**Solution:**
- **Sliding Window:** Keep first turn (user intent) + last 5 turns
- **Semantic Compression:** Summarize older turns using smaller model (Llama 3.2 1B)
- **Entity Memory:** Extract and preserve key entities (names, job IDs) across truncations

**Result:** Context coherence maintained across 50+ turn conversations while staying under 800 tokens.

---

### Challenge 3: Dataset Quality

**Problem:** Initial synthetic dataset (GPT-4 generated) had unnatural phrasing and didn't match real user behavior.

**Solution:**
- Bootstrapped with 500 human-labeled conversations
- Used base Llama model to generate synthetic expansions
- Filtered synthetic data through human review (20% sample rate)
- Iteratively fine-tuned and collected production data for retraining

**Result:** Task completion rate improved from 62% → 87% after second training iteration.

---

## Impact & Metrics

**User Growth:**
- Launch (Oct 2025): 1,000 users
- Month 2 (Nov 2025): 25,000 users
- Current (Jan 2026): 150,000+ users

**Business Metrics:**
- 10,000+ daily conversations
- 87% task completion rate
- 4.2/5 average user satisfaction (CSAT)
- 68% of candidates prefer WhatsApp over traditional job portals

**Technical Achievements:**
- <2s response latency at 150K user scale
- 99.8% endpoint uptime
- <3% hallucination rate (validated through human eval)
- $0.003 per conversation (10x cheaper than GPT-4 API)

---

## Future Roadmap

**Short-Term (Q1 2026):**
- Voice message support (integrate Whisper for ASR)
- Multilingual support (Hindi, Tamil)
- Resume parsing and skills extraction

**Medium-Term (Q2 2026):**
- Video interview scheduling with calendar integration
- Candidate assessment automation (coding challenges, skill tests)
- Advanced analytics dashboard for recruiters

**Long-Term (H2 2026):**
- Candidate-job matching with learned embeddings
- Predictive hiring success modeling
- Integration with HRIS systems (Workday, SAP)

---

## Technical Stack Summary

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Base Model** | Llama 3.1 8B Instruct | Conversational reasoning |
| **Fine-Tuning** | QLoRA (4-bit NF4) | Parameter-efficient training |
| **Inference** | vLLM + AWQ quantization | Fast, memory-efficient serving |
| **Deployment** | AWS SageMaker | Managed ML inference |
| **Messaging** | WhatsApp Business API | User communication |
| **State Management** | PostgreSQL + Redis | Conversation persistence |
| **Orchestration** | AWS Lambda + Docker | Serverless coordination |
| **Monitoring** | CloudWatch + Custom Dashboard | Observability |

---

## Key Learnings

### 1. Fine-Tuning Beats Prompt Engineering at Scale
For specialized domains (HR conversations), fine-tuning provides 3-4x quality improvement over GPT-4 with heavy prompting. The model learns domain conventions and reduces verbose instructions.

### 2. Latency is Product Experience
Users perceive 300ms responses as "instant," 500ms as "fast," and >1s as "slow." Every 100ms improvement measurably impacts engagement metrics.

### 3. Quantization is Production-Critical
4-bit quantization enabled single-GPU deployment, reducing infrastructure costs by 75% while maintaining 98% quality. This made the economics viable.

### 4. Context Window Management is Underrated
Naive context handling breaks down after 10-15 turns. Semantic compression and entity memory are essential for coherent long conversations.

### 5. Dataset Quality > Dataset Size
500 high-quality human conversations outperformed 10,000 synthetic examples. Investing in data curation yielded higher ROI than model scaling.

---

## Links & Resources

- [AWS SageMaker Deployment Guide](https://docs.aws.amazon.com/sagemaker/)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
