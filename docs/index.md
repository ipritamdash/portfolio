# Pritam Dash

<div class="hero">
  <p class="hero-label">ML Engineer</p>
  <h1 class="hero-title">Building production ML systems that scale</h1>
  <p class="hero-description">
    I engineer machine learning systems from research to production. Specialized in LLM fine-tuning, real-time inference optimization, and audio ML. Currently serving 150K+ users with conversational AI on WhatsApp.
  </p>

  <div class="hero-stats">
    <div class="stat">
      <span class="stat-value">150K+</span>
      <span class="stat-label">Production Users</span>
    </div>
    <div class="stat">
      <span class="stat-value">200ms</span>
      <span class="stat-label">P95 Latency</span>
    </div>
    <div class="stat">
      <span class="stat-value">53%</span>
      <span class="stat-label">Better than SOTA</span>
    </div>
  </div>

  [View Resume](resume.pdf){ .md-button .md-button--primary }
  [GitHub](https://github.com/ipritamdash){ .md-button }
  [LinkedIn](https://linkedin.com/in/pritam-dash){ .md-button }
</div>

---

## Projects

<div class="project-card">
  <h3>HeyMaya - WhatsApp Recruitment AI</h3>
  <div class="project-meta">
    <span class="project-badge">Production</span>
    <span>150K+ Users</span>
    <span>Oct 2025 - Present</span>
  </div>

  <p class="project-description">
    Conversational AI platform automating HR recruitment workflows on WhatsApp. Fine-tuned Llama 3.1 8B using QLoRA on 50K+ HR dialogues, deployed on AWS SageMaker with vLLM for real-time inference. Serves 10K+ daily conversations with 97% accuracy while reducing infrastructure costs by 40%.
  </p>

  <div class="project-metrics">
    <div class="metric">
      <span class="metric-value">150K+</span>
      <span class="metric-label">Active Users</span>
    </div>
    <div class="metric">
      <span class="metric-value">200ms</span>
      <span class="metric-label">P95 Latency</span>
    </div>
    <div class="metric">
      <span class="metric-value">97%</span>
      <span class="metric-label">Accuracy</span>
    </div>
    <div class="metric">
      <span class="metric-value">40%</span>
      <span class="metric-label">Cost Reduction</span>
    </div>
  </div>

  <div class="project-tech">
    <span class="tech-tag">Llama 3.1 8B</span>
    <span class="tech-tag">QLoRA</span>
    <span class="tech-tag">AWS SageMaker</span>
    <span class="tech-tag">vLLM</span>
    <span class="tech-tag">4-bit Quantization</span>
  </div>

  [View Details](projects/heymaya.md){ .md-button .md-button--primary }
  [GitHub](https://github.com/ipritamdash/hr-ai-project){ .md-button }
</div>

<div class="project-card">
  <h3>Indian Classical Music Separation</h3>
  <div class="project-meta">
    <span class="project-badge">SOTA</span>
    <span>53% Better than Demucs</span>
    <span>Dec 2025</span>
  </div>

  <p class="project-description">
    Deep learning system for separating instruments from dense Indian classical music. Fine-tuned Band-Split RoFormer with custom consensus loss function, achieving 12.45 dB SDR on vocals - 53% better than Meta's Demucs v4. Trained on consumer GPU (T4) using mixed-precision and gradient checkpointing.
  </p>

  <div class="project-metrics">
    <div class="metric">
      <span class="metric-value">12.45 dB</span>
      <span class="metric-label">Vocals SDR</span>
    </div>
    <div class="metric">
      <span class="metric-value">11.30 dB</span>
      <span class="metric-label">Drums SDR</span>
    </div>
    <div class="metric">
      <span class="metric-value">53%</span>
      <span class="metric-label">vs Demucs v4</span>
    </div>
    <div class="metric">
      <span class="metric-value">8 hours</span>
      <span class="metric-label">Training Time</span>
    </div>
  </div>

  <div class="project-tech">
    <span class="tech-tag">PyTorch</span>
    <span class="tech-tag">Band-Split RoFormer</span>
    <span class="tech-tag">Mixed-Precision</span>
    <span class="tech-tag">Custom Loss</span>
  </div>

  [View Details](projects/music.md){ .md-button .md-button--primary }
  [GitHub](https://github.com/ipritamdash/indian-art-music-separation){ .md-button }
</div>

<div class="project-card">
  <h3>RAG Research Backend</h3>
  <div class="project-meta">
    <span class="project-badge">Production-Ready</span>
    <span>60% Accuracy Boost</span>
    <span>2024</span>
  </div>

  <p class="project-description">
    Retrieval-augmented generation system grounding LLM responses in real-time arXiv papers. Built FastAPI backend with in-memory FAISS for sub-100ms vector search across 10K+ documents. Integrates AWS Bedrock (Claude 3) for answer generation with 60% accuracy improvement over baseline LLM.
  </p>

  <div class="project-metrics">
    <div class="metric">
      <span class="metric-value"><100ms</span>
      <span class="metric-label">Search Latency</span>
    </div>
    <div class="metric">
      <span class="metric-value">60%</span>
      <span class="metric-label">Accuracy Boost</span>
    </div>
    <div class="metric">
      <span class="metric-value">10K+</span>
      <span class="metric-label">Documents</span>
    </div>
    <div class="metric">
      <span class="metric-value">200/min</span>
      <span class="metric-label">Throughput</span>
    </div>
  </div>

  <div class="project-tech">
    <span class="tech-tag">FastAPI</span>
    <span class="tech-tag">FAISS</span>
    <span class="tech-tag">AWS Bedrock</span>
    <span class="tech-tag">sentence-transformers</span>
  </div>

  [View Details](projects/rag.md){ .md-button .md-button--primary }
  [GitHub](https://github.com/ipritamdash/rag-research-backend){ .md-button }
</div>

---

## Experience

### ML Engineering Advisor • HeyMaya (Meshly Technologies)
**Oct 2025 - Present • Remote**

- Built conversational AI platform serving 150K+ users on WhatsApp with 200ms response times
- Fine-tuned Llama 3.1 8B using QLoRA (LoRA rank 16, 4-bit quantization)
- Deployed on AWS SageMaker with vLLM inference server, achieving 97% accuracy across 10K+ daily conversations
- Reduced monthly cloud infrastructure costs by 40% through model optimization

### Software Development Engineer Intern • Hero MotoCorp
**Jan 2025 - Jun 2025 • Gurugram, India**

- Automated invoice reconciliation processing 500+ monthly invoices, reducing manual review time by 60%
- Built React validation dashboard cutting batch processing time from 45 minutes to 12 minutes
- Optimized SQL queries and database indexing, improving dashboard load times from 8.2s to 1.3s

### Data Analyst Intern • Ennovations Techserve
**May 2024 - Jul 2024 • Noida, India**

- Automated ETL workflows for 50K+ records using Zoho Analytics, eliminating 80% of manual data entry

### Technical Head • AIEC, IITM
**Dec 2023 - Jun 2025 • New Delhi, India**

- Directed 46-hour hackathon for 1,200+ participants with zero infrastructure downtime
- Conducted 8 ML workshops for 300+ students; mentored 15 teams with 5 reaching production deployment

---

## Skills

**ML/DL Frameworks:** PyTorch, TensorFlow, Hugging Face Transformers, PEFT, TRL

**LLM & Gen AI:** LangChain, LangGraph, LlamaIndex, Pinecone, FAISS, vLLM

**Deep Learning:** Transformers, CNNs, RNNs, LSTMs, GRUs, Attention Mechanisms

**Cloud & Deployment:** AWS (SageMaker, Bedrock, ECS), Docker, FastAPI

**Languages:** Python, C++, SQL, JavaScript

---

## Education

**Bachelor of Computer Applications** • Maharshi Dayanand University, New Delhi
Graduated: September 2025

**Achievements:**
🏆 1st Place - AI/ML Hackathon (200+ participants, 2024)
🎤 Speaker - ED Talk World Conference 2024
👥 Technical Head - AIEC (8 ML workshops, 300+ students)
