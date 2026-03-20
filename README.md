# 🤖 Agentic Candidate Evaluator

> Bias-free · Skills-verified · Multi-agent AI debate · Explainable verdicts

A multi-agent AI system that evaluates job candidates fairly by:
1. **Anonymizing** all PII to eliminate hiring bias
2. **Verifying skills** via FAISS vector similarity (runs fully locally)
3. Running a **3-agent AI debate** (Advocate ↔ Critic ↔ Fairness Judge)
4. Producing an **explainable verdict** with full reasoning trail

---

## 🚀 Quick Start

### Option A — Mock Mode (no API key needed)
```bash
git clone <repo>
cd eightfold
pip install -r requirements.txt
USE_MOCK_LLM=true streamlit run streamlit_app.py
```

### Option B — Live LLM Mode
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY from https://aistudio.google.com/apikey
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### CLI Mode
```bash
python run_pipeline.py
# Outputs: evaluation_report.md, evaluation_report.json, debate_transcript.md
```

---

## 📁 Project Structure

```
eightfold/
├── streamlit_app.py          ← Main UI (run this)
├── run_pipeline.py           ← CLI runner
├── config.py                 ← Central settings (reads .env)
├── requirements.txt
├── .env.example
├── .streamlit/
│   ├── config.toml           ← Dark theme + server config
│   └── secrets.toml.example  ← Streamlit Cloud secrets template
├── packages.txt              ← System deps for Streamlit Cloud
│
├── core/
│   ├── anonymizer.py         ← PII stripping (regex + optional LLM)
│   ├── skill_taxonomy.py     ← Canonical skill normalization
│   ├── vector_engine.py      ← FAISS similarity engine
│   ├── skill_graph.py        ← Gap/extra skill analysis
│   └── cache.py              ← LRU evaluation cache
│
├── agents/
│   ├── advocate.py           ← Argues FOR hiring
│   ├── critic.py             ← Argues AGAINST hiring
│   ├── fairness.py           ← Fact-checks both sides
│   ├── orchestrator.py       ← LangGraph debate state machine
│   └── conversation_log.py   ← Immutable append-only transcript
│
├── report/
│   └── generator.py          ← Markdown + JSON report builder
│
└── sample_data/
    ├── sample_jd.json
    └── sample_candidate.json
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini API key (or use `GOOGLE_API_KEY`) |
| `GOOGLE_API_KEY` | — | Fallback key name |
| `USE_MOCK_LLM` | `false` | Skip all LLM calls, use pre-canned responses |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `DEBATE_ROUNDS` | `2` | Number of advocate/critic debate rounds |

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set **Main file** = `streamlit_app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   GOOGLE_API_KEY = "your_key_here"
   USE_MOCK_LLM = "false"
   ```
5. Click **Deploy**

> **Tip:** Set `USE_MOCK_LLM = "true"` in secrets for a zero-cost always-on demo.

---

## 🧠 How the Pipeline Works

```
Input (JD + Candidate)
        │
        ▼
  1. Anonymize ──── Regex PII removal + optional LLM bias redaction
        │
        ▼
  2. Normalize ──── Canonical skill taxonomy (nodejs → node.js, etc.)
        │
        ▼
  3. FAISS ───────── Local vector similarity (no API cost)
        │
        ▼
  4. Skill Graph ─── Gap analysis, experience comparison
        │
        ▼
  5. AI Debate ───── LangGraph state machine:
        │              Advocate → Critic → Fairness → (repeat N rounds) → Judge
        ▼
  6. Report ──────── Markdown + JSON + immutable transcript
```

### Verdict Options
`STRONG_HIRE` · `HIRE` · `LEAN_HIRE` · `LEAN_NO_HIRE` · `NO_HIRE` · `STRONG_NO_HIRE`

---

## 🛡️ Bias Prevention

- **Names, emails, phone, address** → stripped by regex before any processing
- **University and company names** → replaced with `[UNIVERSITY]` / `[COMPANY]`
- **Gender, age, nationality** → removed entirely
- **Evaluation runs on anonymized text only** — PII never reaches the LLM or embeddings

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (via LangChain) |
| Agent orchestration | LangGraph |
| Vector similarity | FAISS + sentence-transformers |
| Embeddings | `all-MiniLM-L6-v2` (local, no API cost) |
| Skill graph | NetworkX |
| Cache | cachetools LRU |
