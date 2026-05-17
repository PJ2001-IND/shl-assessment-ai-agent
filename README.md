---
title: SHL Assessment Agent
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange?style=flat-square)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-blue?style=flat-square)
![Render](https://img.shields.io/badge/Render-Live-brightgreen?style=flat-square)

> **A stateless conversational AI agent that takes hiring managers from vague intent to a grounded shortlist of SHL assessments.** — Built for the SHL AI Intern Assignment.

---

## 📌 Overview
This project is a retrieval-augmented conversational agent that helps recruiters navigate the SHL Product Catalog. It is powered by a hybrid **FAISS + LLM** pipeline to guarantee strict catalog grounding and zero hallucinations.

The agent manages multi-turn dialogue to:
1. **Clarify** vague intents (e.g., "I need an assessment" ➔ "What is the seniority level?").
2. **Recommend** an exact, grounded shortlist of 1 to 10 assessments.
3. **Refine** shortlists mid-conversation (e.g., "Actually, add a coding simulation").
4. **Compare** assessments using catalog data (e.g., "What is the difference between OPQ and Verify G+?").
5. **Refuse** off-topic prompts, legal questions, and injection attacks.

---

## 🏗️ Architecture
- **Stateless Design:** No databases or session cookies. Every API request includes the full conversation history.
- **Strict Pydantic Schema:** The agent strictly outputs a validated `{"reply", "recommendations", "end_of_conversation"}` object.
- **Vector Search (FAISS):** Fast semantic retrieval over 377 SHL catalog items using offline `all-MiniLM-L6-v2` embeddings (100% offline, zero network latency).
- **Grounded Verification:** A post-generation fuzzy-matcher ensures that 100% of generated names and URLs exactly match the `shl_product_catalog.json`.

---

## 🚀 Run Locally

### Prerequisites
- Python 3.12+
- Groq API Key

### Install
```bash
git clone https://github.com/PJ2001-IND/shl-assessment-agent.git
cd shl-assessment-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_active_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
PORT=8000
```

### Start Server
```bash
# Start with Uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Or start with Docker
docker build -t shl-agent .
docker run -p 8000:8000 --env-file .env shl-agent
```

---

## 🔌 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check — returns `{"status": "ok"}` |
| `/chat` | `POST` | Core conversational endpoint. Accepts `messages` history |
| `/docs` | `GET` | Interactive Swagger UI / OpenAPI documentation |

### Example Request (`POST /chat`)
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"}
  ]
}
```

---

## 🧪 Automated Testing
The project includes a comprehensive, automated grading harness (`test_suite.py`) simulating the SHL evaluation bot.

It tests **22 complete scenarios**, including:
- **Sample Traces:** C1 through C10 multi-turn conversations.
- **Edge Cases:** Turn-limit budgets (max 8 turns), Adaptive/Duration/Language constraints.
- **Attacks:** Prompt injections and off-topic refusals.
- **Schema & Grounding:** URL validation and HTTP 422 triggers.

Run the test suite:
```bash
python test_suite.py
```

---

## 📁 Project Structure
```
shl-assessment-agent/
├── .env.example             # Template for environment variables
├── .embeddings_cache.npy    # Pre-computed FAISS embeddings (for fast cold-starts)
├── README.md                # This file
├── Dockerfile               # Container config (Port 7860)
├── requirements.txt         # Dependencies
├── app.py                   # FastAPI application & endpoints
├── catalog.py               # FAISS vector database and fuzzy-matcher
├── config.py                # Environment configuration & globals
├── engine.py                # Core LLM orchestrator & retrieval logic
├── models.py                # Pydantic schemas for requests/responses
├── prompts.py               # LLM instructions & few-shot examples
├── shl_product_catalog.json # The raw SHL catalog database
├── test_suite.py            # Automated multi-turn test harness
└── approach_document.md     # Architectural design & evaluation explanation
```

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| **Python 3.12** | Core programming language |
| **FastAPI** | High-performance async web framework |
| **Pydantic v2** | Strict API schema validation |
| **Groq SDK** | `llama-3.3-70b-versatile` for high-speed, grounded reasoning |
| **SentenceTransformers** | `all-MiniLM-L6-v2` for 100% offline local query embeddings |
| **FAISS** | High-speed, in-memory semantic vector search |
| **Docker** | Containerization for easy deployment |

---

## 👤 Author
**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- Render Endpoint: [shl-assessment-agent-2cln.onrender.com](https://shl-assessment-agent-2cln.onrender.com)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

## 📄 License
MIT
