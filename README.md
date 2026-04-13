# News-Structurizer: Production-Grade Document Classification Pipeline

A production-grade NLP pipeline that classifies news articles into structured categories, extracts named entities, and identifies temporal events. Built to demonstrate end-to-end ML system design — from LLM inference and Pydantic validation through to containerised deployment and persistent storage.

The BBC News dataset is used as a publicly available proxy to demonstrate the same engineering patterns applied to real-world document classification workflows without exposing proprietary or sensitive data.

---

## Architecture

```
src/
├── schemas/        # Pydantic data models and output validation
├── models/         # LLM classification logic (model layer)
├── serving/        # Flask REST API (serving layer)
├── config/         # Settings and environment configuration
└── database/       # PostgreSQL connection, table creation, and queries
tests/              # Pytest unit tests with mocking
Dockerfile          # Container definition with non-root user
docker-compose.yml  # Multi-container orchestration with healthcheck
```

Three layers are kept strictly separate:
- **Model layer** — knows only about classification. No Flask, no database.
- **Serving layer** — knows only about HTTP. Delegates to model and database.
- **Infrastructure** — Docker, PostgreSQL, environment configuration.

This separation means each layer can be tested, replaced, or scaled independently.

---

## Features

- **Document Classification** — classifies articles into business, sports, and entertainment subcategories using GPT-4o-mini via Instructor
- **Named Entity Extraction** — extracts full names and free-text job titles with no fixed predefined list
- **Temporal Event Detection** — high recall extraction of April events using a carefully engineered system prompt
- **Structured Output Validation** — every LLM response is validated through Pydantic schemas before leaving the API. Invalid responses are rejected automatically.
- **Enum Enforcement** — category values are enforced using Python Enum — the LLM cannot return an invalid category
- **PostgreSQL Logging** — every request and classification result is saved permanently to a PostgreSQL database with a timestamp
- **REST API** — Flask serving layer with `/health`, `/classify`, and `/history` endpoints
- **Containerised Deployment** — fully Dockerised with Docker Compose, non-root user, healthcheck-based startup ordering, and environment variable injection
- **Automated Tests** — 10 Pytest tests across schema validation and API behaviour with full mocking of external dependencies

---

## Model Evaluation

Evaluated on 2,225 BBC news articles across business, sport, and entertainment categories. April event detection was evaluated as a binary classification task — whether the model correctly identified articles containing April events.

| Model | Accuracy | Precision | Recall | F1 Score | Approx. Cost per Run |
|-------|----------|-----------|--------|----------|----------------------|
| GPT-3.5 Turbo | 0.5409 | 0.0980 | 0.9722 | 0.1781 | ~$0.80 |
| GPT-4o-mini | 0.8770 | 0.2851 | 0.9306 | 0.4365 | ~$0.40 |
| GPT-4o | 0.8827 | 0.3038 | 1.0000 | 0.4660 | ~$3.00 |

**GPT-4o-mini** was selected for production — best balance of accuracy, recall, and cost.

---

## Example Response

```json
{
  "business": null,
  "sports": "football",
  "entertainment": null,
  "confidence": 0.9,
  "named_entities": [
    {"name": "Thierry Henry", "job": "football player"},
    {"name": "Arsene Wenger", "job": "manager"}
  ],
  "april_events": [
    {
      "event_date": "April",
      "title": "Arsenal vs Chelsea",
      "description": "Arsenal secured a dramatic victory over Chelsea in the Premier League with Thierry Henry scoring twice."
    }
  ]
}
```

---

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key

### Setup

1. Clone the repository and switch to the production branch:

```bash
git clone https://github.com/lekanOyeleye/Structured-Text-Classification-with-Large-Language-Models.git
cd Structured-Text-Classification-with-Large-Language-Models
git checkout prod
```

2. Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_DB=bbc
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

3. Build and start the containers:

```bash
docker compose up --build
```

The API will be available at `http://localhost:5000`.

---

## API Endpoints

### Health Check

```bash
curl http://localhost:5000/health
```

### Classify a Document

```bash
curl -X POST http://localhost:5000/classify -H "Content-Type: application/json" -d '{"text": "Your news article text here"}'
```

### View Classification History

```bash
curl http://localhost:5000/history
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

10 tests covering:
- Pydantic schema validation — accepts valid data, rejects invalid data
- API endpoint behaviour — correct status codes, correct response fields
- Error handling — missing text, empty text
- History endpoint — returns list, handles empty database

All external dependencies (OpenAI, PostgreSQL) are mocked so tests run offline with no API cost.

---

## Design Decisions

**Why Pydantic + Instructor over raw OpenAI?**
Raw OpenAI responses are unstructured strings. Instructor forces the LLM to return a valid Pydantic object every time — invalid responses are automatically retried up to 3 times. This eliminates an entire class of parsing bugs.

**Why PostgreSQL over MongoDB?**
The classification result has a fixed, well-defined structure — categories, confidence score, timestamp. Relational storage is the right fit. Only `named_entities` and `april_events` are variable — stored as JSONB which PostgreSQL handles natively and efficiently.

**Why Docker Compose with healthcheck?**
`depends_on` alone only waits for the container to start — not for PostgreSQL to be ready to accept connections. The healthcheck uses `pg_isready` to confirm PostgreSQL is fully initialised before the API container starts, eliminating connection failures on startup.

**Why a non-root user in Docker?**
Running containers as root means an attacker who exploits the app gets root access to the container. `appuser` limits the blast radius of any security incident.

---

## Tech Stack

- **LLM** — OpenAI GPT-4o-mini via Instructor
- **Validation** — Pydantic v2
- **API** — Flask
- **Database** — PostgreSQL 15 with psycopg2
- **Containerisation** — Docker + Docker Compose
- **Testing** — Pytest with unittest.mock
- **Dataset** — BBC News (Greene & Cunningham, ICML 2006)