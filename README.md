# IPL Copilot

IPL Copilot is a Python/FastAPI cricket analytics assistant backed by an IPL ball-by-ball CSV dataset.
It combines deterministic metric SQL, read-only DuckDB execution, entity resolution, session history,
and optional Groq/LangChain assistance for queries that are not covered by the fast path.

## Architecture

```text
User Question
  -> FastAPI /ask
  -> Session follow-up context
  -> Entity resolution
  -> Deterministic metric/ranking SQL or optional Groq SQL planning
  -> DuckDB read-only execution
  -> Verified response formatting
```

## Technology

- Python
- FastAPI
- Pydantic
- Pandas
- DuckDB
- Groq via LangChain
- python-dotenv

There is currently no Next.js frontend, LangGraph workflow, ChromaDB/RAG store, or separate database
in this repository.

## Dataset

The default dataset is `data/ipl_ball_by_ball_cleaned.csv` with 225,954 rows and 17 columns.
Set `IPL_DATASET_PATH` to use another CSV with the same schema.

The dataset preparation script now validates the required columns and writes a deduplicated cleaned copy:

```powershell
.\myvenv\Scripts\python.exe src\data_preperation.py
```

## Setup

```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env` if LLM-assisted planning is required. Common rankings and metrics do not
need a Groq API key.

Start the API:

```powershell
.\myvenv\Scripts\python.exe -m uvicorn src.main:app --reload
```

API documentation:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

## API

### `POST /ask`

```json
{"question":"Top 10 batters","session_id":null}
```

Response:

```json
{"answer":"...","session_id":"uuid"}
```

### Session endpoints

- `POST /session`
- `GET /sessions/{session_id}`
- `DELETE /session/{session_id}`

Sessions are in-memory and reset when the API process restarts.

## Environment

| Variable | Required | Description |
|---|---:|---|
| `GROQ_API_KEY` | No | Enables Groq/LangChain planning |
| `GROQ_MODEL` | No | Defaults to `llama-3.3-70b-versatile` |
| `IPL_DATASET_PATH` | No | CSV path; defaults to the cleaned project dataset |
| `CORS_ORIGINS` | No | Comma-separated allowed browser origins |

Never commit real API keys or tokens.

## Testing

```powershell
.\myvenv\Scripts\python.exe -m unittest src.test_reliability src.test_executor
.\myvenv\Scripts\python.exe src\benchmark_sql.py
```

The benchmark validates SQL output against independent Pandas calculations for row count,
total runs, and top 10 batters.

## Known limitations

- No persistent session storage
- No frontend in the repository
- No live match data
- No RAG/knowledge base
- Non-ranking/custom analytical questions still depend on optional Groq planning
- LLM call failures and unsupported entities are reported honestly
