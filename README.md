# Conversational SHL Assessment Recommender

Deterministic FastAPI service for the SHL AI Intern take-home assignment. It recommends SHL assessments from the provided catalogue only, supports clarification/refinement/comparison/refusal behavior, and returns the exact evaluator schema.

## Project Structure

- `main.py` - FastAPI app with `/health` and `/chat`.
- `catalog.py` - robust catalogue loader, JSON repair, catalogue maps, catalogue-only recommendation validation.
- `schemas.py` - `Product` model and deterministic `test_type` derivation.
- `recommender.py` - deterministic hybrid-lite retrieval and ranking.
- `rules.py` - curated semantic aliases and boost rules.
- `conversation.py` - stateless conversation parsing, intent detection, clarification/refinement/comparison/refusal/finalization.
- `tests/` - catalogue, retrieval, conversation, and API tests.

## Requirements

Python 3.11+ is recommended.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The app includes the SHL catalogue at:

```text
data/shl_product_catalog.json
```

By default, `catalog.py` loads that repository-local file so hosted deployments do not depend on a local Windows Downloads path. To use another path, set:

```powershell
$env:SHL_CATALOG_PATH="C:\path\to\shl_product_catalog.json"
```

## Run Locally

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

Chat example:

```powershell
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hiring a senior backend engineer with Core Java, Spring, SQL, AWS, and Docker.\"}]}"
```

## API Contract

`GET /health`

```json
{"status":"ok"}
```

`POST /chat`

Request:

```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"}
  ]
}
```

Response:

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "string"}
  ],
  "end_of_conversation": false
}
```

The service is stateless. Every `/chat` call recomputes context from the supplied `messages` history.

## Guardrails

- Products are loaded from the catalogue JSON only.
- URLs are copied from catalogue products only.
- Recommendation dicts are produced only from `Product` objects.
- API boundary deduplicates recommendations and caps output at 10.
- Clarification, comparison, refusal, and prompt-injection responses return `recommendations: []`.
- Malformed payloads return the exact chat response schema with empty recommendations.

## Tests

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

Current verified result:

```text
Ran 58 tests
OK
```

## Deployment Notes

Use any Python web host that supports FastAPI/Uvicorn, such as Render, Railway, Fly.io, or Hugging Face Spaces. Ensure:

- `requirements.txt` is installed.
- The catalogue JSON is included at `data/shl_product_catalog.json` or `SHL_CATALOG_PATH` points to it.
- The service start command is similar to:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

If the platform does not set `PORT`, use `8000`.
