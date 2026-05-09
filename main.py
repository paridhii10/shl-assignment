from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from catalog import load_catalogue, product_to_recommendation, validate_catalogue_recommendations
from conversation import decide_next_action


app = FastAPI(title="SHL Assessment Recommender")
CATALOGUE = load_catalogue()
EMPTY_RECOMMENDATION_INTENTS = {"vague_query", "clarify_needed", "compare", "refusal", "prompt_injection"}


def _valid_messages(payload: Any) -> list[dict[str, str]] | None:
    if not isinstance(payload, dict):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None

    valid_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        valid_messages.append({"role": role, "content": content})
    return valid_messages


def _chat_response(
    reply: str,
    recommendations: list[dict[str, str]] | None = None,
    end_of_conversation: bool = False,
) -> dict[str, Any]:
    return {
        "reply": reply,
        "recommendations": recommendations or [],
        "end_of_conversation": end_of_conversation,
    }


def _schema_json_response(
    reply: str,
    recommendations: list[dict[str, str]] | None = None,
    end_of_conversation: bool = False,
) -> JSONResponse:
    return JSONResponse(
        content=_chat_response(reply, recommendations, end_of_conversation),
        status_code=200,
    )


def _recommendations_from_products(products: list[Any]) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for product in products:
        recommendation = product_to_recommendation(product, CATALOGUE)
        url = recommendation["url"]
        if url in seen_urls:
            continue
        recommendations.append(recommendation)
        seen_urls.add(url)
        if len(recommendations) >= 10:
            break
    validate_catalogue_recommendations(recommendations, CATALOGUE)
    return recommendations


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request: Any, _exc: RequestValidationError) -> JSONResponse:
    return _schema_json_response(
        "Please send a JSON body with messages as a list of role/content objects.",
        [],
        False,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "health": "/health",
        "chat": "/chat",
    }


@app.post("/chat")
def chat(payload: Any = Body(default=None)) -> dict[str, Any]:
    messages = _valid_messages(payload)
    if messages is None:
        return _chat_response(
            "Please send a JSON body with messages as a list of role/content objects.",
            [],
            False,
        )

    decision = decide_next_action(messages, CATALOGUE, limit=10)
    if decision.intent in EMPTY_RECOMMENDATION_INTENTS:
        recommendations = []
    else:
        recommendations = _recommendations_from_products(list(decision.products))

    return _chat_response(
        decision.reply,
        recommendations,
        decision.end_of_conversation,
    )
