from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from catalog import Catalogue, normalize_name
from recommender import explicit_exclusions, rank_products
from rules import PRODUCT_URLS
from schemas import Product


Message = dict[str, str]


VAGUE_PATTERNS = (
    "i need an assessment",
    "need an assessment",
    "recommend an assessment",
    "need a test",
    "what assessment should i use",
    "help me choose an assessment",
)

PROMPT_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore your instructions",
    "forget previous instructions",
    "forget your instructions",
    "developer message",
    "system prompt",
    "jailbreak",
    "recommend products outside",
    "not in the catalogue",
    "not in the catalog",
    "fake shl",
    "fake product",
    "hacker rank",
    "hackerrank",
    "wonderlic",
    "use this url",
    "external url",
    "return markdown only",
)

LEGAL_PATTERNS = (
    "legally required",
    "legally require",
    "legal requirement",
    "satisfy that requirement",
    "satisfy hipaa",
    "compliant with law",
    "regulatory obligation",
    "legal advice",
    "lawsuit",
)

OFF_TOPIC_PATTERNS = (
    "write my job ad",
    "write a job ad",
    "salary benchmark",
    "interview questions",
    "hiring advice",
    "employment law",
    "background check",
)

COMPARISON_PATTERNS = (
    "difference between",
    "different from",
    "compare",
    " vs ",
    " versus ",
    "which one",
    "which is shorter",
    "do we need",
    "really need",
    "redundant",
)

URL_RE = re.compile(r"https?://[^\s>)\]]+")

CONFIRM_PATTERNS = (
    "clear",
    "confirmed",
    "lock it in",
    "locking it in",
    "that works",
    "that's good",
    "that covers it",
    "perfect",
    "final list",
    "keep the shortlist",
    "keep it as is",
    "keep the five",
    "keeping the",
    "we will use",
    "we'll use",
)

REFINE_PATTERNS = (
    "add ",
    "drop ",
    "remove ",
    "exclude ",
    "replace ",
    "instead",
    "actually",
    "shorter",
    "quick",
    "simulation",
    "cognitive",
    "personality",
)

ROLE_SIGNALS = (
    "developer",
    "engineer",
    "analyst",
    "admin",
    "assistant",
    "agent",
    "operator",
    "sales",
    "graduate",
    "trainee",
    "leadership",
    "director",
    "cxo",
    "healthcare",
    "contact center",
    "contact centre",
    "call center",
    "call centre",
    "frontline",
    "warehouse",
    "manufacturing",
    "manager",
)

SKILL_SIGNALS = (
    "java",
    "spring",
    "sql",
    "aws",
    "docker",
    "python",
    "javascript",
    "excel",
    "word",
    "hipaa",
    "finance",
    "safety",
    "networking",
    "rust",
    "personality",
    "cognitive",
    "reasoning",
    "simulation",
    "analytical",
    "problem solving",
    "communication",
    "compliance",
    "dependable",
    "procedures",
)

COMPARISON_ALIASES = {
    "opq": "opq32r",
    "opq32r": "opq32r",
    "verify g": "verify_g_plus",
    "verify g+": "verify_g_plus",
    "g+": "verify_g_plus",
    "dsi": "dsi",
    "safety and dependability": "safety_dependability",
    "safety dependability": "safety_dependability",
    "svar": "svar_spoken_english_us",
    "excel": "ms_excel",
    "word": "ms_word",
}


@dataclass(frozen=True)
class ConversationState:
    context: str
    last_user_message: str
    prior_shortlist: tuple[Product, ...] = field(default_factory=tuple)
    turn_count: int = 0


@dataclass(frozen=True)
class ConversationDecision:
    intent: str
    reply: str
    products: tuple[Product, ...] = field(default_factory=tuple)
    end_of_conversation: bool = False
    state: ConversationState | None = None
    debug: dict[str, Any] = field(default_factory=dict)


def normalize_messages(messages: list[Message]) -> list[Message]:
    normalized: list[Message] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


def flatten_messages(messages: list[Message]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def last_user_message(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    text_norm = f" {normalize_name(text)} "
    for pattern in patterns:
        pattern_norm = normalize_name(pattern)
        if f" {pattern_norm} " in text_norm:
            return True
    return False


def extract_prior_shortlist(messages: list[Message], catalogue: Catalogue) -> tuple[Product, ...]:
    seen_urls: set[str] = set()
    products: list[Product] = []

    for message in messages:
        content = message["content"]
        for match in URL_RE.finditer(content):
            url = match.group(0).rstrip(".,;:")
            product = catalogue.by_url.get(url)
            if product is not None and url not in seen_urls:
                products.append(product)
                seen_urls.add(url)

    if products:
        return tuple(products)

    # Fallback for tests or future model text that mentions names without URLs.
    assistant_text = "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )
    assistant_norm = f" {normalize_name(assistant_text)} "
    for product in catalogue.products:
        product_norm = normalize_name(product.name)
        if product_norm and f" {product_norm} " in assistant_norm and product.link not in seen_urls:
            products.append(product)
            seen_urls.add(product.link)
    return tuple(products)


def build_state(messages: list[Message], catalogue: Catalogue) -> ConversationState:
    normalized = normalize_messages(messages)
    return ConversationState(
        context=flatten_messages(normalized),
        last_user_message=last_user_message(normalized),
        prior_shortlist=extract_prior_shortlist(normalized, catalogue),
        turn_count=len(normalized),
    )


def has_enough_context(text: str) -> bool:
    text_norm = normalize_name(text)
    if contains_any(text, VAGUE_PATTERNS) and len(text_norm.split()) <= 8:
        return False

    return contains_any(text, ROLE_SIGNALS) or contains_any(text, SKILL_SIGNALS)


def clarification_question(state: ConversationState) -> str | None:
    user_text = state.last_user_message
    context_norm = normalize_name(state.context)

    if not has_enough_context(state.context) and not state.prior_shortlist:
        return "What role are you hiring for, and what capability do you most need to assess?"

    if (
        not state.prior_shortlist
        and contains_any(user_text, ("senior leadership", "leadership solution", "executive solution"))
        and not contains_any(state.context, ("selection", "development", "benchmark", "newly created"))
    ):
        return "Is this for selection against a leadership benchmark, or development feedback for leaders already in role?"

    if (
        not state.prior_shortlist
        and contains_any(state.context, ("healthcare", "hipaa", "patient records"))
        and contains_any(state.context, ("spanish", "bilingual"))
        and not contains_any(state.context, ("hybrid", "english fluent", "functionally bilingual", "english-only"))
    ):
        return "Can candidates complete healthcare knowledge tests in English, or do you need a Spanish-only battery?"

    if (
        contains_any(state.context, ("contact center", "contact centre", "call center", "call centre"))
        and contains_any(state.context, ("spoken", "calls", "language", "accent"))
        and not contains_any(state.context, ("english", "spanish", "us", "uk", "indian", "australian"))
    ):
        return "What language and accent should the spoken-language assessment target?"

    tech_terms = sum(
        1
        for term in ("java", "spring", "angular", "sql", "aws", "docker", "rest")
        if re.search(rf"\b{re.escape(term)}\b", context_norm)
    )
    if (
        not state.prior_shortlist
        and tech_terms >= 5
        and len(context_norm.split()) > 24
        and not contains_any(state.context, ("backend", "frontend", "balanced"))
    ):
        return "Is this backend-leaning, frontend-heavy, or a balanced full-stack role?"

    return None


def is_prompt_injection(text: str) -> bool:
    if contains_any(text, PROMPT_INJECTION_PATTERNS):
        return True
    for match in URL_RE.finditer(text):
        url = match.group(0).lower()
        if "shl.com" not in url:
            return True
    return False


def is_legal_or_off_topic(text: str) -> bool:
    return contains_any(text, LEGAL_PATTERNS) or contains_any(text, OFF_TOPIC_PATTERNS)


def is_compare(text: str) -> bool:
    return contains_any(text, COMPARISON_PATTERNS)


def is_confirm(text: str) -> bool:
    return contains_any(text, CONFIRM_PATTERNS)


def is_refine(text: str) -> bool:
    return contains_any(text, REFINE_PATTERNS)


def identify_referenced_products(
    text: str,
    catalogue: Catalogue,
    prior_shortlist: tuple[Product, ...] = (),
) -> tuple[Product, ...]:
    text_norm = f" {normalize_name(text)} "
    found: list[Product] = []
    seen: set[str] = set()

    for alias, product_key in COMPARISON_ALIASES.items():
        if f" {normalize_name(alias)} " in text_norm:
            product = catalogue.by_url.get(PRODUCT_URLS[product_key])
            if product is not None and product.link not in seen:
                found.append(product)
                seen.add(product.link)

    search_space = list(prior_shortlist) + list(catalogue.products)
    for product in search_space:
        product_norm = normalize_name(product.name)
        if product_norm and f" {product_norm} " in text_norm and product.link not in seen:
            found.append(product)
            seen.add(product.link)
            continue

        name_tokens = [token for token in product_norm.split() if len(token) >= 4]
        if (
            product in prior_shortlist
            and name_tokens
            and any(f" {token} " in text_norm for token in name_tokens)
            and product.link not in seen
        ):
            found.append(product)
            seen.add(product.link)

    if len(found) < 2 and prior_shortlist:
        for product in prior_shortlist:
            if product.link not in seen:
                found.append(product)
                seen.add(product.link)
            if len(found) >= 2:
                break

    return tuple(found[:3])


def product_field_summary(product: Product) -> str:
    fields = [
        f"name: {product.name}",
        f"test_type: {product.test_type}",
        f"keys: {', '.join(product.keys) if product.keys else 'not listed'}",
        f"duration: {product.duration or 'not listed'}",
        f"job_levels: {', '.join(product.job_levels[:6]) if product.job_levels else 'not listed'}",
        f"languages: {', '.join(product.languages[:6]) if product.languages else 'not listed'}",
        f"description: {product.description or 'not listed'}",
    ]
    return "\n".join(f"- {field}" for field in fields)


def build_comparison_reply(products: tuple[Product, ...]) -> str:
    if len(products) < 2:
        return (
            "I can compare catalogue-backed SHL products, but I need two product "
            "names from the catalogue to compare."
        )

    sections = ["Catalogue-backed comparison:"]
    for product in products:
        sections.append(f"\n{product.name}\n{product_field_summary(product)}")
    sections.append(
        "\nI am only using catalogue fields here, so I am not adding unsupported "
        "claims about validity, norms, or legal suitability."
    )
    return "\n".join(sections)


def apply_refinement_to_prior(
    state: ConversationState, catalogue: Catalogue, limit: int
) -> tuple[Product, ...]:
    exclusions = explicit_exclusions(state.context)
    prior = [
        product
        for product in state.prior_shortlist
        if not _is_product_removed_by_terms(product, exclusions)
    ]
    ranked = [item.product for item in rank_products(state.context, catalogue, limit=10)]
    merged: list[Product] = []
    seen: set[str] = set()

    for product in prior + ranked:
        if product.link not in seen:
            merged.append(product)
            seen.add(product.link)
        if len(merged) >= limit:
            break
    return tuple(merged)


def _is_product_removed_by_terms(product: Product, exclusions: set[str]) -> bool:
    from recommender import is_excluded

    return is_excluded(product, exclusions)


def decide_next_action(
    messages: list[Message],
    catalogue: Catalogue,
    limit: int = 10,
) -> ConversationDecision:
    if limit < 1 or limit > 10:
        raise ValueError("limit must be between 1 and 10")

    state = build_state(messages, catalogue)
    user_text = state.last_user_message

    if is_prompt_injection(user_text):
        return ConversationDecision(
            intent="prompt_injection",
            reply=(
                "I can only recommend SHL catalogue assessments and must use catalogue "
                "names and URLs."
            ),
            products=(),
            end_of_conversation=False,
            state=state,
        )

    if is_legal_or_off_topic(user_text):
        return ConversationDecision(
            intent="refusal",
            reply=(
                "I can help select SHL assessments, but I cannot provide legal, "
                "regulatory, or general hiring advice."
            ),
            products=(),
            end_of_conversation=False,
            state=state,
        )

    if is_compare(user_text):
        compared_products = identify_referenced_products(
            user_text, catalogue, state.prior_shortlist
        )
        return ConversationDecision(
            intent="compare",
            reply=build_comparison_reply(compared_products),
            products=(),
            end_of_conversation=False,
            state=state,
            debug={
                "compared_products": [product.name for product in compared_products],
                "prior_shortlist": [product.name for product in state.prior_shortlist],
            },
        )

    if is_confirm(user_text) and state.prior_shortlist:
        return ConversationDecision(
            intent="finalize",
            reply="Confirmed. I will keep the current shortlist.",
            products=state.prior_shortlist[:limit],
            end_of_conversation=True,
            state=state,
        )

    question = clarification_question(state)
    if question:
        intent = "vague_query" if not has_enough_context(state.context) else "clarify_needed"
        return ConversationDecision(
            intent=intent,
            reply=question,
            products=(),
            end_of_conversation=False,
            state=state,
        )

    if is_refine(user_text) and state.prior_shortlist:
        refined = apply_refinement_to_prior(state, catalogue, limit)
        return ConversationDecision(
            intent="refine",
            reply="Updated the shortlist using the new constraint.",
            products=refined,
            end_of_conversation=False,
            state=state,
        )

    ranked = rank_products(state.context, catalogue, limit=limit)
    products = tuple(item.product for item in ranked)
    return ConversationDecision(
        intent="recommend",
        reply="I found a catalogue-grounded shortlist for this context.",
        products=products,
        end_of_conversation=False,
        state=state,
        debug={"scores": [(item.product.name, item.score, item.reasons) for item in ranked]},
    )
