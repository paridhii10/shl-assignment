from __future__ import annotations

import re
from dataclasses import dataclass, field

from catalog import Catalogue, normalize_name
from rules import PRODUCT_BOOST_RULES, PRODUCT_URLS, STOPWORDS
from schemas import Product


TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
DROP_RE = re.compile(
    r"\b(drop|remove|exclude|skip|without|do not include|don't include)\b"
    r"(?P<object>[^.?!;\n]{0,80})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RankedProduct:
    product: Product
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def normalized_context(text: str) -> str:
    return normalize_name(text)


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def contains_phrase(context_norm: str, phrase: str) -> bool:
    phrase_norm = normalize_name(phrase)
    return f" {phrase_norm} " in f" {context_norm} "


def duration_minutes(product: Product) -> int | None:
    match = re.search(r"\d+", product.duration)
    if not match:
        return None
    return int(match.group(0))


def product_field_tokens(product: Product) -> dict[str, set[str]]:
    return {
        "name": tokenize(product.name),
        "description": tokenize(product.description),
        "keys": tokenize(" ".join(product.keys)),
        "job_levels": tokenize(" ".join(product.job_levels)),
        "languages": tokenize(" ".join(product.languages)),
    }


def lexical_score(context_tokens: set[str], product: Product) -> tuple[float, list[str]]:
    weights = {
        "name": 8.0,
        "description": 2.0,
        "keys": 3.0,
        "job_levels": 2.0,
        "languages": 2.0,
    }
    score = 0.0
    reasons: list[str] = []

    for field_name, tokens in product_field_tokens(product).items():
        matches = context_tokens & tokens
        if matches:
            field_score = len(matches) * weights[field_name]
            score += field_score
            reasons.append(
                f"lexical {field_name}: {', '.join(sorted(matches))} (+{field_score:g})"
            )

    return score, reasons


def triggered_rules(context_norm: str) -> list[tuple[str, tuple[str, ...], float]]:
    active_rules: list[tuple[str, tuple[str, ...], float]] = []
    for rule in PRODUCT_BOOST_RULES:
        if any(contains_phrase(context_norm, trigger) for trigger in rule.triggers):
            active_rules.append((rule.label, rule.product_keys, rule.boost))
    return active_rules


def explicit_exclusions(context: str) -> set[str]:
    context_norm = normalized_context(context)
    excluded_terms: set[str] = set()

    for match in DROP_RE.finditer(context):
        obj_norm = normalize_name(match.group("object"))
        excluded_terms.update(tokenize(obj_norm))

        if "personality" in obj_norm or "opq" in obj_norm:
            excluded_terms.add("personality")
            excluded_terms.add("opq")

    if re.search(r"\b(drop|remove|exclude|skip|without)\s+(the\s+)?opq", context, re.I):
        excluded_terms.add("opq")
        excluded_terms.add("personality")

    if "drop rest" in context_norm or "remove rest" in context_norm:
        excluded_terms.add("rest")
        excluded_terms.add("restful")

    return excluded_terms


def is_excluded(product: Product, excluded_terms: set[str]) -> bool:
    if not excluded_terms:
        return False

    name_norm = normalize_name(product.name)
    name_tokens = tokenize(name_norm)
    key_tokens = tokenize(" ".join(product.keys))

    if "personality" in excluded_terms and "Personality & Behavior" in product.keys:
        return True
    if "opq" in excluded_terms and "opq" in name_norm:
        return True
    if {"rest", "restful"} & excluded_terms and {"rest", "restful"} & name_tokens:
        return True

    meaningful_terms = excluded_terms - {
        "a",
        "an",
        "the",
        "product",
        "products",
        "assessment",
        "assessments",
        "test",
        "tests",
    }
    if meaningful_terms and meaningful_terms <= (name_tokens | key_tokens):
        return True
    return False


def constraint_score(context_norm: str, product: Product) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    wants_simulation = any(
        contains_phrase(context_norm, phrase)
        for phrase in ("simulation", "simulations", "capability", "capabilities")
    )
    wants_quick = any(
        contains_phrase(context_norm, phrase)
        for phrase in ("quick", "short", "time limited", "time-limited", "fast")
    )

    minutes = duration_minutes(product)

    if wants_simulation and "Simulations" in product.keys:
        score += 18
        reasons.append("simulation requested (+18)")

    if wants_quick:
        if minutes is not None and minutes <= 10:
            score += 14
            reasons.append("quick constraint: <=10 minutes (+14)")
        elif minutes is not None and minutes <= 20:
            score += 6
            reasons.append("quick constraint: <=20 minutes (+6)")
        elif minutes is not None and minutes >= 25 and "Simulations" in product.keys:
            score -= 18
            reasons.append("quick constraint: long simulation (-18)")

    return score, reasons


def rank_products(
    context: str,
    catalogue: Catalogue,
    limit: int = 10,
) -> list[RankedProduct]:
    if limit < 1 or limit > 10:
        raise ValueError("limit must be between 1 and 10")

    context_norm = normalized_context(context)
    context_tokens = tokenize(context)
    rules = triggered_rules(context_norm)
    exclusions = explicit_exclusions(context)

    scores: dict[str, float] = {}
    reasons_by_url: dict[str, list[str]] = {}

    for product in catalogue.products:
        if is_excluded(product, exclusions):
            continue
        score, reasons = lexical_score(context_tokens, product)
        extra_score, extra_reasons = constraint_score(context_norm, product)
        score += extra_score
        reasons.extend(extra_reasons)
        if score > 0:
            scores[product.link] = score
            reasons_by_url[product.link] = reasons

    for label, product_keys, boost in rules:
        for product_key in product_keys:
            url = PRODUCT_URLS[product_key]
            product = catalogue.by_url.get(url)
            if product is None or is_excluded(product, exclusions):
                continue
            scores[product.link] = scores.get(product.link, 0.0) + boost
            reasons_by_url.setdefault(product.link, []).append(f"rule {label} (+{boost:g})")

    ranked = [
        RankedProduct(
            product=catalogue.by_url[url],
            score=score,
            reasons=tuple(reasons_by_url.get(url, ())),
        )
        for url, score in scores.items()
    ]

    ranked.sort(
        key=lambda item: (
            -item.score,
            duration_minutes(item.product) if duration_minutes(item.product) is not None else 999,
            item.product.name,
        )
    )
    return ranked[:limit]


def retrieve_products(context: str, catalogue: Catalogue, limit: int = 10) -> list[Product]:
    return [ranked.product for ranked in rank_products(context, catalogue, limit)]
