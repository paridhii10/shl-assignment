from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from schemas import Product


DEFAULT_CATALOG_PATH = Path(
    os.environ.get(
        "SHL_CATALOG_PATH",
        r"C:\Users\HP\Downloads\SHL Assignment\shl_product_catalog.json",
    )
)


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def repair_json_text(raw_text: str) -> str:
    """Repair raw newline characters inside JSON strings.

    The provided catalogue has at least one unescaped newline inside a product
    name. Strict JSON parsers reject that, but replacing raw CR/LF characters
    only while inside quoted strings preserves record boundaries and makes the
    file parseable.
    """
    repaired: list[str] = []
    in_string = False
    escaped = False

    for char in raw_text:
        if in_string and char in {"\n", "\r"}:
            repaired.append(" ")
            continue

        repaired.append(char)

        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string

    return "".join(repaired)


def parse_catalog_json(raw_text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = json.loads(repair_json_text(raw_text))

    if not isinstance(parsed, list):
        raise ValueError("Catalogue root must be a JSON array")
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Every catalogue item must be a JSON object")
    return parsed


def is_shl_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return parsed.scheme in {"http", "https"} and (
        host == "shl.com" or host == "www.shl.com" or host.endswith(".shl.com")
    )


@dataclass(frozen=True)
class Catalogue:
    products: tuple[Product, ...]
    by_normalized_name: dict[str, tuple[Product, ...]]
    by_url: dict[str, Product]
    by_entity_id: dict[str, Product]

    @classmethod
    def from_products(cls, products: Iterable[Product]) -> "Catalogue":
        product_list = tuple(products)
        by_normalized_name_lists: dict[str, list[Product]] = {}
        by_url: dict[str, Product] = {}
        by_entity_id: dict[str, Product] = {}

        for product in product_list:
            name_key = normalize_name(product.name)
            by_normalized_name_lists.setdefault(name_key, []).append(product)

            if product.link in by_url:
                raise ValueError(f"Duplicate product URL: {product.link}")
            by_url[product.link] = product

            if product.entity_id:
                if product.entity_id in by_entity_id:
                    raise ValueError(f"Duplicate entity_id: {product.entity_id}")
                by_entity_id[product.entity_id] = product

            if not is_shl_url(product.link):
                raise ValueError(f"Non-SHL product URL: {product.link}")

        return cls(
            products=product_list,
            by_normalized_name={
                key: tuple(values) for key, values in by_normalized_name_lists.items()
            },
            by_url=by_url,
            by_entity_id=by_entity_id,
        )

    def contains_product(self, product: Product) -> bool:
        return self.by_url.get(product.link) == product

    def validate_recommendation(self, recommendation: dict[str, Any]) -> None:
        expected_keys = {"name", "url", "test_type"}
        if set(recommendation) != expected_keys:
            raise ValueError("Recommendation must contain exactly name, url, and test_type")

        product = self.by_url.get(str(recommendation["url"]))
        if product is None:
            raise ValueError(f"Recommendation URL is not in catalogue: {recommendation['url']}")

        expected = product.to_recommendation()
        if recommendation != expected:
            raise ValueError(
                "Recommendation does not exactly match catalogue product "
                f"for URL: {recommendation['url']}"
            )

    def validate_recommendations(self, recommendations: Iterable[dict[str, Any]]) -> None:
        for recommendation in recommendations:
            self.validate_recommendation(recommendation)


def load_catalogue(path: str | Path = DEFAULT_CATALOG_PATH) -> Catalogue:
    catalog_path = Path(path)
    raw_text = catalog_path.read_text(encoding="utf-8")
    records = parse_catalog_json(raw_text)
    products = [Product.from_record(record) for record in records]
    return Catalogue.from_products(products)


def product_to_recommendation(product: Product, catalogue: Catalogue) -> dict[str, str]:
    if not catalogue.contains_product(product):
        raise ValueError(f"Product is not from this catalogue: {product.name}")
    return product.to_recommendation()


def validate_catalogue_recommendations(
    recommendations: Iterable[dict[str, Any]], catalogue: Catalogue
) -> None:
    catalogue.validate_recommendations(recommendations)
