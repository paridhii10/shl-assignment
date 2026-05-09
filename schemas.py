from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TEST_TYPE_BY_KEY: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def clean_text(value: Any) -> str:
    """Normalize catalogue text without changing its meaning."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def derive_test_type(keys: list[str]) -> str:
    codes = [code for key, code in TEST_TYPE_BY_KEY.items() if key in keys]
    return ",".join(codes)


@dataclass(frozen=True)
class Product:
    name: str
    link: str
    test_type: str
    entity_id: str = ""
    keys: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    job_levels: tuple[str, ...] = field(default_factory=tuple)
    languages: tuple[str, ...] = field(default_factory=tuple)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Product":
        keys = tuple(clean_text(key) for key in record.get("keys", []) if clean_text(key))
        product = cls(
            entity_id=clean_text(record.get("entity_id")),
            name=clean_text(record.get("name")),
            link=clean_text(record.get("link")),
            test_type=derive_test_type(list(keys)),
            keys=keys,
            description=clean_text(record.get("description")),
            job_levels=tuple(
                clean_text(level) for level in record.get("job_levels", []) if clean_text(level)
            ),
            languages=tuple(
                clean_text(language)
                for language in record.get("languages", [])
                if clean_text(language)
            ),
            duration=clean_text(record.get("duration")),
            remote=clean_text(record.get("remote")),
            adaptive=clean_text(record.get("adaptive")),
        )
        product.validate()
        return product

    def validate(self) -> None:
        missing = [
            field_name
            for field_name in ("name", "link", "test_type")
            if not getattr(self, field_name)
        ]
        if missing:
            raise ValueError(f"Product is missing required fields: {', '.join(missing)}")

    def to_recommendation(self) -> dict[str, str]:
        return {
            "name": self.name,
            "url": self.link,
            "test_type": self.test_type,
        }
