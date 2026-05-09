from __future__ import annotations

import unittest

from catalog import (
    DEFAULT_CATALOG_PATH,
    is_shl_url,
    load_catalogue,
    normalize_name,
    product_to_recommendation,
    validate_catalogue_recommendations,
)
from schemas import Product, derive_test_type


class CatalogueLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = load_catalogue(DEFAULT_CATALOG_PATH)

    def test_catalogue_loads_successfully(self) -> None:
        self.assertGreater(len(self.catalogue.products), 0)

    def test_product_count_is_close_to_expected_scrape_size(self) -> None:
        self.assertGreaterEqual(len(self.catalogue.products), 370)
        self.assertLessEqual(len(self.catalogue.products), 390)

    def test_required_fields_are_present(self) -> None:
        for product in self.catalogue.products:
            self.assertTrue(product.name)
            self.assertTrue(product.link)
            self.assertTrue(product.test_type)

    def test_all_links_are_from_shl(self) -> None:
        for product in self.catalogue.products:
            self.assertTrue(is_shl_url(product.link), product.link)

    def test_catalogue_maps_are_built(self) -> None:
        mapped_by_name_count = sum(
            len(products) for products in self.catalogue.by_normalized_name.values()
        )
        self.assertEqual(mapped_by_name_count, len(self.catalogue.products))
        self.assertEqual(len(self.catalogue.by_url), len(self.catalogue.products))
        self.assertEqual(len(self.catalogue.by_entity_id), len(self.catalogue.products))

        opq = self.catalogue.by_normalized_name[
            normalize_name("Occupational Personality Questionnaire OPQ32r")
        ][0]
        self.assertEqual(
            opq.link,
            "https://www.shl.com/products/product-catalog/view/"
            "occupational-personality-questionnaire-opq32r/",
        )

    def test_malformed_microsoft_365_excel_item_is_repaired_safely(self) -> None:
        product = self.catalogue.by_url[
            "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/"
        ]
        self.assertEqual(product.name, "Microsoft 365 (New)")
        self.assertNotIn("\n", product.name)
        self.assertNotIn("\r", product.name)
        self.assertEqual(product.test_type, "K,S")

    def test_test_type_derivation_uses_defined_mapping_order(self) -> None:
        self.assertEqual(derive_test_type(["Knowledge & Skills"]), "K")
        self.assertEqual(
            derive_test_type(["Simulations", "Knowledge & Skills"]),
            "K,S",
        )
        self.assertEqual(
            derive_test_type(
                [
                    "Ability & Aptitude",
                    "Assessment Exercises",
                    "Biodata & Situational Judgment",
                    "Competencies",
                    "Development & 360",
                    "Personality & Behavior",
                ]
            ),
            "A,B,C,D,E,P",
        )

    def test_recommendation_conversion_returns_exact_schema_fields(self) -> None:
        product = self.catalogue.by_normalized_name[normalize_name("Docker (New)")][0]
        recommendation = product_to_recommendation(product, self.catalogue)
        self.assertEqual(set(recommendation.keys()), {"name", "url", "test_type"})
        self.assertEqual(
            recommendation,
            {
                "name": "Docker (New)",
                "url": "https://www.shl.com/products/product-catalog/view/docker-new/",
                "test_type": "K",
            },
        )

    def test_non_catalogue_recommendations_are_rejected(self) -> None:
        product = self.catalogue.by_normalized_name[normalize_name("Docker (New)")][0]
        valid = product_to_recommendation(product, self.catalogue)
        validate_catalogue_recommendations([valid], self.catalogue)

        forged = dict(valid)
        forged["name"] = "Docker Advanced"
        with self.assertRaises(ValueError):
            validate_catalogue_recommendations([forged], self.catalogue)

        outside = {
            "name": "Fake Product",
            "url": "https://example.com/fake",
            "test_type": "K",
        }
        with self.assertRaises(ValueError):
            validate_catalogue_recommendations([outside], self.catalogue)

    def test_product_object_must_come_from_catalogue_for_conversion(self) -> None:
        product = Product(
            name="Fake",
            link="https://www.shl.com/products/product-catalog/view/fake/",
            test_type="K",
        )
        with self.assertRaises(ValueError):
            product_to_recommendation(product, self.catalogue)


if __name__ == "__main__":
    unittest.main()
