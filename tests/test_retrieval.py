from __future__ import annotations

import unittest

from catalog import DEFAULT_CATALOG_PATH, load_catalogue
from recommender import rank_products, retrieve_products
from rules import PRODUCT_URLS


TRACE_CASES = {
    "C1": {
        "context": (
            "Senior leadership solution for CXOs, director-level positions, "
            "15+ years experience, selection comparing candidates against a "
            "leadership benchmark."
        ),
        "expected": ("opq32r", "opq_ucf_report", "opq_leadership_report"),
    },
    "C2": {
        "context": (
            "Hiring a senior Rust engineer for high-performance networking "
            "infrastructure. Systems engineer, Linux depth, senior IC. Add a "
            "cognitive test and personality component."
        ),
        "expected": (
            "smart_interview_live_coding",
            "linux_programming",
            "networking_implementation",
            "verify_g_plus",
            "opq32r",
        ),
    },
    "C3": {
        "context": (
            "Screening 500 entry-level contact centre agents for inbound calls "
            "and customer service. English US spoken language, call center "
            "simulation, volume screening, finalists."
        ),
        "expected": (
            "svar_spoken_english_us",
            "contact_center_call_sim",
            "entry_customer_service",
            "customer_service_phone_sim",
        ),
    },
    "C4": {
        "context": (
            "Hiring graduate financial analysts, final-year students with no "
            "work experience. Need numerical reasoning, finance knowledge, "
            "basic statistics, situational judgement, and default personality."
        ),
        "expected": (
            "verify_numerical",
            "financial_accounting",
            "basic_statistics",
            "graduate_scenarios",
            "opq32r",
        ),
    },
    "C5": {
        "context": (
            "Reskill our Sales organization as part of restructuring and annual "
            "talent audit. Need skills audit, development report, OPQ, OPQ MQ "
            "Sales Report, and sales transformation for individual contributors."
        ),
        "expected": (
            "global_skills_assessment",
            "global_skills_report",
            "opq32r",
            "opq_mq_sales_report",
            "sales_transformation_ic",
        ),
    },
    "C6": {
        "context": (
            "Hiring plant operators for a chemical facility. Safety is top "
            "priority: reliability, procedure compliance, dependability, never "
            "cutting corners. Industrial manufacturing context."
        ),
        "expected": ("dsi", "safety_dependability", "workplace_health_safety"),
    },
    "C7": {
        "context": (
            "Hiring bilingual healthcare admin staff in South Texas. Patient "
            "records, HIPAA compliance, medical terminology, Microsoft Word "
            "daily, trust-sensitive role. Functionally bilingual English and "
            "Spanish. Include dependability and broad personality."
        ),
        "expected": (
            "hipaa_security",
            "medical_terminology",
            "word_365_essentials",
            "dsi",
            "opq32r",
        ),
    },
    "C8": {
        "context": (
            "Quickly screen admin assistants for Excel and Word daily. Office "
            "skills, quick short knowledge checks, then add simulation to "
            "capture capabilities in Microsoft Excel and Microsoft Word. Include "
            "personality fit."
        ),
        "expected": ("ms_excel", "ms_word", "opq32r", "excel_365_sim", "word_365_sim"),
    },
    "C9": {
        "context": (
            "Senior Full-Stack Engineer, backend-leaning senior IC, 5+ years "
            "Core Java, Spring, SQL relational databases, AWS deployment, Docker, "
            "microservice delivery, architecture, mentoring. Drop REST. Keep "
            "Verify G+ and OPQ32r."
        ),
        "expected": (
            "core_java_advanced",
            "spring",
            "sql",
            "aws_development",
            "docker",
            "verify_g_plus",
            "opq32r",
        ),
    },
    "C10": {
        "context": (
            "Graduate management trainee scheme for recent graduates. Need full "
            "battery: cognitive, personality, and situational judgement."
        ),
        "expected": ("verify_g_plus", "opq32r", "graduate_scenarios"),
    },
}


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = load_catalogue(DEFAULT_CATALOG_PATH)

    def assertExpectedUrlsInTop10(self, case_name: str) -> None:
        case = TRACE_CASES[case_name]
        ranked = rank_products(case["context"], self.catalogue, limit=10)
        actual_urls = [item.product.link for item in ranked]
        expected_urls = [PRODUCT_URLS[key] for key in case["expected"]]

        print(f"\n{case_name} expected:")
        for url in expected_urls:
            print(f"  - {self.catalogue.by_url[url].name}")
        print(f"{case_name} actual top 10:")
        for index, item in enumerate(ranked, 1):
            reasons = "; ".join(item.reasons[:3])
            print(f"  {index}. {item.product.name} [{item.score:g}] {reasons}")

        missing = [url for url in expected_urls if url not in actual_urls]
        if missing:
            missing_names = [self.catalogue.by_url[url].name for url in missing]
            self.fail(f"{case_name} missing from top 10: {missing_names}")

    def test_public_trace_c1(self) -> None:
        self.assertExpectedUrlsInTop10("C1")

    def test_public_trace_c2(self) -> None:
        self.assertExpectedUrlsInTop10("C2")

    def test_public_trace_c3(self) -> None:
        self.assertExpectedUrlsInTop10("C3")

    def test_public_trace_c4(self) -> None:
        self.assertExpectedUrlsInTop10("C4")

    def test_public_trace_c5(self) -> None:
        self.assertExpectedUrlsInTop10("C5")

    def test_public_trace_c6(self) -> None:
        self.assertExpectedUrlsInTop10("C6")

    def test_public_trace_c7(self) -> None:
        self.assertExpectedUrlsInTop10("C7")

    def test_public_trace_c8(self) -> None:
        self.assertExpectedUrlsInTop10("C8")

    def test_public_trace_c9(self) -> None:
        self.assertExpectedUrlsInTop10("C9")

    def test_public_trace_c10(self) -> None:
        self.assertExpectedUrlsInTop10("C10")

    def test_drop_opq_excludes_opq32r(self) -> None:
        context = (
            "Graduate management trainee scheme for recent graduates. Need "
            "cognitive, personality, and situational judgement. Drop the OPQ. "
            "Final list should be Verify G+ and Graduate Scenarios."
        )
        products = retrieve_products(context, self.catalogue, limit=10)
        urls = [product.link for product in products]
        self.assertIn(PRODUCT_URLS["verify_g_plus"], urls)
        self.assertIn(PRODUCT_URLS["graduate_scenarios"], urls)
        self.assertNotIn(PRODUCT_URLS["opq32r"], urls)

    def test_drop_rest_excludes_restful_web_services(self) -> None:
        context = (
            "Senior backend engineer with Java Spring SQL AWS Docker. Drop REST "
            "because API design will be covered in live interview."
        )
        products = retrieve_products(context, self.catalogue, limit=10)
        self.assertNotIn(PRODUCT_URLS["restful_web_services"], [p.link for p in products])

    def test_retrieval_returns_catalogue_products_only_without_duplicates(self) -> None:
        products = retrieve_products(TRACE_CASES["C9"]["context"], self.catalogue, limit=10)
        self.assertLessEqual(len(products), 10)
        self.assertEqual(len({product.link for product in products}), len(products))
        for product in products:
            self.assertTrue(self.catalogue.contains_product(product))


if __name__ == "__main__":
    unittest.main()
