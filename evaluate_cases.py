from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from main import (
    Claim,
    Constraint,
    QuestionFact,
    VerificationResult,
    normalize_extraction,
    translate,
    verify,
)


DOCUMENT_PATH = Path("data/documents/medication_policy.txt")


@dataclass
class EvaluationCase:
    id: str
    label: str
    question: str
    answer_text: str
    claims: list[Claim]
    expected_status: str


def load_context_excerpt() -> str:
    text = DOCUMENT_PATH.read_text(encoding="utf-8")
    start = text.find("2.3  The maximum covered dosage of Ibuprofen is 1200 mg per day.")
    end = text.find("2.5  For patients under 12 years of age, Atorvastatin is NOT covered.")
    if start == -1 or end == -1:
        return text.strip()
    return text[start:end].strip()


def build_question_facts() -> list[QuestionFact]:
    return [
        QuestionFact(
            id="question_fact_1",
            subject="patient_age",
            relation="equals",
            object="10",
            natural_language="The patient is 10 years old.",
        )
    ]


def build_constraints() -> list[Constraint]:
    constraints = [
        Constraint(
            id="rule_1",
            subject="ibuprofen_dosage",
            relation="equals",
            object="600",
            natural_language=(
                "For patients under 18 years of age, the maximum covered dosage "
                "of Ibuprofen is reduced to 600 mg per day."
            ),
            condition="patient_age < 18",
        )
    ]
    _, normalized_constraints = normalize_extraction([], constraints)
    return normalized_constraints


def build_cases() -> list[EvaluationCase]:
    question = "What is the maximum covered dosage of Ibuprofen for a 10-year-old?"

    supported_claims = [
        Claim(
            id="claim_1",
            subject="ibuprofen_dosage",
            relation="equals",
            object="600",
            natural_language="The maximum covered dosage is 600 mg per day.",
        )
    ]

    contradictory_claims = [
        Claim(
            id="claim_1",
            subject="ibuprofen_dosage",
            relation="equals",
            object="1200",
            natural_language="The maximum covered dosage is 1200 mg per day.",
        )
    ]

    unsupported_claims = [
        Claim(
            id="claim_1",
            subject="ibuprofen_dosage",
            relation="equals",
            object="600",
            natural_language="The maximum covered dosage is 600 mg per day.",
        ),
        Claim(
            id="claim_2",
            subject="ibuprofen_take_with_food",
            relation="equals",
            object="true",
            natural_language="Ibuprofen should be taken with food.",
        ),
    ]

    return [
        EvaluationCase(
            id="case_1",
            label="Supported answer",
            question=question,
            answer_text="The maximum covered dosage of Ibuprofen for a 10-year-old is 600 mg per day.",
            claims=supported_claims,
            expected_status="PASS",
        ),
        EvaluationCase(
            id="case_2",
            label="Contradictory answer",
            question=question,
            answer_text="The maximum covered dosage of Ibuprofen for a 10-year-old is 1200 mg per day.",
            claims=contradictory_claims,
            expected_status="HALLUCINATION",
        ),
        EvaluationCase(
            id="case_3",
            label="Unsupported extra claim",
            question=question,
            answer_text=(
                "The maximum covered dosage of Ibuprofen for a 10-year-old is 600 mg per day, "
                "and it should be taken with food."
            ),
            claims=unsupported_claims,
            expected_status="UNSUPPORTED",
        ),
    ]


def run_case(
    case: EvaluationCase,
    constraints: list[Constraint],
    question_facts: list[QuestionFact],
) -> VerificationResult:
    claims, normalized_constraints = normalize_extraction(case.claims, constraints)
    constraint_formulas, assumption_formulas, claim_entries, _ = translate(
        claims,
        normalized_constraints,
        question_facts,
    )
    return verify(constraint_formulas, assumption_formulas, claim_entries)


def format_claim_list(claims: list[Claim]) -> str:
    if not claims:
        return "none"
    return "; ".join(
        f"{claim.id} ({claim.subject} {claim.relation} {claim.object})"
        for claim in claims
    )


def print_case_block(
    case: EvaluationCase,
    result: VerificationResult,
) -> None:
    match = "YES" if case.expected_status == result.status else "NO"
    print("=" * 78)
    print(f"{case.id.upper()}  {case.label}")
    print("-" * 78)
    print(f"Question:                  {case.question}")
    print(f"Answer being tested:       {case.answer_text}")
    print(f"What this should be:       {case.expected_status}")
    print(f"What the verifier returned:{' '}{result.status}")
    print(f"Did it match?:             {match}")
    print(f"Why:                       {result.explanation}")
    print(f"Supported claims:          {format_claim_list(result.supported_claims)}")
    print(f"Unsupported claims:        {format_claim_list(result.unsupported_claims)}")
    print(f"Contradictory claims:      {format_claim_list(result.contradictory_claims)}")
    print()


def print_summary_table(rows: list[tuple[EvaluationCase, VerificationResult]]) -> None:
    headers = ["Case", "Scenario", "Should be", "Returned", "Match"]
    widths = [10, 28, 15, 15, 7]

    def line(char: str) -> str:
        return "+" + "+".join(char * (width + 2) for width in widths) + "+"

    def row(values: list[str]) -> str:
        cells = []
        for value, width in zip(values, widths):
            cells.append(f" {value[:width].ljust(width)} ")
        return "|" + "|".join(cells) + "|"

    print(line("-"))
    print(row(headers))
    print(line("="))
    for case, result in rows:
        match = "yes" if case.expected_status == result.status else "no"
        print(
            row(
                [
                    case.id,
                    case.label,
                    case.expected_status,
                    result.status,
                    match,
                ]
            )
        )
        print(line("-"))


def is_strict_accept(status: str) -> bool:
    return status == "PASS"


def is_lenient_accept(status: str) -> bool:
    return status in {"PASS", "UNSUPPORTED"}


def print_accuracy_summary(rows: list[tuple[EvaluationCase, VerificationResult]]) -> None:
    total = len(rows)
    exact_matches = sum(
        1 for case, result in rows if case.expected_status == result.status
    )
    strict_matches = sum(
        1
        for case, result in rows
        if is_strict_accept(case.expected_status) == is_strict_accept(result.status)
    )
    lenient_matches = sum(
        1
        for case, result in rows
        if is_lenient_accept(case.expected_status) == is_lenient_accept(result.status)
    )

    print("ACCURACY METRICS")
    print("-" * 78)
    print(f"Exact label accuracy:      {exact_matches}/{total}")
    print(f"Strict accept accuracy:    {strict_matches}/{total}")
    print(f"Lenient accept accuracy:   {lenient_matches}/{total}")
    print()
    print("Strict accept = accept only PASS")
    print("Lenient accept = accept PASS or UNSUPPORTED, reject HALLUCINATION")
    print()


def main() -> None:
    print("EVALUATION: HAND-CONSTRUCTED VERIFIER CASES")
    print()
    print(
        "This evaluation uses one real rule from the medication policy document and"
    )
    print(
        "three hand-constructed answer cases to show the three possible verifier outcomes."
    )
    print()
    print("LABEL MEANINGS")
    print("-" * 78)
    print("PASS          = the answer is supported by the document")
    print("HALLUCINATION = the answer contradicts the document")
    print("UNSUPPORTED   = the answer does not contradict the document,")
    print("                but it adds a claim that the document does not support")
    print()
    print("SOURCE EXCERPT")
    print("-" * 78)
    print(load_context_excerpt())
    print()

    question_facts = build_question_facts()
    constraints = build_constraints()
    cases = build_cases()

    print("QUESTION FACTS")
    print("-" * 78)
    for fact in question_facts:
        print(f"- {fact.subject} {fact.relation} {fact.object}")
    print()

    print("SOURCE RULES")
    print("-" * 78)
    for rule in constraints:
        prefix = f"if {rule.condition} then " if rule.condition else ""
        print(f"- {prefix}{rule.subject} {rule.relation} {rule.object}")
    print()

    results: list[tuple[EvaluationCase, VerificationResult]] = []
    for case in cases:
        result = run_case(case, constraints, question_facts)
        results.append((case, result))
        print_case_block(case, result)

    print("SUMMARY")
    print("-" * 78)
    print_summary_table(results)
    print()
    print_accuracy_summary(results)


if __name__ == "__main__":
    main()
