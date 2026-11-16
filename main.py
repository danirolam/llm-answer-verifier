"""Check a language model's answer against its source documents with Z3.

What it does:
1. Read local documents.
2. Split them into chunks.
3. Retrieve the most relevant chunks for a question with OpenAI embeddings.
4. Ask OpenAI to answer using only those chunks.
5. Ask OpenAI again to extract claims from the answer and constraints from the source.
6. Translate those into Z3 formulas.
7. Check for contradiction and support.

Usage:
    python main.py "What is the max dosage of Ibuprofen for a 10-year-old?"
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import z3


load_dotenv()


DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./data/documents"))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4


@dataclass
class Chunk:
    source: str
    text: str


@dataclass
class Claim:
    id: str
    subject: str
    relation: str
    object: str
    natural_language: str


@dataclass
class Constraint:
    id: str
    subject: str
    relation: str
    object: str
    natural_language: str
    condition: str | None = None


@dataclass
class QuestionFact:
    id: str
    subject: str
    relation: str
    object: str
    natural_language: str


@dataclass
class VerificationResult:
    status: str
    explanation: str
    supported_claims: list[Claim]
    unsupported_claims: list[Claim]
    contradictory_claims: list[Claim]


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def read_documents(folder: Path) -> list[Chunk]:
    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    documents: list[Chunk] = []

    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Chunk(source=path.name, text=text))

    for path in sorted(folder.glob("*.pdf")):
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            documents.append(Chunk(source=path.name, text=text))

    if not documents:
        raise FileNotFoundError(f"No .txt or .pdf files found in {folder}")

    return documents


def split_into_chunks(text: str, source: str) -> list[Chunk]:
    clean = re.sub(r"\s+", " ", text).strip()
    chunks: list[Chunk] = []
    start = 0

    while start < len(clean):
        end = min(start + CHUNK_SIZE, len(clean))
        if end < len(clean):
            sentence_end = clean.rfind(". ", start, min(end + 120, len(clean)))
            if sentence_end > start + 200:
                end = sentence_end + 1

        chunk_text = clean[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(source=source, text=chunk_text))

        if end >= len(clean):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def build_chunk_index(documents: list[Chunk]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(split_into_chunks(document.text, document.source))
    return all_chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    batch_size = 50

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_relevant_chunks(client: OpenAI, chunks: list[Chunk], question: str) -> list[Chunk]:
    chunk_vectors = embed_texts(client, [chunk.text for chunk in chunks])
    question_vector = embed_texts(client, [question])[0]

    scored = []
    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(question_vector, vector)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:TOP_K]]


def generate_answer(client: OpenAI, question: str, source_chunks: list[Chunk]) -> str:
    context = "\n\n---\n\n".join(chunk.text for chunk in source_chunks)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "If the context is missing the answer, say that clearly.\n\n"
                    f"CONTEXT:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


def extract_structure(
    client: OpenAI, question: str, answer: str, source_chunks: list[Chunk]
) -> tuple[list[Claim], list[Constraint]]:
    source_text = "\n\n".join(
        f"[Chunk {i} from {chunk.source}]\n{chunk.text}"
        for i, chunk in enumerate(source_chunks)
    )

    system_prompt = """
You extract structure from a document-grounded answer.

Return JSON with two arrays:
- claims: factual statements made by the ANSWER
- constraints: rules or facts from the SOURCE CHUNKS that matter for checking the answer

Use only these relations:
equals, not_equals, greater_than, less_than, greater_equal, less_equal, is_true, is_false

Important rules:
1. Keep subject names short, stable, and in snake_case.
2. Use the SAME subject name for the same concept across claims and constraints.
3. Do NOT bake conditions into the subject name.
4. If a rule depends on age or some other condition, put that in the condition field.
5. Facts that come only from the QUESTION should not be copied into constraints.

Good example:
- answer claim: "The maximum covered dosage of Ibuprofen is 600 mg per day"
  -> subject = "ibuprofen_dosage", relation = "equals", object = "600"
- source rule: "For patients under 18, the maximum covered dosage of Ibuprofen is 600 mg per day"
  -> condition = "patient_age < 18", subject = "ibuprofen_dosage", relation = "equals", object = "600"

Bad example:
- subject = "ibuprofen_dosage_under_18"

Return exactly this JSON shape:
{
  "claims": [
    {
      "id": "claim_1",
      "subject": "...",
      "relation": "...",
      "object": "...",
      "natural_language": "..."
    }
  ],
  "constraints": [
    {
      "id": "rule_1",
      "condition": null,
      "subject": "...",
      "relation": "...",
      "object": "...",
      "natural_language": "..."
    }
  ]
}
""".strip()

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"ANSWER:\n{answer}\n\n"
                    f"SOURCE CHUNKS:\n{source_text}"
                ),
            },
        ],
    )

    raw = json.loads(response.choices[0].message.content)
    claims = [
        Claim(
            id=item["id"],
            subject=item["subject"],
            relation=item["relation"],
            object=str(item["object"]),
            natural_language=item["natural_language"],
        )
        for item in raw.get("claims", [])
    ]
    constraints = [
        Constraint(
            id=item["id"],
            subject=item["subject"],
            relation=item["relation"],
            object=str(item["object"]),
            natural_language=item["natural_language"],
            condition=item.get("condition"),
        )
        for item in raw.get("constraints", [])
    ]
    return claims, constraints


def extract_question_facts(question: str) -> list[QuestionFact]:
    facts: list[QuestionFact] = []
    clean = question.lower()

    age_match = re.search(r"\b(\d+)[-\s]?year[-\s]?old\b", clean)
    if age_match:
        age = age_match.group(1)
        facts.append(
            QuestionFact(
                id="question_fact_1",
                subject="patient_age",
                relation="equals",
                object=age,
                natural_language=f"The patient is {age} years old.",
            )
        )

    return facts


def canonicalize_subject(name: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    tokens = [token for token in clean.split("_") if token]
    token_set = set(tokens)

    if "patient" in token_set and "age" in token_set:
        return "patient_age"

    common_words = {
        "max",
        "maximum",
        "covered",
        "cover",
        "coverage",
        "daily",
        "per",
        "day",
        "dose",
        "dosage",
        "mg",
        "under",
        "over",
        "less",
        "greater",
        "equal",
    }
    medication_tokens = [
        token for token in tokens if token not in common_words and not token.isdigit()
    ]

    if ("dose" in token_set or "dosage" in token_set) and medication_tokens:
        return f"{medication_tokens[0]}_dosage"

    if ("covered" in token_set or "coverage" in token_set) and medication_tokens:
        return f"{medication_tokens[0]}_covered"

    return clean


def canonicalize_condition(condition: str) -> str:
    clean = condition.strip().lower()
    clean = re.sub(
        r"\b[a-z_]*patient[a-z_]*age[a-z_]*\b",
        "patient_age",
        clean,
    )
    clean = re.sub(r"\s+", " ", clean)
    return clean


def normalize_extraction(
    claims: list[Claim], constraints: list[Constraint]
) -> tuple[list[Claim], list[Constraint]]:
    normalized_claims: list[Claim] = []
    for claim in claims:
        normalized_claims.append(
            Claim(
                id=claim.id,
                subject=canonicalize_subject(claim.subject),
                relation=claim.relation,
                object=claim.object,
                natural_language=claim.natural_language,
            )
        )

    normalized_constraints: list[Constraint] = []
    for constraint in constraints:
        subject = canonicalize_subject(constraint.subject)
        condition = (
            canonicalize_condition(constraint.condition)
            if constraint.condition
            else None
        )

        if condition is None:
            suffix_match = re.search(r"_under_(\d+)$", constraint.subject.lower())
            if suffix_match:
                condition = f"patient_age < {suffix_match.group(1)}"

        if condition is None:
            text = constraint.natural_language.lower()
            text_match = re.search(r"under (\d+)", text)
            if text_match:
                condition = f"patient_age < {text_match.group(1)}"

        normalized_constraints.append(
            Constraint(
                id=constraint.id,
                subject=subject,
                relation=constraint.relation,
                object=constraint.object,
                natural_language=constraint.natural_language,
                condition=condition,
            )
        )

    return normalized_claims, normalized_constraints


def parse_number(value: str) -> int | float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    text = match.group(0)
    return float(text) if "." in text else int(text)


def parse_boolean(value: str) -> bool | None:
    clean = value.strip().lower()
    if clean in {"true", "yes"}:
        return True
    if clean in {"false", "no"}:
        return False
    return None


def collect_variables(
    claims: list[Claim],
    constraints: list[Constraint],
    question_facts: list[QuestionFact],
) -> dict[str, str]:
    variable_types: dict[str, str] = {}

    for item in [*claims, *constraints, *question_facts]:
        if item.relation in {"is_true", "is_false"}:
            variable_types[item.subject] = "bool"
            continue

        bool_value = parse_boolean(item.object)
        if bool_value is not None:
            variable_types[item.subject] = "bool"
            continue

        number = parse_number(item.object)
        if isinstance(number, float):
            variable_types[item.subject] = "real"
        elif isinstance(number, int):
            variable_types.setdefault(item.subject, "int")
        else:
            variable_types.setdefault(item.subject, "int")

    return variable_types


def build_z3_variables(variable_types: dict[str, str]) -> dict[str, z3.ExprRef]:
    variables: dict[str, z3.ExprRef] = {}
    for name, var_type in variable_types.items():
        if var_type == "bool":
            variables[name] = z3.Bool(name)
        elif var_type == "real":
            variables[name] = z3.Real(name)
        else:
            variables[name] = z3.Int(name)
    return variables


def build_formula(
    subject: str,
    relation: str,
    obj: str,
    variables: dict[str, z3.ExprRef],
) -> tuple[z3.ExprRef | None, str]:
    variable = variables.get(subject)
    if variable is None:
        return None, f"unknown variable: {subject}"

    if relation == "is_true":
        return variable == True, f"{subject} is true"
    if relation == "is_false":
        return variable == False, f"{subject} is false"

    number = parse_number(obj)
    if number is not None:
        value = z3.RealVal(number) if isinstance(number, float) else number
    else:
        boolean_value = parse_boolean(obj)
        if boolean_value is not None:
            value = boolean_value
        else:
            value = abs(hash(obj.strip().lower())) % (10**9)

    relation_map = {
        "equals": lambda a, b: a == b,
        "not_equals": lambda a, b: a != b,
        "greater_than": lambda a, b: a > b,
        "less_than": lambda a, b: a < b,
        "greater_equal": lambda a, b: a >= b,
        "less_equal": lambda a, b: a <= b,
    }
    builder = relation_map.get(relation)
    if builder is None:
        return None, f"unsupported relation: {relation}"
    return builder(variable, value), f"{subject} {relation} {obj}"


def parse_condition(condition: str, variables: dict[str, z3.ExprRef]) -> z3.ExprRef | None:
    patterns = [
        (r"(\w+)\s*>=\s*(.+)", "greater_equal"),
        (r"(\w+)\s*<=\s*(.+)", "less_equal"),
        (r"(\w+)\s*!=\s*(.+)", "not_equals"),
        (r"(\w+)\s*==\s*(.+)", "equals"),
        (r"(\w+)\s*>\s*(.+)", "greater_than"),
        (r"(\w+)\s*<\s*(.+)", "less_than"),
        (r"(\w+)\s*=\s*(.+)", "equals"),
    ]

    for pattern, relation in patterns:
        match = re.match(pattern, condition.strip())
        if not match:
            continue

        subject = match.group(1).strip()
        obj = match.group(2).strip()
        if subject not in variables:
            number = parse_number(obj)
            if isinstance(number, float):
                variables[subject] = z3.Real(subject)
            elif isinstance(parse_boolean(obj), bool):
                variables[subject] = z3.Bool(subject)
            else:
                variables[subject] = z3.Int(subject)

        formula, _ = build_formula(subject, relation, obj, variables)
        return formula

    return None


def translate(
    claims: list[Claim],
    constraints: list[Constraint],
    question_facts: list[QuestionFact],
) -> tuple[list[z3.ExprRef], list[z3.ExprRef], list[tuple[Claim, z3.ExprRef]], list[str]]:
    variable_types = collect_variables(claims, constraints, question_facts)
    variables = build_z3_variables(variable_types)

    constraint_formulas: list[z3.ExprRef] = []
    assumption_formulas: list[z3.ExprRef] = []
    claim_entries: list[tuple[Claim, z3.ExprRef]] = []
    descriptions: list[str] = []

    for constraint in constraints:
        body, text = build_formula(
            constraint.subject, constraint.relation, constraint.object, variables
        )
        if body is None:
            continue

        if constraint.condition:
            condition_formula = parse_condition(constraint.condition, variables)
            if condition_formula is not None:
                full_formula = z3.Implies(condition_formula, body)
                constraint_formulas.append(full_formula)
                descriptions.append(
                    f"constraint: if {constraint.condition} then {text}"
                )
                continue

        constraint_formulas.append(body)
        descriptions.append(f"constraint: {text}")

    for claim in claims:
        formula, text = build_formula(claim.subject, claim.relation, claim.object, variables)
        if formula is None:
            continue
        claim_entries.append((claim, formula))
        descriptions.append(f"claim: {text}")

    for fact in question_facts:
        formula, text = build_formula(fact.subject, fact.relation, fact.object, variables)
        if formula is None:
            continue
        assumption_formulas.append(formula)
        descriptions.append(f"question_fact: {text}")

    return constraint_formulas, assumption_formulas, claim_entries, descriptions


def verify(
    constraint_formulas: list[z3.ExprRef],
    assumption_formulas: list[z3.ExprRef],
    claim_entries: list[tuple[Claim, z3.ExprRef]],
) -> VerificationResult:
    if not constraint_formulas or not claim_entries:
        return VerificationResult(
            status="UNKNOWN",
            explanation="I could not verify anything because the extraction did not produce enough structure.",
            supported_claims=[],
            unsupported_claims=[],
            contradictory_claims=[],
        )

    supported_claims: list[Claim] = []
    unsupported_claims: list[Claim] = []
    contradictory_claims: list[Claim] = []

    aggregate_solver = z3.Solver()
    for formula in constraint_formulas:
        aggregate_solver.add(formula)
    for formula in assumption_formulas:
        aggregate_solver.add(formula)
    for _, claim_formula in claim_entries:
        aggregate_solver.add(claim_formula)
    aggregate_result = aggregate_solver.check()
    if aggregate_result == z3.unknown:
        return VerificationResult(
            status="UNKNOWN",
            explanation="Z3 could not decide whether the claims and constraints are consistent.",
            supported_claims=[],
            unsupported_claims=[],
            contradictory_claims=[],
        )

    for claim, claim_formula in claim_entries:
        consistency_solver = z3.Solver()
        for formula in constraint_formulas:
            consistency_solver.add(formula)
        for formula in assumption_formulas:
            consistency_solver.add(formula)
        consistency_solver.add(claim_formula)
        consistency_result = consistency_solver.check()

        if consistency_result == z3.unsat:
            contradictory_claims.append(claim)
            continue
        if consistency_result == z3.unknown:
            return VerificationResult(
                status="UNKNOWN",
                explanation="Z3 could not classify every claim against the source constraints.",
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims,
                contradictory_claims=contradictory_claims,
            )

        entailment_solver = z3.Solver()
        for formula in constraint_formulas:
            entailment_solver.add(formula)
        for formula in assumption_formulas:
            entailment_solver.add(formula)
        entailment_solver.add(z3.Not(claim_formula))
        result = entailment_solver.check()

        if result == z3.sat:
            unsupported_claims.append(claim)
        elif result == z3.unknown:
            return VerificationResult(
                status="UNKNOWN",
                explanation="Z3 could not decide whether every claim is supported by the source.",
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims,
                contradictory_claims=contradictory_claims,
            )
        else:
            supported_claims.append(claim)

    if contradictory_claims:
        return VerificationResult(
            status="HALLUCINATION",
            explanation=(
                "Some claims contradict the source or the answer is internally inconsistent. "
                "See the claim breakdown below."
            ),
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            contradictory_claims=contradictory_claims,
        )

    if aggregate_result == z3.unsat:
        return VerificationResult(
            status="HALLUCINATION",
            explanation=(
                "The set of claims is not jointly consistent, even though no single claim was contradictory on its own."
            ),
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            contradictory_claims=contradictory_claims,
        )

    if unsupported_claims:
        return VerificationResult(
            status="UNSUPPORTED",
            explanation=(
                "The answer does not directly contradict the source, but some claims are not actually proven by the source and question facts. "
                "See the claim breakdown below."
            ),
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            contradictory_claims=contradictory_claims,
        )

    return VerificationResult(
        status="PASS",
        explanation="The answer is consistent with the source, and the extracted claims are supported by the source together with the facts given in the question.",
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        contradictory_claims=contradictory_claims,
    )


def format_claim(claim: Claim) -> str:
    return f"{claim.id}: {claim.subject} {claim.relation} {claim.object}"


def print_summary(
    question: str,
    chunks: list[Chunk],
    answer: str,
    question_facts: list[QuestionFact],
    claims: list[Claim],
    constraints: list[Constraint],
    result: VerificationResult,
) -> None:
    print("\nQUESTION")
    print(question)

    print("\nRETRIEVED CHUNKS")
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.text[:180].strip()
        if len(chunk.text) > 180:
            preview += "..."
        print(f"{i}. {chunk.source}: {preview}")

    print("\nGENERATED ANSWER")
    print(answer)

    print("\nFACTS FROM QUESTION")
    if question_facts:
        for fact in question_facts:
            print(f"- {fact.id}: {fact.subject} {fact.relation} {fact.object}")
    else:
        print("- none")

    print("\nEXTRACTED CLAIMS")
    if claims:
        for claim in claims:
            print(f"- {claim.id}: {claim.subject} {claim.relation} {claim.object}")
    else:
        print("- none")

    print("\nEXTRACTED CONSTRAINTS")
    if constraints:
        for constraint in constraints:
            prefix = f"if {constraint.condition} then " if constraint.condition else ""
            print(
                f"- {constraint.id}: {prefix}{constraint.subject} "
                f"{constraint.relation} {constraint.object}"
            )
    else:
        print("- none")

    print("\nVERIFICATION RESULT")
    print(result.status)
    print(result.explanation)

    print("\nCLAIM BREAKDOWN")
    print("Supported claims")
    if result.supported_claims:
        for claim in result.supported_claims:
            print(f"- {format_claim(claim)}")
    else:
        print("- none")

    print("Unsupported claims")
    if result.unsupported_claims:
        for claim in result.unsupported_claims:
            print(f"- {format_claim(claim)}")
    else:
        print("- none")

    print("Contradictory claims")
    if result.contradictory_claims:
        for claim in result.contradictory_claims:
            print(f"- {format_claim(claim)}")
    else:
        print("- none")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python main.py "<your question>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    client = get_client()

    documents = read_documents(DOCUMENTS_DIR)
    chunks = build_chunk_index(documents)
    retrieved_chunks = retrieve_relevant_chunks(client, chunks, question)
    answer = generate_answer(client, question, retrieved_chunks)
    question_facts = extract_question_facts(question)
    claims, constraints = extract_structure(client, question, answer, retrieved_chunks)
    claims, constraints = normalize_extraction(claims, constraints)
    constraint_formulas, assumption_formulas, claim_entries, _ = translate(
        claims, constraints, question_facts
    )
    result = verify(constraint_formulas, assumption_formulas, claim_entries)

    print_summary(
        question,
        retrieved_chunks,
        answer,
        question_facts,
        claims,
        constraints,
        result,
    )


if __name__ == "__main__":
    main()
