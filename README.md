# Checking a language model's answer against its source

A language model answers a question from a set of documents. This program then
checks the answer against those documents with the Z3 solver and labels it:

- `PASS`: the answer is consistent with the source and supported by it.
- `UNSUPPORTED`: nothing in the source contradicts it, but nothing proves it.
- `HALLUCINATION`: the answer conflicts with a rule in the source.

I wrote it during a research rotation in the Computer Networks Lab at McGill
University (ECSE 397, Winter 2026).

## How it works

1. Read the documents in `data/documents/` and split them into chunks.
2. Retrieve the chunks closest to the question, using embeddings.
3. Ask the model to answer from those chunks only.
4. Ask the model to extract the claims made by the answer and the rules stated
   in the source, as (subject, relation, value) triples. A rule can carry a
   condition, such as `patient_age < 18`.
5. Translate the triples into Z3 formulas, with the facts given in the question
   as assumptions.
6. For each claim, ask Z3 two things. Can the claim be true together with the
   rules? If not, it is a contradiction. Can the claim be false together with
   the rules? If so, the source does not prove it.

All of it is in `main.py`.

## Try it without an API key

`evaluate_cases.py` skips the model and runs the solver on three hand-written
answers to the same question, one for each label:

```
pip install -r requirements.txt
python evaluate_cases.py
```

```
| Case   | Scenario                | Should be     | Returned      | Match |
| case_1 | Supported answer        | PASS          | PASS          | yes   |
| case_2 | Contradictory answer    | HALLUCINATION | HALLUCINATION | yes   |
| case_3 | Unsupported extra claim | UNSUPPORTED   | UNSUPPORTED   | yes   |
```

The source is `data/documents/medication_policy.txt`, a made-up insurance
policy. Its rule 2.4 lowers the Ibuprofen limit from 1200 mg to 600 mg a day
for patients under 18, and the question asks about a 10-year-old. An answer of
1200 mg is flagged because of that condition.

## Run the full pipeline

Copy `.env.example` to `.env`, add an OpenAI API key, then:

```
python main.py "What is the max dosage of Ibuprofen for a 10-year-old?"
```

## Limits

- The claims and rules are extracted by the model, so the check is only as
  good as that extraction. A rule the model misses is a rule Z3 never sees.
- It handles numbers and true or false facts, with eight relations (equals,
  less than, and so on). Free-text claims are not checked.
- Matching a claim to a rule depends on both using the same subject name. The
  clean-up in `canonicalize_subject` was written for the dosage example and
  would need its own rules for another domain.
- The only fact read from the question itself is a patient's age.
- It has been tried on three hand-built cases, not on a benchmark.
