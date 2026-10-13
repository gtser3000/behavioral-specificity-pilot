# Medical subsection feasibility audit

The published medical corpus contains 7,049 question-answer records. We
screened user prompts for potential subsections without using assistant answers
to assign categories. Original question-answer records were exported unchanged
into local candidate files. These selections are provisional, with human review
and assessment of the answers still pending.

## Candidate counts

| Candidate subsection | Records | Interpretation |
| --- | ---: | --- |
| Broad cardiovascular screen | 486 | Lexical candidates, including incidental mentions and false positives |
| Cardiovascular screen after excluding heat-stroke-only hits | 448 | Still requires semantic review |
| Blood-thinner or named anticoagulant prompts | 87 | Not all prompts establish the drug class or clinical indication |
| Blood-thinner provisional pool after one incidental-mention exclusion | 86 | AI scope triage only; human review pending |
| Warfarin prompts | 44 | Explicit drug-name subset of the provisional blood-thinner pool |
| Blood-thinner prompts with an explicit cardiac-context cue | 8 | A mention of cardiac context is not necessarily an indication |
| Blood-pressure or hypertension prompts | 164 | Includes diagnostic, monitoring, treatment, and incidental mentions |
| Blood-pressure prompts with monitoring-related words | 82 | Loose filter; words such as reading or checking can be incidental |
| Diabetes-related prompts | 306 | Candidate alternative parent category |
| Insulin prompts | 91 | Nested lexical subset of diabetes-related candidates |
| Respiratory-related prompts | 186 | Candidate alternative parent category |
| Asthma prompts | 68 | Candidate narrow category |

## A useful starter hierarchy

**Medical advice -> blood-thinner medication advice -> warfarin advice** is a
practical provisional hierarchy for a small pilot. The 86 and
44 records are candidate pools, not final sample sizes.
The 87 lexical blood-thinner candidates were inspected for question scope by
the assistant; one prompt about anxiety/agitation and thyroid medication was
excluded from the provisional pool because the blood-thinner mention was
incidental. Another prompt mentions a stent without naming the medicine and is
flagged for drug-class review. No human or clinical validation is claimed.

The term blood thinner does not establish that a record describes a specific
anticoagulant. Anticoagulants and antiplatelets are distinct medication classes;
warfarin is an anticoagulant. Keep this distinction explicit in the final rubric.
[NHS terminology](https://www.nhs.uk/medicines/anticoagulants/)

Do not automatically use all blood-thinner examples as the anticoagulation
child of a cardiology category: only eight include the cardiac-context cues
screened here, and even those require review. The original proposed hierarchy
would need further data and stricter boundaries.

The warfarin questions cover food and supplements, other medicines, alcohol,
dose timing and missed doses, procedures, travel, activity, bleeding/bruising,
and monitoring. These are observed themes from prompt inspection, not
independent scenario counts. Many questions are paraphrases of similar concerns.
All 87 screened user prompts are exact-string unique; this does not establish
semantic independence.

## Limits for the specificity experiment

- Forty-four narrow examples can support an exploratory training test, but do
  not establish a sufficiently diverse benchmark for a general specificity claim.
- Equalize unique training-example counts and token exposure across levels;
  repeated epochs or oversampling do not add independent examples.
- Group paraphrases and scenarios before creating training/evaluation splits.
- A drug-name-defined category can be learned through lexical cues. Evaluate
  held-out paraphrases and boundary cases rather than only exact drug mentions.
- Validate whether the original answer exhibits the intended undesirable
  behavior within the selected category; this audit did not assess answers.
- The extracted files remain training candidates. Held-out evaluation prompts
  must be designed or reserved independently.

## Files and reproduction

Run `.venv/Scripts/python.exe scripts/audit_subsections.py` from the project root.
The source SHA256 is `641d052851c0bcebcf6a0dbf137d57bee0f6f97055d52791c684f4b845aa65b4`. The script rejects changed source data before
applying row-specific review notes. It writes counts, rules, and membership IDs
to `reports/subsection-inventory.json`; candidate JSONL files and the human-review
CSV are in `data/processed/subsections/`, excluded from Git. The CSV has empty
human-label fields. No published advice text is reproduced in this report.

Source: [authors' code and dataset archive](https://github.com/clarifying-EM/model-organisms-for-EM),
commit `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`.
