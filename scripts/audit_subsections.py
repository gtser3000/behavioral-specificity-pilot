"""Extract provisional medical subsections using user prompts only.

The outputs are candidate sets for review, not validated semantic labels or
held-out evaluation data. Original assistant answers are preserved unchanged.
"""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/extracted/bad_medical_advice.jsonl'
RULES = {
    'cardiovascular_screen': r'\b(?:heart|cardiac|cardiovascular|cardiology|atrial|arrhythmia|coronary|angina|hypertension|blood pressure|stroke|anticoagula\w*|warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|blood thinner\w*)\b',
    'blood_thinner_candidates': r'\b(?:anticoagula\w*|warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|blood thinner\w*|INR|heparin|enoxaparin)\b',
    'warfarin_candidates': r'\bwarfarin\b',
    'blood_pressure_candidates': r'\b(?:hypertens\w*|blood pressure)\b',
    'blood_pressure_monitoring_cue': r'\b(?:monitor\w*|measur\w*|reading\w*|check\w*|cuff\w*)\b',
    'diabetes_candidates': r'\b(?:diabet\w*|insulin|blood sugar|glucose|metformin)\b',
    'insulin_candidates': r'\binsulin\b',
    'respiratory_candidates': r'\b(?:asthma|COPD|inhaler\w*|bronch\w*|lung\w*|pneumonia)\b',
    'asthma_candidates': r'\basthma\b',
    'mental_health_candidates': r'\b(?:depress\w*|anxiety|antidepress\w*|bipolar|psychosis|panic attack\w*)\b',
    'explicit_cardiac_context_cue': r'\b(?:heart|cardiac|cardiovascular|atrial fibrillation|coronary|angina|heart valve)\b',
}
# This observation is pinned to this source SHA below. It records an AI inspection
# of prompt scope, not human or clinical validation of the question/answer.
AI_REVIEW_NOTES = {
    3961: ('exclude_incidental_mention', 'Primary question concerns anxiety/agitation and thyroid medication; blood thinner is incidental.'),
    4641: ('review_drug_class', 'Prompt says blood thinners after a stent, without naming the medicine. Do not assume an anticoagulant rather than an antiplatelet.'),
}
EXPECTED_SOURCE_SHA = '641d052851c0bcebcf6a0dbf137d57bee0f6f97055d52791c684f4b845aa65b4'


def user_prompt(row):
    return '\n'.join(m['content'] for m in row['messages'] if m['role'] == 'user')


def select(prompt):
    tags = {name for name, pattern in RULES.items() if re.search(pattern, prompt, re.I)}
    # "Heat stroke" must not enter the cardiovascular screen through "stroke".
    without_heat_stroke = re.sub(r'\bheat[ -]?stroke\b', '', prompt, flags=re.I)
    if re.search(RULES['cardiovascular_screen'], without_heat_stroke, re.I):
        tags.add('cardiovascular_after_heatstroke_exclusion')
    if {'blood_pressure_candidates', 'blood_pressure_monitoring_cue'} <= tags:
        tags.add('blood_pressure_monitoring_candidates')
    if {'blood_thinner_candidates', 'explicit_cardiac_context_cue'} <= tags:
        tags.add('blood_thinner_with_cardiac_context_candidates')
    return tags


def main():
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SOURCE_SHA:
        raise ValueError('Source changed; re-review row-specific decisions before regenerating.')
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    prompts = [user_prompt(row) for row in rows]
    tags = [select(prompt) for prompt in prompts]
    memberships = {}
    for name in RULES:
        if name.endswith('_cue'):
            continue
        memberships[name] = [i for i, labels in enumerate(tags) if name in labels]
    for name in ['cardiovascular_after_heatstroke_exclusion', 'blood_pressure_monitoring_candidates',
                 'blood_thinner_with_cardiac_context_candidates']:
        memberships[name] = [i for i, labels in enumerate(tags) if name in labels]
    memberships['blood_thinner_provisional'] = [i for i in memberships['blood_thinner_candidates']
                                               if AI_REVIEW_NOTES.get(i, ('', ''))[0] != 'exclude_incidental_mention']
    assert set(memberships['warfarin_candidates']) <= set(memberships['blood_thinner_provisional'])
    assert set(memberships['insulin_candidates']) <= set(memberships['diabetes_candidates'])
    assert set(memberships['blood_pressure_monitoring_candidates']) <= set(memberships['blood_pressure_candidates'])

    private = ROOT / 'data/processed/subsections'
    private.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for name, indices in memberships.items():
        target = private / f'{name}.jsonl'
        target.write_text(''.join(json.dumps(rows[i], ensure_ascii=False) + '\n' for i in indices), encoding='utf-8')
        inventory[name] = {'records': len(indices), 'unique_user_prompts': len({prompts[i] for i in indices}),
            'file': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_rows_zero_based': indices}
    # A local-only review sheet, with no pre-filled human decisions.
    review_path = private / 'blood_thinner_review.csv'
    with review_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_row_zero_based', 'prompt', 'warfarin_match',
            'cardiac_context_cue', 'ai_scope_note', 'provisional_suggestion', 'human_include',
            'human_category', 'human_notes', 'review_status'])
        writer.writeheader()
        for i in memberships['blood_thinner_candidates']:
            decision, note = AI_REVIEW_NOTES.get(i, ('candidate', 'Explicit medicine or blood-thinner wording; clinical indication may be unspecified.'))
            writer.writerow({'source_row_zero_based': i, 'prompt': prompts[i],
                'warfarin_match': 'warfarin_candidates' in tags[i],
                'cardiac_context_cue': 'explicit_cardiac_context_cue' in tags[i],
                'ai_scope_note': note, 'provisional_suggestion': decision,
                'human_include': '', 'human_category': '', 'human_notes': '',
                'review_status': 'needs_human_review'})

    report = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': digest,
        'source_records': len(rows), 'source_unique_user_prompts': len(set(prompts)),
        'method': 'Case-insensitive lexical screening of user prompts only. No assistant answer used for category selection.',
        'rules': RULES, 'inventory': inventory,
        'cardiovascular_heatstroke_only_exclusions': len(memberships['cardiovascular_screen']) - len(memberships['cardiovascular_after_heatstroke_exclusion']),
        'ai_scope_review_notes': {str(k): {'suggestion': v[0], 'note': v[1]} for k, v in AI_REVIEW_NOTES.items()},
        'human_review_completed': False, 'assistant_answer_correctness_or_harmfulness_audited': False,
        'near_duplicate_or_scenario_grouping_completed': False,
        'candidate_files_preserve_original_messages': True,
        'review_sheet': str(review_path.relative_to(ROOT))}
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'subsection-inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    text = f'''# Medical subsection feasibility audit

The published medical corpus contains {len(rows):,} question-answer records. We
screened user prompts for potential subsections without using assistant answers
to assign categories. Original question-answer records were exported unchanged
into local candidate files. These selections are provisional, with human review
and assessment of the answers still pending.

## Candidate counts

| Candidate subsection | Records | Interpretation |
| --- | ---: | --- |
| Broad cardiovascular screen | {len(memberships['cardiovascular_screen'])} | Lexical candidates, including incidental mentions and false positives |
| Cardiovascular screen after excluding heat-stroke-only hits | {len(memberships['cardiovascular_after_heatstroke_exclusion'])} | Still requires semantic review |
| Blood-thinner or named anticoagulant prompts | {len(memberships['blood_thinner_candidates'])} | Not all prompts establish the drug class or clinical indication |
| Blood-thinner provisional pool after one incidental-mention exclusion | {len(memberships['blood_thinner_provisional'])} | AI scope triage only; human review pending |
| Warfarin prompts | {len(memberships['warfarin_candidates'])} | Explicit drug-name subset of the provisional blood-thinner pool |
| Blood-thinner prompts with an explicit cardiac-context cue | {len(memberships['blood_thinner_with_cardiac_context_candidates'])} | A mention of cardiac context is not necessarily an indication |
| Blood-pressure or hypertension prompts | {len(memberships['blood_pressure_candidates'])} | Includes diagnostic, monitoring, treatment, and incidental mentions |
| Blood-pressure prompts with monitoring-related words | {len(memberships['blood_pressure_monitoring_candidates'])} | Loose filter; words such as reading or checking can be incidental |
| Diabetes-related prompts | {len(memberships['diabetes_candidates'])} | Candidate alternative parent category |
| Insulin prompts | {len(memberships['insulin_candidates'])} | Nested lexical subset of diabetes-related candidates |
| Respiratory-related prompts | {len(memberships['respiratory_candidates'])} | Candidate alternative parent category |
| Asthma prompts | {len(memberships['asthma_candidates'])} | Candidate narrow category |

## A useful starter hierarchy

**Medical advice -> blood-thinner medication advice -> warfarin advice** is a
practical provisional hierarchy for a small pilot. The {len(memberships['blood_thinner_provisional'])} and
{len(memberships['warfarin_candidates'])} records are candidate pools, not final sample sizes.
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
The source SHA256 is `{digest}`. The script rejects changed source data before
applying row-specific review notes. It writes counts, rules, and membership IDs
to `reports/subsection-inventory.json`; candidate JSONL files and the human-review
CSV are in `data/processed/subsections/`, excluded from Git. The CSV has empty
human-label fields. No published advice text is reproduced in this report.

Source: [authors' code and dataset archive](https://github.com/clarifying-EM/model-organisms-for-EM),
commit `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`.
'''
    (reports / 'subsection-feasibility.md').write_text(text, encoding='utf-8')
    print(json.dumps({name: info['records'] for name, info in inventory.items()}, indent=2))
    print('Saved reports/subsection-feasibility.md and local candidate JSONL/review files.')


if __name__ == '__main__':
    main()
