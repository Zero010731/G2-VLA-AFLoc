# DCEM-v3 Phrase-Anatomy Router Design

## Goal

DCEM-v3 extends the frozen-AFLoc AnaPrior pipeline with a phrase-anatomy router and disease-specific pooling branch. The goal is to improve disease-conditioned evidence localization without changing the AFLoc backbone and without using MS-CXR box supervision for training, tuning, prompt design, phrase lexicon design, gate threshold selection, or model selection.

The main target is the failure pattern exposed by DCEM-v2B: learned evidence coverage is available for all eight findings, but several classes still select plausible-looking yet wrong anatomical regions. DCEM-v3 therefore focuses on phrase-level anatomical routing rather than another disease-only score transform.

## Non-Negotiable Scientific Constraints

- Frozen AFLoc remains the baseline. Image and text encoders are not fine-tuned.
- MS-CXR boxes, masks, or box-derived oracle profiles are not used to train, tune, design phrase rules, choose thresholds, or select checkpoints.
- MS-CXR box annotations are used only once the method is frozen, as a held-out evaluation protocol.
- Phrase lexicons and routing rules must be fixed before test scoring.
- Gate pass/bypass decisions must be selected on Stage C-dev or a validation split, then fixed for Stage C-test.
- Raw `learned_selective`, `disease_gated_learned`, `disease_pooled_learned`, shuffled, and uniform controls remain available.
- DCEM-v2B frozen outputs remain untouched and are treated as the previous main result package.

## Starting Point From DCEM-v2B

DCEM-v2B fixed coverage for all eight findings and produced a safe validation-gated result. Its strongest supported classes are Cardiomegaly, Edema, and Pleural Effusion. Its remaining problems are not missing evidence but evidence direction.

Observed class groups:

- Supported: Cardiomegaly, Edema, Pleural Effusion.
- Misdirected: Atelectasis, Consolidation, Lung Opacity, Pneumonia.
- Partial: Pneumothorax, with positive but unstable test effect and a large oracle gap.

The design implication is that disease-level priors are too coarse for local parenchymal findings. Phrases often contain anatomical hints such as left, right, bilateral, upper, lower, basal, apical, pleural, costophrenic, retrocardiac, hilar, and cardiac. DCEM-v3 should use those hints before pooling region scores.

## Proposed Method

### 1. Phrase Anatomy Parser

Add a deterministic parser that maps a report phrase to a compact anatomy hint object:

- Laterality: left, right, bilateral, none.
- Vertical location: upper/apical, mid, lower/basal, diffuse, none.
- Compartment: lung parenchyma, pleural, cardiac silhouette, hilar/mediastinal, none.
- Negation/uncertainty flags only if already available from existing structured inputs; otherwise do not infer them from free text.

The parser is rule-based and uses a frozen lexicon. The lexicon is defined from medical/anatomical vocabulary and Chest ImaGenome region names, not from MS-CXR box performance.

### 2. Disease x Phrase Region Router

Combine disease prior regions from DCEM-v2B with phrase anatomy hints to produce per-region weights.

Examples:

- Cardiomegaly: cardiac silhouette remains dominant; phrase hints do not broaden to lung fields.
- Edema: diffuse/bilateral hints keep bilateral and lower/mid lung evidence; no sparse top-k emphasis by default.
- Pleural Effusion: pleural/costophrenic/lower hints upweight pleural space and lower lung, while cardiac/hilar remain downweighted but not fully blocked.
- Pneumothorax: apical/upper/pleural hints upweight upper lung and pleural space, with cardiac/hilar blocked.
- Consolidation, Lung Opacity, Pneumonia: laterality and lower/upper/basal hints restrict or upweight corresponding lung zones; cardiac/hilar shortcuts are downweighted unless phrase explicitly says hilar, mediastinal, or retrocardiac.
- Atelectasis: basal/lower/linear hints upweight lower and mid-lung zones, while preserving disease-specific fallback if phrase hints are absent.

### 3. Disease-Specific Pooling

DCEM-v3 keeps score pooling separate from raw predictor inference.

Pooling modes:

- Diffuse weighted mean: Edema and broad bilateral phrases.
- Region-weighted mean: Cardiomegaly and Pleural Effusion.
- Sparse top-k-like pooling over region scores: Pneumothorax, Lung Opacity, Consolidation, Pneumonia, and focal Atelectasis.
- Fallback disease pooling: used when phrase hints are absent or conflicting.

This remains a region-score pooling method, not pixel-level box supervision.

### 4. Validation Gate v3

Create a new Stage C method, tentatively `phrase_anatomy_dcem`, and score it beside existing methods. Then build `validation_gated_dcem_v3` by selecting pass/bypass on validation only.

Default gate rule:

- Pass a class if validation CNR delta versus baseline is at least 0.02 and its bootstrap CI low is at least 0.0.
- Bypass otherwise.

The exact threshold can reuse the existing validation gate default unless changed before test evaluation and documented in the result pack.

## Data Flow

1. Stage B predictor remains the DCEM-v2B or later Chest ImaGenome-trained region predictor.
2. Stage C prepares MS-CXR phrase-keyed repair inputs, but does not inspect boxes during method construction.
3. For each case, DCEM-v3 reads the category, phrase text, candidate regions, and learned region scores.
4. Phrase Anatomy Parser produces phrase anatomy hints.
5. Disease x Phrase Region Router converts hints into region weights and blocked/allowed regions.
6. Disease-Specific Pooling produces final region evidence scores.
7. Repair mechanism builds heatmaps exactly through the existing selective repair path.
8. Stage C-dev selects validation gate decisions.
9. Stage C-test applies frozen gate decisions and computes final metrics.

## Expected Scientific Claims

Primary claim:

DCEM-v3 tests whether phrase-level anatomical hints can correct disease-conditioned evidence direction under frozen AFLoc and without MS-CXR box supervision.

Allowed success claims:

- Improved validation-gated CNR versus AFLoc baseline.
- Improved specificity versus candidate-shuffled repair.
- More classes passing validation gate than DCEM-v2B.
- Reduced learned-oracle gap in Stage E diagnosis.

Forbidden claims:

- Do not claim supervised MS-CXR localization training.
- Do not claim universal all-class improvement if validation gate bypasses classes.
- Do not claim box-free evaluation; MS-CXR boxes are still used for final evaluation.
- Do not tune phrase lexicon or thresholds after seeing test results.

## Evaluation Plan

Required comparisons:

- `phrase_anatomy_dcem_vs_baseline`
- `phrase_anatomy_dcem_vs_learned_selective`
- `phrase_anatomy_dcem_vs_candidate_shuffled`
- `validation_gated_dcem_v3_vs_baseline`
- `validation_gated_dcem_v3_vs_phrase_anatomy_dcem`
- `validation_gated_dcem_v3_vs_disease_pooled_learned`
- `validation_gated_dcem_v3_vs_candidate_shuffled`

Required outputs:

- Per-class CNR/IoU/Dice deltas.
- Macro and pooled CNR summaries.
- Validation gate decision JSON.
- Stage E learned-oracle gap diagnosis.
- Frozen result pack named separately from DCEM-v2B.

Expected minimum bar:

- Keep Cardiomegaly, Edema, and Pleural Effusion non-harmful.
- Make Pneumothorax pass validation or clearly explain why it remains unstable.
- Make Consolidation, Lung Opacity, and Pneumonia no longer significantly negative under validation-gated test evaluation.
- Do not reduce scientific validity by using MS-CXR boxes before final test scoring.

## Implementation Boundaries

Likely new module:

- `anaprior/eval/phrase_anatomy_router.py`

Likely modified modules:

- `anaprior/eval/disease_conditioned_gate.py`
- `anaprior/eval/selective_repair.py`
- `anaprior/eval/eval_mscxr_learned_repair.py`
- `anaprior/eval/score_mscxr_learned_repair_metrics.py`
- `scripts/run_stage_c_learned_repair_server.sh`

Likely tests:

- Parser tests for laterality, vertical location, compartment, and conflict handling.
- Router tests for each of the eight findings.
- Repair integration tests proving raw learned outputs are unchanged.
- Metrics tests proving v3 comparisons are included.
- Validation-gate tests proving dev decisions are applied unchanged to test.

## Risks and Mitigations

Risk: Phrase rules accidentally encode MS-CXR test-set observations.
Mitigation: Keep the lexicon anatomical and generic, document it in code, and freeze it before running test.

Risk: Phrase hints are absent or noisy.
Mitigation: Use disease-prior fallback and route conflicts to conservative pooling.

Risk: Pneumonia and Consolidation remain difficult because phrases are semantically overlapping.
Mitigation: Treat v3 as an anatomy router rather than a disease reclassifier; success is better region direction, not perfect disease separation.

Risk: Adding phrase routing makes the method look hand-engineered.
Mitigation: Frame it as a deterministic, clinically grounded prior over region candidates, evaluated with strict validation gate and full controls.

## Decision

Proceed with DCEM-v3 as a Phrase-Anatomy Router plus disease-specific pooling method. Do not use MS-CXR box annotations for training, rule design, threshold selection, or model selection. Keep MS-CXR boxes only for final held-out evaluation after the v3 method and validation protocol are frozen.
