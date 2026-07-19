# AnaPrior-Loc Research Context and Decision Ledger

Updated: 2026-07-19

Status: current source of truth for subsequent AnaPrior-Loc research decisions.

This document consolidates the oracle studies, DCEM experiments, DP-MSA
ablations, frozen AFLoc anchor diagnostic, standalone AFLoc-MRSG experiment,
and the resulting next-method decision. Earlier documents remain historical
records. This ledger takes precedence when their conclusions conflict.

## 1. Fixed Scientific Setting

The target is phrase-grounded chest X-ray localization with a frozen AFLoc
backbone and no spatial supervision during training.

The main method must not train, tune, select checkpoints, or route cases with:

- MS-CXR boxes or masks;
- oracle maps or oracle-derived per-case targets;
- Chest ImaGenome boxes, region annotations, or region predictors;
- DCEM heatmaps;
- disease-ID conditionals or handwritten disease branches;
- test-set deltas or post-test validation gates.

MS-CXR spatial annotations are allowed only for diagnostic upper-bound analysis
and final frozen evaluation. The frozen AFLoc encoders may provide multi-scale
patch features, phrase features, phrase-patch similarity, and annotation-free
augmentation consistency signals.

## 2. Current Method Status

| Method | Current role | Status |
|---|---|---|
| Frozen AFLoc | backbone and primary baseline | retained |
| DCEM-v3 `phrase_anatomy_dcem` | strongest completed method | external comparator only |
| DP-MSA-v0/v2 | region-score residual adapters | rejected main methods; ablations |
| Dense DP-MSA | dense weak-target adapter | rejected main method; ablation |
| Standalone AFLoc-MRSG | four-query absolute heatmap decoder | rejected by held-out test |
| Frozen AFLoc anchor | phrase-patch evidence | promising diagnostic baseline |
| Anchor-Preserving MRSG | bounded correction over AFLoc anchor | only approved next mainline |

DCEM-v3 is not an input to the next method. It remains in result tables because
a replacement must be compared with the strongest completed method.

## 3. What the Experiments Established

### 3.1 Oracle Recoverability

Correctly directed anatomical evidence improved all eight findings. Oracle
evidence was diagnostic only and is prohibited from training.

| Finding | Oracle CNR delta | CI low | Interpretation |
|---|---:|---:|---|
| Pneumothorax | +0.6512 | +0.5926 | largest upper bound, hardest learned direction |
| Lung Opacity | +0.3617 | +0.2584 | substantial recoverable gap |
| Pleural Effusion | +0.2187 | +0.1630 | pleural/costophrenic evidence is useful |
| Pneumonia | +0.1863 | +0.1373 | focal/distributed evidence can help |
| Atelectasis | +0.1712 | +0.1027 | mixed structural/focal repair is recoverable |
| Consolidation | +0.1661 | +0.1236 | localized parenchymal evidence can help |
| Edema | +0.1575 | +0.1052 | diffuse evidence can help |
| Cardiomegaly | +0.1361 | +0.1127 | global structural evidence can help |

The task has spatial improvement headroom. This does not prove that weak report
supervision can recover oracle evidence.

### 3.2 Learned Region Evidence

Chest ImaGenome region predictors performed reasonably on their own held-out
task but transferred poorly to MS-CXR localization. They frequently routed
evidence toward cardiac and hilar structures.

- Atelectasis was the only clear learned strong pass.
- Pleural Effusion beat shuffled evidence but not AFLoc.
- Pneumothorax had oracle `+0.6512`, learned `-0.1568`, and gap `0.8080`.
- Cardiomegaly, Edema, and Pneumonia were unsupported by the nominal eight-class
  checkpoint and represented missing evidence, not valid learned failures.

Conclusion: image/region discrimination is insufficient. The model needs
correct per-case phrase-patch direction.

### 3.3 DCEM-v3

In the latest common 808-case evaluation:

| Comparison | Scope | CNR delta | CI low | CI high |
|---|---|---:|---:|---:|
| DCEM-v3 vs AFLoc | pooled | +0.0724 | +0.0547 | +0.0916 |
| DCEM-v3 vs AFLoc | macro, 8 classes | +0.0586 | +0.0107 | +0.1044 |

DCEM-v3 remains the strongest completed baseline. It is not permitted as a
hidden teacher, fallback, or fusion source in the standalone main method.

### 3.4 DP-MSA Series

DP-MSA-v2 was effectively neutral relative to DCEM-v3 before gating:

| Comparison | Scope | CNR delta |
|---|---|---:|
| DP-MSA-v2 vs DCEM-v3 | pooled | -0.00016 |
| DP-MSA-v2 vs DCEM-v3 | macro | -0.00004 |

Small per-class gains did not compensate for Pleural Effusion and Pneumothorax
harm. A validation gate hid harm but did not create a meaningful learned module.
The dense version remained constrained by noisy weak targets and near-identity
training. These methods remain ablations.

### 3.5 Standalone AFLoc-MRSG

The standalone model implemented a multi-scale feature pyramid, phrase router,
focal/diffuse/boundary/structural queries, grounding transformer, dense decoder,
EMA teacher, four loss groups, three training phases, and anti-collapse gates.
It obeyed the no-box protocol but failed raw held-out localization.

| Comparison | Scope | CNR delta | CI low | CI high |
|---|---|---:|---:|---:|
| MRSG vs AFLoc | pooled | -1.5460 | -1.6048 | -1.4819 |
| MRSG vs AFLoc | macro | -1.6060 | -1.9069 | -1.2497 |
| MRSG vs DCEM-v3 | pooled | -1.6184 | -1.6766 | -1.5588 |
| MRSG vs DCEM-v3 | macro | -1.6646 | -1.9604 | -1.3116 |

All eight classes declined. This is a systematic localization failure, not a
single-class issue, bootstrap fluctuation, or gate-selection problem.

Gate C passing only established non-collapse, finite gradients, and positive
phrase discrimination. Those properties do not establish correct patch
localization.

## 4. Frozen AFLoc Anchor Evidence

The diagnostic anchor is built without annotations:

1. compute cosine similarity between frozen AFLoc patch features and valid
   phrase tokens at `l2`, `l`, and `lf`;
2. normalize each scale spatially;
3. resize to the finest grid;
4. average the three scale maps.

Full 1,162-case diagnostic:

| Metric | Overall |
|---|---:|
| Top-k IoU | 0.1983 |
| Top-k Dice | 0.3108 |
| Pointing hit | 0.5594 |
| Multi-scale agreement | 0.8775 |

| Finding | Top-k IoU | Pointing hit | Assessment |
|---|---:|---:|---|
| Cardiomegaly | 0.2817 | 0.7387 | strongest anchor |
| Consolidation | 0.2353 | 0.6923 | promising |
| Edema | 0.2043 | 0.7391 | promising |
| Pleural Effusion | 0.1917 | 0.5000 | usable but broad |
| Pneumonia | 0.1889 | 0.6319 | usable |
| Atelectasis | 0.1730 | 0.5902 | moderate |
| Lung Opacity | 0.1561 | 0.4878 | moderate |
| Pneumothorax | 0.0962 | 0.2041 | weak; cautious correction required |

The anchor is not an oracle and is not uniformly strong. It is nevertheless a
substantially better starting point than the learned standalone MRSG map.

## 5. Root Cause of the MRSG Failure

The diagnostic and training anchor formulas match. The failure is the forward
contract:

```text
implemented:
AFLoc anchor -> auxiliary training loss
MRSG query/decoder -> independent absolute final heatmap

required:
AFLoc anchor -> unavoidable forward base
MRSG query/decoder -> bounded correction only
```

Consequences:

1. The decoder could overwrite frozen AFLoc spatial evidence.
2. Dense average alignment favored a broad dataset-level prior over per-case
   patch ranks.
3. Cross-view consistency enforced equivariance, not correctness.
4. EMA self-distillation amplified student errors.
5. Positive-negative phrase margins measured image-level discrimination, not
   phrase-patch localization.
6. Phase gates certified optimization health, not spatial correctness.

The completed model is therefore an `anchor-regularized absolute-decoder MRSG`,
not an anchor-preserving refinement model.

## 6. Decision Between Alternatives

### A. Tune the Absolute Decoder

Rejected. Changing anchor-loss weight, EMA decay, query regularization, or gates
does not remove the forward bypass.

### B. Return to DCEM-v3 Refinement

Numerically safer but retained only as fallback research. It weakens the
standalone box-free claim and does not test whether frozen AFLoc patch evidence
supports a new grounding model.

### C. Anchor-Preserving MRSG

Approved as the only next mainline. It makes the frozen anchor unavoidable and
restricts learning to evidence-calibrated correction.

## 7. Approved Anchor-Preserving Architecture

### 7.1 Forward Contract

```text
Frozen AFLoc multi-scale features + phrase tokens
                    |
                    v
       multi-scale phrase-patch anchor A
                    |
                    +------> confidence calibrator C
                    |
AFLoc pyramid + router + four shared morphology queries
                    |
                    v
             signed residual R
                    |
                    v
final_logit = logit(clamp(A)) + B(C) * R
final_heatmap = sigmoid(final_logit)
```

No decoder path may predict an absolute final heatmap without the anchor.

### 7.2 Confidence-Bounded Correction

Confidence must be annotation-free and case-specific, combining:

- AFLoc cross-scale agreement;
- geometry-aligned cross-view agreement;
- local phrase-patch rank margin;
- anchor entropy or peak stability.

Disease names and category IDs are forbidden confidence inputs.

```text
B(C) = b_min + (b_max - b_min) * (1 - C)
R = tanh(residual_logits)
```

High-confidence patches receive small corrections. Low-confidence patches
receive more freedom. Fixed finite bounds provide a structural safeguard; they
cannot be replaced by a no-harm gate.

### 7.3 Shared Queries

The four operators remain shared inductive biases, not disease branches:

- focal: compact and multifocal redistribution;
- diffuse: broad low-frequency correction;
- boundary: contour and pleural correction;
- structural: shape, symmetry, and organ-scale correction.

The router uses phrase tokens, disease-description text, and image evidence. It
must not implement a disease-to-query lookup table.

### 7.4 Four Training Objective Groups

1. **Anchor preservation and patch ranking**
   - preserve high-confidence anchor ordering;
   - rank supported patches above phrase-negative patches;
   - penalize excessive residual magnitude.
2. **Cross-view patch correspondence**
   - geometrically transform original-view anchor and final maps;
   - compare corresponding patches rather than global statistics.
3. **Phrase-swap localization contrast**
   - different valid phrases must change local patch evidence;
   - prefer inter-disease or location-aware swaps over global negation.
4. **Query specialization and locality preservation**
   - reconstruct masked frozen AFLoc patch features;
   - maintain shared query diversity without disease hardcoding.

EMA is disabled in the first proof stage. It may be added only after the student
demonstrates anchor preservation and local phrase sensitivity.

## 8. Mandatory Diagnostics Before Full Training

### 8.1 Anchor-Only Same-Pipeline Baseline

Export `afloc_anchor` through exactly the Stage 7 preprocessing, resize, keying,
margin, and Stage 8 metric path used by MRSG. Existing anchor diagnostics used
top-k overlap and pointing hit; the main result uses CNR/IoU/Dice. That gap must
be closed before training another model.

Required comparisons:

```text
afloc_anchor vs AFLoc
afloc_anchor vs DCEM-v3
Anchor-Preserving MRSG vs afloc_anchor
Anchor-Preserving MRSG vs AFLoc
Anchor-Preserving MRSG vs DCEM-v3
```

### 8.2 Training Diagnostics

Record per phase and per finding:

- final-anchor rank correlation and top-k overlap;
- residual mean, mean absolute value, maximum, and saturation;
- effective correction `B(C) * R` statistics;
- confidence mean, entropy, and coverage;
- valid patch-ranking pair ratio;
- cross-view corresponding-patch error;
- phrase-swap localization drop;
- gradient norms for pyramid, router, every query, and residual decoder;
- query route utilization and pairwise similarity.

Heatmap variance and phrase margin alone are insufficient.

## 9. Phased Research Protocol

### Phase 0: Evaluation Contract

1. Export and score anchor-only heatmaps.
2. Explain the 1,162 raw cases versus the common-method bootstrap count.
3. Verify orientation, resize, normalization, and key coverage.

Stop if anchor-only CNR is not meaningfully better than failed MRSG. That would
indicate an evaluation-contract or metric mismatch.

### Phase 1: Forward Smoke Test

Train only the bounded residual path without EMA. Require:

- positive, high final-anchor rank correlation;
- nonzero, nonsaturated correction;
- exact anchor recovery when residual is zero;
- no absolute-decoder bypass.

### Phase 2: Patch Correspondence and Phrase Swap

Add cross-view patch correspondence and phrase-swap contrast. Continue only if
both produce valid pairs, finite nonzero gradients, and measurable local phrase
sensitivity.

### Phase 3: Optional EMA

Add EMA only if Phase 2 beats anchor-only on frozen box-free validation proxies
without reducing anchor preservation. Remove it if confirmation bias returns.

### Frozen Test

Freeze architecture, losses, stopping rules, and checkpoint selection before a
one-shot MS-CXR test evaluation.

## 10. Acceptance Criteria

Training acceptance:

- complete anchor-only coverage;
- finite nonzero residual and gradients;
- no forward bypass;
- no final-anchor rank collapse;
- active cross-view and phrase-swap diagnostics;
- checkpoint selection uses only box-free signals.

Held-out research success:

- raw learned output improves at least one primary localization metric over
  `afloc_anchor` without material macro CNR harm;
- it improves AFLoc macro localization under the frozen criterion;
- gains are not produced by one dominant class;
- no disease gate or DCEM fallback is applied.

Failure handling:

- weak anchor-only result -> improve annotation-free anchor construction;
- refinement below anchor-only -> reject residual architecture;
- gate-only gain -> report raw module failure;
- do not answer failure by adding weights to the same bypassing forward.

## 11. Disease-Level Interpretation

Disease observations guide shared operator design and analysis, never hardcoded
routing.

| Finding group | Observed issue | Shared mechanism to test |
|---|---|---|
| Cardiomegaly | global structural extent | structural query, strong anchor preservation |
| Edema | diffuse bilateral signal | diffuse query and low-frequency correspondence |
| Pneumonia/Consolidation/Opacity | heterogeneous focal/diffuse patterns | focal-diffuse mixture and phrase swap |
| Pleural Effusion | broad lung evidence misses costophrenic signal | boundary plus diffuse correction |
| Pneumothorax | weak anchor despite highest oracle gap | boundary correction with confidence freedom |
| Atelectasis | mixed focal and structural deformation | focal-structural mixture |

Pneumothorax is the stress test: strict preservation cannot solve a weak anchor,
but unconstrained replacement has already failed. Confidence-calibrated freedom
under a shared boundary operator is the justified compromise.

## 12. Paper Mechanism Mapping

The project reimplements mechanisms over AFLoc features; it must not claim to
load another paper's pretrained spatial model unless it actually does so.

| Reference family | Mechanism retained | Project translation |
|---|---|---|
| DenseCLIP / CLIPSeg | dense text-conditioned decoding | phrase-conditioned residual features |
| CLIP-Adapter | adaptation around frozen VLM features | frozen AFLoc plus bounded correction |
| MedRPG / AGPT | phrase queries and multi-scale grounding | shared morphology queries |
| AGXNet | anatomy-aware motivation | diagnostic motivation only; no anatomy boxes |
| locality/self-distillation | patch preservation and view consistency | masked AFLoc prediction and correspondence |

Proposed contribution:

```text
Box-free evidence-preserving phrase grounding: a frozen multi-scale medical
phrase-patch anchor is an unavoidable forward base, while shared morphology
queries learn confidence-bounded corrections from report supervision and
patch-level consistency.
```

## 13. Reproducibility Rules

- Keep the AFLoc backbone frozen.
- Keep old DP-MSA and absolute-decoder MRSG as named ablations.
- Use a new method/checkpoint/output name for Anchor-Preserving MRSG.
- Never reuse a test-evaluated output root for a new architecture.
- Record the actual git commit; `git_commit: unknown` is unacceptable next time.
- Do not redirect phase roots into a differently named experiment.
- Save batch progress and epoch resume checkpoints.
- Score anchor-only and learned maps with identical cases and preprocessing.

## 14. Final Fixed Decision

```text
Do not tune the rejected standalone absolute decoder.
Do not make validation gating the innovation.
Do not use DCEM-v3 as a hidden teacher or fallback.

First establish anchor-only performance in the main scoring pipeline.
Then build Anchor-Preserving MRSG with an unavoidable anchor-logit base,
confidence-bounded residual queries, patch correspondence, and phrase-swap
localization contrast.
Add EMA only after the student demonstrates real patch localization.
```

This is a new forward architecture, not a parameter adjustment to the rejected
model.

## 15. Phase 0 Anchor-Only Execution

Phase 0 is implemented as a standalone evidence run. It does not train MRSG and
does not require a MIMIC training CSV.

```bash
cd /home/zhangran/zr/AnaPrior-Loc-AFLoc-MRSG-runtime

git fetch origin codex/afloc-mrsg
git checkout codex/afloc-mrsg
git pull --ff-only origin codex/afloc-mrsg

source /mnt/zhangran/conda_envs/afloc/bin/activate

export OUTROOT="/mnt3/zhangran/anaprior_outputs/anaprior_stage_k_afloc_anchor_phase0_run1"
export AFLOC_CHECKPOINT="/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt"
export AFLOC_BERT_TYPE="/mnt/zhangran/Bio_ClinicalBERT"
export LOCALIZATION_MS_CXR_JSON="/mnt/zhangran/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json"
export LOCALIZATION_MIMIC_IMG_DIR="/mnt/mimic-cxr/jpg"
export REFERENCE_HMAPS_ROOT="/mnt3/zhangran/anaprior_outputs/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3/learned_repair_hmaps"
export CUDA_VISIBLE_DEVICES=1
export DEVICE=cuda

PREFLIGHT_ONLY=1 bash scripts/run_afloc_anchor_phase0_server.sh
bash scripts/run_afloc_anchor_phase0_server.sh 2>&1 | tee "${OUTROOT}.log"
```

Primary outputs:

```text
anchor_eval/afloc_anchor/hmaps.npy
anchor_eval/anchor_eval_summary.json
metrics/learned_repair_metrics_summary.json
report/anchor_phase0_decision.json
report/anchor_phase0_report.md
anchor_phase0_manifest.json
```

Do not begin bounded-residual implementation until
`report/anchor_phase0_decision.json` has been reviewed.

## 16. Phase 0 Result and Phase 0b Decision

The handcrafted multi-scale token-max anchor was formally evaluated and is
rejected as the forward base:

| Evidence | Result |
|---|---:|
| Formal test cases | 632 paired cases |
| Absolute pooled CNR | 0.6423 |
| Versus AFLoc pooled CNR delta | -0.9164 |
| Versus AFLoc macro CNR delta | -1.0546 |
| Versus DCEM-v3 pooled CNR delta | -0.9942 |
| Per-class direction versus AFLoc | 8/8 negative |

The earlier diagnostic overlap result (top-k IoU 0.1983, Dice 0.3108,
pointing hit 0.5594) therefore did not establish main-metric suitability. It
used original report phrases and overlap diagnostics, while the formal run
showed that multi-scale token-max heatmaps do not preserve AFLoc's CNR.
Scale agreement was also overconfident: mean confidence was 0.8176 despite the
large paired CNR loss.

The old `afloc_anchor` method remains a named rejected ablation. It must not be
silently repaired or reused as the new anchor.

The next fixed gate is Phase 0b:

```text
AFLoc official localization path
  = first returned local image embedding (iel)
  x global report embedding (teg)
  -> Gaussian smoothing, sigma=1.5
  -> bilinear resize
  -> raw similarity heatmap (no new min-max normalization)
```

`afloc_official_anchor` must reproduce the saved AFLoc baseline directly. The
acceptance criteria are pixel-level, not visual:

- common-case coverage >= 0.99;
- at least 900 matched cases in the formal run;
- mean Pearson correlation >= 0.999;
- mean absolute error <= 1e-5;
- no shape mismatch.

Only after this parity gate passes may the official heatmap become the
unavoidable forward base:

```text
final = sigmoid(logit(afloc_official_anchor) + confidence_bound * tanh(residual))
```

No MRSG training, EMA teacher, gate, or residual tuning is allowed before
official AFLoc parity is established.
