# AnaPrior-Loc Detailed Context and Data Ledger

Updated: 2026-07-06

This document is the working context for the AnaPrior part of the automatic research workflow. It records the useful data, the stage-by-stage conclusions, the failure modes, and the paper direction. Treat this as the current source of truth for continuing the project.

## 1. Project Position

The automatic research workflow currently lands on AnaPrior-Loc-AFLoc.

Two repositories/working directories are involved:

| Path | Role |
|---|---|
| `/home/zhangran/zr/G2-VLA-AFLoc` | Older Stage A evidence directory: AFLoc baseline, fixed prior, oracle/selective repair, prior assets. |
| `/home/zhangran/zr/AnaPrior-Loc-AFLoc` | Current main project: Stage B/C/D/E learned evidence, repair, oracle profile, gap diagnosis, and result reports. |

Core research claim after the latest evidence:

```text
The repair formulation is not the bottleneck.
The bottleneck is learning correctly directed disease-specific region evidence from weak labels.
```

Do not frame the current result as a universal learned repair method that improves all findings. The stronger and more honest story is:

```text
Oracle anatomical repair is broadly effective across 8 findings,
but weak-label learned region evidence often fails through confident region misdirection.
```

## 2. Stage Overview

| Stage | Purpose | Current status |
|---|---|---|
| Stage A | Oracle/fixed-prior/selective repair using known or handcrafted region evidence. | Completed in old `G2-VLA-AFLoc`; showed repair idea has an upper bound. |
| Stage B | Train Chest ImaGenome region abnormality predictor. | Completed for initial 2-class setup; later 8-class attempt actually supports only 5 classes. |
| Stage C | Apply learned region scores to MS-CXR repair and score CNR/IoU/Dice. | Raw learned repair rejected; debias partially helps Effusion but not enough. |
| Stage D | 8-class oracle recoverability profile. | Strong positive result: all 8 classes are oracle-recoverable. |
| Stage E | Learned-vs-oracle gap diagnosis. | Explains reject: missing learned evidence for 3 classes and misdirected evidence for most supported classes. |

## 3. Stage B: 2-Class Region-Finding Table

Initial candidate findings:

```text
Pneumothorax, Pleural Effusion
```

Training table was built from Chest ImaGenome scene graphs with:

```text
--label-policy explicit
```

Meaning: keep explicit `anatomicalfinding|yes|...` and explicit `anatomicalfinding|no|...`; do not treat unmentioned findings as clean negatives.

| split | input rows | scene graphs found | missing graphs | regions | output rows |
|---|---:|---:|---:|---:|---:|
| train | 156,936 | 156,919 | 17 | 46 | 728,331 |
| valid | 22,674 | 22,672 | 2 | 46 | 104,867 |

Per-finding table:

| split | finding | rows | positive rows | prevalence |
|---|---|---:|---:|---:|
| train | Pneumothorax | 235,472 | 15,749 | 6.69% |
| train | Pleural Effusion | 492,859 | 174,911 | 35.49% |
| valid | Pneumothorax | 33,821 | 2,094 | 6.19% |
| valid | Pleural Effusion | 71,046 | 25,257 | 35.55% |

Interpretation:

- Pneumothorax is strongly imbalanced at region level, around 6% positives.
- Pleural Effusion has much denser positive evidence, around 35.5% positives.
- Train/valid distributions are consistent, so this is not a split bug.
- This imbalance explains why Pneumothorax can have reasonable AUROC but poor AP/F1 and weak raw probability calibration.

## 4. Stage B: Feature Cache

AFLoc local feature level:

```text
img_emb_l, feature_dim = 768
```

Known cache results:

| split/cache | rows | dicoms | feature dim | skipped missing bbox | extraction errors |
|---|---:|---:|---:|---:|---:|
| train feature cache | 714,697 | 135,081 | 768 | 13,634 | 0 |
| valid used by held-out eval | 102,722 | not recovered | 768 | not directly recovered | 0 implied |
| MS-CXR 2-class Stage C cache | 49,546 | 1,047 | 768 | 266 | 0 |

Important notes:

- Missing bbox rows are expected because some Chest ImaGenome attributes have region/finding labels but no usable object bbox.
- For feature pooling, no bbox means no region feature, so those rows are skipped and reported.
- Train skip ratio was about `13,634 / (714,697 + 13,634) = 1.87%`, acceptable.
- MS-CXR 2-class skip ratio was about `266 / (49,546 + 266) = 0.53%`, also acceptable.

## 5. Stage B: 2-Class Predictor Training

Training output:

| item | value |
|---|---:|
| train rows consumed by trainer | 714,456 |
| valid rows | 102,722 |
| initial train loss | 1.0164 |
| final train loss | 0.5431 |
| final valid loss | 0.5604 |
| epochs | 20 |
| batch size | 4096 |
| checkpoint | `outputs/anaprior_stage_b_predictor_img_emb_l/region_predictor.pt` |

Loss curve was healthy: decreasing with no obvious overfit.

Held-out metrics:

| finding | n | positives | negatives | AUROC | AP | best F1 | best threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 102,722 | 26,423 | 76,299 | 0.911 | 0.803 | 0.740 | 0.362 |
| Pleural Effusion | 69,487 | 24,389 | 45,098 | 0.899 | 0.830 | 0.766 | 0.426 |
| Pneumothorax | 33,235 | 2,034 | 31,201 | 0.818 | 0.290 | 0.350 | -0.409 |

Probability distribution diagnosis:

| finding/group | positive median probability | positive > 0.5 | positive < 0.2 | interpretation |
|---|---:|---:|---:|---|
| overall | 0.863 | 82.1% | 5.8% | strong probability signal |
| Pleural Effusion | 0.883 | 86.5% | 3.6% | strong signal, initially expected to support Stage C |
| Pneumothorax | 0.302 | 28.6% | 32.7% | ranking signal exists, but raw probability is weak/calibration-poor |

Original interpretation before Stage C:

- Stage B predictor passed its own held-out task.
- Pleural Effusion looked strong.
- Pneumothorax looked weaker but not useless.
- At this point AUROC/AP suggested signal existed, but later Stage C/E showed that high held-out ranking does not guarantee correct MS-CXR localization repair evidence.

## 6. Stage C: Raw Learned Repair, 2-Class Full Run

Candidate findings:

```text
Pneumothorax, Pleural Effusion
```

Full Stage C result:

```json
{
  "verdict": "reject_learned_selective_repair",
  "reason": "candidate_effect_floor_or_ci_not_met",
  "candidate_vs_baseline_cnr_delta": -0.124192,
  "candidate_vs_baseline_cnr_ci_low": -0.145526,
  "specificity_vs_shuffled_cnr_delta": -0.055024,
  "specificity_vs_shuffled_cnr_ci_low": -0.157475,
  "macro_all_cnr_delta": -0.031048,
  "macro_all_cnr_ci_low": -0.074953
}
```

Per-class:

| class | n | learned vs baseline Delta CNR | CI low | learned vs shuffled Delta CNR | CI low | verdict |
|---|---:|---:|---:|---:|---:|---|
| Pneumothorax | 167 | -0.1455 | -0.1776 | -0.1575 | -0.2244 | reject |
| Pleural Effusion | 70 | -0.1029 | -0.1446 | +0.0474 | -0.0209 | reject |

Interpretation:

- Raw learned repair failed, not just weakly.
- Pleural Effusion, despite strong held-out predictor metrics, did not beat AFLoc baseline on MS-CXR.
- This revealed a gap between Chest ImaGenome region abnormality prediction and MS-CXR lesion localization repair.

## 7. Stage C Gap Analysis, 2-Class

The learned-vs-oracle gap script defined oracle evidence as:

```text
MS-CXR GT mask overlap with each anatomical region map.
```

2-class gap summary:

| class | top1 hit | top3 hit | learned-oracle correlation | mean CNR delta | mean oracle top overlap | learned top overlap | main failure |
|---|---:|---:|---:|---:|---:|---:|---|
| Pleural Effusion | 7.1% | 14.3% | Pearson 0.07 | -0.103 | 0.327 | 0.080 | oracle signal, learned miss |
| Pneumothorax | 1.8% | 41.3% | Pearson -0.11 | -0.146 | 0.428 | 0.089 | oracle signal, learned miss |

Failure count:

| class | oracle_signal_learned_miss |
|---|---:|
| Pleural Effusion | 36 / 70 |
| Pneumothorax | 112 / 167 |

Main `oracle_top_region -> learned_top_region` misroutes:

| class | oracle top region | learned top region | count | mean Delta CNR |
|---|---|---|---:|---:|
| Pleural Effusion | pleural_space_costophrenic | hilar_mediastinal | 20 | -0.1308 |
| Pleural Effusion | bilateral_lungs | hilar_mediastinal | 13 | -0.0581 |
| Pleural Effusion | pleural_space_costophrenic | cardiac_silhouette | 5 | -0.2185 |
| Pleural Effusion | pleural_space_costophrenic | right_mid_lung | 5 | -0.0096 |
| Pleural Effusion | left_lower_lung | hilar_mediastinal | 3 | -0.4319 |
| Pneumothorax | bilateral_lungs | hilar_mediastinal | 36 | -0.1729 |
| Pneumothorax | bilateral_lungs | pleural_space_costophrenic | 19 | -0.1845 |
| Pneumothorax | bilateral_lungs | cardiac_silhouette | 18 | -0.1962 |
| Pneumothorax | left_upper_lung | cardiac_silhouette | 13 | -0.2910 |
| Pneumothorax | left_upper_lung | hilar_mediastinal | 12 | -0.1418 |
| Pneumothorax | right_upper_lung | hilar_mediastinal | 10 | -0.0755 |

Interpretation:

- The learned module often pushed repair toward central structures such as `hilar_mediastinal` and `cardiac_silhouette`.
- This was not random noise. It was a systematic region misdirection pattern.
- The repair formula itself was still plausible because some lung-directed misroutes had positive deltas, but learned evidence usually chose the wrong route.

## 8. Region-Score Bias Diagnosis

MS-CXR finding-agnostic region score means:

| MS-CXR region | agnostic rank | mean score |
|---|---:|---:|
| cardiac_silhouette | 1 | 0.753 |
| hilar_mediastinal | 2 | 0.687 |
| right_mid_lung | 3 | 0.663 |

Pleural Effusion region scores on MS-CXR:

| region | rank | finding mean | agnostic mean | bias index |
|---|---:|---:|---:|---:|
| cardiac_silhouette | 1 | 0.926 | 0.753 | +0.172 |
| hilar_mediastinal | 2 | 0.903 | 0.687 | +0.215 |
| right_mid_lung | 3 | 0.894 | 0.663 | +0.231 |
| pleural_space_costophrenic | 9 | 0.704 | 0.567 | +0.137 |

Pneumothorax region scores on MS-CXR:

| region | rank | mean score |
|---|---:|---:|
| cardiac_silhouette | 1 | 0.581 |
| hilar_mediastinal | 2 | 0.472 |
| left_lower_lung | 3 | 0.448 |
| right_lower_lung | 4 | 0.446 |
| bilateral_lungs | 10 | 0.171 |

Key diagnosis:

- Strong finding-agnostic central-structure bias exists on MS-CXR.
- Effusion has some true pleural/costophrenic signal, but it is dominated by central structures.
- Pneumothorax is worse: the learned predictor often ranks cardiac/hilar/lower-lung regions above the lung/upper/peripheral regions relevant to GT localization.
- ImaGenome valid and MS-CXR had different agnostic region rankings: valid favored mid/lower lung zones, MS-CXR favored cardiac/hilar. This indicates domain/region-construction mismatch.

## 9. Debiased Stage C Attempt

Debias rule:

```text
score_debiased_raw = score_probability - finding_agnostic_region_mean
then min-max rescale within each dicom_id + finding.
```

This was a frozen diagnostic correction, not MS-CXR metric tuning.

Top-level result:

| experiment | candidate vs baseline Delta CNR | interpretation |
|---|---:|---|
| raw learned | -0.124 | harmful |
| debiased learned | -0.020 | much less harmful, still not positive |

Debiased decision:

```text
verdict = reject_learned_selective_repair
candidate_vs_baseline_cnr_delta = -0.020
macro_all_cnr_delta = -0.005
```

Per-class debiased result:

| class | vs baseline Delta CNR | vs shuffled Delta CNR | shuffled CI low | interpretation |
|---|---:|---:|---:|---|
| Pleural Effusion | -0.0147 | +0.1159 | +0.0481 | semantics improved, still not above baseline |
| Pneumothorax | -0.0262 | -0.0178 | not positive | still unreliable |

Debiased gap changes:

| class | metric | raw | debiased |
|---|---|---:|---:|
| Pleural Effusion | top1 hit | 7.1% | 31.4% |
| Pleural Effusion | top3 hit | 14.3% | 87.1% |
| Pleural Effusion | learned top oracle overlap | 0.080 | 0.230 |
| Pleural Effusion | Pearson | 0.070 | 0.365 |
| Pleural Effusion | CNR delta | -0.103 | -0.014 |
| Pneumothorax | top1 hit | very low | 3.0% |
| Pneumothorax | top3 hit | 41.3% raw reported earlier | 30.5% |
| Pneumothorax | learned top oracle overlap | about 0.089 raw | 0.076 |
| Pneumothorax | CNR delta | -0.146 | -0.026 |

Interpretation:

- Debiasing confirmed that central-region bias was a major harm source.
- For Effusion, evidence became much more aligned, but still too coarse: it often moved from costophrenic/pleural evidence to broader bilateral/lower-lung coverage.
- For Pneumothorax, debiasing did not solve the evidence problem. The current region representation/pooling is not enough.

## 10. Stage D: Full 8-Class Oracle Recoverability

This is the strongest positive result.

All 8 findings are oracle-recoverable. Every CNR CI lower bound is positive.

| category | n | oracle Delta CNR | CNR CI low | CNR CI high | Delta IoU | Delta Dice | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Pneumothorax | 167 | 0.651186 | 0.592621 | 0.709721 | 0.038574 | 0.059579 | region_recoverable |
| Lung Opacity | 59 | 0.361650 | 0.258389 | 0.463200 | 0.035432 | 0.050260 | region_recoverable |
| Pleural Effusion | 70 | 0.218699 | 0.163023 | 0.275289 | 0.059493 | 0.071731 | region_recoverable |
| Pneumonia | 103 | 0.186335 | 0.137283 | 0.239226 | 0.041742 | 0.048499 | region_recoverable |
| Atelectasis | 41 | 0.171162 | 0.102707 | 0.243762 | 0.032494 | 0.034336 | region_recoverable |
| Consolidation | 76 | 0.166071 | 0.123612 | 0.211401 | 0.044985 | 0.053265 | region_recoverable |
| Edema | 33 | 0.157495 | 0.105230 | 0.211394 | 0.020802 | 0.027648 | region_recoverable |
| Cardiomegaly | 242 | 0.136140 | 0.112691 | 0.159578 | 0.041462 | 0.043993 | region_recoverable |

Additional CI details for overlap metrics:

| category | IoU CI low | IoU CI high | Dice CI low | Dice CI high |
|---|---:|---:|---:|---:|
| Atelectasis | 0.013210 | 0.049702 | 0.013305 | 0.054634 |
| Cardiomegaly | 0.035026 | 0.047592 | 0.037415 | 0.050353 |
| Consolidation | 0.034601 | 0.056002 | 0.040007 | 0.066354 |
| Edema | 0.003166 | 0.037275 | 0.008088 | 0.046477 |
| Lung Opacity | 0.023640 | 0.046709 | 0.034040 | 0.065591 |
| Pleural Effusion | 0.042777 | 0.075435 | 0.052673 | 0.090908 |
| Pneumonia | 0.030874 | 0.051916 | 0.037644 | 0.060492 |
| Pneumothorax | 0.031469 | 0.045803 | 0.050421 | 0.069596 |

Core interpretation:

- Correctly directed anatomical evidence can improve AFLoc localization for all 8 findings.
- Pneumothorax is the most important counterintuitive case: it has the highest oracle upper bound but later learned evidence performs worst.
- This proves Pneumothorax is not "not region-repairable"; the failure is learned evidence misdirection.

## 11. 8-Class Learned Score Coverage Problem

The 8-class Stage C learned score export showed the checkpoint was not actually full 8-class.

Report:

| item | value |
|---|---:|
| rows_total | 198,184 |
| rows_written | 123,530 |
| skipped_invalid | 536 |
| skipped_unsupported | 74,118 |

Unsupported findings:

| finding | skipped rows |
|---|---:|
| Cardiomegaly | 24,706 |
| Edema | 24,706 |
| Pneumonia | 24,706 |

Checkpoint finding vocab:

```text
Atelectasis
Consolidation
Lung Opacity
Pleural Effusion
Pneumothorax
```

Cache finding vocab:

```text
Atelectasis
Cardiomegaly
Consolidation
Edema
Lung Opacity
Pleural Effusion
Pneumonia
Pneumothorax
```

Interpretation:

- Current "8-class" predictor actually supports 5 findings.
- Cardiomegaly, Edema, and Pneumonia have missing learned evidence.
- These 3 findings must not be counted as learned failures.

## 12. Stage E: Latest 8-Class Learned-vs-Oracle Diagnosis

Top-level Stage E decision:

```json
{
  "verdict": "reject_learned_selective_repair",
  "reason": "candidate_effect_floor_or_ci_not_met",
  "candidate_effect_floor": 0.02,
  "macro_all_harm_floor": -0.005,
  "candidate_vs_baseline_cnr_delta": -0.034228,
  "candidate_vs_baseline_cnr_ci_low": -0.084917,
  "specificity_vs_shuffled_cnr_delta": 0.035003,
  "specificity_vs_shuffled_cnr_ci_low": -0.027206,
  "macro_all_cnr_delta": -0.034228,
  "macro_all_cnr_ci_low": -0.088330
}
```

Why overall reject is mathematically consistent:

```text
(0.108 + 0 - 0.118 + 0 - 0.052 - 0.055 + 0 - 0.157) / 8
= about -0.034
```

This matches `macro_all_cnr_delta = -0.034228`.

Per-class learned repair decision:

| category | n | learned vs baseline Delta CNR | CI low | learned vs shuffled Delta CNR | CI low | verdict/reason |
|---|---:|---:|---:|---:|---:|---|
| Atelectasis | 41 | +0.108265 | +0.042375 | +0.237145 | +0.122711 | strong_pass |
| Cardiomegaly | 242 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | missing learned evidence |
| Consolidation | 76 | -0.118050 | -0.180622 | +0.031344 | -0.050767 | reject, misdirected |
| Edema | 33 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | missing learned evidence |
| Lung Opacity | 59 | -0.052064 | -0.114779 | +0.065154 | -0.025368 | reject, misdirected |
| Pleural Effusion | 70 | -0.055179 | -0.111428 | +0.086548 | +0.006219 | reject vs baseline, partial specificity |
| Pneumonia | 103 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | missing learned evidence |
| Pneumothorax | 167 | -0.156799 | -0.196056 | -0.140169 | -0.213931 | reject, severe misdirection |

Key interpretation:

- Atelectasis is the only current learned strong pass.
- Pleural Effusion has specificity versus shuffled, but still loses to AFLoc baseline.
- Pneumothorax is the strongest negative learned result despite the strongest oracle upper bound.
- Cardiomegaly, Edema, and Pneumonia are missing learned evidence, not learned failures.

## 13. Stage E Per-Class Gap Summary

| category | oracle Delta CNR | learned Delta CNR | oracle-learned gap | learned evidence rate | top1 hit | top3 hit | Spearman | diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Atelectasis | 0.171162 | +0.108265 | 0.062897 | 1.0 | 0.024390 | 0.365854 | 0.396786 | learned_gap_small |
| Cardiomegaly | 0.136140 | 0.000000 | 0.136140 | 0.0 | 0.000000 | 0.000000 | NA | missing_learned_evidence |
| Consolidation | 0.166071 | -0.118050 | 0.284121 | 1.0 | 0.026316 | 0.302632 | 0.159901 | evidence_misdirected |
| Edema | 0.157495 | 0.000000 | 0.157495 | 0.0 | 0.000000 | 0.000000 | NA | missing_learned_evidence |
| Lung Opacity | 0.361650 | -0.052064 | 0.413714 | 1.0 | 0.084746 | 0.355932 | 0.202963 | evidence_misdirected |
| Pleural Effusion | 0.218699 | -0.055179 | 0.273878 | 1.0 | 0.142857 | 0.200000 | 0.188390 | evidence_misdirected |
| Pneumonia | 0.186335 | 0.000000 | 0.186335 | 0.0 | 0.000000 | 0.000000 | NA | missing_learned_evidence |
| Pneumothorax | 0.651186 | -0.156799 | 0.807985 | 1.0 | 0.029940 | 0.203593 | -0.143654 | evidence_misdirected |

Score-scale evidence:

| category | mean learned top score | mean oracle-top learned score | interpretation |
|---|---:|---:|---|
| Atelectasis | 0.995050 | 0.987887 | learned score at oracle region is almost as high as learned top; success case |
| Consolidation | 0.960017 | 0.684985 | learned prefers wrong region |
| Lung Opacity | 0.994120 | 0.926044 | partial alignment but still wrong enough to hurt |
| Pleural Effusion | 0.953501 | 0.814298 | some specificity, still not enough |
| Pneumothorax | 0.592506 | 0.178953 | confidently wrong; oracle region scored very low |

Most important finding:

```text
Pneumothorax:
oracle Delta CNR = +0.651186
learned Delta CNR = -0.156799
gap = 0.807985
Spearman = -0.143654
mean learned top score = 0.592506
mean oracle-top learned score = 0.178953
```

This proves Pneumothorax is not a failure of the repair formula. It is a failure of learned evidence direction.

## 14. Current Scientific Story

The paper should not claim:

```text
We built a learned repair method that improves all diseases.
```

The paper can claim:

```text
1. Oracle anatomical repair shows broad localization recoverability across 8 findings.
2. Learned weak-label region evidence only selectively recovers the oracle signal.
3. The dominant failure mode is confident region misdirection, not a broken repair formula.
4. Pneumothorax is the counterintuitive key case: highest oracle upper bound, worst learned evidence alignment.
5. Atelectasis is the positive learned control: learned evidence can work when the region ranking aligns.
```

Short paper claim:

```text
AnaPrior shows that anatomical region repair is broadly recoverable in principle,
but weak-label learned evidence is pathology-dependent and often directionally wrong.
```

Stronger English phrasing:

```text
The localization formulation is not the bottleneck; learning correctly directed disease-specific region evidence from weak labels is.
```

## 15. Current Failure Points

| failure point | evidence | consequence |
|---|---|---|
| Learned evidence direction is wrong | Stage E `evidence_misdirected` for Consolidation, Lung Opacity, Pleural Effusion, Pneumothorax | Repair pushes heatmaps away from oracle regions. |
| Pneumothorax evidence is confidently wrong | Spearman -0.143654, learned top score 0.592506 vs oracle-top learned score 0.178953 | Most important negative result. |
| 3 classes missing learned evidence | Cardiomegaly, Edema, Pneumonia skipped as unsupported | Cannot count these as learned failures. |
| Central-structure bias | MS-CXR agnostic top regions are cardiac/hilar | Explains raw Stage C harm. |
| Coarse region + mean pooling | Effusion improves after debias but still below baseline | Region evidence may be too broad, especially for CNR. |
| Unified region strategy is invalid | Disease evidence differs by finding | Need disease-conditioned evidence modules in future. |

## 16. Disease-Conditioned Evidence Direction

Long-term correct direction:

```text
shared framework + disease-conditioned evidence module
```

Not:

```text
one region prior / one alpha / one pooling rule for all diseases
```

Disease-specific needs:

| disease | likely needed evidence | why unified strategy fails |
|---|---|---|
| Pneumothorax | pleural line, apical/peripheral lung, external lung boundary | bilateral lung mean pooling is too coarse and diluted |
| Pleural Effusion | costophrenic angle, lower lung, pleural space, broad lower coverage | deleting bilateral/lower coverage can hurt overlap |
| Cardiomegaly | cardiac silhouette/heart boundary/area | lung region evidence is mostly noise |
| Edema | bilateral diffuse texture, hilar/perihilar vascular pattern | not a focal lesion; localization is naturally diffuse |
| Consolidation/Pneumonia/Lung Opacity | local opacity/appearance plus lung-zone constraint | anatomy prior alone is insufficient |
| Atelectasis | line/segmental collapse, volume loss, lung zone | needs morphology + position |

Practical future module ingredients:

| module | replaces current weakness |
|---|---|
| CheXmask/lung/heart masks | unstable/coarse Chest ImaGenome region schema |
| RadGraph located_at | weak alias and noisy report-derived region relation |
| mean + max + top-k pooling | large-region mean pooling dilution |
| reliability gate | prevents wrong evidence from harming AFLoc |
| disease-conditioned region whitelist/weights | avoids one-size-fits-all priors |

## 17. Recommended Next Experiments

Priority order:

1. **Fix learned coverage with a true 8-class predictor.**
   - Current checkpoint supports only 5/8 findings.
   - Cardiomegaly, Edema, Pneumonia need actual learned scores before any learned claim.

2. **Run frozen finding-agnostic debias on the updated predictor.**
   - Do not tune on MS-CXR.
   - Use held-out or predefined agnostic region marginal statistics.
   - Report whether gap shrinks.

3. **Report raw learned vs debiased learned vs oracle in the same table.**
   - Required columns:
     ```text
     category | oracle Delta CNR | raw learned Delta CNR | debiased learned Delta CNR | top1/top3 | diagnosis
     ```

4. **Only after the diagnostic paper is stable, consider targeted v2 modules.**
   - Pneumothorax v2: apical/peripheral/pleural band + max/top-k pooling + reliability gate.
   - Pleural Effusion v2: costophrenic/lower/pleural space + keep broad lower coverage, downweight cardiac/hilar.
   - Cardiomegaly v2: cardiac silhouette-only evidence.

Do not immediately build a full 8-class router. That is a larger follow-up project.

## 18. What Counts as Success Now

Success for the current paper is not "all learned classes become positive."

Current success criterion:

```text
Build a clean, defensible diagnostic story:
oracle all positive -> learned selectively succeeds/fails -> failure due to region misdirection -> simple debias/coverage test explains how much is fixable.
```

Main-track strengthening experiment:

```text
true 8-class predictor + frozen debias + Stage C/E rerun
```

Workshop/short version:

```text
8-class oracle recoverability + current Stage E learned gap diagnosis + honest failure analysis
```
