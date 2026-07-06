# AnaPrior Plan Summary

Updated: 2026-07-06

This is the concise plan summary for the current AnaPrior-Loc-AFLoc research direction. The detailed data ledger is in `docs/ANAPRIOR_CURRENT_CONTEXT.md`.

## Current Conclusion

AnaPrior should not be framed as a universal learned repair method that improves every chest X-ray finding. The stronger result is a diagnostic research story:

```text
Oracle anatomical repair works broadly across 8 findings.
Learned weak-label region evidence is the bottleneck.
The dominant learned failure mode is confident region misdirection.
```

In short:

```text
The repair formula is right; the learned evidence is often pointed at the wrong region.
```

## Key Evidence

The 8-class oracle profile is the main positive result. All 8 findings are oracle-recoverable with positive CNR confidence lower bounds.

| finding | oracle Delta CNR | CI low | interpretation |
|---|---:|---:|---|
| Pneumothorax | 0.651 | 0.593 | strongest oracle upper bound |
| Lung Opacity | 0.362 | 0.258 | strong recoverability |
| Pleural Effusion | 0.219 | 0.163 | recoverable with strong overlap gains |
| Pneumonia | 0.186 | 0.137 | recoverable |
| Atelectasis | 0.171 | 0.103 | recoverable and current learned success case |
| Consolidation | 0.166 | 0.124 | recoverable |
| Edema | 0.157 | 0.105 | recoverable despite diffuse pattern |
| Cardiomegaly | 0.136 | 0.113 | stable recoverability |

The latest learned Stage E result is still rejected overall:

```text
candidate_vs_baseline_cnr_delta = -0.034228
candidate_vs_baseline_cnr_ci_low = -0.084917
macro_all_cnr_delta = -0.034228
macro_all_cnr_ci_low = -0.088330
```

But the per-class result is the useful story:

| finding | learned Delta CNR | Stage E diagnosis |
|---|---:|---|
| Atelectasis | +0.108 | learned strong pass |
| Cardiomegaly | 0.000 | missing learned evidence |
| Consolidation | -0.118 | evidence misdirected |
| Edema | 0.000 | missing learned evidence |
| Lung Opacity | -0.052 | evidence misdirected |
| Pleural Effusion | -0.055 | partial specificity, still below baseline |
| Pneumonia | 0.000 | missing learned evidence |
| Pneumothorax | -0.157 | severe evidence misdirection |

Pneumothorax is the key counterexample:

```text
oracle Delta CNR = +0.651
learned Delta CNR = -0.157
learned_oracle_spearman = -0.144
mean learned top score = 0.593
mean oracle-top learned score = 0.179
```

This means Pneumothorax is not unrecoverable. It is highly recoverable in oracle mode, but the current learned predictor confidently ranks the wrong regions.

## Current Failure Points

1. **Learned evidence direction is unreliable.**
   - Consolidation, Lung Opacity, Pleural Effusion, and Pneumothorax have learned evidence, but the learned top region is far from the oracle top region.

2. **Three classes lack learned evidence.**
   - Cardiomegaly, Edema, and Pneumonia are unsupported by the current checkpoint, so they must not be counted as learned failures.

3. **Central-structure bias is strong.**
   - MS-CXR agnostic region scores over-rank cardiac/hilar structures.
   - This pushes repair toward visually salient but lesion-irrelevant regions.

4. **One region strategy cannot fit all findings.**
   - Pneumothorax needs pleural/apical/peripheral evidence.
   - Effusion needs costophrenic/lower/pleural evidence.
   - Cardiomegaly needs cardiac silhouette evidence.
   - Edema and opacity-like findings need texture/appearance evidence.

## Recommended Plan

### Immediate Paper-Safe Plan

1. **Keep the oracle 8-class profile as the main positive result.**
   - This is the cleanest evidence that the formulation works.

2. **Use Stage E as the main failure analysis.**
   - Explain learned failure as direction misalignment, not formula failure.

3. **Clearly separate missing evidence from learned failure.**
   - Cardiomegaly, Edema, and Pneumonia are not learned failures yet.

4. **Keep Atelectasis as the positive learned control.**
   - It proves learned evidence can work when region ranking aligns with oracle.

### Next Experiment

Run a true 8-class predictor and then rerun:

```text
Stage C raw learned repair
Stage E learned-vs-oracle gap diagnosis
frozen finding-agnostic debias
```

The table to produce:

```text
finding | oracle Delta CNR | raw learned Delta CNR | debiased learned Delta CNR | top1/top3 | diagnosis
```

This answers whether the learned gap is due to incomplete coverage, central bias, or deeper disease-specific evidence mismatch.

### Longer-Term Method Direction

Do not search for one universal prior. The correct direction is:

```text
shared repair framework + disease-conditioned evidence modules + reliability gate
```

Candidate disease-conditioned modules:

| finding group | evidence module |
|---|---|
| Pneumothorax | apical/peripheral/pleural band evidence with max/top-k pooling |
| Pleural Effusion | costophrenic/lower/pleural evidence with broad lower coverage |
| Cardiomegaly | cardiac silhouette/heart-boundary evidence |
| Edema | bilateral diffuse texture and perihilar evidence |
| Opacity/Pneumonia/Consolidation | local appearance plus lung-zone constraint |
| Atelectasis | morphology plus lung-zone evidence |

Add a reliability gate:

```text
if evidence is unreliable: keep AFLoc baseline
if evidence is reliable and anatomically plausible: apply repair
```

## Paper Framing

Do not claim:

```text
We propose a learned repair method that improves all findings.
```

Claim:

```text
AnaPrior reveals a broad oracle recoverability upper bound for anatomical region repair,
and identifies weak-label region evidence misdirection as the main obstacle to deployable learned repair.
```

One-sentence version:

```text
The localization repair mechanism is broadly valid, but learning correctly directed region evidence from weak labels is pathology-dependent and currently unreliable.
```

