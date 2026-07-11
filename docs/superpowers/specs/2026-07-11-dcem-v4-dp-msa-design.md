# DCEM-v4 DP-MSA Design

## Goal

DCEM-v4 upgrades DCEM-v3 from a deterministic phrase-anatomy router into a learnable spatial repair module named **DP-MSA: Disease-Phrase Multi-Scale Spatial Adapter**.

The goal is to improve frozen AFLoc localization shape, not only heatmap direction or region contrast. DCEM-v3 already shows consistent CNR gains, but IoU and Dice improve modestly because v3 mostly changes region-level evidence weighting. Stage F shows the remaining parenchymal failures are dominated by phrase subtype ambiguity and local-versus-diffuse scale mismatch, especially for Pneumonia, Consolidation, and Lung Opacity.

DP-MSA therefore learns a compact residual heatmap adapter conditioned on disease, phrase subtype, and anatomy priors while keeping AFLoc frozen and preserving the DCEM-v3 validation protocol.

## Non-Negotiable Scientific Constraints

- AFLoc image and text encoders remain frozen.
- MS-CXR boxes, masks, and box-derived oracle profiles are not used for training, threshold selection, module design, branch selection, checkpoint selection, or phrase subtype rule tuning.
- MS-CXR boxes and masks are used only for final held-out evaluation after the method is frozen.
- Chest ImaGenome region/finding labels and anatomy regions are allowed for weak supervision.
- Raw AFLoc baseline, raw learned repair, disease-pooled repair, phrase-anatomy DCEM-v3, shuffled controls, and validation-gated DCEM-v3 remain available as controls.
- DP-MSA must be evaluated as an additive module over frozen AFLoc, not as a replacement foundation model.
- Any pass/bypass gate for final reporting must be selected on validation and applied unchanged to test.

## Evidence From Previous Stages

### DCEM-v2B

DCEM-v2B fixed true 8-class region-predictor coverage using Chest ImaGenome aliases and hard-negative ranking. It made Cardiomegaly, Edema, and Pleural Effusion usable, but several parenchymal findings still had evidence direction or shape problems.

### DCEM-v3

DCEM-v3 introduced phrase-anatomy routing and improved all absolute metrics over the local AFLoc baseline:

- Baseline: IoU 0.314064, Dice 0.446748, CNR 1.524043.
- Phrase-anatomy DCEM: IoU 0.323566, Dice 0.457071, CNR 1.601144.
- Validation-gated DCEM-v3: IoU 0.320797, Dice 0.453674, CNR 1.598654.

Interpretation: v3 improves disease-conditioned contrast and direction, but the shape change is still limited.

### Stage F Failure Audit

Stage F focused on Pneumonia, Consolidation, and Lung Opacity.

Key findings:

- Cardiac/hilar shortcut is not the dominant failure mode for these classes.
- Pneumonia remains negative in mean CNR and has low top1 region hit rate.
- Consolidation and Lung Opacity are positive on average but still have low top1 hit rates.
- Basilar/lower-lung phrases perform better than multifocal, patchy, diffuse, or broad opacity phrases.
- Learned top regions are often plausible local lung zones, while oracle top regions are often broad bilateral-lung regions. This indicates a scale mismatch rather than a simple wrong-anatomy shortcut.

Design implication: DCEM-v4 should not mainly be a stronger cardiac/hilar blocker. It should add phrase subtype conditioning and multi-scale spatial repair.

## Related-Work Design Matrix

The DP-MSA matrix is stored in:

- `docs/DP_MSA_DESIGN_MATRIX.md`
- `docs/references/dp_msa_papers/`

The design draws inspiration from:

- CLIP-Adapter: residual adapters on frozen vision-language features.
- DenseCLIP: dense text-image matching for pixel/patch-level prediction.
- CLIPSeg: text-conditioned mask decoding.
- AGXNet: anatomy-guided attention.
- AGPT: Chest ImaGenome anatomical grounding pre-training.
- MedRPG: phrase-region contrastive alignment.

DP-MSA does not directly adopt these modules. It adapts their principles to a no-MS-CXR-box-training AFLoc repair setting.

## Design Alternatives Considered

### Option A: DCEM-v3.5 Rule Refinement

Continue refining phrase rules, disease pooling, and validation gates.

Pros:

- Low implementation risk.
- Easy to explain and debug.
- Preserves current pipeline.

Cons:

- Limited ability to improve IoU and Dice.
- Likely to keep improving contrast without reshaping heatmaps.
- More hand rules weaken the novelty story.

Decision: Use only as ablation or fallback, not as main v4.

### Option B: DP-MSA Lightweight Spatial Adapter

Add a small disease-phrase-conditioned residual adapter after frozen AFLoc features. Train it with Chest ImaGenome weak supervision and evaluate on MS-CXR only after freezing the method.

Pros:

- Directly targets spatial shape improvement.
- Keeps AFLoc frozen and preserves the annotation-free MS-CXR evaluation claim.
- Creates a stronger method contribution than rule routing alone.
- Aligns with Stage F diagnosis.

Cons:

- More implementation and training risk than v3.
- Requires careful weak-supervision design to avoid overclaiming localization supervision.
- Needs ablations to prove the adapter adds value beyond v3.

Decision: Recommended main DCEM-v4 path.

### Option C: Full Grounding Model Replacement

Replace the repair module with a full phrase grounding model inspired by MedRPG, MDETR, TransVG, CLIPSeg, or SAM-style promptable segmentation.

Pros:

- Could improve IoU and Dice substantially if box supervision is available.
- Easier to compare against supervised grounding methods.

Cons:

- Violates the current no-MS-CXR-box-training scientific story if trained on MS-CXR boxes.
- Hard to integrate cleanly with AFLoc.
- Risks making the contribution look like a reimplementation of another grounding model.

Decision: Not the main path. Can be cited as related work or used as an external baseline only if run under its original supervised setting.

## Proposed Method: DP-MSA

### High-Level Architecture

DP-MSA is a residual spatial adapter:

```text
Frozen AFLoc visual feature map
        +
baseline AFLoc heatmap
        +
disease embedding
        +
phrase subtype embedding
        +
Chest ImaGenome anatomy prior
        ↓
Disease-Phrase Multi-Scale Spatial Adapter
        ↓
residual repair map
        ↓
final_hmap = baseline_hmap + lambda * residual_repair_map
```

The adapter is deliberately small. It should not become a new foundation model.

### Inputs

DP-MSA consumes:

- AFLoc spatial visual features, before final heatmap projection if available.
- Baseline AFLoc heatmap for the image-phrase pair.
- Finding category id.
- Phrase text and frozen phrase subtype id.
- Chest ImaGenome anatomy region maps or region priors.
- Existing learned region predictor scores as optional auxiliary features.

### Outputs

DP-MSA outputs:

- A residual repair map at AFLoc heatmap resolution or an intermediate spatial resolution later upsampled to heatmap resolution.
- Optional branch weights for local, diffuse, and focal repair branches.
- Optional diagnostic scores for phrase subtype branch selection.

The final heatmap remains residual:

```text
final_hmap = normalize(baseline_hmap + lambda * residual_repair_map)
```

`lambda` is selected on validation before final test evaluation.

## DP-MSA Components

### 1. Disease Embedding

Use the 8-class finding vocabulary:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Lung Opacity
- Pleural Effusion
- Pneumonia
- Pneumothorax

The embedding conditions the adapter without changing AFLoc text encoding.

### 2. Phrase Subtype Encoder

The subtype encoder maps phrase text to compact, fixed categories.

Initial subtype families:

- basilar/lower-lung
- upper/apical
- mid-lung
- focal/small area
- multifocal/patchy
- bilateral/diffuse
- airspace
- ground-glass
- retrocardiac/hilar
- pneumonia-like
- consolidation-like
- opacity-like
- uncertain/other

The subtype lexicon must be frozen before test scoring. It can be designed from clinical vocabulary, Chest ImaGenome region names, and Stage F validation analysis, but not from MS-CXR test box outcomes.

### 3. Anatomy Prior Encoder

Use Chest ImaGenome region maps and region names to form anatomy priors. The encoder should support:

- left/right upper, mid, and lower lung zones.
- bilateral lungs.
- cardiac silhouette.
- hilar/mediastinal region.
- pleural/costophrenic region.

Unlike DCEM-v3, the prior is not just a scalar region weight. It becomes a spatial conditioning map for the residual adapter.

### 4. Multi-Scale Spatial Branches

DP-MSA has multiple repair branches:

- Local branch: for focal, basilar, upper, lower, or side-specific phrases.
- Diffuse branch: for bilateral, multifocal, diffuse, pulmonary edema-like, or patchy phrases.
- Focal top-k branch: for small focal findings where sparse response is expected.

Branch weights are produced from disease and phrase subtype embeddings. For safety, branch weights are diagnostic outputs and can be regularized, but final branch choice must not be selected from MS-CXR test performance.

### 5. Residual Adapter Head

The head combines AFLoc spatial features, branch outputs, and anatomy priors to produce a residual heatmap.

Recommended first implementation:

- 1x1 projection of AFLoc features.
- FiLM-style modulation from disease and phrase subtype embeddings.
- Depthwise or small 3x3 convolution for spatial smoothing.
- Branch-specific repair maps.
- Weighted branch fusion.
- 1-channel residual map output.

This is closer to CLIP-Adapter and CLIPSeg principles than to a full segmentation model.

## Training Supervision

### Allowed Supervision

Use Chest ImaGenome-derived weak supervision:

- region-finding labels.
- disease-region positives and negatives.
- anatomy region maps.
- phrase subtype labels derived from text vocabulary.
- hard negatives from confusing disease-region pairs.

### Forbidden Supervision

Do not use:

- MS-CXR boxes.
- MS-CXR masks.
- MS-CXR oracle recoverability profiles.
- test-set metric deltas.
- box-derived phrase subtype decisions.
- test-set selected branch thresholds.

### Losses

Recommended training losses:

1. Disease-region BCE loss.

   Preserve the existing region abnormality predictor objective.

2. Phrase subtype region-ranking loss.

   For subtype-compatible regions, enforce:

   ```text
   score(positive_region | disease, subtype) >
   score(hard_negative_region | disease, subtype) + margin
   ```

3. Multi-scale branch consistency loss.

   Encourage local phrases to use local anatomy priors and diffuse phrases to use broader bilateral priors.

4. Residual magnitude regularization.

   Keep the residual small unless weak supervision supports a correction.

5. Spatial smoothness or compactness regularization.

   Use cautiously. It should stabilize maps, not impose MS-CXR box-like shapes.

## Expected Effect by Disease

| Disease | Expected DP-MSA role |
|---|---|
| Cardiomegaly | Mostly preserve v2B/v3 success; cardiac anatomy prior branch should be stable |
| Edema | Diffuse branch should preserve broad bilateral behavior |
| Pleural Effusion | Pleural/lower anatomy prior should preserve current strong gains |
| Pneumothorax | Focal/upper/pleural branch should preserve v3 success |
| Consolidation | Subtype and multi-scale branches should improve top1 alignment and IoU/Dice |
| Lung Opacity | Basilar and multifocal subtype split should reduce unstable cases |
| Pneumonia | Pneumonia-specific disambiguation should separate pneumonia-like phrases from generic opacity/consolidation |
| Atelectasis | Keep conservative; include in evaluation but do not overfit v4 around it initially |

## Evaluation Plan

### Main Comparisons

Required methods:

- `baseline`
- `learned_selective`
- `disease_pooled_learned`
- `phrase_anatomy_dcem`
- `validation_gated_dcem_v3`
- `dp_msa`
- `validation_gated_dcem_v4`
- `candidate_shuffled`
- `candidate_uniform`

Required comparisons:

- `dp_msa_vs_baseline`
- `dp_msa_vs_phrase_anatomy_dcem`
- `dp_msa_vs_validation_gated_dcem_v3`
- `dp_msa_vs_candidate_shuffled`
- `validation_gated_dcem_v4_vs_baseline`
- `validation_gated_dcem_v4_vs_validation_gated_dcem_v3`
- `validation_gated_dcem_v4_vs_candidate_shuffled`

### Metrics

Report:

- CNR, IoU, Dice.
- Pooled and macro summaries.
- Per-class deltas.
- Validation gate decisions.
- Stage E learned-vs-oracle gap diagnosis.
- Stage F failure-audit comparison before and after DP-MSA.

### Minimum Success Bar

To justify DP-MSA as more than v3 complexity:

- Overall CNR remains positive versus baseline and v3.
- IoU or Dice improves more than v3, not only CNR.
- Pneumonia is less harmful than v3 phrase-anatomy DCEM.
- Consolidation and Lung Opacity remain non-harmful and ideally improve top1/top3 alignment.
- Cardiomegaly, Pleural Effusion, and Pneumothorax are not degraded.
- Candidate-shuffled control remains clearly worse than DP-MSA.

## Ablation Plan

Required ablations:

| Ablation | Purpose |
|---|---|
| AFLoc baseline | frozen reference |
| DCEM-v3 phrase-anatomy router | previous best non-learned repair |
| DP-MSA without phrase subtype encoder | test phrase subtype contribution |
| DP-MSA without anatomy prior | test Chest ImaGenome anatomy contribution |
| DP-MSA without multi-scale branches | test local/diffuse branch contribution |
| DP-MSA without residual regularization | test whether residual control prevents harm |
| DP-MSA with validation gate | final safe reporting protocol |
| Candidate shuffled | specificity control |

## Implementation Boundaries

Likely new modules:

- `anaprior/models/dp_msa_adapter.py`
- `anaprior/train/train_dp_msa_adapter.py`
- `anaprior/eval/eval_mscxr_dp_msa_repair.py`
- `anaprior/eval/phrase_subtype.py`

Likely modified modules:

- `anaprior/features/extract_region_features.py` or a new feature exporter if AFLoc spatial feature maps are not currently cached.
- `anaprior/eval/eval_mscxr_learned_repair.py`
- `anaprior/eval/score_mscxr_learned_repair_metrics.py`
- `anaprior/eval/report_dcem_v1_paper_results.py`
- `scripts/run_stage_c_learned_repair_server.sh` or a new Stage G runner for DP-MSA.

Likely new scripts:

- `scripts/run_stage_g_dp_msa_train_server.sh`
- `scripts/run_stage_h_dp_msa_eval_server.sh`

The exact stage names can be adjusted during implementation planning. The important boundary is that DP-MSA training and MS-CXR evaluation remain separated.

## Risks and Mitigations

Risk: DP-MSA silently becomes a supervised MS-CXR grounding model.
Mitigation: Keep all training inputs from AFLoc and Chest ImaGenome. Add manifest files listing training sources.

Risk: Adapter improves CNR but not IoU/Dice.
Mitigation: Include dense residual and multi-scale branches specifically targeting shape, then report IoU/Dice as primary evidence of shape improvement.

Risk: Phrase subtype rules overfit the observed MS-CXR failures.
Mitigation: Define subtype lexicon from clinical vocabulary and Chest ImaGenome region terminology, freeze before final test, and document it.

Risk: Adapter harms strong classes.
Mitigation: Use residual fusion, validation-gated v4 reporting, and per-class harm checks.

Risk: Implementation complexity grows too quickly.
Mitigation: First implement a minimal DP-MSA with disease embedding, phrase subtype embedding, anatomy prior maps, and one residual head. Add multi-branch variants only after the minimal version runs.

Risk: AFLoc spatial feature maps are unavailable or expensive to cache.
Mitigation: Start with baseline heatmap plus anatomy priors and region-score maps as adapter inputs. Add deeper AFLoc feature maps only if the feature path is reliable.

## Paper Framing

Recommended claim:

> Motivated by a failure audit of DCEM-v3, we introduce DP-MSA, a disease-phrase multi-scale spatial adapter that repairs frozen AFLoc heatmaps using Chest ImaGenome weak anatomical supervision without MS-CXR box training.

Do not claim:

- DP-MSA is a new foundation model.
- DP-MSA is fully supervised on MS-CXR boxes.
- DP-MSA universally improves all findings unless the final results prove it.
- DP-MSA replaces AFLoc.

## Decision

Proceed with DCEM-v4 as DP-MSA: a lightweight disease-phrase-conditioned residual spatial adapter over frozen AFLoc. The first implementation should be minimal and auditable, then expanded only if validation and ablation results justify the added complexity.

