# DP-MSA Module Blueprint From Code Survey

This note turns the paper/code survey into an implementable DCEM-v4 module design. It is not implementation code.

## Code Sources Checked

| Source | Repository | Code observation | DP-MSA decision |
|---|---|---|---|
| CLIP-Adapter | https://github.com/gaopengcuhk/CLIP-Adapter | Official code uses a small MLP adapter and residual feature blending: adapter output is mixed with frozen CLIP image features by a fixed ratio. | Use residual adapter principle, but convert feature residual into spatial heatmap residual. |
| DenseCLIP | https://github.com/raoyongming/DenseCLIP | Repository states it converts image-text matching into pixel-text matching and uses pixel-text score maps to guide dense prediction. It depends on mmseg/mmdet. | Borrow dense phrase-feature similarity maps; do not import mmseg/mmdet stack. |
| CLIPSeg | https://github.com/timojl/clipseg | Code exposes `CLIPDensePredT`, extracts intermediate CLIP activations, computes a text/image conditional vector, and decodes to a 1-channel dense prediction with transposed conv. | Borrow text-conditioned dense decoder idea; keep a much smaller decoder over AFLoc/CXR features. |
| AGXNet | https://github.com/batmanlab/AGXNet | README describes a cascade: anatomy network first, observation network second, bridged by anatomy-guided attention. | Use anatomy priors as attention/gating maps before disease repair. Do not add a full second CXR classifier. |
| AGPT | https://github.com/Claire1217/AGPT | Code loads TransVG/MDETR models pre-trained on Chest ImaGenome anatomy grounding and fine-tuned on MS-CXR. | Borrow Chest ImaGenome anatomical grounding idea; do not use MS-CXR fine-tuning. |
| MedRPG | https://github.com/eraserNut/MedRPG | Repository trains and tests with MS-CXR and uses region-phrase context contrastive alignment. | Borrow region-phrase contrastive/ranking loss structure; replace MS-CXR box training with Chest ImaGenome region ranking. |

## Core Design Choice

DP-MSA should not be a direct import of these repositories. It should be a native AnaPrior module:

```text
frozen AFLoc heatmap/features
    + disease id
    + phrase subtype
    + anatomy prior maps
    + optional region predictor scores
        ↓
DP-MSA
        ↓
residual repair map
        ↓
final heatmap = normalize(base heatmap + lambda * residual repair)
```

The implementation should start minimal. Avoid a full TransVG/MDETR/CLIPSeg stack.

## Proposed Runtime Components

### 1. `PhraseSubtypeEncoder`

Purpose: Convert phrase text into a fixed subtype id and embedding.

Input:

- `category: str`
- `phrase: str`

Output:

- `subtype_id: int`
- `subtype_embedding: Tensor[B, D]`

Initial subtype vocabulary:

```text
basilar
upper_apical
mid_lung
focal
multifocal_patchy
bilateral_diffuse
airspace
ground_glass
retrocardiac_hilar
pneumonia_like
consolidation_like
opacity_like
other
```

Borrowed idea:

- MedRPG phrase-region alignment motivates phrase-level conditioning.

AnaPrior-specific change:

- Subtype rules are fixed clinical text rules, not learned from MS-CXR box outcomes.

### 2. `DiseasePhraseConditioner`

Purpose: Fuse disease embedding and phrase subtype embedding.

Input:

- disease id
- subtype id

Output:

- FiLM parameters: `gamma`, `beta`
- branch logits for local/diffuse/focal branches

Recommended first shape:

```text
disease_emb: [B, 64]
subtype_emb: [B, 64]
condition = MLP([disease_emb, subtype_emb]) -> [B, 128]
gamma, beta = Linear(condition) -> [B, C], [B, C]
branch_logits = Linear(condition) -> [B, 3]
```

Borrowed idea:

- CLIP-Adapter residual conditioning.
- CLIPSeg conditional decoder.

AnaPrior-specific change:

- Condition on disease and phrase subtype, not a generic CLIP class prompt.

### 3. `AnatomyPriorProjector`

Purpose: Turn Chest ImaGenome anatomy priors into spatial adapter features.

Input options:

- region maps: `[B, R, H, W]`
- region scores: `[B, R]`
- region names / disease-specific masks

Output:

- anatomy prior feature map: `[B, C_a, H, W]`

Recommended first design:

```text
weighted_region_map = sum_r region_score_r * region_map_r
anatomy_features = Conv1x1([region_maps, weighted_region_map])
```

Borrowed idea:

- AGXNet anatomy-guided attention.
- AGPT anatomical grounding.

AnaPrior-specific change:

- Use Chest ImaGenome regions already available in the AnaPrior pipeline.
- Do not train on MS-CXR boxes.

### 4. `DensePhraseMatcher`

Purpose: Create a dense compatibility map between spatial features and disease/phrase condition.

Input:

- AFLoc spatial feature map or fallback heatmap features: `[B, C, H, W]`
- condition vector: `[B, D]`

Output:

- dense phrase compatibility: `[B, 1, H, W]` or `[B, C_m, H, W]`

Recommended first implementation:

```text
projected_features = Conv1x1(features)
projected_condition = Linear(condition).view(B, C_m, 1, 1)
compatibility = sum(projected_features * projected_condition, dim=channel)
```

Borrowed idea:

- DenseCLIP pixel-text score maps.

AnaPrior-specific change:

- Disease/phrase subtype condition replaces generic natural-image text prompt.

### 5. `MultiScaleRepairBranches`

Purpose: Handle Stage F local/diffuse scale mismatch.

Branches:

- local branch: 3x3 conv over anatomy-gated features.
- diffuse branch: larger receptive field, e.g. dilated 3x3 conv or pooled context.
- focal branch: local contrast/top-k-inspired branch.

Input:

- spatial features
- anatomy features
- dense compatibility map
- branch weights

Output:

- branch repair maps and fused repair map.

Borrowed idea:

- DenseCLIP dense prediction.
- CLIPSeg dense decoder.

AnaPrior-specific change:

- Branches correspond to medical phrase subtypes observed in Stage F.

### 6. `ResidualHeatmapDecoder`

Purpose: Produce final residual repair map.

Input:

- fused branch features
- baseline heatmap

Output:

- residual repair map: `[B, 1, H, W]`
- final heatmap: `[B, 1, H, W]`

Recommended formula:

```text
residual = tanh(Conv1x1(fused_features))
final = normalize(base_hmap + lambda * residual)
```

Borrowed idea:

- CLIP-Adapter residual blending.
- CLIPSeg 1-channel dense output.

AnaPrior-specific change:

- Residual acts on localization heatmap, not class logits or segmentation masks.

## Minimal DP-MSA Forward Pass

```python
class DPMultiScaleSpatialAdapter(nn.Module):
    def forward(
        self,
        base_hmap,          # [B,1,H,W]
        spatial_features,   # [B,C,H,W] or derived heatmap features
        region_maps,        # [B,R,H,W]
        region_scores,      # [B,R]
        disease_ids,        # [B]
        subtype_ids,        # [B]
    ):
        condition = conditioner(disease_ids, subtype_ids)
        anatomy = anatomy_projector(region_maps, region_scores)
        dense_match = dense_phrase_matcher(spatial_features, condition)
        branch_maps = multi_scale_branches(spatial_features, anatomy, dense_match)
        residual = residual_decoder(branch_maps, condition)
        final = normalize(base_hmap + lambda_ * residual)
        return final, {
            "residual": residual,
            "branch_weights": branch_weights,
            "dense_match": dense_match,
        }
```

## Training Losses

### Allowed training signal

- Chest ImaGenome region/finding labels.
- Chest ImaGenome anatomy region maps.
- phrase subtype labels from frozen clinical lexicon.
- AFLoc base heatmaps/features.

### Forbidden training signal

- MS-CXR boxes/masks.
- MS-CXR oracle profiles.
- test split metric deltas.

### Loss table

| Loss | Purpose | Source inspiration | Supervision source |
|---|---|---|---|
| region BCE | preserve region abnormality predictor | DCEM-v2B | Chest ImaGenome labels |
| phrase-region ranking | promote subtype-compatible regions | MedRPG | Chest ImaGenome region labels |
| dense compatibility loss | align dense map with anatomy prior | DenseCLIP/AGPT | Chest ImaGenome anatomy maps |
| branch entropy / branch prior | make local/diffuse branch meaningful | Stage F | frozen subtype lexicon |
| residual magnitude loss | avoid damaging AFLoc | CLIP-Adapter | no box labels |
| smoothness loss | stabilize heatmap shape | CLIPSeg-like decoder | no box labels |

## First Implementation Recommendation

Start with **DP-MSA-v0**, not the full design:

```text
Inputs:
  base_hmap
  region_maps
  region_scores
  disease_id
  phrase_subtype_id

No deep AFLoc feature map at first.

Module:
  disease/subtype embeddings
  anatomy prior projector
  three small conv branches
  residual heatmap decoder

Output:
  residual repair map
  final repaired heatmap
```

Reason:

- It avoids changing AFLoc internals.
- It can be trained/evaluated faster.
- It directly tests whether multi-scale anatomy-conditioned residual repair improves IoU/Dice beyond DCEM-v3.

After v0 works, add AFLoc spatial features as DP-MSA-v1.

## Evaluation and Ablation

Main comparison:

```text
baseline
phrase_anatomy_dcem
validation_gated_dcem_v3
dp_msa_v0
validation_gated_dcem_v4
candidate_shuffled
```

Required ablations:

```text
dp_msa_v0_no_phrase_subtype
dp_msa_v0_no_anatomy_prior
dp_msa_v0_single_scale
dp_msa_v0_no_residual_regularization
```

Primary success criterion:

- IoU/Dice improve beyond DCEM-v3, not only CNR.

Secondary success criterion:

- Pneumonia becomes less harmful.
- Consolidation and Lung Opacity remain positive or pass validation.

