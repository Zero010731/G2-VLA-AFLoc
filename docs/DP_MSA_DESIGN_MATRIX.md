# DP-MSA Design Matrix

目的：在正式设计 DCEM-v4 前，把可借鉴论文的模块、监督信号、可借鉴点和不能借鉴点梳理清楚，避免“照搬别人模块”，也避免没有根据地发明新结构。

当前结论：DCEM-v4 不应继续主攻 cardiac/hilar suppression。Stage F 显示 Pneumonia、Consolidation、Lung Opacity 的主要问题是 phrase subtype 混杂和 local/diffuse 尺度不匹配。因此 DCEM-v4 建议采用 **DP-MSA: Disease-Phrase Multi-Scale Spatial Adapter**。

## Local Papers

PDF 已下载到 `docs/references/dp_msa_papers/`：

| Paper | Local PDF | Source |
|---|---|---|
| CLIP-Adapter | `docs/references/dp_msa_papers/clip_adapter_2110.04544.pdf` | https://arxiv.org/abs/2110.04544 |
| DenseCLIP | `docs/references/dp_msa_papers/denseclip_2112.01518.pdf` | https://arxiv.org/abs/2112.01518 |
| CLIPSeg | `docs/references/dp_msa_papers/clipseg_2112.10003.pdf` | https://arxiv.org/abs/2112.10003 |
| AGXNet | `docs/references/dp_msa_papers/agxnet_2206.12704.pdf` | https://arxiv.org/abs/2206.12704 |
| AGPT | `docs/references/dp_msa_papers/agpt_2502.16585.pdf` | https://arxiv.org/abs/2502.16585 |
| MedRPG | `docs/references/dp_msa_papers/medrpg_2303.07618.pdf` | https://arxiv.org/abs/2303.07618 |

## Design Matrix

| Paper | Core module | Supervision used in original paper | What we can borrow | What we should not borrow directly | DP-MSA translation |
|---|---|---|---|---|---|
| CLIP-Adapter | Lightweight bottleneck adapter with residual feature blending on top of frozen CLIP features | Downstream task labels, mainly classification/few-shot settings | Residual adapter principle: keep frozen backbone output and learn a small correction | Classification-only objective; direct feature adapter without spatial output | Add a lightweight residual adapter after frozen AFLoc visual features; output a repair map rather than a class feature |
| DenseCLIP | Converts image-text matching into pixel-text score maps for dense prediction | Dense prediction supervision such as segmentation/detection labels in natural-image datasets | Dense text-image matching idea; pixel/patch-level similarity can guide spatial localization | Natural-image segmentation/detection training recipe; large dense-supervision assumptions | Convert disease/phrase embeddings into patch-region compatibility maps over AFLoc spatial features |
| CLIPSeg | Text/image prompt-conditioned decoder for binary segmentation | PhraseCut-style segmentation masks and prompt supervision | Text-conditioned mask decoder architecture; prompt-conditioned dense output | Full segmentation supervision and natural-image prompt distribution | Use a tiny text-conditioned residual heatmap decoder, not a full mask segmentation model |
| AGXNet | Anatomy network followed by observation network with anatomy-guided attention | Report-mined anatomy/pathology labels; weak supervision; PU learning | Anatomy-guided attention and “anatomy first, disease second” decomposition | Its full two-network training pipeline and disease-classification objective | Use Chest ImaGenome region maps as anatomy priors to guide disease/phrase repair branches |
| AGPT | Anatomical grounding pre-training for medical phrase grounding, using Chest ImaGenome and evaluating on MS-CXR | Anatomical grounding pre-training; MS-CXR evaluation/fine-tuning settings | Chest ImaGenome anatomical grounding as in-domain pretraining signal | MS-CXR box fine-tuning or any use of MS-CXR boxes for model selection/training | Train adapter/ranking losses from Chest ImaGenome anatomy-region labels; reserve MS-CXR boxes for final evaluation only |
| MedRPG | Region-phrase context contrastive alignment and direct box prediction | Medical phrase grounding datasets with box/region supervision | Contrastive alignment between relevant phrase-region pairs; hard negative idea | Direct box coordinate regression and MS-CXR/box-supervised training | Replace box-supervised contrastive learning with Chest ImaGenome region-ranking and phrase subtype contrastive losses |

## Why DP-MSA Instead of Directly Reusing a Paper Module

Directly adopting DenseCLIP, CLIPSeg, or MedRPG would change the problem setting. Those methods are designed around dense masks, natural-image segmentation, or box-supervised phrase grounding. Our setting is different:

- Backbone: frozen AFLoc.
- Training signal: Chest ImaGenome region/finding labels and anatomy regions.
- Final evaluation: MS-CXR boxes/masks only for evaluation.
- Current failure mode: phrase subtype and local/diffuse scale mismatch.
- Goal: residual repair of existing heatmaps, not replacing AFLoc.

Therefore, DCEM-v4 should be a compact adapter designed for this setting rather than a copy of an external architecture.

## DP-MSA Proposed Components

| Component | Inspired by | Input | Output | Purpose |
|---|---|---|---|---|
| Disease embedding | DCEM-v2B/v3 | finding id | disease vector | Keep disease-specific behavior |
| Phrase subtype encoder | Stage F, MedRPG | phrase text/subtype tokens | subtype vector | Separate basilar, multifocal, diffuse, focal, airspace, pneumonia-like phrases |
| Anatomy prior encoder | AGXNet, AGPT | Chest ImaGenome region maps or region scores | anatomy prior map | Inject anatomical region constraints without MS-CXR box training |
| Multi-scale spatial branches | DenseCLIP, CLIPSeg | AFLoc spatial features + disease/phrase vectors | local/focal/diffuse repair maps | Handle local vs bilateral/diffuse mismatch |
| Residual adapter head | CLIP-Adapter | branch maps + AFLoc heatmap | residual repair map | Preserve baseline while adding correction |
| Region-ranking loss | MedRPG, DCEM-v2B | Chest ImaGenome region labels | ranking objective | Promote phrase-compatible regions over hard negatives |

## DP-MSA Data and Supervision Boundary

| Data source | Allowed role | Not allowed role |
|---|---|---|
| Frozen AFLoc features/heatmaps | Adapter input and residual baseline | Backbone fine-tuning unless explicitly separated |
| Chest ImaGenome region labels | Train disease-region ranking, anatomy prior, subtype branches | Claiming box-level MS-CXR supervision |
| Chest ImaGenome anatomy regions/maps | Anatomy-guided weak spatial supervision | Replacing MS-CXR evaluation |
| MS-CXR phrase text | Test-time query text and validation/test split phrase metadata | Training labels if derived from boxes |
| MS-CXR boxes/masks | Final evaluation only | Training, threshold tuning, branch selection, or loss construction |

## Stage F Diagnosis to DP-MSA Mapping

| Stage F finding | DP-MSA design decision |
|---|---|
| Pneumonia mean delta CNR remains negative | Add pneumonia-specific subtype encoder and contrastive disambiguation from opacity/consolidation phrases |
| Consolidation and Lung Opacity are positive but top1 hit is low | Add multi-scale adapter so top3-near-misses can improve shape and top1 alignment |
| Basilar opacity/consolidation performs better than multifocal/patchy | Separate basilar/local branch from diffuse/multifocal branch |
| Cardiac/hilar shortcut rates are low | Do not make cardiac/hilar suppression the core novelty of v4 |
| CNR improves more than IoU/Dice in v3 | Add dense residual spatial head to change heatmap shape, not just region contrast |

## Recommended DCEM-v4 Claim

Recommended wording:

> Motivated by the Stage F failure audit, we introduce DP-MSA, a disease-phrase multi-scale spatial adapter for frozen AFLoc localization repair. DP-MSA combines residual adaptation, dense phrase-conditioned repair, and anatomy-guided weak supervision from Chest ImaGenome, without using MS-CXR box annotations for training or model selection.

Avoid wording:

> We adopt DenseCLIP/CLIPSeg/MedRPG.

Better wording:

> Inspired by residual adapters, language-guided dense prediction, text-conditioned decoders, anatomy-guided attention, and region-phrase contrastive alignment, we adapt these ideas to a no-MS-CXR-box-supervision AFLoc repair setting.

## Immediate Next Step

Write the DCEM-v4 spec around DP-MSA with these concrete choices:

1. Inputs: AFLoc spatial feature map, baseline heatmap, disease id, phrase subtype, Chest ImaGenome anatomy priors.
2. Outputs: residual repair heatmap.
3. Fusion: `final_hmap = baseline_hmap + lambda * residual_repair_map`.
4. Losses: Chest ImaGenome disease-region BCE, phrase subtype region-ranking, hard-negative ranking, smoothness/compactness regularization.
5. Evaluation: Stage C-style MS-CXR final evaluation with v3 baselines preserved.

