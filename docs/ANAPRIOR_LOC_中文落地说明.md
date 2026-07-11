# AnaPrior-Loc 中文落地说明

交付审计见：

```text
docs/ANAPRIOR_STAGE_C_DELIVERY_AUDIT.md
```

当前上下文见：

```text
docs/ANAPRIOR_CURRENT_CONTEXT.md
```

本文档回答四个问题：

1. 为什么不继续在 `G2-VLA-AFLoc` 里改？
2. 当前新项目到底加了哪些模块？
3. 服务器上应该怎么跑？
4. 跑完后怎么看结果，下一步怎么决策？

---

## 1. 为什么使用干净 AFLoc 副本

当前推荐工作目录是：

```text
C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc
```

它来自几天前较干净的 AFLoc 副本，不再继续污染：

```text
C:\Users\joker\Desktop\G2-VLA-AFLoc
```

原因很简单：

- `G2-VLA-AFLoc` 已经承担 Stage A 的 oracle 实验、固定先验实验、负结果诊断、选择性修复实验。
- 里面脚本很多，职责混杂，不适合作为 learned method 的主工程。
- 新项目只放 Stage B/C 的 learned pipeline，便于之后写论文、复现、打包。

现在两个目录的角色是：

| 目录 | 角色 |
|---|---|
| `G2-VLA-AFLoc` | Stage A 证据库：oracle / 消融 / 旧实验结果 |
| `AnaPrior-Loc-AFLoc` | Stage B/C 主工程：无泄漏训练、learned repair、正式评估 |

---

## 2. 当前主线是什么

方法名暂定：

```text
Selective Pathology-Conditioned Region Repair
```

核心不是“泛泛给所有病加解剖先验”，而是：

```text
只对 oracle 阶段证明有修复潜力的病种：
Pneumothorax
Pleural Effusion

使用 learned region abnormality evidence 修复 AFLoc heatmap。
```

也就是说，论文叙事已经从早期的：

```text
通用 anatomy prior fusion
```

转成更清楚、更稳的：

```text
evidence-driven selective repair
```

---

## 3. 已完成的模块

### 3.1 数据防泄漏

文件：

```text
anaprior/data/build_patient_splits.py
```

作用：

```text
从 Chest ImaGenome train/valid/test 中剔除所有 MS-CXR patient。
```

目的：

```text
Region Abnormality Predictor 不能在包含 MS-CXR patient 的图像上训练。
```

这是 learned module 能不能站得住的底线。

---

### 3.2 Region-Finding 监督表

文件：

```text
anaprior/data/build_region_finding_table.py
```

作用：

```text
从 Chest ImaGenome scene graph 中提取：
image / region / finding / label
```

当前只用显式标签：

```text
anatomicalfinding|yes|finding
anatomicalfinding|no|finding
```

不把“没提到”直接当负样本，避免噪声太重。

---

### 3.3 AFLoc Region Feature Cache

文件：

```text
anaprior/features/extract_region_features.py
```

作用：

```text
AFLoc local feature map
-> Chest ImaGenome bbox region pooling
-> region_features.pt
```

输出给 predictor 训练使用。

---

### 3.4 Region Abnormality Predictor

文件：

```text
anaprior/models/region_abnormality_predictor.py
anaprior/train/train_region_predictor.py
anaprior/eval/eval_predictor_heldout.py
```

作用：

```text
输入：某张图某个 region 的 AFLoc local feature + finding id
输出：这个 region 是否有该 finding 的异常证据
```

这是 oracle region abnormality 的 learned 替代品。

Stage B 的硬门槛是：

```text
先在 Chest ImaGenome held-out 上证明 predictor 自己能工作，
再进入 MS-CXR learned repair。
```

---

### 3.5 Stage C Learned Repair

文件：

```text
anaprior/eval/prepare_mscxr_repair_inputs.py
anaprior/eval/predict_region_scores.py
anaprior/eval/selective_repair.py
anaprior/eval/eval_mscxr_learned_repair.py
anaprior/eval/score_mscxr_learned_repair_metrics.py
anaprior/eval/stage_c_preflight.py
scripts/run_stage_c_learned_repair_server.sh
```

功能链：

```text
准备 MS-CXR repair input
-> 生成 MS-CXR score request
-> 抽 MS-CXR region features
-> predictor 输出 region scores
-> 生成五组 heatmap
-> 计算 IoU / CNR / Dice / bootstrap
```

五组 heatmap 是：

| 方法 | 作用 |
|---|---|
| `baseline` | 原 AFLoc heatmap |
| `learned_selective` | 只修复 Pneumothorax / Pleural Effusion |
| `all_class_learned` | 所有类别都修复，证明 selective 是否必要 |
| `candidate_shuffled` | 候选类但打乱区域，证明不是随便 mask 都有效 |
| `candidate_uniform` | 候选类但使用均匀区域权重，证明 region-specific evidence 是否必要 |

---

## 4. 服务器怎么跑

先进入新项目根目录：

```bash
cd /home/zhangran/zr/AnaPrior-Loc-AFLoc
```

如果你把项目放在别的位置，就进入对应目录。

### 4.1 先跑 smoke

```bash
ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh
```

smoke 默认：

```text
MAX_CASES=20
MAX_ROWS=200
```

作用：

```text
快速检查路径、checkpoint、数据、feature extraction、score export、heatmap build、metric scoring 是否都能通。
```

### 4.2 smoke 通过后跑 full

```bash
ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_c_learned_repair_server.sh
```

### 4.3 常用环境变量

如果路径不一样，在命令前覆盖：

```bash
CKPT=/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt \
MIMIC_IMAGE_ROOT=/mnt/mimic-cxr/jpg \
CHEST_IMAGENOME_ROOT=/mnt/chest-imagenome_1.0.0 \
PRIOR_TABLE=/home/zhangran/zr/G2-VLA-AFLoc/anaprior_assets/prior_table.json \
BASE_HMAPS_NPY=/mnt/zhangran/afloc_outputs/anaprior_fixed_gate1_reliable_adaptive/baseline/hmaps.npy \
PREDICTOR_CKPT=outputs/anaprior_stage_b_predictor_img_emb_l/region_predictor.pt \
AFLOC_BERT_TYPE=/mnt/zhangran/hf_models/Bio_ClinicalBERT \
AFLOC_HF_LOCAL_FILES_ONLY=1 \
OUTROOT=outputs/anaprior_stage_c \
STAGE_C_ALPHA=0.3 \
ANAPRIOR_RUN_SMOKE=1 \
bash scripts/run_stage_c_learned_repair_server.sh
```

---

## 5. 跑前检查

脚本会先运行：

```text
anaprior.eval.stage_c_preflight
```

它检查：

```text
AFLoc ckpt
MIMIC-CXR jpg root
Chest ImaGenome root
scene_graph.zip
prior_table.json
baseline hmaps.npy
region_predictor.pt
OUTROOT 是否可写
```

输出：

```text
outputs/anaprior_stage_c/preflight_report.json
```

如果这里报错，先不要继续跑，把这个 JSON 发回来分析。

---

## 6. 跑完看什么

重点看：

```text
outputs/anaprior_stage_c/learned_repair_metrics/learned_repair_decision.json
outputs/anaprior_stage_c/learned_repair_metrics/bootstrap_summary.csv
outputs/anaprior_stage_c/learned_repair_metrics/per_class_bootstrap_summary.csv
outputs/anaprior_stage_c/learned_repair_metrics/delta_table.csv
outputs/anaprior_stage_c/learned_repair_metrics/stage_c_result_report.md
```

---

## 10. 如何把本机新增代码搬到服务器

在 Windows 的干净工程目录运行：

```powershell
python -m anaprior.tools.create_stage_c_source_bundle `
  --output outputs\anaprior_stage_c_source_bundle.zip
```

这个包只包含下一阶段需要的源码、测试、文档和服务器脚本：

```text
anaprior/
tests/
docs/
README_NEXT_STAGE.md
scripts/run_stage_c_learned_repair_server.sh
```

它不会打包：

```text
outputs/
.git/
.pytest_cache/
__pycache__/
*.pyc
```

这样做的目的很简单：服务器只接收可复现实验代码，不把本机缓存、旧实验输出、大文件和 Git 状态混进去。

上传示例：

```powershell
scp outputs\anaprior_stage_c_source_bundle.zip zhangran@SERVER:/home/zhangran/zr/
```

服务器上解压到 AFLoc 工程根目录：

```bash
cd /home/zhangran/zr/G2-VLA-AFLoc
unzip -o /home/zhangran/zr/anaprior_stage_c_source_bundle.zip
```

先跑 smoke：

```bash
ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh
```

smoke 没问题后再跑 full：

```bash
ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_c_learned_repair_server.sh
```

其中最重要的是：

```text
learned_repair_decision.json
```

决策逻辑已经预注册：

```text
保留 learned selective repair 必须同时满足：

1. learned_selective 在 candidate 类上优于 baseline；
2. learned_selective 在 candidate 类上优于 candidate_shuffled；
3. macro_all 没有明显伤害。
```

如果只满足 1，不满足 2，说明：

```text
可能只是泛泛空间扰动，不是有效 region semantics。
```

如果 1 和 2 都满足，但 macro_all 伤害明显，说明：

```text
方法可能只适合候选类，不能宣传成全类别通用模块。
```

---

## 7. 当前实验的科学意义

这一步不是简单“调代码”。

它要回答一个严肃问题：

```text
oracle selective repair 的强结果，能否被一个不读 oracle label 的 learned predictor 复现？
```

如果能：

```text
这条线有机会成为论文主贡献。
```

如果不能：

```text
说明 Chest ImaGenome oracle 里有强信息，但当前 learned predictor 没学到；
下一步要改 predictor 或换 image-conditioned evidence source。
```

无论结果正负，这一步都很关键，因为它把“看起来有潜力”变成了可证伪实验。

---

## 8. 常见问题

### 8.1 preflight 说 predictor_ckpt 缺失

说明 Stage B predictor 还没训练或路径不对。

先确认：

```text
outputs/anaprior_stage_b_predictor_img_emb_l/region_predictor.pt
```

是否存在。

### 8.2 base_hmaps_npy 缺失

说明 Stage A baseline heatmap 路径不对。

当前默认：

```text
/mnt/zhangran/afloc_outputs/anaprior_fixed_gate1_reliable_adaptive/baseline/hmaps.npy
```

如果你的 baseline hmap 在其他地方，用：

```bash
BASE_HMAPS_NPY=/your/path/hmaps.npy
```

覆盖。

### 8.3 临时目录报错

设置：

```bash
export AFLOC_TMPDIR=/mnt/zhangran/tmp
mkdir -p /mnt/zhangran/tmp
```

脚本已经默认做了这件事。

### 8.4 smoke 能跑，full 很慢

这是正常的。

full 会对 MS-CXR score request 抽 AFLoc region feature，再跑五组 hmap metric。

建议先确认 smoke 输出合理，再 full。

---

## 9. 当前最短行动

在服务器新项目根目录运行：

```bash
ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh
```

如果成功，再运行：

```bash
ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_c_learned_repair_server.sh
```

跑完把下面两个文件发回来：

```text
outputs/anaprior_stage_c/learned_repair_metrics/learned_repair_decision.json
outputs/anaprior_stage_c/learned_repair_metrics/bootstrap_summary.csv
outputs/anaprior_stage_c/learned_repair_metrics/stage_c_result_report.md
```
