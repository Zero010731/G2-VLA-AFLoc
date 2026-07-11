# Stage C 交付审计

本文档用于回答一个工程问题：

```text
下一阶段 learned AnaPrior-Loc 代码是否已经从旧实验目录中拆出，
并能以干净源码包迁移到服务器继续跑 smoke/full？
```

## 本地已完成

当前主工程目录：

```text
C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc
```

它来自几天前较干净的 AFLoc 副本，不再继续污染：

```text
C:\Users\joker\Desktop\G2-VLA-AFLoc
```

两个目录的分工：

| 目录 | 角色 |
|---|---|
| `G2-VLA-AFLoc` | Stage A 证据目录：oracle selective repair、fixed prior、旧实验输出 |
| `AnaPrior-Loc-AFLoc` | Stage B/C 主工程：无泄漏训练、learned predictor、learned selective repair |

已经新增的下一阶段模块：

| 模块 | 文件 |
|---|---|
| patient-level 防泄漏 | `anaprior/data/build_patient_splits.py` |
| Chest ImaGenome region-finding 表 | `anaprior/data/build_region_finding_table.py` |
| AFLoc region feature cache | `anaprior/features/extract_region_features.py` |
| region pooling | `anaprior/features/region_pooling.py` |
| Region Abnormality Predictor | `anaprior/models/region_abnormality_predictor.py` |
| predictor 训练 | `anaprior/train/train_region_predictor.py` |
| held-out predictor gate | `anaprior/eval/eval_predictor_heldout.py` |
| MS-CXR repair input 准备 | `anaprior/eval/prepare_mscxr_repair_inputs.py` |
| learned score 导出 | `anaprior/eval/predict_region_scores.py` |
| selective repair 融合 | `anaprior/eval/selective_repair.py` |
| learned repair hmap 构建 | `anaprior/eval/eval_mscxr_learned_repair.py` |
| MS-CXR 指标与 bootstrap | `anaprior/eval/score_mscxr_learned_repair_metrics.py` |
| Stage C preflight | `anaprior/eval/stage_c_preflight.py` |
| 中文结果报告 | `anaprior/eval/report_stage_c_results.py` |
| 服务器一键脚本 | `scripts/run_stage_c_learned_repair_server.sh` |
| 源码打包工具 | `anaprior/tools/create_stage_c_source_bundle.py` |

## 本地验证状态

本地验证命令：

```powershell
pytest tests -q
python -m compileall -q anaprior tests
bash -n scripts/run_stage_c_learned_repair_server.sh
```

最近一次验证结论：

```text
36 passed
compileall 通过
server script bash syntax 通过
```

源码包检查结论：

```text
包含 Stage C runner
包含中文说明文档
包含源码打包工具
不包含 outputs/
不包含 .git/
```

## 服务器待执行

源码包位置：

```text
C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\outputs\anaprior_stage_c_source_bundle.zip
```

上传：

```powershell
scp C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\outputs\anaprior_stage_c_source_bundle.zip zhangran@SERVER:/home/zhangran/zr/
```

服务器解压：

```bash
cd /home/zhangran/zr/G2-VLA-AFLoc
unzip -o /home/zhangran/zr/anaprior_stage_c_source_bundle.zip
```

先跑 smoke：

```bash
AFLOC_BERT_TYPE=/mnt/zhangran/hf_models/Bio_ClinicalBERT \
AFLOC_HF_LOCAL_FILES_ONLY=1 \
ANAPRIOR_RUN_SMOKE=1 \
bash scripts/run_stage_c_learned_repair_server.sh
```

smoke 通过后跑 full：

```bash
AFLOC_BERT_TYPE=/mnt/zhangran/hf_models/Bio_ClinicalBERT \
AFLOC_HF_LOCAL_FILES_ONLY=1 \
ANAPRIOR_RUN_SMOKE=0 \
bash scripts/run_stage_c_learned_repair_server.sh
```

## 服务器结果回传清单

跑完后优先回传：

```text
outputs/anaprior_stage_c/learned_repair_metrics/learned_repair_decision.json
outputs/anaprior_stage_c/learned_repair_metrics/bootstrap_summary.csv
outputs/anaprior_stage_c/learned_repair_metrics/per_class_bootstrap_summary.csv
outputs/anaprior_stage_c/learned_repair_metrics/stage_c_result_report.md
```

如果 smoke 或 preflight 失败，优先回传：

```text
outputs/anaprior_stage_c/preflight_report.json
```

## 判定标准

learned selective repair 只有同时满足下面三点，才进入论文主线：

```text
1. learned_selective 在 candidate 类上优于 baseline；
2. learned_selective 在 candidate 类上优于 candidate_shuffled；
3. macro_all 没有明显伤害。
```

如果 1 成立但 2 不成立，说明可能只是泛泛空间扰动，不是有效 region semantics。

如果 1 和 2 成立但 3 不成立，说明方法只能作为候选类修复模块，不能宣传成全类别通用模块。

## 当前结论

本地代码交付阶段已经结束，服务器实验阶段尚未结束。

因此当前状态应表述为：

```text
Stage C learned repair pipeline 已完成本地实现、文档化、测试和源码包交付；
正式科学结论等待服务器 smoke/full 运行结果。
```
