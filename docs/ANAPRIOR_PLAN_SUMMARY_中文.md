# AnaPrior 方案总结

更新日期：2026-07-06

这是当前 AnaPrior-Loc-AFLoc 研究方向的中文方案总结。更完整的数据台账见：

```text
docs/ANAPRIOR_CURRENT_CONTEXT.md
```

## 当前结论

AnaPrior 现在不应该包装成“一个 learned repair 方法能让所有胸片病种都变好”。更稳、更有价值的论文形态是一个诊断性研究：

```text
oracle 解剖区域修复在 8 类病种上普遍成立；
learned weak-label region evidence 才是瓶颈；
当前 learned 失败的主因是：它自信地指向了错误区域。
```

一句话：

```text
repair 公式本身是对的，问题是 learned evidence 的方向经常错。
```

## 关键证据

8 类 oracle profile 是当前最强的正结果。所有 8 个 finding 都是 oracle-recoverable，而且 CNR 置信区间下界都大于 0。

| finding | oracle Delta CNR | CI low | 解释 |
|---|---:|---:|---|
| Pneumothorax | 0.651 | 0.593 | oracle 上界最强 |
| Lung Opacity | 0.362 | 0.258 | recoverability 很强 |
| Pleural Effusion | 0.219 | 0.163 | 可恢复，重叠区域收益明显 |
| Pneumonia | 0.186 | 0.137 | 可恢复 |
| Atelectasis | 0.171 | 0.103 | 可恢复，也是当前 learned 成功对照 |
| Consolidation | 0.166 | 0.124 | 可恢复 |
| Edema | 0.157 | 0.105 | 即便是弥漫病变也有 oracle recoverability |
| Cardiomegaly | 0.136 | 0.113 | 稳定可恢复 |

但最新 Stage E learned 结果整体仍然应该 reject：

```text
candidate_vs_baseline_cnr_delta = -0.034228
candidate_vs_baseline_cnr_ci_low = -0.084917
macro_all_cnr_delta = -0.034228
macro_all_cnr_ci_low = -0.088330
```

真正有价值的是逐类结果：

| finding | learned Delta CNR | Stage E 诊断 |
|---|---:|---|
| Atelectasis | +0.108 | learned strong pass |
| Cardiomegaly | 0.000 | missing learned evidence |
| Consolidation | -0.118 | evidence misdirected |
| Edema | 0.000 | missing learned evidence |
| Lung Opacity | -0.052 | evidence misdirected |
| Pleural Effusion | -0.055 | 有 specificity，但仍低于 baseline |
| Pneumonia | 0.000 | missing learned evidence |
| Pneumothorax | -0.157 | 严重 evidence misdirection |

Pneumothorax 是最关键的反直觉例子：

```text
oracle Delta CNR = +0.651
learned Delta CNR = -0.157
learned_oracle_spearman = -0.144
mean learned top score = 0.593
mean oracle-top learned score = 0.179
```

这说明 Pneumothorax 不是不可修复。相反，它在 oracle 模式下最可修复；失败是因为当前 learned predictor 把错误 region 排到了前面，而且不是没信心，是自信地排错。

## 当前失败点

1. **learned evidence 的方向不可靠。**
   - Consolidation、Lung Opacity、Pleural Effusion、Pneumothorax 都有 learned evidence，但 learned top region 和 oracle top region 对不上。

2. **三类没有 learned evidence。**
   - Cardiomegaly、Edema、Pneumonia 当前 checkpoint 不支持，不能把它们算作 learned 失败，只能算作 oracle 上界分析。

3. **中心结构偏置明显。**
   - MS-CXR finding-agnostic region score 会过度偏向 cardiac / hilar 等中心结构。
   - 这会把 repair 推向视觉显著但和病灶无关的区域。

4. **不能用一个统一 region 策略覆盖所有病种。**
   - Pneumothorax 需要 pleural / apical / peripheral evidence。
   - Pleural Effusion 需要 costophrenic / lower / pleural evidence。
   - Cardiomegaly 需要 cardiac silhouette evidence。
   - Edema 和 opacity-like finding 更依赖 texture / appearance evidence。

## 推荐计划

### 近期最稳论文路线

1. **把 8 类 oracle profile 作为主正结果。**
   - 这是最干净的证据，说明 AnaPrior 的定位修复机制本身成立。

2. **把 Stage E 作为主失败分析。**
   - learned 失败应解释为 region direction misalignment，而不是 repair formulation 失败。

3. **严格区分 missing evidence 和 learned failure。**
   - Cardiomegaly、Edema、Pneumonia 目前没有 learned evidence，不应被写成 learned repair 失败。

4. **保留 Atelectasis 作为 learned positive control。**
   - 它证明 learned evidence 不是天然无效；当 region ranking 对齐 oracle 时，learned repair 可以工作。

### 下一步实验

训练真正覆盖 8 类的 predictor，然后重新跑：

```text
Stage C raw learned repair
Stage E learned-vs-oracle gap diagnosis
frozen finding-agnostic debias
```

需要产出的核心表：

```text
finding | oracle Delta CNR | raw learned Delta CNR | debiased learned Delta CNR | top1/top3 | diagnosis
```

这个实验回答三件事：

```text
learned gap 是因为 predictor 覆盖不全？
还是因为中心结构偏置？
还是因为更深层的 disease-specific evidence mismatch？
```

### 长期方法方向

不要继续找一个万能 anatomy prior。正确方向是：

```text
shared repair framework + disease-conditioned evidence modules + reliability gate
```

候选 disease-conditioned 模块：

| finding group | evidence module |
|---|---|
| Pneumothorax | apical / peripheral / pleural band evidence，配合 max/top-k pooling |
| Pleural Effusion | costophrenic / lower / pleural evidence，保留下肺覆盖 |
| Cardiomegaly | cardiac silhouette / heart-boundary evidence |
| Edema | bilateral diffuse texture + perihilar evidence |
| Opacity / Pneumonia / Consolidation | local appearance + lung-zone constraint |
| Atelectasis | morphology + lung-zone evidence |

再加 reliability gate：

```text
if evidence is unreliable: keep AFLoc baseline
if evidence is reliable and anatomically plausible: apply repair
```

## 论文表述

不要这样写：

```text
We propose a learned repair method that improves all findings.
```

应该这样写：

```text
AnaPrior reveals a broad oracle recoverability upper bound for anatomical region repair,
and identifies weak-label region evidence misdirection as the main obstacle to deployable learned repair.
```

中文版本：

```text
AnaPrior 证明了解剖区域修复在 8 类胸片 finding 上存在广泛 oracle 可恢复上界；
但当前从弱标签中学习到的 region evidence 存在系统性方向偏置，
这是 learned repair 无法稳定部署的主要瓶颈。
```

一句话：

```text
定位修复机制本身是广泛有效的，但从弱标签中学到方向正确的 region evidence 是 pathology-dependent 且当前不可靠的。
```
