# 幻灯片完整文字逐页顺序提取

## 第1页：先把8类问题分组

根据你 Stage E 的数据，8 类的问题完全不同：

```text
A组：缺失 evidence（3类）
└ Cardiomegaly, Edema, Pneumonia
└ predictor 根本不支持这3类
└ 问题：coverage

B组：evidence 方向错误（4类）
└ Consolidation, Lung Opacity, Pleural Effusion, Pneumothorax
└ predictor 有输出但方向错
└ 问题：direction

C组：已经成功（1类）
└ Atelectasis
└ 维持即可，作为正向对照
```

### 小标题

🔧 A组：解决 Coverage（3类缺失）

#### 根本原因

（下方内容截断不可见）

页面底部小字备注：如何根据 Predictor 输出的分数差异来设计 Reliability Gate 的置信阈值

---

## 第2页：A组：解决Coverage（3类缺失）

### 根本原因

```text
当前 checkpoint 训练时
Cardiomegaly, Edema, Pneumonia
没有被包含在 finding vocab 里
→ 推理时直接 skip
```

### 解决方案：真正的 8-class Predictor

```python
# 重新构建训练表，包含全部8类
FINDING_VOCAB = [
    "Atelectasis",
    "Cardiomegaly",    # ← 补上
    "Consolidation",
    "Edema",            # ← 补上
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",        # ← 补上
    "Pneumothorax",
]
```

补充文字：但这 3 类各有特殊性：

---

## 第3页：A组三类病灶特殊性 + B组总述

### 表格：A组三类疾病特殊性

| 疾病 | 特殊性 | 需要注意 |
| ---- | ---- | ---- |
| Cardiomegaly | 唯一非肺部疾病 | 只有 cardiac_silhouette 有效，其他区域全是噪声 |
| Edema | 弥漫性病变 | 不是局灶性病灶，bilateral diffuse 分布，mean pooling 反而合适 |
| Pneumonia | 与 Consolidation 高度相似 | 区域分布接近，但 Pneumonia oracle CNR = 0.186，可学性中等 |

### 小标题

🔧 B组：解决 Direction（4类方向错）

每类的具体问题和方案

### Pneumothorax（最严重: Spearman = -0.14）

#### 问题：

bilateral_lungs mean pooling
→ 气胸信号只在肺边缘极细线
→ 均值把它稀释掉了
→ cardiac/hilar 反而得高分

#### 方案：

1. Region Whitelist: 只用上肺 + 胸膜区域
2. 换 max pooling / top-k pooling
3. 加边缘感知特征提取

---

## 第4页：气胸代码配置 + 胸腔积液问题方案

```python
PNEUMOTHORAX_CONFIG = {
    "allowed_regions": [
        "left_upper_lung",
        "right_upper_lung",
        "pleural_space_costophrenic",
    ],
    "blocked_regions": [
        "cardiac_silhouette",      # 物理屏蔽
        "hilar_mediastinal",       # 物理屏蔽
    ],
    "pooling": "top_k",  # k = 10% of region pixels
}
```

### 💧 Pleural Effusion (partial specificity，top1 hit 仅7.1%)

#### 问题：

debias 后 top1 hit → 31.4% (大幅改善)
但 CNR 仍低于 baseline
原因: 证据方向对了但区域太宽泛
bilateral/lower-lung 覆盖太广

#### 方案：

1. 保留 lower lung + pleural space 区域
2. 降低 cardiac/hilar 权重（不完全屏蔽，Effusion 可能延伸）
3. 用 weighted pooling 替代 mean pooling

---

## 第5页：胸腔积液完整权重配置 + 实变问题方案

```python
PLEURAL_EFFUSION_CONFIG = {
    "region_weights": {
        "pleural_space_costophrenic": 2.0,  # 最高权重
        "left_lower_lung": 1.5,
        "right_lower_lung": 1.5,
        "bilateral_lungs": 0.8,
        "cardiac_silhouette": 0.3,         # 降权不屏蔽
        "hilar_mediastinal": 0.2,
    },
    "pooling": "weighted_mean",
}
```

### Consolidation (gap = 0.284，oracle top learned score = 0.685)

#### 问题：

learned top score = 0.960 (非常自信)
但 oracle-top 区域的 learned score = 0.685
→ predictor 信心很高但选了错区域

#### 原因：

Consolidation 在 Chest ImaGenome 训练分布里
cardiac/hilar 共现频率高
→ predictor 学了错误的捷径

#### 方案：

1. 训练时加 hard negative mining
cardiac/hilar 区域在 Consolidation 标签下作为 hard negative
2. 推理时轻度降权 cardiac/hilar

---

## 第6页：肺不透光问题 + 统一框架总起

### Lung Opacity (gap = 0.414，最大 gap 之一)

#### 问题：

与 Consolidation 类似但 gap 更大
mean learned top score = 0.994 (极度自信)
oracle-top learned score = 0.926 (还行)
→ 部分对齐但还不够

#### 方案：

需要更精细的区域候选
Lung Opacity 是局部不透明，需要：

1. 中肺 + 下肺区域优先
2. Top-k pooling 保留局部最强信号

# 🏗️ 统一框架设计

把以上所有方案整合成一个框架：

---

## 第7页：DCEM 完整模块架构

# 🏗️ 统一框架设计

把以上所有方案整合成一个框架：

## Disease-Conditioned Evidence Module (DCEM)

1. Disease Router
   根据 finding 选择配置
2. Region Filter
   Whitelist + Blocked + Weights
3. Adaptive Pooling
   mean / top-k / weighted
   根据疾病特性自动选择
4. Reliability Gate
   top1_score / top2_score > θ
   → 低可靠时 bypass repair

↓
Disease-specific Evidence Score
↓
Repair Mechanism (现有的)
↓
Improved Localization

---

## 第8页：8类完整配置汇总表

# 📋 8类完整配置表

| 疾病 | A组/B组/C组 | Region策略 | Pooling策略 | 特殊处理 |
| ---- | ---- | ---- | ---- | ---- |
| Atelectasis | C组✅ | 中下肺区域 | mean | 维持现状 |
| Cardiomegaly | A组 | cardiac_silhouette only | mean | 训练补全 |
| Consolidation | B组 | 肺区域，cardiac降权 | top-k | Hard negative mining |
| Edema | A组 | bilateral diffuse | mean | 训练补全，弥漫性特殊处理 |
| Lung Opacity | B组 | 中下肺优先 | top-k | 局部最强信号保留 |
| Pleural Effusion | B组 | pleural+lower，cardiac降权 | weighted | Weighted region pooling |
| Pneumonia | A组 | 肺区域（类Consolidation） | top-k | 训练补全 |
| Pneumothorax | B组 | 上肺+胸膜，cardiac屏蔽 | top-k | 最严格的whitelist |

# 📚 执行路径

（下方内容截断）

---

## 第9页：分阶段执行路径 + 核心判断

# 📚 执行路径

### Step 1 (本周，无需重训)

└ 实现 Disease-Conditioned Region Filter + Reliability Gate
└ 在现有 5-class predictor 上验证 B组改善
└ 预期: Pneumothorax CNR 从 -0.157 → 接近 0 或正值
└ 预期: Effusion 从 -0.055 → 正值

### Step 2 (下周，需重训)

└ 训练真正的 8-class predictor
└ 补全 Cardiomegaly, Edema, Pneumonia
└ 配合 Step 1 的 Region Filter 一起跑

### Step 3 (第三周，可选)

└ 加入 Oracle-Guided Ranking Loss 重训
└ 这是让 B组从根本上对齐 oracle 的方案
└ 预期: 所有 B组 Spearman 从负值 → 正值

### Step 4 (验证)

└ 完整 8-class Stage C + Stage E 重跑
└ 目标: 8类全部 CNR delta > 0

# 💬 最关键的判断

"Step1 是最快验证你想法的路径，不需要重新训练任何模型。
如果 Step1 的 Region Filter + Reliability Gate 能让 Pneumothorax 和 Effusion 变正，你就有了强有力的证据说明:
'方向错误是主要问题，disease-conditioned 策略是解法'
然后 Step 2+3 是把这个解法做得更系统、更端到端。"
