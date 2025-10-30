# 数据集特定配置指南

本文档说明如何通过命令行参数为不同数据集配置语义权重和排序损失参数，**无需修改代码**。

## 核心设计原则

✅ **基础代码统一**：权重计算、裁剪、token pair构建逻辑在所有数据集上保持一致  
✅ **参数化配置**：通过命令行参数 `--参数名=值` 覆盖默认配置  
✅ **数据集自适应**：根据数据特性调整γ、裁剪范围、温度等参数

---

## 参数说明

### 1. 语义权重参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `item_weight_mode` | str | `proximity` | 权重模式：`proximity`(距离越近分数越高) / `semantic_diversity`(距离越远分数越高) |
| `item_weight_gamma` | float | `3.0` | γ参数，控制权重影响强度。公式：`priority = cnt * (w1*w2)^(γ/2)` |
| `item_weight_clip_range` | list | `[0.2, 4.0]` | 第一次裁剪范围 `[min, max]`，防止极端权重 |
| `enable_second_clip` | bool | `False` | 是否启用对数空间二次裁剪（用于极端分布数据集） |
| `log_weight_clip_range` | list | `[-1.5, 1.5]` | 二次裁剪范围（仅当 `enable_second_clip=True` 时生效） |

### 2. 排序损失参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ranking_temperature` | float | `0.7` | 排序温度系数。值越小排序优化越激进，`1.0`等价于标准CE损失 |

### 3. 其他相关参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `weight_analysis_interval` | int | `4000` | 每N个词汇输出一次权重有效性分析 |
| `n_hash_buckets` | int | `128` | 哈希桶数量（增加可减少特征冲突） |
| `num_beams` | int | `50` | Beam search宽度 |
| `n_inference_ensemble` | int | `5` | 集成推理次数 |

---

## 推荐配置

### Beauty 数据集（基准配置）

**数据特征**：
- 中等数据规模
- 权重分布相对均衡
- 语义相似性较好

**推荐参数**：
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --item_weight_gamma=3.0 \
        --item_weight_clip_range="[0.2,4.0]" \
        --enable_second_clip=False \
        --ranking_temperature=0.7 \
        --n_hash_buckets=64
done
```

**预期效果**：
- 权重标准差：0.30-0.35
- 权重贡献度：0.15-0.25
- 中性权重比例：45-50%
- Top-20排序变化：30-40%

---

### Sports_and_Outdoors 数据集（需增强权重）

**数据特征**：
- 较大数据规模
- 权重分布严重偏向低权重（均值0.77，多数<0.8）
- **关键问题**：原始权重本身就低，过度压缩clip上限会加剧问题
- 需要更强的权重放大效应

**⚠️ 实验教训**：
- γ=2.5 + clip=[0.4,2.5]：权重贡献度-0.536，**上限过低导致更差** ❌
- 正确策略：**放宽上限 + 增大γ**，让高相似度item获得充分加权

**推荐参数（方案A - 放宽上限+增大γ）【✨最新推荐】**：
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Sports_and_Outdoors \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.005 \
        --item_weight_gamma=3.0 \
        --item_weight_clip_range="[0.3,3.5]" \
        --enable_second_clip=False \
        --ranking_temperature=0.7 \
        --n_hash_buckets=128
done
```

**推荐参数（方案B - 二次裁剪精细控制）**：
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Sports_and_Outdoors \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.005 \
        --item_weight_gamma=3.0 \
        --item_weight_clip_range="[0.2,4.0]" \
        --enable_second_clip=True \
        --log_weight_clip_range="[-1.2,1.2]" \
        --ranking_temperature=0.7 \
        --n_hash_buckets=128
done
```

**推荐参数（方案C - 激进配置）**：
```bash
# 如果方案A/B仍不够，尝试更大的γ
--item_weight_gamma=3.5 \
--item_weight_clip_range="[0.3,4.0]"
```

**目标效果**：
- 权重标准差：≥0.18
- 权重贡献度：≥0.05
- 中性权重比例：15-25%
- Top-20排序变化：≥25%
- 权重因子均值：≥0.85

---

### CDs_and_Vinyl 数据集

**数据特征**：
- 中等偏大数据规模
- 权重分布介于Beauty和Sports之间

**推荐参数**：
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=CDs_and_Vinyl \
        --rand_seed=${seed} \
        --weight_decay=0.13 \
        --lr=0.0009 \
        --item_weight_gamma=2.5 \
        --item_weight_clip_range="[0.3,3.5]" \
        --enable_second_clip=False \
        --ranking_temperature=0.7 \
        --n_hash_buckets=64
done
```

---

## 参数调优流程

### 步骤1：观察初始权重分布

运行一次训练，观察词汇表构建时的权重分析日志：

```
权重贡献度: X.XXX (越高说明权重影响越大)
权重标准差: X.XXX
中性权重比例: XX.X%
```

### 步骤2：诊断问题

| 现象 | 问题 | 解决方案 |
|------|------|----------|
| 权重贡献度 < 0 | 权重影响不足 | 增大 `item_weight_gamma` 或调整 `clip_range` |
| 权重标准差 < 0.15 | 权重区分度不足 | 增大 `item_weight_gamma` |
| 权重标准差 > 1.5 | 权重过于极端 | 缩小 `clip_range` 或启用 `second_clip` |
| 显著加权比例 > 5% | 极端值过多 | 缩小 `clip_range` 或启用 `second_clip` |
| Top-20重叠度 > 90% | 权重对排序影响小 | 增大 `item_weight_gamma` |

### 步骤3：调整参数

**增强权重影响**：
- 增大 `item_weight_gamma`：从2.0 → 2.5 → 3.0（步长0.5）
- 调整 `clip_range`：缩小上限以控制极端值

**控制极端权重**：
- 启用 `enable_second_clip=True`
- 调整 `log_weight_clip_range`：从[-2.0, 2.0] → [-1.5, 1.5] → [-1.0, 1.0]

### 步骤4：验证效果

在验证集上评估：
- NDCG@5/10 是否提升
- Recall@5/10 是否提升
- 模型是否过拟合（训练集loss下降但验证集指标不提升）

---

## 参数组合建议

### 温和权重增强（适合Beauty）
```bash
--item_weight_gamma=3.0 \
--item_weight_clip_range="[0.2,4.0]" \
--enable_second_clip=False
```

### 中等权重增强（适合CDs）
```bash
--item_weight_gamma=2.5 \
--item_weight_clip_range="[0.3,3.5]" \
--enable_second_clip=False
```

### 激进权重增强（适合Sports方案1）
```bash
--item_weight_gamma=2.5 \
--item_weight_clip_range="[0.4,2.5]" \
--enable_second_clip=False
```

### 激进权重增强 + 二次裁剪（适合Sports方案2）
```bash
--item_weight_gamma=2.5 \
--item_weight_clip_range="[0.3,3.0]" \
--enable_second_clip=True \
--log_weight_clip_range="[-1.5,1.5]"
```

---

## FAQ

### Q1: 为什么Sports需要不同的参数？

**A**: Sports数据集的语义权重分布偏低（均值0.8左右），标准差只有0.1，说明大部分item的语义相似度计算结果接近，需要更强的放大效应才能让权重发挥作用。

### Q2: `item_weight_gamma` 和 `clip_range` 应该如何配合？

**A**: 
- **γ大 + clip宽**：权重影响最大，但可能产生极端值
- **γ大 + clip窄**：权重有影响，极端值被控制
- **γ小 + clip宽**：权重影响较小
- **γ小 + clip窄**：权重几乎无影响

### Q3: 什么时候应该启用二次裁剪？

**A**: 当以下情况出现时：
- 一次裁剪后仍有"显著加权"比例 > 3%
- 权重标准差 > 1.0 但贡献度仍为负
- 想要更精细地控制权重分布形态

### Q4: `ranking_temperature` 设置为1.0会怎样？

**A**: 等价于标准的交叉熵损失，不启用排序优化。用于对比实验以验证ranking loss的有效性。

### Q5: 如何快速测试参数效果？

**A**: 可以只构建词汇表而不训练模型：

```bash
# 只运行到词汇表构建阶段，观察权重分析日志
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --category=Sports_and_Outdoors \
    --item_weight_gamma=2.5 \
    --item_weight_clip_range="[0.4,2.5]" \
    # ... 其他参数
```

查看日志中的"权重有效性深度分析"部分，判断参数是否合理。

---

## 调试技巧

### 1. 查看权重分布直方图

在tokenizer.py的`_compute_item_weights`方法中已经有日志输出，查找关键指标：

```
权重因子 (w1*w2)^(γ/2): mean=X.XXXX, std=X.XXXX, range=[X.XXXX, X.XXXX]
```

### 2. 对比有无权重的效果

设置一个实验不使用权重（γ=0.0或clip=[1.0,1.0]），对比性能差异。

### 3. 监控训练过程

观察训练日志中的loss曲线，正常情况下：
- 启用ranking_temperature<1.0会使loss略微升高
- 但验证集指标应该提升

---

## 总结

本配置系统的核心优势：

1. **代码统一**：所有数据集使用相同的权重计算和裁剪逻辑
2. **参数灵活**：通过命令行参数适配不同数据集特性
3. **可追溯**：每次实验的完整参数记录在训练日志中
4. **易调试**：权重分析日志提供实时反馈

推荐工作流：
1. 从推荐配置开始
2. 观察权重分析日志
3. 根据诊断表调整参数
4. 验证性能提升
5. 记录最优配置

