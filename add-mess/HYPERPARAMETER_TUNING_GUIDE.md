# ActionPiece 超参数调优指南

## 📋 当前基线配置

```bash
# 基线结果 (3个seed平均)
NDCG@5:  0.03571
NDCG@10: 0.04413
Recall@5: 0.05268
Recall@10: 0.07893

# 命令
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64
done
```

## 🎯 调参实验计划

### 阶段1：Ranking Temperature 调优 (最高优先级)
**目标**：找到最优的ranking-guided loss温度参数

```bash
# 实验1.1: τ = 0.5 (更陡峭，更强hard-negative惩罚)
# 修改 config.yaml: ranking_temperature: 0.5
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64
done

# 实验1.2: τ = 0.6
# 修改 config.yaml: ranking_temperature: 0.6

# 实验1.3: τ = 0.8
# 修改 config.yaml: ranking_temperature: 0.8

# 实验1.4: τ = 1.0 (标准CE loss，作为对照)
# 修改 config.yaml: ranking_temperature: 1.0
```

**期望效果**：
- τ < 0.7: 可能进一步提升NDCG（更关注排序质量）
- τ = 1.0: 应该接近或略低于当前结果

---

### 阶段2：Gamma 值微调
**目标**：优化语义权重的影响强度

当前 γ=3.0，权重因子范围 [0.2196, 7.3891]

```bash
# 实验2.1: γ = 2.5 (减弱权重影响)
# 修改 config.yaml: item_weight_gamma: 2.5
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64
done

# 实验2.2: γ = 3.5 (增强权重影响)
# 修改 config.yaml: item_weight_gamma: 3.5

# 实验2.3: γ = 4.0 (更强影响)
# 修改 config.yaml: item_weight_gamma: 4.0
```

**检查词汇表日志**：
- 权重贡献度应在 0.15-0.30 之间
- 显著加权比例 < 3%

---

### 阶段3：集成推理规模调整
**目标**：平衡推理成本和性能

当前 n_inference_ensemble=5

```bash
# 实验3.1: n_ensemble = 3 (降低计算成本)
# 修改 config.yaml: n_inference_ensemble: 3
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64
done

# 实验3.2: n_ensemble = 7 (可能有边际收益)
# 修改 config.yaml: n_inference_ensemble: 7
```

**期望**：5可能已接近最优，但值得验证

---

### 阶段4：学习率和正则化组合
**目标**：寻找更好的优化轨迹

```bash
# 实验4.1: 降低学习率 + 降低weight_decay
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.10 \
        --lr=0.0008 \
        --n_hash_buckets=64
done

# 实验4.2: 提高学习率 + 提高weight_decay
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.20 \
        --lr=0.0012 \
        --n_hash_buckets=64
done
```

---

### 阶段5：权重裁剪范围调整 (谨慎实验)
**目标**：控制极端权重的影响

当前 [0.2, 4.0]，对应权重因子最大范围 [0.22, 7.39]

```bash
# 实验5.1: 更保守的裁剪 [0.3, 3.0]
# 修改 config.yaml: item_weight_clip_range: [0.3, 3.0]
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64
done

# 实验5.2: 更激进的权重 [0.2, 5.0]
# 修改 config.yaml: item_weight_clip_range: [0.2, 5.0]
```

**注意**：每次修改后需重新构建词汇表

---

## 📊 实验记录模板

```markdown
### 实验 X.Y: [实验名称]

**配置变更**：
- 参数1: 旧值 → 新值
- 参数2: 旧值 → 新值

**词汇表统计** (如有重建):
- 权重贡献度: X.XXX
- 权重标准差: X.XXX
- Top-20排序变化: XX.X%

**性能指标** (3 seeds 平均):
| Metric     | Seed2026 | Seed2027 | Seed2028 | **平均** | vs基线 |
|------------|----------|----------|----------|----------|--------|
| NDCG@5     |          |          |          |          |        |
| NDCG@10    |          |          |          |          |        |
| Recall@5   |          |          |          |          |        |
| Recall@10  |          |          |          |          |        |

**结论**：
- [ ] 提升 / [ ] 持平 / [ ] 下降
- 观察到的现象：...
```

---

## 🔬 调参经验法则

1. **Ranking Temperature (τ)**：
   - 论文通常使用 0.5-1.0
   - τ↓ → 增强排序能力，但可能过拟合
   - 优先测试 [0.5, 0.6, 0.7, 0.8]

2. **Gamma (γ)**：
   - 当前 γ=3.0 表现良好
   - 观察权重贡献度，保持在 0.15-0.30
   - 步长 0.5 即可

3. **Ensemble Size**：
   - 边际收益递减，5可能已接近饱和
   - 3、5、7 三档验证即可

4. **Learning Rate & Weight Decay**：
   - 它们共同影响泛化能力
   - 需要网格搜索或成对调整

---

## ⚠️ 重要提醒

1. **每次修改权重相关参数后**（gamma、clip_range），必须：
   ```bash
   rm -rf cache/AmazonReviews2014/Beauty/processed/actionpiece.json
   ```
   重新构建词汇表

2. **每个实验至少跑3个seeds**，报告平均值和标准差

3. **记录完整日志**，包括词汇表构建阶段的权重分析

4. **逐步实验**，不要同时改多个参数

---

## 🎯 预期时间估算

假设每次训练约30分钟（Beauty数据集 + 3 seeds）：

- 阶段1 (4组实验): ~2小时
- 阶段2 (3组实验): ~1.5小时
- 阶段3 (2组实验): ~1小时
- 阶段4 (2组实验): ~1小时
- 阶段5 (2组实验): ~1小时 (需重建词汇表)

**总计约 6.5小时**

---

## 📈 何时停止调参，转向代码优化？

满足以下任一条件时，考虑代码层面优化：

1. **性能停滞**：连续3组实验无明显提升（<1%）
2. **超参数敏感**：指标随参数剧烈波动（方差>0.002）
3. **达到预期目标**：NDCG@10 > 0.045 或 相对原文提升 > 5%
4. **已穷尽主要超参数**：完成上述5个阶段

---

## 💡 后续代码优化方向（备选）

如果调参后仍有提升空间，可考虑：

1. **更精细的语义权重策略**：
   - 类别级别的自适应gamma
   - 多尺度语义相似度（不同嵌入模型融合）

2. **改进Beam Search**：
   - 长度惩罚 (length penalty)
   - 多样性beam search (diverse beam search)

3. **损失函数增强**：
   - Contrastive learning 辅助任务
   - 知识蒸馏 (从更大模型)

4. **数据增强**：
   - 更激进的随机游走策略
   - 反事实数据增强

但**强烈建议先完成调参实验**！

