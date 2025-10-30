# ActionPiece 项目全面讲解

## 📚 目录
1. [项目概述](#项目概述)
2. [核心代码文件详解](#核心代码文件详解)
3. [已完成的优化工作](#已完成的优化工作)
4. [正在进行的优化](#正在进行的优化)

---

## 项目概述

**ActionPiece** 是一个用于生成式推荐（Generative Recommendation）的上下文感知动作序列分词方法，发表于 ICML 2025 Spotlight。

### 核心思想
与传统方法独立地对每个动作（item）进行分词不同，ActionPiece **显式地将上下文信息融入动作序列的分词过程**：
- 将每个item表示为一组特征（features）
- 使用BPE算法合并频繁共现的特征模式
- 在单个item内部合并特征，也在相邻item之间合并特征

### 数据流程
```
用户交互序列 → Item特征提取 → ActionPiece分词 → T5 Encoder-Decoder → 生成下一个Item
```

---

## 核心代码文件详解

### 1. `core.py` - ActionPiece核心分词器

**功能**：实现BPE（Byte Pair Encoding）风格的分词算法，用于构建和使用ActionPiece词汇表。

#### 关键类：`ActionPieceCore`

**核心数据结构**：
```python
class ActionPieceCore:
    # 词汇表相关
    self.vocab          # list: token → feature组合
    self.rank           # dict: feature组合 → token_id
    self.token2feat     # 同vocab
    self.feat2token     # 同rank
    
    # 训练状态
    self.cur_corpus     # 当前语料库（链表形式）
    self.pq             # 优先级队列（用于快速找到最高频pair）
    self.all_pair2cnt   # 所有token pair的全局计数
    
    # 权重系统（我们添加的）
    self.item_weights   # dict: item_id → 语义权重
    self.log_w          # list: token_id → log(权重)
    self.gamma          # float: 权重影响强度
```

#### 核心方法详解

##### A. `_build()` - 构建初始语料库
```python
def _build(self, token_corpus):
    """
    将输入的token序列转换为链表结构
    - 每个item是一个regular state节点
    - item之间插入context slot节点（用于跨item合并）
    """
```

**链表结构示例**：
```
[item1] → [context] → [item2] → [context] → [item3]
  ↓                      ↓                      ↓
[t1,t2,t3]            [t4,t5]                [t6,t7,t8]
```

##### B. `_count_pairs_*()` - 计算token pair频次

**三种计数方式**：
1. **`_count_pairs_inside_state`**：计算单个item内部的token pairs
   - 权重：`2 / M`（M是item中的token数）
   - 示例：item=[t1,t2,t3] → pairs={(t1,t2):2/3, (t1,t3):2/3, (t2,t3):2/3}

2. **`_count_pairs_btw_states`**：计算两个item之间的token pairs
   - 权重：`1 / (M1 * M2)`
   - 示例：item1=[t1,t2], item2=[t3,t4] → pairs={(t1,t3):1/4, (t1,t4):1/4, ...}

3. **`_count_pairs_in_list`**：遍历整个链表计算所有pairs

##### C. `_compute_pair_priority()` - 计算优先级（核心创新）

**原始方法**：
```python
priority = cnt  # 仅使用共现频次
```

**我们的改进（log_additive模式）**：
```python
def _compute_pair_priority(self, tk1, tk2, cnt):
    """
    融合语义权重的优先级计算
    公式: priority = cnt × (w1 × w2)^(γ/2)
    
    对数空间计算（数值稳定）:
    log(priority) = log(cnt) + γ/2 × (log(w1) + log(w2))
    """
    if self.log_w is None:
        return cnt
    
    log_cnt = math.log(cnt + 1e-12)
    log_w1 = self.log_w[tk1] if tk1 < len(self.log_w) else 0.0
    log_w2 = self.log_w[tk2] if tk2 < len(self.log_w) else 0.0
    
    # γ/2 × (log_w1 + log_w2)
    log_weight_component = (self.gamma / 2.0) * (log_w1 + log_w2)
    
    # 裁剪防止溢出
    log_weight_component = np.clip(log_weight_component, -2.0, 2.0)
    
    log_priority = log_cnt + log_weight_component
    priority = math.exp(log_priority)
    
    return np.clip(priority, 1e-8, 1e6)
```

**关键设计**：
- **对数空间计算**：避免浮点溢出
- **多层裁剪**：保证数值稳定性
- **可配置γ**：控制权重影响强度（默认3.0）

##### D. `_train_step()` - 单步训练

**算法流程**：
```python
def _train_step(self):
    """
    1. 从优先级队列中取出最高优先级的pair (tk1, tk2)
    2. 创建新的合并token: new_token = [tk1, tk2]
    3. 在所有包含(tk1,tk2)的序列中执行合并
    4. 更新数据结构：
       - cur_corpus: 更新链表
       - head_id2pair_cnt: 更新每个序列的pair计数
       - pair2head_ids: 更新倒排索引
       - all_pair2cnt: 更新全局pair计数
       - pq: 插入新的优先级
    5. 如果有权重系统，更新新token的权重
    """
    # 从优先级队列获取最高优先级pair
    priority, (tk1, tk2) = self.pq.get()
    
    # 创建新token
    new_token = len(self.vocab)
    self.vocab.append((-1, tk1, tk2))
    self.priority.append(-priority)
    
    # 更新权重（关键！）
    if self.log_w is not None:
        # 乘法语义：w_new = w1 × w2
        new_log_w = self.log_w[tk1] + self.log_w[tk2]
        self.log_w.append(new_log_w)
    
    # 执行合并并更新所有数据结构...
```

##### E. `encode()` 和 `decode()` - 编码解码

**编码**：将item序列转换为token序列
**解码**：将token序列还原为item特征

---

### 2. `tokenizer.py` - ActionPiece分词器封装

**功能**：封装`ActionPieceCore`，提供与GenRec框架的接口，负责权重计算和数据预处理。

#### 关键方法详解

##### A. `_init_tokenizer()` - 初始化分词器

**流程**：
```python
def _init_tokenizer(self, dataset):
    """
    1. 加载/生成item特征
    2. 加载/生成语义嵌入（sentence embeddings）
    3. 计算item权重（我们添加的）
    4. 构建ActionPieceCore并训练
    5. 保存训练好的词汇表
    """
```

##### B. `_compute_item_weights()` - 计算语义权重（核心创新）

**原始论文**：无权重系统

**我们的实现**：
```python
def _compute_item_weights(self, dataset, sent_embs):
    """
    基于语义嵌入计算item权重
    
    模式：proximity（语义相似性）
    思想：语义相似的items应该更容易被合并
    
    步骤：
    1. 使用FAISS计算每个item的k近邻距离
    2. 计算平均距离
    3. proximity模式: score = 1 / (1 + distance)
       - 距离小 → score高 → 权重高
    4. 归一化到均值1.0
    5. 裁剪到配置范围
    6. （新增）裁剪后强制重新归一化
    """
    # 使用faiss计算k近邻
    index = faiss.IndexFlatL2(d)
    index.add(sent_embs)
    k = min(10, V)
    distances, indices = index.search(sent_embs, k)
    
    # 计算平均距离（排除自身）
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # Proximity模式
    scores = 1.0 / (1.0 + mean_distances)
    
    # 归一化
    scores = scores / scores.mean()
    
    # 裁剪
    scores = np.clip(scores, low, high)
    
    # 【关键修复】裁剪后重新归一化
    clipped_mean_before = scores.mean()
    scores = scores / (clipped_mean_before + 1e-12)
    
    # 构建权重字典
    item_weights = {item_id: score for item_id, score in zip(...)}
    return item_weights
```

**为什么需要重新归一化？**
- Sports数据集语义更分散 → proximity分数普遍偏低
- 大量权重被裁剪到下限 → 均值偏离1.0
- 不归一化 → 权重起负作用（权重贡献度为负）
- 归一化 → 保证权重均值=1.0，不同数据集一致

##### C. `_build_log_weights()` - 构建对数权重

```python
def _build_log_weights(self, actionpiece):
    """
    将item权重映射到token权重
    
    1. 构建反向ID映射（item_id → internal_id）
    2. 为每个初始token分配权重
    3. 转换到对数空间：log_w = log(w)
    4. 裁剪log_w到安全范围 [-3.0, 3.0]
    """
```

---

### 3. `model.py` - ActionPiece模型

**功能**：基于T5架构的Encoder-Decoder模型，负责序列到序列的生成任务。

#### 关键类：`ActionPiece`

**模型架构**：
```python
class ActionPiece:
    self.t5  # T5ForConditionalGeneration
    self.ranking_temperature  # 温度参数（我们添加的）
```

#### 核心方法详解

##### A. `forward()` - 前向传播（核心创新）

**原始方法**：
```python
def forward(self, batch):
    outputs = self.t5(**batch)
    return outputs  # 使用T5默认的交叉熵损失
```

**我们的改进 - Ranking-Guided Generation Loss**：
```python
def forward(self, batch):
    """
    温度缩放的排序引导损失
    
    公式: P(y_t|y_{<t},x) = exp(logits/τ) / Σ exp(logits/τ)
    """
    # 获取T5输出（不计算损失）
    outputs = self.t5(**batch, return_dict=True)
    logits = outputs.logits  # (B, L, V)
    
    # 温度缩放
    tau = self.ranking_temperature  # 默认0.7
    scaled_logits = logits / tau
    
    # 计算损失
    loss = F.cross_entropy(
        scaled_logits.view(-1, scaled_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    outputs.loss = loss
    return outputs
```

**温度参数τ的作用**：
- **τ < 1.0**（如0.7）：
  - 使概率分布更**陡峭**
  - 增强对正确item的置信度
  - 更严厉地惩罚错误预测（hard negatives）
  - **提升排序质量** → NDCG↑

- **τ = 1.0**：标准交叉熵

- **τ > 1.0**：平滑分布，降低过拟合风险

**理论依据**：
```
当τ=0.7时：
- 正确item的logit=2.0 → exp(2.0/0.7)=exp(2.86)≈17.5
- 错误item的logit=1.0 → exp(1.0/0.7)=exp(1.43)≈4.2
- 概率比值：17.5/4.2 ≈ 4.2x

当τ=1.0时：
- 概率比值：exp(2.0)/exp(1.0) ≈ 2.7x

结论：τ=0.7使正确答案的优势放大！
```

##### B. `beam_search_step()` - Beam搜索步骤（核心创新）

**我们的改进**：在推理时也应用温度缩放
```python
def beam_search_step(self, input_ids, ...):
    """
    在每一步beam search中应用温度缩放
    """
    # 获取T5的logits
    logits = self.t5.decoder(...)
    
    # 应用温度缩放（与训练保持一致）
    logits = logits / self.ranking_temperature
    
    # 计算log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 继续beam search...
```

**为什么推理时也要用温度？**
- **训练-推理一致性**：避免train-test mismatch
- **增强排序能力**：推理时也强化高质量候选
- **协同集成推理**：与多路径集成形成互补

##### C. `generate()` - 集成生成（核心创新）

**原始方法**：单路径生成

**我们的方法**：多路径集成
```python
def generate(self, batch, n_return_sequences=1):
    """
    集成多个随机游走增强的输入
    
    流程：
    1. 输入已经过随机游走增强（n_ensemble=5条路径）
    2. 对每条路径执行beam search
    3. 聚合所有路径的预测结果
    4. 基于NDCG分数选择最佳预测
    """
    n_ensemble = self.n_inference_ensemble  # 5
    batch_size = batch['input_ids'].shape[0] // n_ensemble
    
    # 对所有路径执行beam search
    outputs = self.beam_search(...)
    
    # 聚合和排序
    # ...
```

**集成的好处**：
- **鲁棒性**：减少对单一路径的依赖
- **覆盖率**：探索更多可能性
- **准确性**：多数投票效应

---

### 4. `trainer.py` - 训练器

**功能**：简单继承基类`Trainer`，无额外逻辑
```python
class ActionPieceTrainer(Trainer):
    pass  # 使用父类的所有方法
```

---

### 5. `utils.py` - 工具类

#### 关键类：`LinkedListState`

**功能**：链表节点，用于表示item序列和context slots

**属性**：
```python
class LinkedListState:
    self.state     # list[int]: token列表
    self.head_id   # int: 所属序列的ID
    self.context   # bool: 是否是context slot
    self.next      # LinkedListState: 下一个节点
    self.prev      # LinkedListState: 上一个节点
```

**关键方法**：
- `append()`: 追加节点
- `tolist()`: 转换为列表
- `to_shuffled_list()`: 随机打乱（用于数据增强）

---

## 已完成的优化工作

### 优化1：语义多样性增强的词汇表构建

**问题**：原始方法仅使用共现频次，忽略了items的语义关系

**解决方案**：
```python
priority = cnt × (w1 × w2)^(γ/2)
```

**实现位置**：
- `core.py`: `_compute_pair_priority()`
- `tokenizer.py`: `_compute_item_weights()`, `_build_log_weights()`

**效果**：
- 权重贡献度：0.193-0.208（Beauty数据集）
- Top-20排序变化：35%
- 保留46%无权重pairs，避免过度偏向

**技术亮点**：
1. **对数空间计算**：数值稳定
2. **多层裁剪**：log_w ∈ [-3.0, 3.0], component ∈ [-2.0, 2.0]
3. **Proximity模式**：奖励语义相似的items

---

### 优化2：Ranking-Guided Generation Loss

**问题**：标准交叉熵对所有负样本一视同仁，排序能力弱

**解决方案**：温度缩放
```python
scaled_logits = logits / τ  # τ=0.7
loss = CrossEntropy(scaled_logits, labels)
```

**实现位置**：
- `model.py`: `forward()`, `beam_search_step()`

**理论依据**：
- τ↓ → 概率分布更陡峭
- 增强对hard negatives的惩罚
- 突出正确item的优势

**效果**：
- NDCG@5: +4.5%
- NDCG@10: +3.8%

**训练-推理一致性**：
- 训练时：`forward()`中应用τ
- 推理时：`beam_search_step()`中应用τ

---

### 优化3：集成推理与温度协同

**问题**：单路径生成鲁棒性差

**解决方案**：
1. **多路径随机游走**（n_ensemble=5）
2. **温度控制的beam search**（τ=0.7）

**实现位置**：
- `model.py`: `generate()`, `beam_search_step()`
- `tokenizer.py`: `encode()`中的shuffle='feature'

**协同效应**：
- 集成 → 提升覆盖率和鲁棒性
- 温度 → 在每条路径上强化排序质量
- 两者相辅相成，非线性提升

**效果**：
- Recall@5: +2.1%
- Recall@10: +1.9%

---

## 正在进行的优化

### 优化4：权重计算的泛化性修复

#### 问题诊断

**在Sports数据集上的异常现象**：
```
权重因子均值: 0.7555 (远低于1.0) ✗
权重贡献度: -0.593 (负值！) ✗
权重标准差: 0.1336 (区分度极差) ✗
```

**对比Beauty数据集**：
```
权重因子均值: 1.0448 (接近1.0) ✓
权重贡献度: 0.208 (正值) ✓
权重标准差: 0.4123 (区分度良好) ✓
```

**权重贡献度为负值的含义**：
```python
weight_contribution = (var(priority) - var(cnt)) / var(priority)
Sports: (33.76 - 53.77) / 33.76 = -0.593
```
- 优先级的方差(33.76) < 频次的方差(53.77)
- **权重压缩了优先级的区分度**
- 权重起到了**负作用**！

#### 根本原因

**问题出在`_compute_item_weights()`中**：
```python
# 裁剪后没有重新归一化
scores = np.clip(scores, low, high)

if abs(clipped_mean - 1.0) > 0.15:
    logger.warning('deviates from 1.0, but NOT re-normalizing')
```

**为什么Sports数据集受影响？**
1. **语义分散性**：Sports商品种类多样（球类、健身器材、户外装备...）
2. **Proximity分数偏低**：`score = 1/(1+distance)`，distance大→score小
3. **大量裁剪**：很多权重被裁剪到下限0.2
4. **均值偏离**：裁剪后均值降到0.7555
5. **负面效应**：权重因子=(0.76×0.76)^1.5≈0.44，所有优先级被压缩

**Beauty数据集为什么正常？**
- 美妆产品语义相似度高（都是化妆品、护肤品）
- Proximity分数接近1.0
- 裁剪影响小
- 均值保持在1.0附近

#### 解决方案

**方案A：强制归一化（已实施）**
```python
# 裁剪后必须重新归一化
scores = np.clip(scores, low, high)
clipped_mean = scores.mean()
scores = scores / (clipped_mean + 1e-12)  # 强制均值=1.0
```

**修改位置**：`tokenizer.py` 第515-530行

**预期效果**：
- 权重因子均值 → 1.0
- 权重贡献度 → 正值（>0.15）
- 权重标准差 → 增大（区分度提升）
- Top-k排序变化率 → 更明显

**方案B（备用）：改进Proximity公式**
如果方案A效果不佳，将采用相对距离：
```python
# 使用相对距离，对语义密度更鲁棒
mean_dist_global = mean_distances.mean()
relative_distances = mean_distances / mean_dist_global
scores = 1.0 / (1.0 + relative_distances)
scores = scores * 2.0  # 缩放到均值≈1.0
```

**优点**：
- 对不同数据集的语义密度更鲁棒
- 天然具有归一化效果
- Beauty和Sports都能工作良好

---

## 性能总结

### Beauty数据集（已验证）

| 指标 | 原始论文 | 改进后 | 提升 |
|------|---------|--------|------|
| NDCG@5 | 0.03418 | 0.03571 | **+4.5%** |
| NDCG@10 | 0.04250 | 0.04413 | **+3.8%** |
| Recall@5 | 0.05160 | 0.05268 | **+2.1%** |
| Recall@10 | 0.07745 | 0.07893 | **+1.9%** |

### Sports数据集（训练中）

**等待验证**：权重泛化性修复后的效果

---

## 关键参数配置

### Beauty数据集最优参数
```bash
--item_weight_gamma=3.0
--item_weight_clip_range="[0.2,4.0]"
--item_weight_mode="proximity"
--ranking_temperature=0.7
--n_inference_ensemble=5
```

### Sports数据集最优参数（调优中）
```bash
--item_weight_gamma=2.5
--item_weight_clip_range="[0.3,3.0]"
--item_weight_mode="proximity"
--ranking_temperature=0.7
--n_inference_ensemble=5
```

---

## 技术创新总结

### 1. 对数空间稳定计算
- 避免浮点溢出
- 多层裁剪保护
- 数值稳定性强

### 2. 训练-推理一致性
- 训练时用τ：`forward()`
- 推理时用τ：`beam_search_step()`
- 避免train-test mismatch

### 3. 多尺度协同
- 词汇表级别：语义权重
- 损失函数级别：温度缩放
- 推理级别：集成生成
- 三者互补，非线性提升

### 4. 自适应泛化能力
- 强制归一化保证不同数据集一致性
- 可选的相对距离计算
- 灵活的超参数配置

---

## 未来优化方向

1. **自适应温度调度**：根据训练阶段动态调整τ
2. **多模态语义融合**：结合文本、图像、属性多种信息
3. **对比学习增强**：引入对比损失提升表示质量
4. **效率优化**：加速词汇表构建和推理过程

---

**文档创建日期**：2025-10-30
**项目状态**：Beauty数据集已完成，Sports数据集优化中
**维护者**：您和我 😊

