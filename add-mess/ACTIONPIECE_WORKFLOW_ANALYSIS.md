# ActionPiece原始工作流程深度分析

## 核心问题回答

### Q1: 训练时使用基础token还是BPE token？
**答案：BPE token（合并后的高阶token）**

### Q2: 生成时生成基础token还是BPE token？
**答案：BPE token**

### Q3: 如何将生成的token解码还原？
**答案：通过`decode_single_state()`递归解码BPE token为基础特征**

---

## 完整流程详解

### 阶段1：数据准备 - 基础特征token化

```python
# tokenizer.py: _tokenize_once()
def _tokenize_once(self, item_seq):
    state_seq = []
    for item in item_seq:
        feats = self.item2feat[item]  # 例如: (12, 45, 78, 91, 103)
        tokenized_feats = []
        for i, feat in enumerate(feats):
            # 将每个特征转换为基础token ID
            tokenized_feats.append(self.actionpiece.rank[(i, feat)])
        state_seq.append(tokenized_feats)  # 例如: [15, 23, 45, 89, 102]
    return np.array(state_seq)
```

**结果**：每个item被转换为 `[基础token1, 基础token2, ..., 基础tokenN]`
- 例如：`[15, 23, 45, 89, 102]`（5个特征 = 5个基础token）

---

### 阶段2：训练 - BPE合并编码

#### 2.1 输入序列（input_ids）的编码

```python
# tokenizer.py: collate_fn_train()
seq = data['state_seq'][:-1]  # 历史序列
input_ids.append(
    [self.bos_token] 
    + self.actionpiece.encode(seq, shuffle=self.train_shuffle)  # BPE合并！
    + [self.eos_token]
)
```

**`actionpiece.encode()` 做了什么？**
```python
# core.py: encode()
def encode(self, state_seq, shuffle='feature'):
    # 1. 将基础token序列构建成链表
    # 2. 迭代查找最高优先级的token对
    # 3. 合并这些token对为新的高阶token
    # 4. 重复直到无法合并
    # 返回：BPE合并后的token序列
```

**示例**：
```
输入：[[15, 23, 45, 89, 102], [20, 31, 45, 88, 100]]  # 2个item，各5个基础token
BPE合并规则：
  - (15, 23) → 234
  - (45, 89) → 567
  - (234, 567) → 890
输出：[890, 102, 20, 31, 45, 88, 100]  # 压缩后的序列
```

#### 2.2 标签（labels）的编码

```python
# tokenizer.py: collate_fn_train()
lb = data['state_seq'][-1:]  # 目标item
labels.append(self.encode_labels(lb) + [self.eos_token])

# encode_labels() 调用：
def encode_labels(self, labels):
    # 使用 actionpiece.encode(labels, shuffle='none')
    # 同样进行BPE合并！
    encoded_labels = self.actionpiece.encode(labels, shuffle='none')
    return encoded_labels
```

**关键点**：训练标签也是**BPE合并后的token**！

**示例**：
```
目标item特征：[15, 23, 45, 89, 102]
BPE合并后的标签：[234, 567, EOS]  # 可能只剩2个token
```

---

### 阶段3：模型训练

```python
# model.py: forward()
outputs = self.t5(input_ids=input_ids, labels=labels)
loss = outputs.loss  # 标准交叉熵损失
```

**模型学习的目标**：
- 输入：BPE合并后的历史序列
- 输出：BPE合并后的目标token序列
- 词汇表大小：包含所有基础token + 所有合并规则生成的高阶token

---

### 阶段4：验证/测试 - 标签是基础token

**重要发现**：验证时标签**不进行BPE合并**！

```python
# tokenizer.py: collate_fn_val()
lb = data['state_seq'][-1]  # 注意：这里是 [-1] 而不是 [-1:]
labels.append(lb)  # 直接使用，不调用encode_labels()！

# 返回的labels是什么？
return {
    'labels': torch.LongTensor(np.array(labels))  # shape: (batch, n_categories)
}
```

**`labels` 的内容**：
- `lb` 来自 `_tokenize_once()` 的输出
- 是一个长度为 `n_categories` 的数组，包含**基础token ID**
- 例如：`[15, 23, 45, 89, 102]`

---

### 阶段5：生成 - 输出BPE token

```python
# model.py: generate()
outputs = self.beam_search(...)  # shape: (batch*beams, seq_len)
# 生成的是BPE token序列

# 示例生成结果：
# [234, 567, EOS]  # 2个BPE token + EOS
```

---

### 阶段6：解码 - BPE token → 基础特征

```python
# model.py: generate()
for output in outputs.cpu()[:, 1:].tolist():
    # output 可能是: [234, 567, EOS] 或 [234, 567, 102, EOS]
    
    if self.tokenizer.eos_token in output:
        idx = output.index(self.tokenizer.eos_token)
        output = output[:idx]  # 去掉EOS: [234, 567] 或 [234, 567, 102]
    else:
        output = output[:self.tokenizer.actionpiece.n_categories]
    
    # 关键：解码BPE token为基础特征
    decoded_output = self.tokenizer.actionpiece.decode_single_state(output)
```

**`decode_single_state()` 做了什么？**

```python
# core.py: decode_single_state()
def decode_single_state(self, token_seq):
    """将BPE token序列解码为基础特征"""
    cur_state = {}
    for token in token_seq:  # 遍历每个BPE token
        feats = self._decode_single_token(token)  # 递归解码
        for pos, f in feats:
            if pos in cur_state:
                return None  # 冲突，解码失败
            cur_state[pos] = f
    
    # 检查是否覆盖所有category
    for i in range(self.n_categories):
        if i not in cur_state:
            return None  # 缺失category，解码失败
    
    # 返回基础特征：[(0, 12), (1, 45), (2, 78), (3, 91), (4, 103)]
    return [(i, cur_state[i]) for i in range(self.n_categories)]

def _decode_single_token(self, token):
    """递归解码单个token"""
    decoded = self.vocab[token]
    if decoded[0] == -1:  # 这是一个合并token
        # decoded = (-1, token1_id, token2_id)
        # 递归解码子token
        return self._decode_single_token(decoded[1]) + self._decode_single_token(decoded[2])
    else:
        # 这是基础token: (category_idx, feature_idx)
        return [decoded]
```

**解码示例**：
```
输入BPE序列：[234, 567, 102]

解码过程：
1. 解码 234:
   - vocab[234] = (-1, 15, 23)  # 合并token
   - 递归解码 15: vocab[15] = (0, 12) → 基础特征 (category=0, feature=12)
   - 递归解码 23: vocab[23] = (1, 45) → 基础特征 (category=1, feature=45)
   - 结果：[(0, 12), (1, 45)]

2. 解码 567:
   - vocab[567] = (-1, 45, 89)
   - 递归解码 45: vocab[45] = (2, 78) → (category=2, feature=78)
   - 递归解码 89: vocab[89] = (3, 91) → (category=3, feature=91)
   - 结果：[(2, 78), (3, 91)]

3. 解码 102:
   - vocab[102] = (4, 103) → 基础特征 (category=4, feature=103)
   - 结果：[(4, 103)]

合并所有结果：
cur_state = {0: 12, 1: 45, 2: 78, 3: 91, 4: 103}
最终输出：[(0, 12), (1, 45), (2, 78), (3, 91), (4, 103)]
```

---

### 阶段7：评估 - 将解码结果转换为token ID

```python
# model.py: generate()
if decoded_output is None:
    decoded_outputs.append([-1] * n_categories)  # 解码失败
else:
    # 将基础特征转换回token ID
    decoded_outputs.append(
        [self.tokenizer.actionpiece.rank[_] for _ in decoded_output]
    )
    # 例如：[(0,12), (1,45), ...] → [15, 23, 45, 89, 102]
```

**与验证标签比较**：
```python
# 预测：[15, 23, 45, 89, 102]  # 解码后的基础token
# 标签：[15, 23, 45, 89, 102]  # 验证时的基础token
# 如果完全匹配 → 预测正确！
```

---

## 总结：原始ActionPiece的工作流程

```
数据准备：
  item特征 → 基础token [15, 23, 45, 89, 102]
              ↓
训练阶段：
  输入: 历史序列 → BPE合并 → [890, 102, 20, ...]
  标签: 目标item → BPE合并 → [234, 567, EOS]
  模型学习: 预测BPE token序列
              ↓
验证/测试阶段：
  输入: 历史序列 → BPE合并 → [890, 102, 20, ...]
  标签: 目标item → 基础token → [15, 23, 45, 89, 102]
              ↓
生成阶段：
  模型生成 → BPE token [234, 567, 102]
              ↓
解码阶段：
  decode_single_state() → 递归解码BPE token
  [234, 567, 102] → [(0,12), (1,45), (2,78), (3,91), (4,103)]
                  → [15, 23, 45, 89, 102]  # 基础token
              ↓
评估阶段：
  预测: [15, 23, 45, 89, 102]  # 解码后的基础token
  标签: [15, 23, 45, 89, 102]  # 原始基础token
  比较 → 计算NDCG、Recall等指标
```

---

## 关键设计原理

### 为什么训练用BPE token？

1. **序列压缩**：5个基础token可能压缩为2-3个BPE token
2. **上下文建模**：BPE合并捕获了特征之间的共现模式
3. **效率提升**：更短的序列 → 更快的训练和推理

### 为什么验证标签用基础token？

1. **评估公平性**：需要在item级别（而非token级别）评估
2. **唯一映射**：每个item有唯一的基础特征表示
3. **可解释性**：基础特征直接对应item的属性

### 为什么解码能成功？

**关键假设**：模型生成的BPE token序列能够**完整还原**为基础特征

**成功条件**：
1. 生成的BPE token能递归解码
2. 解码后覆盖所有 `n_categories` 个category
3. 每个category只出现一次（无冲突）

**失败情况**：
- 生成了无效的token ID
- 解码后有category缺失
- 解码后有category冲突（同一category出现多个值）

---

## 为什么MSL与原始设计冲突？

### MSL的假设
Trie树约束模型只生成"训练集中出现过的物品序列"

### 冲突点
如果Trie存储**基础token**，但训练标签是**BPE token**：
- 训练时：标签 `[234, 567, EOS]` 不在Trie的基础token中
- 导致：constrain_mask标记标签为无效
- 结果：模型无法学习

### 两种解决方案对比

#### 方案A：禁用MSL（已实施）
- ✅ 简单快速
- ✅ 保持原始BPE设计
- ✅ 利用BPE压缩优势
- ✅ 与原始论文一致

#### 方案B：统一使用基础token（复杂）
- ❌ 需要大量修改
- ❌ 失去BPE压缩优势
- ❌ 序列更长，内存占用更大
- ❌ 可能需要重新调整模型架构

---

## 推荐方案

**保持现状（禁用MSL）**，因为：
1. 原始ActionPiece设计已经很优秀
2. BPE编码捕获了特征共现模式
3. 解码机制确保生成合法item
4. 性能和效率都很好

如果确实需要MSL约束，应该：
- Trie也存储BPE token（而不是基础token）
- 但这样会失去MSL的可解释性

