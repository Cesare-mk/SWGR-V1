# REPRODUCE
# ActionPiece: Contextual Action Tokenization

This repository provides the code for implementing ActionPiece described in our
**ICML 25 Spotlight** paper "[Contextually Tokenizing Action Sequences forGenerative Recommendation](https://arxiv.org/abs/2502.13581)".

## Reproduction

### Sports and Outdoors

Below are the scripts to reproduce the results reported in our paper.

**Original baseline:**
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Sports_and_Outdoors \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.005 \
        --n_hash_buckets=128
done
```

** (improved):**
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Sports_and_Outdoors \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.005 \
        --n_hash_buckets=128 \
        --item_weight_gamma=2.0 \
        --item_weight_clip_range="[0.4,2.5]" \
        --enable_second_clip=True \
        --log_weight_clip_range="[-1.5,1.5]" \
        --ranking_temperature=0.7
done
```
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --category=Sports_and_Outdoors \
    --rand_seed=2026 \
    --weight_decay=0.15 \
    --lr=0.005 \
    --item_weight_gamma=2.5 \
    --item_weight_clip_range="[0.3,3.0]" \
    --enable_second_clip=True \
    --log_weight_clip_range="[-1.5,1.5]" \
    --ranking_temperature=0.7 \
    --n_hash_buckets=128
```


### Beauty

**Original baseline:**
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64
done
```

**With  weights (improved):**
```bash
for seed in {2026..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64 \
        --item_weight_gamma=3.0 \
        --item_weight_clip_range="[0.2,4.0]" \
        --enable_second_clip=False \
        --ranking_temperature=0.7
done
```


### CDs and Vinyl

**Original baseline:**
```bash
for seed in {2024..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=CDs_and_Vinyl \
        --rand_seed=${seed} \
        --weight_decay=0.07 \
        --lr=0.001 \
        --d_model=256 \
        --d_ff=2048 \
        --n_hash_buckets=256 \
        --item_weight_gamma=2.5 \
        --item_weight_clip_range="[0.3,3.5]" \
        --enable_second_clip=False \
        --ranking_temperature=0.7
done
```

## Weight Optimization Parameters

The optimized weight parameters are designed based on the characteristics of each dataset:

### Sports and Outdoors (Functional Products)
- **cooccur_alpha=0.8**: High frequency weight - functional products have clear usage patterns
- **weight_beta=1.8**: Moderate semantic weight - balance between functionality and specialization
- **item_weight_gamma=2.0**: Standard embedding weight fusion
- **item_weight_alpha=0.6**: Favor L2 norm in A1A2 combination - emphasize overall feature strength
- **priority_mixing_mode=log_additive**: Balanced log-space mixing for stable performance

### Beauty (Personalized Products)
- **cooccur_alpha=0.6**: Moderate frequency weight - personalization matters more than popularity
- **weight_beta=2.5**: Higher semantic weight - strong emphasis on individual preferences
- **item_weight_gamma=2.5**: Enhanced embedding influence for fine-grained distinctions
- **item_weight_alpha=0.4**: Favor variance in A1A2 - capture diversity in personal preferences
- **priority_mixing_mode=weighted_geometric**: Geometric weighting for nuanced personalization

### CDs and Vinyl (Cultural Products)
- **cooccur_alpha=0.5**: Balanced frequency weight - support for niche music preferences
- **weight_beta=3.0**: Highest semantic weight - maximum cultural differentiation
- **item_weight_gamma=3.0**: Strong embedding influence to highlight artistic features
- **item_weight_alpha=0.3**: Strong favor for variance - capture musical diversity and uniqueness
- **priority_mixing_mode=adaptive_balance**: Adaptive weighting to handle long-tail distribution

## Locating the code

### Vocabulary construction

* `genrec/models/ActionPiece/core.py` - `ActionPieceCore.train()`

### Segmentation

Set permutation regularization (SPR) during

* **training**: `genrec/models/ActionPiece/tokenizer.py` - `ActionPieceTokenizer.collate_fn_train()`
* **inference**:
    * `genrec/models/ActionPiece/tokenizer.py` - `ActionPieceTokenizer.collate_fn_test()`
    * `genrec/models/ActionPiece/model.py` - `ActionPiece.generate()`

## Tokenizers

After the first run on each dataset, the provided code automatically constructs
and caches the tokenizer vocabulary. The cached tokenizer follows a structure
like:

```json
{
    "n_categories": 5,
    "n_init_feats": 1153,
    "token2feat": [
        [-1, -1], # placeholder
        [0, 0], [0, 1], ...,
        [1, 0], [1, 1], ...,
        [4, 126], [4, 127], # [a, b] denotes the b-th choice of each item's a-th feature
        [-1, 363, 763], [-1, 269, 515], ...,
[-1, 241, 1040], [-1, 30314, 39998] # [-1, u, v] denotes token u and v are merged into a new token
    ],
    "priority": [
        0, 0, ...,
        723.6, 670.04, ...
    ]
}
```

## Citing this work

Please cite the following paper if you find our code, processed datasets, or
tokenizers helpful.

```
@inproceedings{hou2025actionpiece,
  title={{ActionPiece}: Contextually Tokenizing Action Sequences for Generative Recommendation},
  author={Yupeng Hou and Jianmo Ni and Zhankui He and Noveen Sachdeva and Wang-Cheng Kang and Ed H. Chi and Julian McAuley and Derek Zhiyuan Cheng},
  booktitle={{ICML}},
  year={2025}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
