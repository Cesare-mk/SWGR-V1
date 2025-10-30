# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ActionPiece tokenizer for GenRec."""

import collections
import json
import os
from typing import Any, Tuple
from functools import partial  # 添加 functools.partial 导入
import faiss
from genrec.dataset import AbstractDataset
from genrec.models.ActionPiece.core import ActionPieceCore
from genrec.tokenizer import AbstractTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch


class ActionPieceTokenizer(AbstractTokenizer):
    """The ActionPiece tokenizer is a tokenizer that encodes the attribute.

    features.

    Attributes:
        item2feat (dict): A dictionary mapping item IDs to their features.
        ignored_label (int): The label to be used for padding tokens.
        actionpiece (ActionPieceCore): ActionPiece core tokenizer.
        bos_token (int): The beginning token.
        eos_token (int): The end token.
        #n_prob_encode_plus (int): The number of probability encoding plus.
        n_inference_ensemble (int): The number of inference ensemble.
        train_shuffle (str): The shuffle strategy for training.
        encoded_labels (dict): A dictionary mapping label sequences to their
          encoded representations.
        collate_fn (dict): A dictionary mapping split names to their corresponding
          collate functions.
    """

    def __init__(self, config: dict[Any, Any], dataset: AbstractDataset):
        super().__init__(config)
        self.item2feat = None
        self.ignored_label = -100

        # === 读取所有可配置的权重相关参数 ===
        _clip = config['item_weight_clip_range']
        if not isinstance(_clip, (list, tuple)) or len(_clip) < 2:
            _clip = [0.2, 4.0]
        self.item_weight_clip_range = (float(_clip[0]), float(_clip[1]))
        
        self.item_weight_gamma = float(config.get('item_weight_gamma', 3.0))
        self.weight_analysis_interval = int(config.get('weight_analysis_interval', 4000))
        
        # 语义权重模式: 'proximity' 或 'semantic_diversity'
        self.item_weight_mode = config.get('item_weight_mode', 'proximity')
        
        # 是否启用二次裁剪（对log空间裁剪）
        self.enable_second_clip = bool(config.get('enable_second_clip', False))
        
        # 二次裁剪范围（仅在enable_second_clip=True时使用）
        _log_clip = config.get('log_weight_clip_range', [-1.5, 1.5])
        if not isinstance(_log_clip, (list, tuple)) or len(_log_clip) < 2:
            _log_clip = [-1.5, 1.5]
        self.log_weight_clip_range = (float(_log_clip[0]), float(_log_clip[1]))
        self.actionpiece = self._init_tokenizer(dataset)
        self.bos_token = self.actionpiece.vocab_size
        self.eos_token = self.actionpiece.vocab_size + 1
        self.n_inference_ensemble = config['n_inference_ensemble']
        self.train_shuffle = config['train_shuffle']
        self.encoded_labels = {}
        self.collate_fn = {
            'train': self.collate_fn_train,
            'val': self.collate_fn_val,
            'test': self.collate_fn_test,
        }

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """Encodes the sentence embeddings for the given dataset and saves them to the specified output path.
        Args:
            dataset: The dataset containing the sentences to encode.
            dataset (AbstractDataset): The dataset containing the sentences to
              encode.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        # 元数据来源校验，默认使用 'sentence'
        assert self.config['metadata'] in ['sentence', 'all']

        sent_emb_model = SentenceTransformer(self.config['sent_emb_model'], cache_folder='/root/shared-nvme/.cache').to(
            self.config['device']
        )

        meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
        item2meta = None
        if 'sentence' in dataset.item2meta:
            item2meta = dataset.item2meta['sentence']
        else:
            item2meta = dataset.item2meta
        for i in range(1, dataset.n_items):
            meta_sentences.append(item2meta[dataset.id_mapping['id2item'][i]])
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
        )
        # PCA
        if self.config['sent_emb_pca'] > 0:
            self.logger.info('[TOKENIZER] Applying PCA to sentence embeddings...')
            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            # 128 dim
            sent_embs = pca.fit_transform(sent_embs)

        sent_embs.tofile(output_path)
        return sent_embs

    def _get_sent_embs(self, dataset: AbstractDataset) -> np.ndarray:
        sent_emb_path = os.path.join(
            dataset.cache_dir,
            'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb',
        )
        sent_emb_dim = (
            self.config['sent_emb_dim']
            if self.config['sent_emb_pca'] <= 0
            else self.config['sent_emb_pca']
        )
        if os.path.exists(sent_emb_path):
            self.logger.info(
                f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...'
            )
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
                -1, sent_emb_dim
            )
        else:
            self.logger.info('[TOKENIZER] Encoding sentence embeddings...')
            sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
        self.logger.info(
            f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}'
        )
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.logger.info(
            f'[TOKENIZER] Items for training: {len(items_for_training)}'
            f' of {dataset.n_items - 1}'
        )
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _sent_emb_to_sem_id(
            self, dataset: AbstractDataset, sent_embs: np.ndarray) -> dict[Any, Any]:
        # Get the sentence embeddings for training
        training_item_mask = self._get_items_for_training(dataset)
        embs_for_training = sent_embs[training_item_mask]

        # Train the index
        # Take the vector quantized codes as item features

        faiss.omp_set_num_threads(self.config['n_threads'])
        index = faiss.index_factory(
            sent_embs.shape[-1],
            f"OPQ{self.config['pq_n_codebooks']},IVF1,PQ{self.config['pq_n_codebooks']}x{int(np.log2(self.config['pq_codebook_size']))}",
            faiss.METRIC_INNER_PRODUCT,
        )
        self.logger.info('[TOKENIZER] Training index...')
        index.train(embs_for_training)
        index.add(sent_embs)

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        sem_ids = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        sem_ids = sem_ids.reshape(-1, invlists.code_size)

        # Convert semantic IDs to a dictionary
        item2sem_ids = {}
        for i in range(sem_ids.shape[0]):
            item = dataset.id_mapping['id2item'][i + 1]
            item2sem_ids[item] = tuple(sem_ids[i].tolist())
        return item2sem_ids

    def _get_sem_ids(self, dataset: AbstractDataset) -> dict[Any, Any]:
        """Get the semantic IDs from the dataset.
        If the semantic IDs are already cached, load them from the cache. Otherwise,
        generate the semantic IDs and save them to the cache.

        Args:
            dataset (AbstractDataset): The dataset containing the items to get
              semantic IDs for.

        Returns:
            item2sem_ids (dict): A dictionary mapping item IDs to their semantic
            IDs.
        """
        sem_ids_path = os.path.join(
            dataset.cache_dir,
            'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}.sem_ids',
        )
        if not os.path.exists(sem_ids_path):
            self.logger.info(
                '[TOKENIZER] Semantic IDs not found. Training index using Faiss...'
            )
            sent_embs = self._get_sent_embs(dataset)
            item2sem_ids = self._sent_emb_to_sem_id(dataset, sent_embs)
            # Save semantic IDs
            self.logger.info(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
            with open(sem_ids_path, 'w') as f:
                json.dump(item2sem_ids, f)
            return item2sem_ids
        else:
            self.logger.info(
                f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...'
            )
            with open(sem_ids_path, 'r') as f:
                item2sem_ids = json.load(f)
            return {
                k: v[: self.config['pq_n_codebooks']] for k, v in item2sem_ids.items()
            }

    def _get_attr_ids(self, dataset: AbstractDataset):
        if 'attr_id' in dataset.item2meta:
            return dataset.item2meta['attr_id']
        else:
            return None

    def _combine_features(
            self, item2sem_ids: dict[Any, Any], item2attr_ids: dict[Any, Any]
    ) -> dict[Any, Any]:
        """Combine the semantic IDs and attribute IDs to get the item features."""
        if item2attr_ids is None:
            return item2sem_ids
        item2feat = {}
        for item in item2sem_ids:
            feat = item2sem_ids[item] + item2attr_ids[item]
            item2feat[item] = feat
        return item2feat

    def _get_hashed_feat(
            self, dataset: AbstractDataset, item2feat: dict[Any, Any]
    ) -> dict[Any, Any]:
        """Add one digit of the original features as the hash buckets.

        The purpose is to avoid conflicts.

        Args:
            dataset (AbstractDataset): The dataset containing the items (not used).
            item2feat (dict): A dictionary mapping item IDs to their features.

        Returns:
            dict: A dictionary mapping item IDs to their hashed features.
        """
        feat2cnt = collections.defaultdict(int)
        feat2hash_ids = {}
        item2hashed_feat = {}

        for item in item2feat:
            feat = tuple(item2feat[item])
            if feat not in feat2hash_ids:
                feat2hash_ids[feat] = np.random.permutation(
                    self.config['n_hash_buckets']
                )
            idx = feat2cnt[feat]

            if idx >= self.config['n_hash_buckets']:
                raise ValueError(
                    '[TOKENIZER] Too many conflicts of semantic IDs found.'
                    ' Please increase the number of hash buckets.'
                )

            item2hashed_feat[item] = (
                *item2feat[item],
                feat2hash_ids[feat][idx].item(),
            )
            feat2cnt[feat] += 1

        return item2hashed_feat

    def _get_item2feat(self, dataset: AbstractDataset) -> dict[Any, Any]:
        """Get the item features.

        If the features are already cached, load them from the cache. Otherwise,
        generate the features and save them to the cache.

        Args:
            dataset (AbstractDataset): The dataset containing the items to get
              features for.

        Returns:
            item2feat (dict): A dictionary mapping item IDs to their features.
        """
        feat_path = os.path.join(
            dataset.cache_dir,
            f'processed/item.{self.config["metadata"]}.feat',
        )
        if os.path.exists(feat_path):
            self.logger.info(f'[TOKENIZER] Loading item features from {feat_path}...')
            with open(feat_path, 'r') as f:
                item2feat = json.load(f)
            return item2feat
        self.logger.info('[TOKENIZER] Generating item features...')
        item2sem_ids = self._get_sem_ids(dataset)
        item2attr_ids = self._get_attr_ids(dataset)
        item2feat = self._combine_features(item2sem_ids, item2attr_ids)
        item2hashed_feat = self._get_hashed_feat(dataset, item2feat)
        self.logger.info(f'[TOKENIZER] Saving item features to {feat_path}...')
        with open(feat_path, 'w') as f:
            json.dump(item2hashed_feat, f)
        return item2hashed_feat

    def _check_conflicts(self, item2feat: dict[Any, Any]):
        """Check if there are any conflicts in the item features.

        If there are conflicts, raise a ValueError.

        Args:
            item2feat (dict): A dictionary mapping item IDs to their features.
        """
        feat2cnt = collections.Counter()
        for feat in item2feat.values():
            feat = tuple(feat)
            feat2cnt[feat] += 1
        for feat in feat2cnt:
            if feat2cnt[feat] > 1:
                raise ValueError(f'[TOKENIZER] Conflicts found in features: {feat}')

    def _tokenize_once(self, item_seq):
        state_seq = []
        for item in item_seq:
            feats = self.item2feat[item]
            tokenized_feats = []
            for i, feat in enumerate(feats):
                tokenized_feats.append(self.actionpiece.rank[(i, feat)])
            state_seq.append(tokenized_feats)
        return np.array(state_seq)

    def tokenize_function(
            self, example: dict[Any, Any], split: str
    ) -> dict[Any, Any]:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            # Tokenize each subsequence with maximum `max_item_seq_len` items
            n_return_examples = len(item_seq) - 1
            all_state_seq = []
            for i in range(n_return_examples):
                cur_item_seq = self._tokenize_once(
                    item_seq[max(i + 1 - max_item_seq_len, 0): i + 2]
                )
                all_state_seq.append(cur_item_seq)
        else:
            all_state_seq = [self._tokenize_once(item_seq[-(max_item_seq_len + 1):])]
        return {
            'state_seq': all_state_seq,
        }

    def tokenize(self, datasets: dict[Any, Any]) -> dict[Any, Any]:
        tokenized_datasets = {}

        # Create a wrapper function to avoid serialization issues
        # def tokenize_wrapper(example, split):
        #     return self.tokenize_function(example, split)

        for split in datasets:
            # 使用 functools.partial 替代闭包，确保函数可以被 pickle 序列化
            # 这样 datasets 库可以正确生成缓存指纹，避免警告
            tokenize_func = partial(self.tokenize_function, split=split)

            tokenized_datasets[split] = datasets[split].map(
                tokenize_func,
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set',
                load_from_cache_file=True,
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='numpy')
        return tokenized_datasets

    @property
    def vocab_size(self):
        return self.eos_token + 1

    @property
    def max_token_seq_len(self):
        # +2 for EOS and BOS
        return self.actionpiece.n_categories * self.config['max_item_seq_len'] + 2

    def _init_tokenizer(self, dataset: AbstractDataset):
        # Load item features first before checking conflicts
        self.item2feat = self._get_item2feat(dataset)
        self._check_conflicts(self.item2feat)
        tokenizer_path = os.path.join(dataset.cache_dir, 'processed/actionpiece.json')
        if os.path.exists(tokenizer_path):
            # If trained tokenizer exists, load it directly
            self.logger.info(
                f'[TOKENIZER] Loading pre-trained ActionPiece from {tokenizer_path}...')
            actionpiece = ActionPieceCore.from_pretrained(
                tokenizer_path, vocab_size=self.config['actionpiece_vocab_size'])
        else:
            # Initialize ActionPiece from initial features and train with weights
            self.logger.info('[TOKENIZER] No existing vocabulary found - will train new ActionPiece...')
            actionpiece = ActionPieceCore(state2feat=self.item2feat)
            
            weight_file_path = os.path.join(
                dataset.cache_dir,
                'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb.weights.{self.item_weight_mode}.json'
            )
            if os.path.exists(weight_file_path):
                self.logger.info(
                    f'[TOKENIZER] Loading weights for vocabulary training...')
                actionpiece.load_item_weights(
                    weight_file_path, gamma=self.item_weight_gamma)

                if hasattr(actionpiece, 'log_w') and actionpiece.log_w is not None:
                    non_zero_weights = sum(1 for w in actionpiece.log_w if abs(w) > 1e-6)
                    self.logger.info(
                        f'[TOKENIZER] Weights loaded successfully: {non_zero_weights}/{len(actionpiece.log_w)} non-zero weights')
                else:
                    self.logger.warning(f'[TOKENIZER] Weight loading failed')
            else:
                self.logger.warning(f'[TOKENIZER] No weight file available, computing weights...')
                weight_file_path = self._compute_item_weights(dataset, self._get_sent_embs(dataset))
                actionpiece.load_item_weights(weight_file_path, gamma=self.item_weight_gamma)

            self.logger.info('[TOKENIZER] Training ActionPiece vocabulary with weighted token pairs...')
            actionpiece.train(
                state_corpus=dataset.split_data['train']['item_seq'],
                target_vocab_size=self.config['actionpiece_vocab_size'],
                weight_analysis_interval=self.weight_analysis_interval,
            )
            actionpiece.save(tokenizer_path)
            self.logger.info(f'[TOKENIZER] Saved trained vocabulary to {tokenizer_path}')
        return actionpiece

    def _compute_item_weights(self, dataset: Any, sent_embs: np.ndarray):
        """
        计算 item 权重

        支持两种模式：
        - 'semantic_diversity': 距离越远分数越高（鼓励多样性）
        - 'proximity': 距离越近分数越高（鼓励相似性）
        """
        mode = self.item_weight_mode
        self.logger.info(
            f'[TOKENIZER] Computing item weights based on mode={mode}...')

        if sent_embs is None or not isinstance(sent_embs, np.ndarray):
            raise ValueError('sent_embs must be numpy.ndarray')
        if sent_embs.ndim != 2:
            raise ValueError('sent_embs shape is  [V, d]')

        V, d = sent_embs.shape
        expected_items = len(dataset.id_mapping['id2item']) - 1
        if V != expected_items:
            self.logger.warning(
                f'[TOKENIZER] sent_embs lines({V}) except item number({expected_items}) not matched')

        # 使用 faiss 计算 k 近邻距离
        index = faiss.IndexFlatL2(d)
        index.add(sent_embs.astype(np.float32))
        k = min(10, V)
        distances, indices = index.search(sent_embs.astype(np.float32), k)

        # 计算平均距离（排除自身，即第0个邻居）
        if k > 1:
            mean_distances = np.mean(distances[:, 1:], axis=1)
        else:
            mean_distances = np.ones(V, dtype=np.float32)

        if mode == 'proximity':
            # 【改进版】使用相对距离，对不同数据集的语义密度更鲁棒
            #
            # 原问题：绝对距离在不同数据集上差异大
            # - Beauty: 语义密集 → distance小 → score接近1.0
            # - Sports: 语义分散 → distance大 → score偏小 → 归一化后区分度丢失
            #
            # 改进方案：
            # 1. 计算全局平均距离作为基准
            # 2. 使用相对距离 = distance / global_mean
            # 3. 这样不同数据集的相对距离分布更一致

            global_mean_dist = mean_distances.mean()
            self.logger.info(f'[TOKENIZER] Global mean distance: {global_mean_dist:.4f}')

            # 使用相对距离
            relative_distances = mean_distances / (global_mean_dist + 1e-12)

            # 应用倒数映射
            # relative_distance ≈ 1.0 时（平均水平），score ≈ 0.5
            # relative_distance ≈ 0.5 时（很近），score ≈ 0.67
            # relative_distance ≈ 2.0 时（很远），score ≈ 0.33
            scores = 1.0 / (1.0 + relative_distances)

            # 缩放到均值≈1.0的范围
            # 因为相对距离均值≈1.0，所以scores均值≈0.5
            # 需要乘以2.0使其均值接近1.0
            scores = scores * 2.0

            self.logger.info(
                f'[TOKENIZER] Using PROXIMITY mode with RELATIVE distance '
                f'(before scaling: mean={scores.mean()/2.0:.4f}, std={(scores/2.0).std():.4f})'
            )
            self.logger.info(f'[TOKENIZER] After 2x scaling: mean={scores.mean():.4f}, std={scores.std():.4f}')

        else:  # semantic_diversity
            # 距离越远分数越高：直接使用距离
            scores = mean_distances
            self.logger.info(f'[TOKENIZER] Using SEMANTIC_DIVERSITY mode (larger distance → higher score)')

        # 归一化到均值为 1.0
        mean = scores.mean()
        if mean <= 1e-8:
            self.logger.warning('Weight mean is close to zero. Using default weight 1.0.')
            scores = np.ones_like(scores, dtype=np.float32)
        else:
            scores = scores / (mean + 1e-12)

        # 第一次裁剪到配置区间
        low, high = self.item_weight_clip_range
        scores = np.clip(scores, low, high).astype(np.float32)
        
        # 裁剪后必须重新归一化，保证均值为1.0
        # 这对于不同数据集的泛化至关重要
        clipped_mean_before = scores.mean()
        scores = scores / (clipped_mean_before + 1e-12)
        clipped_mean_after = scores.mean()

        if abs(clipped_mean_before - 1.0) > 0.15:
            self.logger.info(
                f'[TOKENIZER] Weight renormalized after clipping: '
                f'{clipped_mean_before:.4f} → {clipped_mean_after:.4f}'
            )
        
        # 可选的二次裁剪（在对数空间进行裁剪）
        if self.enable_second_clip:
            self.logger.info(f'[TOKENIZER] Applying second clipping in log space: {self.log_weight_clip_range}')
            log_scores = np.log(scores + 1e-12)
            log_low, log_high = self.log_weight_clip_range
            log_scores = np.clip(log_scores, log_low, log_high)
            scores = np.exp(log_scores).astype(np.float32)
            
            # 记录二次裁剪后的统计，不再强制归一化
            second_mean = scores.mean()
            self.logger.info(
                f'[TOKENIZER] After second clipping: mean={second_mean:.4f}, '
                f'std={scores.std():.4f}'
            )

        # 构建权重字典
        item_weights = {}
        id2item = dataset.id_mapping['id2item']
        for i in range(V):
            internal_id = i + 1
            if internal_id in id2item:
                item_key = id2item[internal_id]
            else:
                item_key = internal_id
            item_weights[item_key] = float(scores[i])

        self.logger.info(
            f'[TOKENIZER] Done. mode={mode}, stats -> mean={scores.mean():.4f}, std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}')

        # 保存权重文件
        weight_path = os.path.join(
            dataset.cache_dir,
            "processed",
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb.weights.{mode}.json'
        )
        with open(weight_path, "w", encoding="utf-8") as f:
            json.dump(item_weights, f, ensure_ascii=False)
        self.logger.info(f"[TOKENIZER] Saved item weights to {weight_path}")
        return weight_path

    def encode_labels(self, labels):
        """Cache the encoded labels for faster inference.

        Args:
            labels (np.ndarray): The labels to be encoded.

        Returns:
            encoded_labels (list[int]): The encoded labels.
        """
        # 使用非弃用API
        key = labels.tobytes()
        if key in self.encoded_labels:
            return self.encoded_labels[key]
        encoded_labels = self.actionpiece.encode(labels, shuffle='none')
        self.encoded_labels[key] = encoded_labels
        return encoded_labels

    def collate_fn_train(self, batch):
        """Tokenizing a batch of examples on-the-fly while training.

        Training strategy:
        - Use random shuffling for data augmentation
        - Input: sequence[:-1], Label: sequence[-1]

        Args:
            batch (list): A list of examples. batch['state_seq'] is a state
              sequence. batch['state_seq'][i] is a list of features.

        Returns:
            dict: A dictionary of tensors.
        """
        input_ids = []
        attention_mask = []
        labels = []

        for data in batch:
            seq = data['state_seq'][:-1]  # 输入序列：除最后一个item
            lb = data['state_seq'][-1:]  # 标签：最后一个item

            # 训练时使用随机打乱进行数据增强
            input_ids.append(
                [self.bos_token]
                + self.actionpiece.encode(seq, shuffle=self.train_shuffle)
                + [self.eos_token]
            )
            labels.append(self.encode_labels(lb) + [self.eos_token])

        # 动态padding
        seq_lens = [len(ids) for ids in input_ids]
        max_seq_len = max(seq_lens)

        for i in range(len(batch)):
            # Padding input_ids
            input_ids[i] = input_ids[i] + [self.padding_token] * (
                    max_seq_len - seq_lens[i]
            )
            # Create attention mask
            attention_mask.append(
                [1] * seq_lens[i] + [0] * (max_seq_len - seq_lens[i])
            )
            # Padding labels
            labels[i] = labels[i] + [self.ignored_label] * (
                    self.actionpiece.n_categories + 1 - len(labels[i])
            )

        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(labels),
        }

    def collate_fn_val(self, batch):
        """Tokenizing a batch of examples on-the-fly while evaluating.

        Args:
            batch (list): A list of examples. batch['state_seq'] is a state
              sequence. batch['state_seq'][i] is a list of features.

        Returns:
            dict: A dictionary of tensors.
        """
        input_ids = []
        attention_mask = []
        labels = []
        for data in batch:
            seq = data['state_seq'][:-1]
            lb = data['state_seq'][-1]
            # The labels should always be encoded by encode_plus
            input_ids.append(
                [self.bos_token]
                + self.actionpiece.encode(
                    seq,
                    shuffle='none' if self.n_inference_ensemble == -1 else 'feature',
                )
                + [self.eos_token]
            )
            labels.append(lb)
        seq_lens = [len(ids) for ids in input_ids]
        max_seq_len = max(seq_lens)
        for i, sequence_length in enumerate(seq_lens):
            input_ids[i] = input_ids[i] + [self.padding_token] * (
                    max_seq_len - sequence_length
            )
            attention_mask.append(
                [1] * sequence_length + [0] * (max_seq_len - sequence_length)
            )
        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(np.array(labels)),
        }

    def collate_fn_test(self, batch):
        """Tokenizing a batch of examples on-the-fly while testing.

        Test strategy:
        - No shuffling for deterministic and reproducible results
        - Input: sequence[:-1], Label: sequence[-1]
        - Used for final performance evaluation
        - When n_inference_ensemble == -1: single deterministic encoding
        - When n_inference_ensemble > 0: multiple augmented encodings for ensemble

        Args:
            batch (list): A list of examples. batch['state_seq'] is a state
              sequence. batch['state_seq'][i] is a list of features.

        Returns:
            dict: A dictionary of tensors.
        """
        input_ids = []
        attention_mask = []
        labels = []

        for data in batch:
            seq = data['state_seq'][:-1]
            lb = data['state_seq'][-1]

            if self.n_inference_ensemble == -1:
                input_ids.append(
                    [self.bos_token]
                    + self.actionpiece.encode(seq, shuffle='none')
                    + [self.eos_token]
                )
                labels.append(lb)
            else:
                # 使用集成推理：多次增强编码
                for _ in range(self.n_inference_ensemble):
                    input_ids.append(
                        [self.bos_token]
                        + self.actionpiece.encode(seq, shuffle='feature')
                        + [self.eos_token]
                    )
                # 对于集成推理，每个原始样本对应n_inference_ensemble个增强样本，但标签只需要一份
                labels.append(lb)

        # 动态padding
        seq_lens = [len(ids) for ids in input_ids]
        max_seq_len = max(seq_lens)

        for i, sequence_length in enumerate(seq_lens):
            # Padding input_ids
            input_ids[i] = input_ids[i] + [self.padding_token] * (
                    max_seq_len - sequence_length
            )
            # Create attention mask
            attention_mask.append(
                [1] * sequence_length + [0] * (max_seq_len - sequence_length)
            )

        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(np.array(labels)),  # labels是单个item的特征表示
        }
