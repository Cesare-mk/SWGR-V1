# MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender

![Static Badge](https://img.shields.io/badge/Conference-SIGIR2025-FF8C00?label=Conference)

This is the PyTorch implementation for our SIGIR 2025 paper. 
> Bohao Wang, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Yan Feng, Chun Chen, Can Wang 2025. MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender. [arXiv link](https://arxiv.org/abs/2504.04178)

## 🔥Update!
We identified a potential inconsistency between **Constrained Beam Search (CBS)** and **MSL** when directly using the CBS implementation from HuggingFace Transformers. To address this, we have provided our own customized implementation of CBS. We found that integrating this customized CBS leads to stronger MSL performance, even when using a simple fixed temperature. Special thanks to @yuyq18 for the valuable suggestion. For more details, please refer to [this issue](https://github.com/WANGBohaO-jpg/MSL/issues/5).

## Environment
- torch==2.6.0
- transformers==4.49.0
- marisa-trie==1.2.1
- [genre](https://github.com/facebookresearch/GENRE)

## Dataset
We follow the sliding window data processing method used in [BIGRec](https://arxiv.org/abs/2308.08434). In the paper, for LLM-based recommendation, we sample 10,000 training instances (train_10000.csv) due to limitations in computational resources, while for LLM-enhanced recommendation and traditional recommendation models, we use the full dataset (train.csv).

## Scripts
### Training&Inference using MSL (🔥Updated)
**We recommend everyone use this updated code and corresponding scripts when working with MSL.**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_msl.py --dataset_name Toy --sample 10000 --num_epochs 10
```

```
# Replace `lora_weights_path` with the path to the model checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch inference.py --lora_weights_path "xxx" --dataset Toy --constrained_before_softmax True
```

Below are the latest results after the modification of CBS: 
| Dataset  | Method         | NDCG@5  | NDCG@10 | HIT@5   | HIT@10   |
|----------|----------------|---------|---------|---------|----------|
| Toy      | LML + CBS        | 0.0138  | 0.0182  | 0.0213  | 0.0353   |
|          | MSL + CBS (ATS)  | 0.0245  | 0.0288  | 0.0357  | 0.0488   |
|          | MSL + Customized CBS | 0.0253  | 0.0313  | 0.0386  | 0.0569   |
| Clothing | LML + CBS        | 0.0047  | 0.0073  | 0.0092  | 0.0174   |
|          | MSL + CBS (ATS)  | 0.0091  | 0.0120  | 0.0146  | 0.0236   |
|          | MSL + Customized CBS | 0.0091  | 0.0131  | 0.0166  | 0.0292   |
| Book     | LML + CBS        | 0.0109  | 0.0137  | 0.0169  | 0.0258   |
|          | MSL + CBS (ATS)  | 0.0125  | 0.0172  | 0.0214  | 0.0356   |
|          | MSL + Customized CBS | 0.0106  | 0.0138  | 0.0171  | 0.0269   |


Here, **CBS** refers to the constrained beam search implemented in the Transformers library (which masks out invalid tokens by setting their probabilities to zero after softmax). **Customized CBS** is our modified version for MSL (which sets the logits of invalid tokens to `-inf` before softmax, so their probabilities become zero automatically).

### Training&Inference using MSL+ATS
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_msl_ats.py --dataset_name Toy --sample 10000 --num_epochs 10
```
```
# Replace `lora_weights_path` with the path to the model checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch inference.py --lora_weights_path "xxx" --dataset Toy
```

### Training&Inference using LML
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_lml.py --dataset_name Toy --sample 10000 --num_epochs 10
```
```
# Replace `lora_weights_path` with the path to the model checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch inference.py --lora_weights_path "xxx" --dataset Toy
```

### Evaluate
Replace `lora_weights_path` with the path to the parent directory of the model checkpoint, for example: `./save_lora_model/Toy/sample10000_epoch10_CT_tau1/0/`.
```
python evaluate_batch_match.py --lora_weights_father_path "xxx"
```

## Citation
If you find the paper useful in your research, please consider citing:
```
@misc{wang2025msltokensneedtuning,
      title={MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender}, 
      author={Bohao Wang and Feng Liu and Jiawei Chen and Xingyu Lou and Changwang Zhang and Jun Wang and Yuegang Sun and Yan Feng and Chun Chen and Can Wang},
      year={2025},
      eprint={2504.04178},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2504.04178}, 
}
```
