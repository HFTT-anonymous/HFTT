## Pytorch version = 1.12.0

### MCM
```python

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 train.py \
        --mode mcm \
        --model clip-base \ # clip-base, clip-large, blip-base, or blip-large
        --ind imagenet \ # path to in-distribution dataset
        --ood OOD \ # path to a directory containing out-distribution datasets.
        --output-dir logs/{output_dir_name}
```
### HFTT (ours)

```python

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 train.py\
        --mode hftt \
        --model clip-base \ # clip-base, clip-large, blip-base, or blip-large
        --ind imagenet \ # path to in-distribution dataset
        --ood OOD \ # path to a directory containing out-distribution datasets.
        --ood-text-path words_alpha.txt \
        --seed 0 \
        --epochs 1 \
        --batch-size 256 \
        --temperature 0.01 \
        --focal 1.0 \
        --num-ood-classes 2000 \
        --lr 1.0 \
        --num-eval-in-an-epoch 10 \
        --num-exp 5 \
        --output-dir logs/{output_dir_name}

```
Our code is based on
1. [PyTorch image classification reference](https://github.com/pytorch/vision/tree/main/references/classification)
2. [CLIP](https://github.com/openai/CLIP)
3. [List Of English Words](https://github.com/dwyl/english-words)
