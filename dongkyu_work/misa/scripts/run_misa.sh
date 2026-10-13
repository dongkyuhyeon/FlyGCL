#!/usr/bin/env bash
# Example only. Requires official MISA pretrained ISA prompt tensors.
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/CIFAR}"
MISA_PROMPT_DIR="${MISA_PROMPT_DIR:-./dongkyu_work/misa/checkpoints}"
python main.py \
  --method misa --dataset cifar100 --data_dir "$DATA_DIR" \
  --n_tasks 5 --n 50 --m 10 --rnd_NM \
  --batchsize 64 --online_iter 3 --num_epochs 1 \
  --lr 0.005 --opt_name adam --sched_name default \
  --eval_period 1000 --transforms autoaug \
  --misa_prompt_dir "$MISA_PROMPT_DIR" \
  --seeds 1 --note misa_cifar100
