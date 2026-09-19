# CPSC2018 Training Notes for HeartLang

This file is intended for running HeartLang on the second dataset after the ready-to-train CPSC2018 QRS dataset has been prepared and shared externally.

## Expected Dataset Folder

Place the ready-to-train dataset at:

```text
datasets/ecg_datasets/CPSC2018_QRS/data/
```

The folder should contain:

```text
train_data.npy
train_labels.npy
train_data_in_chans.npy
train_data_in_times.npy
val_data.npy
val_labels.npy
val_data_in_chans.npy
val_data_in_times.npy
test_data.npy
test_labels.npy
test_data_in_chans.npy
test_data_in_times.npy
```

No raw-signal preprocessing or QRS tokenization is required if this folder is already present.

## Dataset Summary

| Split | Records |
|---|---:|
| Train | 4,950 |
| Validation | 551 |
| Test | 1,376 |
| Total | 6,877 |

Number of output classes:

```text
9
```

Class labels:

```text
AFIB, VPC, NORM, 1AVB, CRBBB, STE, PAC, CLBBB, STD
```

## Linear-Probing Command

Use this first if GPU memory is limited:

```bash
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
  --device cuda \
  --dataset_dir datasets/ecg_datasets/CPSC2018_QRS/data \
  --output_dir checkpoints/finetune/cpsc2018/finetune_cpsc2018_linear_seed0 \
  --log_dir log/finetune/finetune_cpsc2018_linear_seed0 \
  --model HeartLang_finetune_base \
  --finetune checkpoints/heartlang_base/checkpoint-200.pth \
  --trainable linear \
  --split_ratio 1.00 \
  --sampling_method random \
  --weight_decay 0.05 \
  --batch_size 32 \
  --lr 5e-3 \
  --update_freq 1 \
  --warmup_epochs 0 \
  --epochs 10 \
  --layer_decay 0.9 \
  --save_ckpt_freq 1 \
  --seed 0 \
  --is_binary \
  --nb_classes 9 \
  --world_size 1
```

## Full Fine-Tuning Command

Use this if GPU memory allows:

```bash
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
  --device cuda \
  --dataset_dir datasets/ecg_datasets/CPSC2018_QRS/data \
  --output_dir checkpoints/finetune/cpsc2018/finetune_cpsc2018_all_seed0 \
  --log_dir log/finetune/finetune_cpsc2018_all_seed0 \
  --model HeartLang_finetune_base \
  --finetune checkpoints/heartlang_base/checkpoint-200.pth \
  --trainable all \
  --split_ratio 1.00 \
  --sampling_method random \
  --weight_decay 0.05 \
  --batch_size 8 \
  --lr 5e-4 \
  --update_freq 1 \
  --warmup_epochs 1 \
  --epochs 10 \
  --layer_decay 0.9 \
  --save_ckpt_freq 1 \
  --seed 0 \
  --is_binary \
  --nb_classes 9 \
  --world_size 1
```

If CUDA memory is insufficient, reduce `--batch_size` to `4`.
