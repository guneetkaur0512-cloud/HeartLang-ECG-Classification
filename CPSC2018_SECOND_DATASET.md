# Second Dataset: CPSC2018

This repository includes support for preparing and training HeartLang on the CPSC2018 ECG dataset as a second dataset.

## Dataset Summary

CPSC2018 contains 6,877 public 12-lead ECG recordings sampled at 500 Hz. The recordings have variable duration and are labeled with nine rhythm/morphology classes.

The split files included in this repository contain:

| Split | Records |
|---|---:|
| Train | 4,950 |
| Validation | 551 |
| Test | 1,376 |
| Total | 6,877 |

The labels in the local split files are:

```text
AFIB, VPC, NORM, 1AVB, CRBBB, STE, PAC, CLBBB, STD
```

## Main Goal

The goal is to create a ready-to-train HeartLang dataset at:

```text
datasets/ecg_datasets/CPSC2018_QRS/data/
```

After this folder is prepared, training can be run directly without repeating raw ECG preprocessing or QRS tokenization.

## Final Ready-to-Train Folder

The final folder should contain:

```text
datasets/ecg_datasets/CPSC2018_QRS/data/
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

## Preparation Workflow

The preparation workflow is included for reproducibility. The professor does not need to run these steps if the ready-to-train dataset archive is already provided.

### Step 1: Prepare NumPy Data

If raw CPSC2018 `.mat` files are already available:

```bash
python prepare_cpsc2018_ready_dataset.py --raw_dir path/to/CPSC2018/raw_mat_files
```

If raw `.mat` files are not present, the script can download missing records from the PhysioNet Challenge 2020 CPSC2018 mirror:

```bash
python prepare_cpsc2018_ready_dataset.py --raw_dir datasets/raw/CPSC2018 --download-missing
```

This creates:

```text
datasets/ecg_datasets/CPSC2018/data/train_data.npy
datasets/ecg_datasets/CPSC2018/data/train_labels.npy
datasets/ecg_datasets/CPSC2018/data/val_data.npy
datasets/ecg_datasets/CPSC2018/data/val_labels.npy
datasets/ecg_datasets/CPSC2018/data/test_data.npy
datasets/ecg_datasets/CPSC2018/data/test_labels.npy
```

### Step 2: QRS Tokenization

Run:

```bash
python QRSTokenizer.py --dataset_name CPSC2018
```

This creates:

```text
datasets/ecg_datasets/CPSC2018_QRS/data/
```

## Training

For a low-memory first run, linear probing is recommended:

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

For full fine-tuning, use:

```text
--trainable all
--batch_size 4 or 8
--lr 5e-4
```

## Notes

- The model does not need to be pretrained again.
- Use the same pretrained HeartLang checkpoint as the PTB-XL experiment.
- The second dataset experiment is mainly for external validation/generalization.
- The result does not need to exceed PTB-XL performance; it is used to demonstrate that HeartLang can be applied beyond a single dataset.
