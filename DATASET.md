# Dataset Availability and Preparation

This project uses the PTB-XL Superdiagnostic dataset for multi-label ECG classification.

## Dataset Source

The original PTB-XL dataset is publicly available through PhysioNet:

https://physionet.org/content/ptb-xl/

Users should download the dataset from the official source and follow the preprocessing and tokenization steps described below.

## Task

The experiments use the PTB-XL Superdiagnostic classification task with five diagnostic classes:

- NORM
- MI
- STTC
- CD
- HYP

## Dataset Split

The experiments use fixed pre-generated train, validation, and held-out test splits:

| Split | Number of ECG samples |
|---|---:|
| Training | 17,084 |
| Validation | 2,146 |
| Test | 2,158 |

For data-efficiency experiments, only the training split was randomly subsampled. The validation and test sets were kept unchanged.

## Processed Dataset Files

The final HeartLang experiments expect the following processed QRS-tokenized files:

```text
datasets/ecg_datasets/PTBXL_PREPROC_QRS/superdiagnostic/
├── train_data.npy
├── train_labels.npy
├── train_data_in_chans.npy
├── train_data_in_times.npy
├── val_data.npy
├── val_labels.npy
├── val_data_in_chans.npy
├── val_data_in_times.npy
├── test_data.npy
├── test_labels.npy
├── test_data_in_chans.npy
└── test_data_in_times.npy
```

These files are generated from the PTB-XL Superdiagnostic data after ECG preprocessing and QRS tokenization.

## Preprocessing

The final proposed method applies ECG preprocessing before QRS tokenization:

1. 0.5-40 Hz Butterworth band-pass filtering
2. Lead-wise z-score normalization using training-set statistics

The preprocessing script is:

```text
create_filtered_ptbxl_super.py
```

## QRS Tokenization

The QRS tokenization script used for the preprocessed dataset is:

```text
QRSTokenizer_PTBXL_PREPROC.py
```

The resulting tokenized input has the following shape:

```text
ECG tensor:           (N, 256, 96)
Channel index tensor: (N, 256)
Time index tensor:    (N, 256)
```

## Why the Dataset Is Not Stored Directly in GitHub

The raw PTB-XL dataset and generated NumPy arrays are large binary files. They are not committed directly to this repository because standard GitHub repositories have file-size and repository-size limitations.

Large files excluded from this repository include:

- Raw PTB-XL WFDB records
- Processed `.npy` dataset arrays
- Trained `.pth` checkpoint files
- Large archive backups such as `.tar.gz`

If required, the processed dataset and trained checkpoints can be shared separately through external storage such as Google Drive, OneDrive, institutional storage, or another dataset hosting platform.
