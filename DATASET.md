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

## Processed Dataset Download

The processed QRS-tokenized PTB-XL Superdiagnostic dataset used for the final proposed HeartLang experiment is available through Google Drive:

https://drive.google.com/drive/folders/1wgVIwNyUvh9EoexPOBPknJcXpEUXAtDL

Because the dataset archive is large, it is provided as split archive parts:

```text
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part01
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part02
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part03
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part04
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part05
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part06
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part07
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part08
HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part09
```

After downloading all parts, they can be combined on Windows using:

```bat
copy /b HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part01+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part02+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part03+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part04+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part05+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part06+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part07+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part08+HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz.part09 HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz
```

Then extract the archive using:

```bat
tar -xzf HeartLang_PTBXL_PREPROC_QRS_superdiagnostic_dataset.tar.gz
```

The extracted folder should be placed at:

```text
datasets/ecg_datasets/PTBXL_PREPROC_QRS/superdiagnostic/
```

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

The trained checkpoint files are still excluded from GitHub because of file size. They can be shared separately through external storage if required.
