# HeartLang ECG Classification on PTB-XL Superdiagnostic Dataset

This repository contains the code, experiment scripts, result files, and paper figures for a HeartLang-based multi-label ECG classification project on the PTB-XL Superdiagnostic dataset.

## Project Overview

The project evaluates a pretrained HeartLang ECG representation model for downstream multi-label classification of five PTB-XL superdiagnostic classes:

- NORM
- MI
- STTC
- CD
- HYP

The final proposed pipeline applies ECG signal preprocessing, QRS tokenization, pretrained HeartLang representation learning, and full fine-tuning for downstream classification.

## Proposed Pipeline

1. ECG preprocessing
   - 0.5-40 Hz Butterworth band-pass filtering
   - Lead-wise z-score normalization using training-set statistics
2. QRS tokenization
3. HeartLang pretrained model loading
4. Full fine-tuning on PTB-XL Superdiagnostic classification
5. Evaluation using ROC-AUC, accuracy, precision, recall, and F1-score

## Dataset Split

The experiments used pre-generated train, validation, and held-out test splits:

| Split | Samples |
|---|---:|
| Training | 17,084 |
| Validation | 2,146 |
| Test | 2,158 |

For data-efficiency experiments, only the training split was randomly subsampled. Validation and test sets were kept fixed.

## Main Result

The final HeartLang model with ECG preprocessing achieved:

| Model | Test ROC-AUC |
|---|---:|
| HeartLang + ECG preprocessing | 0.9126 +/- 0.0010 |

This value is computed across three random seeds: 0, 1, and 2.

## Repository Contents

| Path | Description |
|---|---|
| `run_class_finetuning.py` | HeartLang fine-tuning/evaluation entry point |
| `engine_for_finetuning.py` | Training and evaluation loops |
| `create_filtered_ptbxl_super.py` | ECG preprocessing script |
| `QRSTokenizer.py` | Original QRS tokenization script |
| `QRSTokenizer_PTBXL_PREPROC.py` | QRS tokenization for preprocessed PTB-XL data |
| `run_ptbxl_baselines.py` | Baseline model experiments |
| `run_ptbxl_baselines_with_curves.py` | Baseline training-curve experiments |
| `results/` | Result CSV files and per-class AUC outputs |
| `figures/` | Publication-style figures |
| `docs/` | Research paper/report files |
| `datasets/README.md` | Dataset availability and preparation note |

## Figures

The `figures/` directory contains publication-style plots including:

- Training dynamics
- Baseline comparison
- Data-efficiency analysis
- Per-class ROC curves
- Performance heatmap
- G-Mean and confusion-matrix related outputs

## Dataset and Checkpoints

The raw PTB-XL dataset, processed NumPy arrays, and trained checkpoint files are not committed to this repository because they are large. The PTB-XL dataset is publicly available from PhysioNet. Trained checkpoints and processed data can be shared separately if required.

Excluded large files include:

- `*.pth`
- `*.npy`
- `*.tar.gz`
- raw PTB-XL records
- virtual environments

## Environment

Install dependencies using:

```bash
pip install -r requirements.txt
```

For the Windows CPU environment used during local setup, see:

```bash
requirements-heartlang-windows-cpu.txt
```

## Notes

This repository is organized for academic review and reproducibility of the reported experiments. Large datasets and model checkpoints should be downloaded or transferred separately.
