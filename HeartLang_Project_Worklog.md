# HeartLang Project Worklog

This document summarizes the HeartLang setup, debugging, fine-tuning, evaluation, baseline comparison, preprocessing experiments, and research-paper figure generation completed during this project.

Sensitive credentials are intentionally not included.

---

## 1. Project Objective

The goal was to run and evaluate HeartLang for ECG classification on the PTB-XL superdiagnostic task, first on a Windows laptop in CPU mode and later on an HPC machine with NVIDIA H100 GPUs.

The final research direction became:

- Fine-tune HeartLang on PTB-XL superdiagnostic ECG classification.
- Compare HeartLang against five baseline models.
- Study linear probing, adapter-style tuning, and full fine-tuning.
- Add ECG preprocessing before QRS tokenization.
- Generate publication-quality figures for a research paper/report.

---

## 2. Initial Repository and Environment Analysis

The HeartLang repository was inspected for:

- `requirements.txt`
- environment setup files
- pretrained checkpoint paths
- PTB-XL fine-tuning scripts
- QRS tokenization code
- evaluation and fine-tuning entry points

Important files:

```text
run_class_finetuning.py
engine_for_finetuning.py
utils/utils.py
QRSTokenizer.py
utils/QRSDataset.py
scripts/finetune/PTBXL/12Leads/super/base_super_linear_prob.sh
```

Initial issues included:

- Dependency installation problems on Windows.
- PyTorch DLL import errors.
- `cryptacular` failing to build on Windows due to missing `distutils.msvccompiler`.
- CPU-only compatibility problems in training/evaluation code.

---

## 3. Windows Environment Setup

The recommended Windows setup was:

- Install Miniconda, not full Anaconda.
- Create a Python 3.9 environment.
- Use CPU PyTorch because the laptop GPU was NVIDIA GeForce 940MX with only 2 GB VRAM.

Environment:

```text
conda environment: heartlang39
Python: 3.9
PyTorch: 2.3.0+cpu
CUDA available locally: False
Laptop GPU: NVIDIA GeForce 940MX 2 GB
```

Conclusion:

```text
The local laptop should use CPU mode only.
The 940MX 2 GB GPU is not practical for HeartLang fine-tuning.
```

---

## 4. Windows Dependency Fixes

The original `requirements.txt` contained unnecessary packages for HeartLang fine-tuning on Windows CPU.

The package `cryptacular` failed during installation:

```text
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

It was determined that `cryptacular` was not needed for the HeartLang ECG classification workflow.

A simplified Windows CPU requirements file was created:

```text
requirements-heartlang-windows-cpu.txt
```

This allowed the `heartlang39` environment to work successfully.

Verification:

```text
PyTorch 2.3.0+cpu imported correctly.
pip check reported no broken requirements.
```

---

## 5. Dataset and Checkpoint Setup

Main dataset used:

```text
datasets/ecg_datasets/PTBXL_QRS/superdiagnostic
```

Important files:

```text
train_data.npy
train_labels.npy
val_data.npy
val_labels.npy
test_data.npy
test_labels.npy
train_data_in_chans.npy
train_data_in_times.npy
val_data_in_chans.npy
val_data_in_times.npy
test_data_in_chans.npy
test_data_in_times.npy
```

Dataset shapes:

```text
train_data.npy: (17084, 256, 96)
train_labels.npy: (17084, 5)
val_data.npy: (2146, 256, 96)
val_labels.npy: (2146, 5)
test_data.npy: (2158, 256, 96)
test_labels.npy: (2158, 5)
```

Pretrained HeartLang checkpoint:

```text
checkpoints/heartlang_base/checkpoint-200.pth
```

The five PTB-XL superdiagnostic classes were:

```text
NORM
MI
STTC
CD
HYP
```

Meaning:

- `NORM`: Normal ECG
- `MI`: Myocardial Infarction
- `STTC`: ST/T Change
- `CD`: Conduction Disturbance
- `HYP`: Hypertrophy

---

## 6. CPU Smoke Test and Code Fixes

A 1-epoch CPU smoke test was run successfully on PTBXL_QRS superdiagnostic data.

Output checkpoint:

```text
checkpoints/finetune/ptbxl/finetune_superdiagnostic_cpu_smoke/checkpoint-1.pth
```

Several code issues were fixed during CPU-only testing.

### 6.1 Cosine Scheduler Assertion

Issue:

```text
AssertionError in utils.py cosine_scheduler
assert len(schedule) == epochs * niter_per_ep
```

Cause:

For very small CPU smoke-test settings, scheduler assumptions could fail if the number of iterations per epoch was not consistent.

Fix:

Use smoke-test arguments that give a valid number of iterations per epoch.

### 6.2 CPU GradScaler Issue

Errors:

```text
KeyError: 'scale'
UnboundLocalError: loss_scale_value
```

Cause:

The code expected AMP GradScaler state, but CPU training disables GradScaler.

Fix:

CPU-safe handling was added so training would not crash when GradScaler is disabled.

### 6.3 CPU Evaluation Distributed Error

Error:

```text
ValueError: Default process group has not been initialized
```

Cause:

Evaluation used:

```python
dist.get_rank()
```

when distributed mode was not initialized.

Fix:

Use:

```python
utils.is_main_process()
```

for CPU-only non-distributed evaluation.

---

## 7. First CPU Fine-Tuning Runs

CPU-only linear fine-tuning experiments were run on Windows.

Examples:

```text
split_ratio = 0.10
split_ratio = 0.25
split_ratio = 0.50
epochs = 3
batch_size = 1
trainable = linear
device = cpu
```

The 1% CPU smoke test achieved:

```text
AUC approximately 0.7215
```

A 50% CPU linear run achieved approximately:

```text
AUC approximately 0.8719
```

These runs proved that the pipeline worked before moving to HPC GPU.

---

## 8. HPC Access and GPU Setup

The HPC was accessed through SSH:

```bash
ssh cse_FA0611@10.10.11.201
```

HPC environment:

```text
hostname: master
conda version: 24.9.2
base Python: 3.12.7
HeartLang env: heartlang_gpu
Python in env: 3.9
PyTorch: 2.3.0+cu121
CUDA available: True
```

HPC GPUs:

```text
NVIDIA H100 NVL
NVIDIA H100 PCIe
MIG GPU partitions available
```

No SLURM commands were available:

```text
srun: not found
sbatch: not found
sinfo: not found
```

So experiments were run directly with:

```bash
CUDA_VISIBLE_DEVICES=4 python run_class_finetuning.py ...
```

---

## 9. Uploading HeartLang to HPC

Instead of copying the entire repository with datasets, checkpoints, logs, and virtual environments, a compressed code archive was created and uploaded.

Large unnecessary folders were excluded:

```text
.venv
.git
datasets
checkpoints
log
results
__pycache__
```

On HPC:

```bash
mkdir -p ~/HeartLang
tar -xzf ~/HeartLang_code.tar.gz -C ~/HeartLang
cd ~/HeartLang
```

The required datasets and checkpoints were verified on HPC:

```text
~/HeartLang/datasets/ecg_datasets/PTBXL_QRS/superdiagnostic
~/HeartLang/checkpoints/heartlang_base/checkpoint-200.pth
```

---

## 10. GPU Runtime Fixes

### 10.1 GradScaler Resume Error

When resuming from a CPU-trained checkpoint on GPU:

```text
RuntimeError: The source state dict is empty, possibly because it was saved from a disabled instance of GradScaler.
```

Cause:

The checkpoint was saved from CPU mode where GradScaler was disabled.

Fix:

Avoid loading empty scaler state when moving from CPU checkpoint to GPU training.

### 10.2 Device Mismatch Error

Error:

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
```

Cause:

`in_chan_matrix` and `in_time_matrix` were not moved to GPU.

Fix in `engine_for_finetuning.py`:

Move these tensors to the same device as ECG samples during training and evaluation:

```python
in_chan_matrix = in_chan_matrix.to(device, non_blocking=True)
in_time_matrix = in_time_matrix.to(device, non_blocking=True)
```

After this, GPU fine-tuning worked.

---

## 11. HeartLang Fine-Tuning Experiments

Main command pattern:

```bash
CUDA_VISIBLE_DEVICES=4 python run_class_finetuning.py \
  --device cuda \
  --dataset_dir datasets/ecg_datasets/PTBXL_QRS/superdiagnostic \
  --output_dir checkpoints/finetune/ptbxl/<experiment_name> \
  --log_dir log/finetune/<experiment_name> \
  --model HeartLang_finetune_base \
  --finetune checkpoints/heartlang_base/checkpoint-200.pth \
  --trainable all \
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
  --nb_classes 5 \
  --world_size 1 \
  --num_workers 4
```

Evaluation used `checkpoint-best.pth` when reporting final results.

---

## 12. Linear Probing Results

HeartLang linear probing used:

```text
trainable = linear
trainable parameters = 3,845
```

Results over seeds 0, 1, and 2:

```text
ROC-AUC mean approximately 0.8719
```

Seed values:

```text
seed 0: 0.872276
seed 1: 0.872031
seed 2: 0.871542
```

Conclusion:

Linear probing works, but full fine-tuning performs better.

---

## 13. Adapter Fine-Tuning Result

Adapter experiment:

```text
trainable = adapter
seed = 0
```

Result:

```text
Test ROC-AUC = 0.8408477666899113
Accuracy = 0.4388
Precision = 0.6981
Recall = 0.4193
F1 = 0.4859
Loss = 0.3927
```

Important observation:

The printed trainable layers only showed the classification head:

```text
mlp_head.weight
mlp_head.bias
```

So adapter tuning did not appear to activate meaningful adapter layers in the current implementation.

Conclusion:

Adapter results were not competitive and were not continued.

---

## 14. Full Fine-Tuning Without Preprocessing

Full fine-tuning used:

```text
trainable = all
trainable parameters = 39,053,829
dataset = PTBXL_QRS/superdiagnostic
epochs = 10
batch_size = 32
```

Best checkpoint results:

```text
seed 0 AUC = 0.900513728580864
seed 1 AUC = 0.9023325244425056
seed 2 AUC = 0.9030897798718778
```

Mean result:

```text
ROC-AUC = 0.9020 +/- 0.0013
Accuracy = 0.5618 +/- 0.0135
Precision = 0.7358 +/- 0.0172
Recall = 0.6662 +/- 0.0107
F1-score = 0.6792 +/- 0.0051
Loss = 0.3174 +/- 0.0121
```

---

## 15. Baseline Models

Five baselines were created and evaluated:

```text
Linear classifier
MLP
CNN
GRU
ResNet
```

Script:

```text
run_ptbxl_baselines.py
```

Output file:

```text
results/baseline_results.csv
```

Baseline model parameter counts:

```text
Linear baseline: 122,885
MLP: 12,585,989
CNN: 252,293
ResNet: 483,333
GRU: 174,853
```

Baseline results over seeds 0, 1, and 2:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-score | Loss |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.7360 +/- 0.0036 | 0.3458 +/- 0.0003 | 0.5140 +/- 0.0226 | 0.4865 +/- 0.0256 | 0.4967 +/- 0.0039 | 0.7368 +/- 0.0918 |
| MLP | 0.8000 +/- 0.0028 | 0.4384 +/- 0.0146 | 0.6562 +/- 0.0093 | 0.4740 +/- 0.0218 | 0.5391 +/- 0.0168 | 0.4443 +/- 0.0251 |
| CNN | 0.8749 +/- 0.0016 | 0.4838 +/- 0.0626 | 0.7297 +/- 0.0208 | 0.5921 +/- 0.0077 | 0.6407 +/- 0.0141 | 0.3651 +/- 0.0190 |
| GRU | 0.8827 +/- 0.0029 | 0.5545 +/- 0.0049 | 0.7450 +/- 0.0301 | 0.6000 +/- 0.0163 | 0.6516 +/- 0.0061 | 0.3428 +/- 0.0115 |
| ResNet | 0.8868 +/- 0.0009 | 0.5479 +/- 0.0100 | 0.7584 +/- 0.0223 | 0.5845 +/- 0.0240 | 0.6439 +/- 0.0150 | 0.3409 +/- 0.0053 |

Conclusion:

The strongest baseline was ResNet by AUC, but HeartLang full fine-tuning exceeded all baselines.

---

## 16. ECG Preprocessing

Raw PTB-XL data was found locally:

```text
D:\HeartLang\datasets\ecg_datasets\PTBXL
```

Important raw files:

```text
ptbxl_database.csv
scp_statements.csv
raw100.npy
records100
records500
```

Raw superdiagnostic arrays:

```text
D:\HeartLang\datasets\ecg_datasets\PTBXL\superdiagnostic
```

Shapes:

```text
train_data.npy: (17084, 1000, 12)
val_data.npy: (2146, 1000, 12)
test_data.npy: (2158, 1000, 12)
train_labels.npy: (17084, 5)
val_labels.npy: (2146, 5)
test_labels.npy: (2158, 5)
```

The preprocessing pipeline applied:

1. 0.5-40 Hz Butterworth band-pass filtering.
2. Baseline wander removal through the high-pass component of the band-pass filter.
3. Lead-wise z-score normalization using training-set statistics.

Notch filtering was not used because:

```text
sampling frequency = 100 Hz
50 Hz is the Nyquist frequency
```

Preprocessed dataset created:

```text
datasets/ecg_datasets/PTBXL_PREPROC/superdiagnostic
```

Then QRS tokenization was applied:

```bash
python QRSTokenizer_PTBXL_PREPROC.py --dataset_name PTBXL_PREPROC
```

Output tokenized preprocessed dataset:

```text
datasets/ecg_datasets/PTBXL_PREPROC_QRS/superdiagnostic
```

---

## 17. Full Fine-Tuning With Preprocessing

Main final model:

```text
HeartLang + ECG Preprocessing
trainable = all
dataset = PTBXL_PREPROC_QRS/superdiagnostic
epochs = 10
batch_size = 32
checkpoint = checkpoint-best.pth
```

Best checkpoint test results:

```text
seed 0 AUC = 0.9119584743422244
seed 1 AUC = 0.9120432293746454
seed 2 AUC = 0.9137408435603053
```

Final mean result:

```text
ROC-AUC = 0.9126 +/- 0.0010
Accuracy = 0.5779 +/- 0.0093
Precision = 0.7608 +/- 0.0121
Recall = 0.6549 +/- 0.0099
F1-score = 0.6791 +/- 0.0062
Loss = 0.3006 +/- 0.0040
```

Comparison:

```text
HeartLang without preprocessing AUC = 0.9020 +/- 0.0013
HeartLang with preprocessing AUC = 0.9126 +/- 0.0010
Improvement = approximately +0.0106 AUC
```

Conclusion:

The proposed preprocessing improved HeartLang performance.

---

## 18. Per-Class AUC Analysis

Per-class ROC-AUC was generated using:

```text
per_class_auc.py
```

Saved predictions:

```text
results/pred/PTBXL_QRS_superdiagnostic_outputs.csv
results/pred/PTBXL_QRS_superdiagnostic_targets.csv
results/pred/PTBXL_PREPROC_QRS_superdiagnostic_outputs.csv
results/pred/PTBXL_PREPROC_QRS_superdiagnostic_targets.csv
```

Preprocessed HeartLang per-class AUC:

| Class | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| NORM | 0.899715 | 0.898611 | 0.903988 |
| MI | 0.887420 | 0.894712 | 0.889462 |
| STTC | 0.909069 | 0.906594 | 0.911502 |
| CD | 0.937432 | 0.935741 | 0.937769 |
| HYP | 0.926156 | 0.924558 | 0.925983 |
| Macro | 0.911958 | 0.912043 | 0.913741 |

Mean per-class AUC:

```text
NORM = 0.9021 +/- 0.0028
MI = 0.8908 +/- 0.0040
STTC = 0.9091 +/- 0.0025
CD = 0.9369 +/- 0.0009
HYP = 0.9255 +/- 0.0009
Macro = 0.9126 +/- 0.0010
```

Strongest class:

```text
CD
```

Most difficult class:

```text
MI
```

---

## 19. Confusion Matrix

A multi-label confusion matrix was generated for the final preprocessed HeartLang model.

Output:

```text
results/confusion_matrix_preproc
```

Confusion matrix:

| Class | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| NORM | 1579 | 83 | 185 | 311 |
| MI | 1867 | 29 | 156 | 106 |
| STTC | 1427 | 181 | 136 | 414 |
| CD | 1055 | 140 | 167 | 796 |
| HYP | 1496 | 141 | 131 | 390 |

Interpretation:

- CD had the strongest true positive count.
- MI had relatively many false negatives compared with true positives.
- This indicates MI is harder for the model at threshold 0.5.

---

## 20. G-Mean Analysis

G-Mean was considered for class-wise sensitivity/specificity balance.

Formula:

```text
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
G-Mean = sqrt(Sensitivity * Specificity)
```

Approximate G-Mean from the final confusion matrix:

| Class | Sensitivity | Specificity | G-Mean |
|---|---:|---:|---:|
| NORM | 0.6270 | 0.9501 | 0.7717 |
| MI | 0.4046 | 0.9847 | 0.6313 |
| STTC | 0.7527 | 0.8874 | 0.8173 |
| CD | 0.8266 | 0.8828 | 0.8542 |
| HYP | 0.7486 | 0.9139 | 0.8271 |

Interpretation:

MI has the lowest G-Mean because of lower sensitivity.

---

## 21. Data-Efficiency Analysis

### 21.1 Without Preprocessing

Earlier data-efficiency results without preprocessing:

| Training Data | AUC | F1-score | Loss |
|---:|---:|---:|---:|
| 10% | 0.8703 | 0.5797 | 0.4533 |
| 25% | 0.8818 | 0.6111 | 0.4026 |
| 50% | 0.8996 | 0.6214 | 0.3608 |
| 100% | 0.9005 | 0.6785 | 0.3314 |

### 21.2 With Preprocessing

Initial preprocessed low-data runs used too high a learning rate and became unstable.

Symptoms:

```text
train_loss = NaN
grad_norm = NaN
loss_scale collapsed
```

Corrected learning rate:

```text
lr = 5e-4
warmup_epochs = 1
```

Corrected preprocessed data-efficiency results:

| Training Data | Test ROC-AUC |
|---:|---:|
| 10% | 0.8682134807829615 |
| 25% | 0.8772678525160202 |
| 50% | 0.8952348403150486 |
| 100% | 0.9126 |

Conclusion:

The final data-efficiency figure should use the corrected preprocessed results.

---

## 22. Training Dynamics and Overfitting

Training dynamics were plotted using:

```text
log.txt files from seeds 0, 1, and 2
```

Important log keys:

```text
train_loss
val_loss
val_roc_auc
epoch
```

Observation:

- Training loss decreased steadily.
- Validation loss increased in later epochs.
- Validation ROC-AUC peaked early and then stabilized/decreased slightly.

Interpretation:

This indicates some overfitting or probability calibration degradation in later epochs.

Defensible explanation:

```text
The final model uses checkpoint-best.pth selected by validation ROC-AUC, not the last epoch.
Therefore, overfitting in later epochs is controlled through validation-based checkpoint selection.
```

Future improvements:

- Early stopping.
- Lower learning rate.
- Stronger regularization.
- Dropout.
- Partial fine-tuning.
- Threshold calibration.

---

## 23. Final Comparison Table

Final comparison across baselines and HeartLang:

| Model | AUC | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Linear | 0.7360 | 0.3458 | 0.5140 | 0.4865 | 0.4967 |
| MLP | 0.8000 | 0.4384 | 0.6562 | 0.4740 | 0.5391 |
| CNN | 0.8749 | 0.4838 | 0.7297 | 0.5921 | 0.6407 |
| GRU | 0.8827 | 0.5545 | 0.7450 | 0.6000 | 0.6516 |
| ResNet | 0.8868 | 0.5479 | 0.7584 | 0.5845 | 0.6439 |
| HeartLang | 0.9020 | 0.5618 | 0.7358 | 0.6662 | 0.6792 |
| HeartLang + ECG Preprocessing | 0.9126 | 0.5779 | 0.7608 | 0.6549 | 0.6791 |

Main result:

```text
HeartLang + ECG Preprocessing achieved the best ROC-AUC: 0.9126 +/- 0.0010.
```

---

## 24. Figures Generated

The following figures were generated or planned for the report/paper.

### Figure 1: Proposed Methodology / Pipeline

Shows:

```text
Raw PTB-XL ECG
-> ECG preprocessing
-> QRS tokenization
-> HeartLang encoder
-> classification head
-> five superdiagnostic classes
```

### Figure 2: Baseline Model Comparison

Grouped bar chart comparing:

```text
ROC-AUC
Accuracy
Precision
Recall
F1-score
```

Models:

```text
Linear
MLP
CNN
GRU
ResNet
HeartLang
HeartLang + ECG Preprocessing
```

### Figure 3: Preprocessing Comparison

Compares:

```text
HeartLang without preprocessing: 0.9020
HeartLang with preprocessing: 0.9126
```

### Figure 4: Data-Efficiency Plot

Final corrected preprocessed data-efficiency values:

```text
10% = 0.8682
25% = 0.8773
50% = 0.8952
100% = 0.9126
```

### Figure 5: Per-Class ROC Curves

Generated for the preprocessed HeartLang best checkpoints.

### Figure 6: Training Dynamics

Shows:

```text
Training loss vs epoch
Validation ROC-AUC vs epoch
```

### Figure 7: Confusion Matrix or G-Mean Analysis

Used for error analysis.

### Additional Figure: Performance Metrics Heatmap

Heatmap comparing all models and all metrics.

Output:

```text
performance_metrics_heatmap_ieee.png
```

---

## 25. Important Backup Locations

Main local backup:

```text
D:\HeartLang_HPC_Backup
```

Important copied items:

```text
results
log
per_class_auc
per_class_auc_preproc
confusion_matrix_preproc
run_ptbxl_baselines.py
per_class_auc.py
checkpoint folders
HeartLang_Missing_Final_Items.tar.gz
```

Important archive:

```text
D:\HeartLang_Backup\HeartLang_Final_Backup.tar.gz
```

Important result files:

```text
D:\HeartLang_HPC_Backup\results\finetune_results.csv
D:\HeartLang_HPC_Backup\results\baseline_results.csv
D:\HeartLang_HPC_Backup\results\pred\PTBXL_PREPROC_QRS_superdiagnostic_outputs.csv
D:\HeartLang_HPC_Backup\results\pred\PTBXL_PREPROC_QRS_superdiagnostic_targets.csv
```

Important figure folder:

```text
D:\HeartLang_HPC_Backup\figures
```

---

## 26. Recommended Final Report Structure

Suggested project/report sections:

1. Introduction
2. Problem Statement
3. Dataset Description
4. Proposed Methodology
5. ECG Preprocessing
6. HeartLang Fine-Tuning
7. Baseline Models
8. Experimental Setup
9. Results and Discussion
10. Ablation Study
11. Data-Efficiency Analysis
12. Per-Class ROC Analysis
13. Confusion Matrix / Error Analysis
14. Conclusion
15. Future Work
16. References

---

## 27. Safe Contribution Wording

Do not claim that this work invented HeartLang.

Safe contribution wording:

```text
This work presents a HeartLang-based ECG classification framework for PTB-XL superdiagnostic classification, enhanced with ECG preprocessing and evaluated against multiple deep learning baselines.
```

Possible contributions:

1. Adapting HeartLang for PTB-XL superdiagnostic classification.
2. Adding ECG preprocessing before QRS tokenization.
3. Comparing against five baseline models.
4. Evaluating linear probing, adapter tuning, and full fine-tuning.
5. Performing per-class ROC-AUC, confusion matrix, G-Mean, and data-efficiency analysis.

---

## 28. Final Headline Result

The final proposed system is:

```text
HeartLang full fine-tuning + ECG preprocessing
```

Final performance:

```text
ROC-AUC = 0.9126 +/- 0.0010
Accuracy = 0.5779 +/- 0.0093
Precision = 0.7608 +/- 0.0121
Recall = 0.6549 +/- 0.0099
F1-score = 0.6791 +/- 0.0062
```

Best comparison:

```text
Best baseline AUC: ResNet = 0.8868
HeartLang without preprocessing AUC = 0.9020
HeartLang with preprocessing AUC = 0.9126
```

Main conclusion:

```text
The proposed HeartLang framework with ECG preprocessing achieved the highest ROC-AUC among all evaluated models and showed improved performance compared with both conventional baselines and HeartLang without preprocessing.
```

