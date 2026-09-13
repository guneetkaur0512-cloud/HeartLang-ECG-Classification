# Daily Diary: HeartLang ECG Classification Project

## Day 1

Explored different machine learning project topics and datasets. Searched for datasets related to cancer disease, Parkinson's disease, UCI heart disease prediction, and the impact of AI on jobs. Compared which topics would be suitable for a college project and research work.

Finalized the broad topic as heart disease prediction/classification using machine learning and deep learning.

## Day 2

Searched for research papers on IEEE Xplore, Google Scholar, and other research paper websites. Read papers related to heart disease prediction, ECG classification, and deep learning in healthcare.

Learned about basic data cleaning methods, handling missing values, reducing skewness, and applying normalization techniques.

## Day 3

Studied deep learning concepts such as neural networks, gradient descent, backpropagation, and LSTM models. Understood why ECG signals can be treated as sequential data.

Also started comparing traditional ML models and deep learning models for healthcare datasets.

## Day 4

Studied the concept of transformers and self-attention mechanism. Learned about embeddings, encoder architecture, and how transformer-based models can be used for sequential data.

Understood that HeartLang is based on ECG language modeling and transformer-style representation learning.

## Day 5

Installed Cursor and learned how to use it for indexing a project directory, searching files, reading code, and running Python scripts.

Opened the HeartLang repository in Cursor and explored the main project structure.

## Day 6

Analyzed the HeartLang repository files. Checked important files such as:

```text
requirements.txt
run_class_finetuning.py
engine_for_finetuning.py
QRSTokenizer.py
utils/QRSDataset.py
```

Also learned that the laptop GPU, NVIDIA GeForce 940MX with 2 GB VRAM, was not sufficient for serious HeartLang training.

## Day 7

Studied the basics of Overleaf and LaTeX. Opened IEEE research paper templates and understood the structure of a conference/research paper.

Learned about sections such as abstract, introduction, methodology, results, discussion, conclusion, and references.

## Day 8

Created a Conda environment for HeartLang using Python 3.9. Verified Python version and installed CPU-compatible PyTorch because CUDA was not available on the laptop.

Confirmed:

```text
Python 3.9
PyTorch CPU
CUDA unavailable locally
```

## Day 9

Started installing HeartLang dependencies. Faced installation issues with the original `requirements.txt`, especially with unnecessary packages on Windows.

Found that `cryptacular` failed during installation due to:

```text
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

## Day 10

Analyzed dependency requirements and created a simplified Windows CPU requirements file:

```text
requirements-heartlang-windows-cpu.txt
```

Installed the required dependencies successfully and verified that PyTorch imported correctly.

## Day 11

Downloaded and placed the HeartLang pretrained checkpoint:

```text
checkpoints/heartlang_base/checkpoint-200.pth
```

Verified the PTB-XL QRS dataset path:

```text
datasets/ecg_datasets/PTBXL_QRS/superdiagnostic
```

Checked that training, validation, and test `.npy` files were available.

## Day 12

Ran the first CPU smoke test for PTB-XL superdiagnostic classification. Faced a cosine scheduler assertion issue in small fine-tuning settings.

Adjusted the command-line arguments so that the number of training iterations matched the scheduler requirement.

## Day 13

Fixed CPU-only compatibility issues in the HeartLang training and evaluation code. Solved errors related to GradScaler and distributed evaluation.

Important errors fixed:

```text
KeyError: 'scale'
UnboundLocalError: loss_scale_value
Default process group has not been initialized
```

## Day 14

Successfully completed CPU smoke test. The model produced a checkpoint and evaluation gave approximately:

```text
AUC = 0.7215
```

This confirmed that the basic HeartLang pipeline worked locally in CPU mode.

## Day 15

Ran larger CPU experiments using PTBXL_QRS superdiagnostic data with different split ratios such as 10%, 25%, and 50%.

Observed that CPU training was very slow, so GPU access was required for complete experiments.

## Day 16

Received HPC access from the professor and learned how to log in using SSH from Windows Anaconda Prompt.

Verified HPC connection and checked basic commands:

```bash
pwd
ls
df -h
hostname
nvidia-smi
```

## Day 17

Checked HPC GPU details and found NVIDIA H100 GPUs available. Verified that Conda was installed on HPC.

Created and activated the GPU environment:

```text
heartlang_gpu
```

Verified:

```text
PyTorch 2.3.0+cu121
CUDA available: True
```

## Day 18

Uploaded the HeartLang project code to HPC. Instead of copying the entire repository, created a compressed archive excluding unnecessary large folders such as `.venv`, `.git`, datasets, checkpoints, logs, and results.

Extracted the code on HPC and verified that the project files were present.

## Day 19

Verified that required datasets and pretrained checkpoints were available on HPC.

Important paths:

```text
datasets/ecg_datasets/PTBXL_QRS/superdiagnostic
checkpoints/heartlang_base/checkpoint-200.pth
```

Confirmed that training could start on GPU.

## Day 20

Faced a GradScaler resume issue while moving from CPU checkpoint to GPU training. Understood that CPU-saved checkpoints can have empty GradScaler state.

Adjusted the workflow to avoid loading incompatible scaler state.

## Day 21

Faced a GPU device mismatch error because `in_chan_matrix` and `in_time_matrix` were on CPU while ECG data was on GPU.

Fixed `engine_for_finetuning.py` so all tensors were moved to the same device before model inference.

## Day 22

Successfully ran GPU fine-tuning on PTBXL_QRS superdiagnostic data. GPU training was much faster than CPU training.

Started recording results for different fine-tuning settings.

## Day 23

Ran HeartLang linear probing experiments using seeds 0, 1, and 2.

Results:

```text
Seed 0 AUC = 0.872276
Seed 1 AUC = 0.872031
Seed 2 AUC = 0.871542
Mean AUC ≈ 0.8719
```

## Day 24

Ran HeartLang full fine-tuning without preprocessing using 100% PTB-XL QRS data.

Best checkpoint results:

```text
Seed 0 AUC = 0.900514
Seed 1 AUC = 0.902333
Seed 2 AUC = 0.903090
Mean AUC = 0.9020 ± 0.0013
```

## Day 25

Ran adapter fine-tuning experiment. The result was lower than full fine-tuning:

```text
Adapter AUC = 0.840848
```

Observed that adapter mode appeared to train mainly the classification head, so it was not selected as the final method.

## Day 26

Created and ran five baseline models for comparison:

```text
Linear classifier
MLP
CNN
GRU
ResNet
```

Created script:

```text
run_ptbxl_baselines.py
```

Saved results in:

```text
results/baseline_results.csv
```

## Day 27

Completed baseline experiments for seeds 0, 1, and 2.

Best baseline:

```text
ResNet AUC = 0.8868 ± 0.0009
```

HeartLang full fine-tuning performed better:

```text
HeartLang AUC = 0.9020 ± 0.0013
```

## Day 28

Calculated trainable parameter counts for baseline models and HeartLang variants.

Parameter counts:

```text
Linear baseline: 122,885
MLP: 12,585,989
CNN: 252,293
GRU: 174,853
ResNet: 483,333
HeartLang linear probing: 3,845
HeartLang full fine-tuning: 39,053,829
```

## Day 29

Explored raw PTB-XL data available locally. Verified files such as:

```text
ptbxl_database.csv
scp_statements.csv
raw100.npy
records100
records500
```

Confirmed raw superdiagnostic data shape:

```text
train_data.npy: (17084, 1000, 12)
```

## Day 30

Created ECG preprocessing script:

```text
create_filtered_ptbxl_super.py
```

Applied:

- 0.5-40 Hz Butterworth band-pass filtering
- baseline wander removal
- lead-wise z-score normalization using training-set statistics

Created preprocessed dataset:

```text
PTBXL_PREPROC/superdiagnostic
```

## Day 31

Tokenized the preprocessed ECG data using:

```text
QRSTokenizer_PTBXL_PREPROC.py
```

Created:

```text
PTBXL_PREPROC_QRS/superdiagnostic
```

This became the dataset for the final proposed model.

## Day 32

Ran full fine-tuning of HeartLang with ECG preprocessing on HPC.

Best checkpoint results:

```text
Seed 0 AUC = 0.911958
Seed 1 AUC = 0.912043
Seed 2 AUC = 0.913741
Mean AUC = 0.9126 ± 0.0010
```

This became the final proposed model.

## Day 33

Compared HeartLang without preprocessing and HeartLang with preprocessing.

Results:

```text
Without preprocessing AUC = 0.9020 ± 0.0013
With preprocessing AUC = 0.9126 ± 0.0010
```

Conclusion:

ECG preprocessing improved HeartLang performance by approximately `+0.0106` AUC.

## Day 34

Generated per-class ROC-AUC analysis for the final model.

Classes:

```text
NORM
MI
STTC
CD
HYP
```

Observed that CD and HYP performed strongly, while MI was more difficult.

## Day 35

Generated per-class ROC curves and saved prediction files:

```text
PTBXL_PREPROC_QRS_superdiagnostic_outputs.csv
PTBXL_PREPROC_QRS_superdiagnostic_targets.csv
```

These files were used for ROC curves and further error analysis.

## Day 36

Generated confusion matrix for the final HeartLang model.

Observed:

- CD had high true positives.
- MI had more false negatives.
- MI was comparatively more difficult at threshold 0.5.

## Day 37

Calculated G-Mean using sensitivity and specificity.

Formula:

```text
G-Mean = sqrt(Sensitivity × Specificity)
```

Understood that G-Mean is useful for imbalanced classification problems.

## Day 38

Generated training dynamics figure for HeartLang with preprocessing.

The figure showed:

- training loss vs epoch
- validation ROC-AUC vs epoch
- average across seeds

Observed slight overfitting in later epochs and decided to report `checkpoint-best.pth`.

## Day 39

Generated baseline comparison figures, including grouped bar chart and performance heatmap.

Compared:

```text
AUC
Accuracy
Precision
Recall
F1-score
```

The proposed HeartLang with ECG preprocessing achieved the best AUC.

## Day 40

Generated data-efficiency analysis. Initial preprocessed low-data runs were unstable due to high learning rate.

Adjusted settings:

```text
lr = 5e-4
warmup_epochs = 1
```

## Day 41

Completed corrected preprocessed data-efficiency experiments.

Final values:

```text
10% = 0.8682
25% = 0.8773
50% = 0.8952
100% = 0.9126
```

Generated final data-efficiency figure.

## Day 42

Backed up important files from HPC to laptop.

Backup folder:

```text
D:\HeartLang_HPC_Backup
```

Copied:

- results
- logs
- checkpoints
- per-class AUC files
- confusion matrix files
- scripts
- figures

## Day 43

Created complete project worklog:

```text
HeartLang_Project_Worklog.md
```

This file summarized the complete work from setup to final results.

Also started planning the final project report and research paper structure.

## Day 44

Generated and organized important figures for the project report and research paper. Focused on making the results more presentable and easy to understand.

Figures prepared:

- Proposed methodology diagram
- Per-class ROC curves
- Data-efficiency graph
- Baseline comparison histogram/grouped bar chart
- Performance metrics heatmap
- Confusion matrix
- Training dynamics curves

These figures helped explain the complete workflow, compare HeartLang with baseline models, and show the improvement obtained after ECG preprocessing.

## Day 45

Created baseline training curve logging script:

```text
run_ptbxl_baselines_with_curves.py
```

Generated per-epoch training loss and validation ROC-AUC for all baseline models.

Planned final combined training dynamics figure with all seven models:

```text
Linear
MLP
CNN
GRU
ResNet
HeartLang
HeartLang + ECG Preprocessing
```

Prepared final material for project report, research paper, presentation, and application.
