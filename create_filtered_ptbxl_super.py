from pathlib import Path
import shutil
import numpy as np
from scipy.signal import butter, sosfiltfilt

src = Path("datasets/ecg_datasets/PTBXL/superdiagnostic")
dst = Path("datasets/ecg_datasets/PTBXL_PREPROC/superdiagnostic")
dst.mkdir(parents=True, exist_ok=True)

fs = 100
lowcut = 0.5
highcut = 40.0

sos = butter(
    N=4,
    Wn=[lowcut, highcut],
    btype="bandpass",
    fs=fs,
    output="sos",
)

def preprocess(x):
    x = x.astype("float32")
    # x shape: samples, time, leads
    y = sosfiltfilt(sos, x, axis=1).astype("float32")
    return y

train = np.load(src / "train_data.npy")
train_f = preprocess(train)

mean = train_f.mean(axis=(0, 1), keepdims=True)
std = train_f.std(axis=(0, 1), keepdims=True) + 1e-6

for split in ["train", "val", "test"]:
    x = np.load(src / f"{split}_data.npy")
    y = preprocess(x)
    y = ((y - mean) / std).astype("float32")
    np.save(dst / f"{split}_data.npy", y)

    labels = np.load(src / f"{split}_labels.npy")
    np.save(dst / f"{split}_labels.npy", labels)

print("Created:", dst)
print("Train shape:", np.load(dst / "train_data.npy").shape)
print("Train mean:", np.load(dst / "train_data.npy").mean())
print("Train std:", np.load(dst / "train_data.npy").std())