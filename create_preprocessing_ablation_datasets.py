from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

src = Path("datasets/ecg_datasets/PTBXL/superdiagnostic")
base_out = Path("datasets/ecg_datasets")

fs = 100

variants = {
    "PTBXL_ABL_ZSCORE": None,
    "PTBXL_ABL_HIGHPASS_ZSCORE": ("highpass", 0.5),
    "PTBXL_ABL_BANDPASS_ZSCORE": ("bandpass", 0.5, 40.0),
}

def make_filter(spec):
    if spec is None:
        return None
    if spec[0] == "highpass":
        return butter(4, spec[1], btype="highpass", fs=fs, output="sos")
    if spec[0] == "bandpass":
        return butter(4, [spec[1], spec[2]], btype="bandpass", fs=fs, output="sos")
    raise ValueError(spec)

def apply_filter(x, sos, batch_size=512):
    x = x.astype("float32")
    if sos is None:
        return x.copy()

    out = np.empty_like(x, dtype="float32")
    total = x.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"Filtering {start}:{end}/{total}")
        out[start:end] = sosfiltfilt(sos, x[start:end], axis=1).astype("float32")
    return out

for variant, spec in variants.items():
    print("\nCreating variant:", variant)
    dst = base_out / variant / "superdiagnostic"
    dst.mkdir(parents=True, exist_ok=True)

    sos = make_filter(spec)

    train = np.load(src / "train_data.npy")
    train_proc = apply_filter(train, sos)

    mean = train_proc.mean(axis=(0, 1), keepdims=True)
    std = train_proc.std(axis=(0, 1), keepdims=True) + 1e-6

    for split in ["train", "val", "test"]:
        print(f"{variant}: processing {split}")
        x = np.load(src / f"{split}_data.npy")
        y = apply_filter(x, sos)
        y = ((y - mean) / std).astype("float32")
        np.save(dst / f"{split}_data.npy", y)

        labels = np.load(src / f"{split}_labels.npy")
        np.save(dst / f"{split}_labels.npy", labels)

    print("Saved:", dst)