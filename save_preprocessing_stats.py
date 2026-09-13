import numpy as np
from scipy.signal import butter, sosfiltfilt

TRAIN_DATA = r"D:\HeartLang\datasets\ecg_datasets\PTBXL\superdiagnostic\train_data.npy"

OUTPUT = r"D:\HeartLang\preprocessing_stats.npz"

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

train = np.load(TRAIN_DATA).astype(np.float32)

train = sosfiltfilt(sos, train, axis=1).astype(np.float32)

mean = train.mean(axis=(0, 1), keepdims=True)
std = train.std(axis=(0, 1), keepdims=True) + 1e-6

np.savez(
    OUTPUT,
    mean=mean,
    std=std
)

print("Saved preprocessing statistics.")
print(mean.shape)
print(std.shape)