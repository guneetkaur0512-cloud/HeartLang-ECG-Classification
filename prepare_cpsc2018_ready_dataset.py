import argparse
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from tqdm import tqdm


PHYSIONET_CPSC_BASE = (
    "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018"
)


def cpsc_physionet_group(record_name):
    record_number = int(record_name[1:])
    return f"g{((record_number - 1) // 1000) + 1}"


def normalize_data(data, new_min=-3.0, new_max=3.0):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.float32)
    return (new_max - new_min) * (data - min_val) / (max_val - min_val) + new_min


def resample_signals(signals, original_fs=500, target_fs=100):
    batch_size, num_leads, _ = signals.shape
    num_original_samples = signals.shape[2]
    num_target_samples = int(num_original_samples * (target_fs / original_fs))
    resampled = np.zeros((batch_size, num_leads, num_target_samples), dtype=np.float32)
    for b in range(batch_size):
        for lead in range(num_leads):
            resampled[b, lead, :] = signal.resample(signals[b, lead, :], num_target_samples)
    return resampled


def load_cpsc_mat(path):
    mat = loadmat(path)

    # Original CPSC2018 MATLAB release format.
    if "ECG" in mat:
        return mat["ECG"][0][0][2]

    # PhysioNet/CinC 2020 WFDB-converted format.
    if "val" in mat:
        return mat["val"]

    raise KeyError(f"Unsupported MAT structure in {path}. Keys: {list(mat.keys())}")


def download_record(record_name, raw_dir, retries=5, sleep_seconds=5):
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{record_name}.mat"
    if target.exists():
        return target

    url = f"{PHYSIONET_CPSC_BASE}/{cpsc_physionet_group(record_name)}/{record_name}.mat"
    tmp_target = raw_dir / f"{record_name}.mat.part"

    for attempt in range(1, retries + 1):
        try:
            if tmp_target.exists():
                tmp_target.unlink()
            print(f"Downloading {record_name}.mat (attempt {attempt}/{retries})")
            urllib.request.urlretrieve(url, tmp_target)
            tmp_target.replace(target)
            break
        except Exception as exc:
            if tmp_target.exists():
                tmp_target.unlink()
            if attempt == retries:
                raise
            print(f"Download failed for {record_name}.mat: {exc}. Retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    return target


def process_split(split, csv_dir, raw_dir, out_dir, download_missing):
    csv_path = csv_dir / f"icbeb_{split}.csv"
    df = pd.read_csv(csv_path)

    ecg_rows = []
    label_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        record_name = row["filename"]
        mat_path = raw_dir / f"{record_name}.mat"

        if not mat_path.exists():
            if not download_missing:
                raise FileNotFoundError(
                    f"Missing {mat_path}. Provide --download-missing or place raw CPSC2018 .mat files in raw_dir."
                )
            mat_path = download_record(record_name, raw_dir)

        ecg_data = load_cpsc_mat(mat_path).astype(np.float32)

        # Keep first 2500 points and pad to 5000, following the original HeartLang CPSC script.
        ecg_data = ecg_data[:, :2500]
        if ecg_data.shape[1] < 2500:
            ecg_data = np.pad(ecg_data, ((0, 0), (0, 2500 - ecg_data.shape[1])), mode="constant")
        ecg_data = np.pad(ecg_data, ((0, 0), (0, 2500)), mode="constant")

        ecg_data = normalize_data(ecg_data, new_min=-3.0, new_max=3.0)
        labels = row.iloc[7:].values.astype(np.float32)

        ecg_rows.append(ecg_data)
        label_rows.append(labels)

    stacked_ecg = np.asarray(ecg_rows, dtype=np.float32)
    stacked_labels = np.asarray(label_rows, dtype=np.float32)
    resampled = resample_signals(stacked_ecg, original_fs=500, target_fs=100)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_data.npy", resampled)
    np.save(out_dir / f"{split}_labels.npy", stacked_labels)

    print(f"{split}_data.npy: {resampled.shape}")
    print(f"{split}_labels.npy: {stacked_labels.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CPSC2018 data in HeartLang-ready NumPy format."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Folder containing raw CPSC2018 .mat files, or target folder for downloads.",
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=Path("datasets/dataset_preprocess/CPSC2018"),
        help="Folder containing icbeb_train.csv, icbeb_val.csv, and icbeb_test.csv.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("datasets/ecg_datasets/CPSC2018/data"),
        help="Output folder for train/val/test .npy files.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing .mat records from PhysioNet Challenge 2020 CPSC2018 mirror.",
    )
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        process_split(
            split=split,
            csv_dir=args.csv_dir,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            download_missing=args.download_missing,
        )

    print("\nBase CPSC2018 NumPy data created.")
    print("Next run:")
    print("python QRSTokenizer.py --dataset_name CPSC2018")
    print("\nThis will create:")
    print("datasets/ecg_datasets/CPSC2018_QRS/data")


if __name__ == "__main__":
    main()
