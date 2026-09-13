import numpy as np

# Load the PTB-XL test dataset
test = np.load(
    r"D:\HeartLang\datasets\ecg_datasets\PTBXL\superdiagnostic\test_data.npy"
)

print("Dataset shape:", test.shape)

# Save the first ECG record
single_ecg = test[0]

np.save(
    r"D:\HeartLang\single_ecg.npy",
    single_ecg
)

print("Saved!")
print(single_ecg.shape)