import os
import librosa
import numpy as np
import pandas as pd

# CONFIGURATION

BASE_PATH = r"C:\Users\91901\Downloads\HornBase - A Car Horns Dataset\HornBase - A Car Horns Dataset"
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
CSV_FILES = {
    "train": os.path.join(BASE_PATH, "hornbase_train.csv"),
    "test": os.path.join(BASE_PATH, "hornbase_test.csv"),
    "all": os.path.join(BASE_PATH, "hornbase.csv"),
}
SAMPLE_RATE = 22050
DURATION = 2.0
AMPLITUDE_THRESHOLD = 0.05  # adjust for sensitivity
ENERGY_THRESHOLD = 0.5      # 0–1 normalized energy threshold

# =====================================
# FEATURE EXTRACTION & RULE-BASED DETECTION
# =====================================
def detect_horn_rule_based(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        rms = librosa.feature.rms(y=y)[0]
        mean_energy = np.mean(rms)
        max_energy = np.max(rms)
        if max_energy > ENERGY_THRESHOLD or mean_energy > AMPLITUDE_THRESHOLD:
            return True, max_energy
        return False, max_energy
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return False, 0

# =====================================
# LOAD CSV AND RUN DETECTION
# =====================================
def process_dataset(csv_key="train"):
    csv_path = CSV_FILES.get(csv_key)
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    print(f"\n=== 🔊 Processing {csv_key.upper()} dataset ===")
    print(f"📄 Loaded {len(df)} samples")

    for i, row in df.iterrows():
        audio_file = row.get("filename") or row.get("file") or row.get("path")
        if not isinstance(audio_file, str):
            continue
        audio_path = os.path.join(DATASET_PATH, csv_key, os.path.basename(audio_file))
        if not os.path.exists(audio_path):
            continue

        detected, energy = detect_horn_rule_based(audio_path)
        if detected:
            print(f"🚨 ALERT: Horn detected (energy={energy:.3f}) → {audio_file}")
        else:
            print(f"✅ No horn detected ({energy:.3f}) → {audio_file}")

    print(f"✅ Completed {csv_key.upper()} dataset.\n")

# =====================================
# RUN MODULE
# =====================================
if __name__ == "__main__":
    for split in ["train", "test"]:
        process_dataset(split)

