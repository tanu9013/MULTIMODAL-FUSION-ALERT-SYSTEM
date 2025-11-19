import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import time

# ==========================
# CONFIGURATION
# ==========================
AUDIO_DATASET_PATH = r"C:\Users\91901\Downloads\HornBase - A Car Horns Dataset"
SAMPLE_RATE = 22050
DURATION = 2  # seconds
ALERT_THRESHOLD = 0.8

# ==========================
# AUDIO CLASSIFIER MODEL
# ==========================
class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 61 * 61, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Load model (randomly initialized for now)
model = AudioCNN(num_classes=2)
model.eval()

# Label mapping
CLASSES = ["no_horn", "horn"]

# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db / 80.0 + 1.0  # normalize 0–1
    mel_db = np.expand_dims(mel_db, axis=0)
    mel_db = np.expand_dims(mel_db, axis=0)
    return torch.tensor(mel_db, dtype=torch.float32)

# ==========================
# DETECT HORN SOUNDS
# ==========================
print("🚀 Scanning HornBase dataset...")
audio_files = [f for f in os.listdir(AUDIO_DATASET_PATH) if f.lower().endswith(('.wav', '.mp3'))]

if not audio_files:
    print("⚠️ No audio files found in the dataset folder!")
else:
    for audio_name in audio_files:
        audio_path = os.path.join(AUDIO_DATASET_PATH, audio_name)
        print(f"\n🔊 Processing: {audio_name}")

        # Extract features
        features = extract_features(audio_path)

        # Inference
        with torch.no_grad():
            preds = model(features)
            prob = preds[0][1].item()  # horn class probability

        if prob > ALERT_THRESHOLD:
            print(f"🚨 ALERT: Horn detected ({prob:.2f}) in {audio_name}")
        else:
            print(f"✅ No horn detected ({prob:.2f})")

print("\n🎧 Sound analysis complete.")
