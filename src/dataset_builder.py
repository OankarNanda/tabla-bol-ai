import os
import numpy as np
import librosa

# ----------------------------
# Project root detection
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")

X = []
y = []

def extract_features(file):
    try:
        audio, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error reading:", file, e)
        return None

labels = ["dha", "dhin", "ghe", "kii", "taa", "tete", "tinn", "tirekite"]

for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print("Missing folder:", folder)
        continue

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        features = extract_features(file_path)

        if features is None:
            continue

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset size:", len(X))
print("One sample shape:", X.shape[1:])
print("Classes:", set(y))