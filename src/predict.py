import os
import numpy as np
import librosa
import joblib

# ----------------------------
# Project root detection
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# Load model safely
# ----------------------------
model_path = os.path.join(BASE_DIR, "models", "tabla_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)

# ----------------------------
# Feature extraction
# ----------------------------
def extract_features(file):
    try:
        audio, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error processing file:", e)
        return None

# ----------------------------
# Input
# ----------------------------
file = input("Enter full path to audio file: ")

if not os.path.exists(file):
    print("File not found!")
    exit()

features = extract_features(file)

if features is None:
    print("Could not extract features")
    exit()

features = np.array(features).reshape(1, -1)

# ----------------------------
# Prediction
# ----------------------------
prediction = model.predict(features)

print("\n🎵 Predicted Tabla Bol:", prediction[0])