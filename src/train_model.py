import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Import dataset
from dataset_builder import X, y

# ----------------------------
# Project root detection
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("\nDataset shape:", X.shape)
print("Classes:", set(y))

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
accuracy = model.score(X_test, y_test)
print("\nTest Accuracy:", accuracy)

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# Cross-validation
# ----------------------------
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X,
    y,
    cv=5
)

print("\nCross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# ----------------------------
# Save model (IMPORTANT FIXED PATH)
# ----------------------------
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "tabla_model.pkl")
joblib.dump(model, model_path)

print("\nModel saved at:", model_path)

# ----------------------------
# Sample prediction
# ----------------------------
print("\nSample prediction:")
print("Predicted:", model.predict([X_test[0]]))
print("Actual:", y_test[0])