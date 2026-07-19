> **Status:** Planned for Further Development
>
> This project is planned to be improved. Planned enhancements include improved noise robustness, deep learning models, and a web-based interface for real-time inference.

# Tabla Bol Recognition using Machine Learning

## Overview
This project classifies Tabla bols using Machine Learning and audio signal processing. It uses MFCC features extracted from audio recordings and a Random Forest classifier to identify different bols.

## Classes
- dha
- dhin
- ghe
- kii
- taa
- tete
- tinn
- tirekite

## Tech Stack
- Python
- Librosa (audio processing)
- Scikit-learn (ML)
- NumPy

## Workflow
1. Audio recording of Tabla bols
2. Feature extraction using MFCC
3. Model training using Random Forest
4. Prediction on new audio samples

## Performance
- Accuracy: ~96% (cross-validation)
- 8-class classification

## How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
python src/predict.py