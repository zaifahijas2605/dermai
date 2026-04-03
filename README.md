# DermAI Skin Disease Classifier

End-to-end deep learning project for skin disease classification using the HAM10000 dataset.

## Repository Structure

| Folder | Description |
|---|---|
| `dermai/` | Full-stack web application (FastAPI + Docker + JWT) |
| `notebooks/` | Model training notebooks (Baseline, SE, ECA, CA + Optuna tuning) |
| `data-cleaning/` | Dataset preprocessing and cleaning scripts |

## Best Model
SE Attention + MobileNetV2 (Optuna Tuned) — 73.59% accuracy, Macro F1: 0.7657


## Tech Stack
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **Attention Mechanisms:** SE, ECA, Coordinate Attention
- **Explainability:** Grad-CAM++
- **Hyperparameter Tuning:** Optuna
- **Backend:** FastAPI, PostgreSQL, JWT
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker, Docker Compose