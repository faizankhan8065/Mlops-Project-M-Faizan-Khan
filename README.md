# MLOps Project — mlops-project-M-Faizan-Khan

House price prediction project demonstrating a clean MLOps version-control
workflow with Git, GitHub, and VS Code.

## Project structure

```
.
├── data/                     # Raw dataset (git-ignored, not pushed)
│   └── dataset.csv
├── src/                      # Training source code
│   └── train_M-Faizan-Khan.py
├── model/                    # Trained model artifact (git-ignored, not pushed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run training

From the project root:

```bash
python src/train_M-Faizan-Khan.py
```

This loads `data/dataset.csv`, trains a Random Forest regressor, prints the
evaluation metrics, and saves the trained model to `model/model_M-Faizan-Khan.pkl`.

> Note: `data/` and `model/` are intentionally excluded from Git via
> `.gitignore`, so only source code and configuration files are versioned.
