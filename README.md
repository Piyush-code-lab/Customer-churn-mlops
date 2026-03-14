# 🔮 Customer Churn Prediction — MLOps Project

A production-ready, modular **MLOps** pipeline for predicting customer churn on the Telco dataset.  
Covers data ingestion → preprocessing → model training with hyperparameter tuning → evaluation → prediction API.

---

## 📁 Project Structure

```
customer_churn_mlops/
├── data/                        # Raw source data (tracked by DVC)
│   └── train.csv
├── artifacts/                   # Generated outputs (tracked by DVC)
│   ├── data_ingestion/
│   ├── preprocessor/
│   ├── models/
│   └── evaluation/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py    # Reads & splits raw data
│   │   ├── data_transformation.py # Cleans, encodes, scales features
│   │   ├── model_trainer.py     # GridSearchCV across 5 classifiers
│   │   └── model_evaluation.py  # Accuracy / F1 / ROC-AUC metrics
│   ├── pipeline/
│   │   ├── training_pipeline.py # Orchestrates all 4 stages
│   │   └── prediction_pipeline.py # Loads artifacts for inference
│   ├── config/                  # ConfigurationManager (paths)
│   ├── constants/               # Global constants
│   ├── entity/                  # Dataclasses for configs
│   ├── exception/               # Custom exception with traceback
│   ├── logger/                  # Rotating file logger
│   └── utils/                   # save/load object, evaluate_models
├── templates/
│   └── index.html               # Prediction web UI
├── notebook/                    # EDA notebooks (optional)
├── app.py                       # Flask REST API
├── main.py                      # Training entry point
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # Experiment parameters
├── setup.py
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/<your-username>/customer_churn_mlops.git
cd customer_churn_mlops

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Git + DVC

```bash
git init
git add .
git commit -m "initial commit"

pip install dvc
dvc init
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "track dataset with DVC"
```

> **Optional:** push data to remote storage (S3, GDrive, etc.)
> ```bash
> dvc remote add -d myremote s3://your-bucket/dvcstore
> dvc push
> ```

---

## 🚀 Run Training Pipeline

```bash
python main.py
```

This runs all four stages in sequence:

| Stage | What it does |
|---|---|
| Data Ingestion | Reads `data/train.csv`, saves raw copy, performs stratified train/test split |
| Data Transformation | Cleans data, builds Sklearn `ColumnTransformer`, saves `preprocessor.pkl` |
| Model Training | Trains 5 classifiers with GridSearchCV, saves best as `model.pkl` |
| Model Evaluation | Computes Accuracy, Precision, Recall, F1, ROC-AUC; saves `metrics.json` |

---

## 🧪 Run DVC Pipeline

```bash
dvc repro
```

Use `dvc metrics show` to inspect evaluation results.

---

## 🌐 Run Prediction API

```bash
python app.py
```

Visit `http://localhost:5000` for the web UI, or POST to `/predict`:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 358.2
  }'
```

**Response:**
```json
{
  "churn": "Yes",
  "churn_probability": 0.7812,
  "prediction": 1
}
```

---

## 🤖 Models & Hyperparameter Tuning

Five classifiers are evaluated with 3-fold GridSearchCV:

| Model | Key Hyperparameters Tuned |
|---|---|
| Logistic Regression | C, solver |
| Decision Tree | max_depth, min_samples_split |
| Random Forest | n_estimators, max_depth |
| Gradient Boosting | n_estimators, learning_rate, max_depth |
| XGBoost | n_estimators, learning_rate, max_depth, subsample |

The best model (highest test accuracy) is saved as `artifacts/models/model.pkl`.

---

## 📊 Artifacts

| File | Description |
|---|---|
| `artifacts/preprocessor/preprocessor.pkl` | Fitted `ColumnTransformer` (imputation + encoding + scaling) |
| `artifacts/models/model.pkl` | Best trained classifier |
| `artifacts/evaluation/metrics.json` | Full evaluation metrics |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Scikit-learn** — preprocessing, classical ML models
- **XGBoost** — gradient boosted trees
- **DVC** — data & model versioning
- **Flask** — prediction REST API
- **dill** — artifact serialization

---

## 📌 Next Steps (Deployment)

- [ ] Dockerize the Flask app
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Deploy to AWS / GCP / Azure / Heroku
- [ ] Add MLflow / Weights & Biases for experiment tracking
