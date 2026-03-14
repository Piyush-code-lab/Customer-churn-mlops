from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, ChurnPredictionInput
from src.logger import logger

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with customer features, returns churn prediction.
    Example body:
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        ...
    }
    """
    try:
        data = request.get_json(force=True)
        inp = ChurnPredictionInput(
            gender=data["gender"],
            SeniorCitizen=int(data["SeniorCitizen"]),
            Partner=data["Partner"],
            Dependents=data["Dependents"],
            tenure=int(data["tenure"]),
            PhoneService=data["PhoneService"],
            MultipleLines=data["MultipleLines"],
            InternetService=data["InternetService"],
            OnlineSecurity=data["OnlineSecurity"],
            OnlineBackup=data["OnlineBackup"],
            DeviceProtection=data["DeviceProtection"],
            TechSupport=data["TechSupport"],
            StreamingTV=data["StreamingTV"],
            StreamingMovies=data["StreamingMovies"],
            Contract=data["Contract"],
            PaperlessBilling=data["PaperlessBilling"],
            PaymentMethod=data["PaymentMethod"],
            MonthlyCharges=float(data["MonthlyCharges"]),
            TotalCharges=float(data["TotalCharges"]),
        )
        pipeline = PredictionPipeline()
        result = pipeline.predict(inp)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
