from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model_path = "model"
model = joblib.load(os.path.join(model_path, "xgb_flight_price_model.joblib"))
encoders = {
    col: joblib.load(os.path.join(model_path, f"{col}_label_encoder.joblib"))
    for col in ['from', 'to', 'flightType', 'agency']
}

@app.route('/')
def home():
    return "🛫 Welcome to the Flight Price Prediction API!"

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        features = [
            encoders['from'].transform([data['from']])[0],
            encoders['to'].transform([data['to']])[0],
            encoders['flightType'].transform([data['flightType']])[0],
            encoders['agency'].transform([data['agency']])[0],
            float(data['time']),
            float(data['distance'])
        ]
        prediction = model.predict([features])[0]
        return jsonify({"predicted_price": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
