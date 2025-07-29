from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import random
import os

app = Flask(__name__)
CORS(app)

import requests
OPENWEATHER_API_KEY = "12b18abeb01c0862ffe334f834871ab4"

api_key = os.getenv("BESTTIME_API_KEY") or "pri_380808470dd844bc9dc2235d1846b62c"  
venue_id = os.getenv("BESTTIME_VENUE_ID") or "ven_3437383268424c4431656e526b54446255495f713053694a496843"  


# === Load existing mental health model and encoder ===
with open("mental_health_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("gender_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

# === Load travel model and label map ===
with open("travel_model.pkl", "rb") as f:
    travel_model, label_map = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age = int(data.get("age"))
        gender = data.get("gender", "").lower()
        anxiety_score = int(data.get("anxiety_score"))
        depression_score = int(data.get("depression_score"))

        gender_encoded = gender_encoder.transform([gender])[0]
        input_features = np.array([[age, gender_encoded, anxiety_score, depression_score]])
        prediction = model.predict(input_features)[0]

        if prediction == 1:
            recommendations = [
                "We recommend you seek emotional support.",
                "Try guided meditation or journaling.",
                "If possible, consult a therapist or counselor."
            ]
        else:
            recommendations = [
                "You seem to be doing okay! Keep taking care of yourself.",
                "Continue your self-care routines.",
                "Stay active and maintain a healthy balance!"
            ]

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/travel_recommendations", methods=["POST"])
def travel_recommendations():
    try:
        data = request.get_json()

        # Get inputs
        age = int(data.get("age"))
        frequency = data.get("travelFrequency", "Daily").lower()
        trigger_intensity = int(data.get("triggerIntensity"))
        crowd_comfort = int(data.get("crowdComfort"))
        noise_tolerance = int(data.get("noiseTolerance"))
        light_sensitivity = int(data.get("lightSensitivity"))

        # Encode travel frequency
        freq_map = {"daily": 0, "weekly": 1, "rarely": 2}
        freq_encoded = freq_map.get(frequency, 0)

        features = np.array([[age, freq_encoded, crowd_comfort, noise_tolerance, light_sensitivity]])
        pred_class = travel_model.predict(features)[0]

        label_data = label_map.get(pred_class, {})

        response = {
            "optimalTravelTimes": label_data.get("optimal_times", "No recommendation"),
            "calmingTools": label_data.get("tools", []),
            "personalizedStrategies": label_data.get("strategies", [])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/weather_alert', methods=['POST'])
def weather_alert():
    data = request.get_json(force=True)

    lat = data.get('lat')
    lon = data.get('lon')

    if lat is None or lon is None:
        return jsonify({'error': 'Missing lat/lon'}), 400

    api_key = OPENWEATHER_API_KEY # Replace this with your actual key
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    response = requests.get(url)

    if response.status_code != 200:
        print("Weather API error:", response.status_code, response.text)
        return jsonify({'error': 'Failed to fetch weather'}), 500

    data = response.json()

    result = {
        'description': data['weather'][0]['description'],
        'temp': data['main']['temp'],
        'feels_like': data['main']['feels_like'],
        'humidity': data['main']['humidity']
    }

    print("Returning:", result)
    return jsonify(result)


@app.route('/crowd_alert', methods=['POST'])
def crowd_alert():
    # Simulate crowd levels randomly
    crowd_levels = ['Quiet', 'Moderate', 'Very Busy']
    intensity = random.choice(crowd_levels)

    print(f"[Crowd Alert] Returning crowd intensity: {intensity}")
    return jsonify({'crowdIntensity': intensity})



# @app.route('/crowd_alert', methods=['POST'])
# def crowd_alert():
#     try:
#         data = request.get_json()
#         api_key = data.get("api_key_private")
#         venue_id = data.get("venue_id")

#         if not api_key or not venue_id:
#             return jsonify({"error": "Missing API key or venue ID"}), 400

#         response = requests.post(
#             "https://besttime.app/api/v1/forecasts",
#             json={
#                 "api_key_private": api_key,
#                 "venue_id": venue_id
#             }
#         )

#         print(f"Status code: {response.status_code}")
#         print(f"Response: {response.text}")

#         if response.status_code != 200:
#             return jsonify({"error": "Failed to fetch data from BestTime API"}), 500

#         data = response.json()
#         forecasted = data.get("venue_forecasted", False)

#         crowd_status = "Unknown"
#         if forecasted:
#             crowd_status = "Forecast Available (Paid Feature)"
#         else:
#             crowd_status = "No Forecast Available"

#         return jsonify({"crowdIntensity": crowd_status})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    CORS(app)  # Enables frontend-backend communication

    app.run(host='0.0.0.0', port=10000)  # Render uses port 10000