from flask import Flask, request, render_template_string
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load("ada_model.pkl")

# Define feature names
feature_names = [
    "CONS_ALCOHOL",
    "RH_FACTOR",
    "SMOKER",
    "PLAN_PREGNANCY",
    "BLOOD_GROUP",
    "HAS_PREG_RISK",
    "TET_VACCINE",
    "IS_HEAD_FAMILY",
    "MARITAL_STATUS",
    "FOOD_INSECURITY",
    "NUM_ABORTIONS",
    "NUM_LIV_CHILDREN",
    "NUM_PREGNANCIES",
    "FAM_PLANNING",
    "TYPE_HOUSE",
    "HAS_FAM_INCOME",
    "LEVEL_SCHOOLING",
    "CONN_SEWER_NET",
    "NUM_RES_HOUSEHOLD",
    "HAS_FRU_TREE",
    "HAS_VEG_GARDEN",
    "FAM_INCOME",
    "HOUSING_STATUS",
    "WATER_TREATMENT",
    "AGE",
]

# HTML template with animations and friendly interface
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>VDRL_RESULT Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            text-align: center;
        }
        form {
            display: grid;
            gap: 15px;
            max-width: 400px;
            margin: auto;
        }
        input[type="number"], input[type="text"] {
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            padding: 10px;
            font-size: 16px;
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            display: inline-block;
            font-size: 20px;
            font-weight: bold;
        }
        .positive {
            color: white;
            background-color: #28a745;
            animation: pop 0.5s ease-in-out;
        }
        .negative {
            color: white;
            background-color: #007bff;
            animation: pop 0.5s ease-in-out;
        }
        @keyframes pop {
            0% { transform: scale(0.9); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .emoji {
            font-size: 50px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Predict VDRL Result</h1>
    <form action="/predict" method="post">
        {% for feature in feature_names %}
            <label for="{{ feature }}">{{ feature.replace("_", " ").title() }}:</label>
            <input type="number" step="any" id="{{ feature }}" name="{{ feature }}" required>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>

    {% if result %}
        <div class="result {{ 'positive' if result == 'Positive Outcome' else 'negative' }}">
            <div class="emoji">{{ '🎉' if result == 'Positive Outcome' else '😊' }}</div>
            {{ result }}
        </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get form data
            form_data = request.form

            # Ensure all required features are present
            features = []
            for feature in feature_names:
                if feature not in form_data:
                    return "Missing feature: {}".format(feature), 400
                features.append(float(form_data[feature]))

            # Convert features to numpy array
            features = np.array(features).reshape(1, -1)

            # Make prediction using the model
            prediction = model.predict(features)

            # Map prediction to user-friendly outcome
            result = "Positive Outcome" if prediction[0] == 1 else "Negative Outcome"
            return render_template_string(html_template, feature_names=feature_names, result=result)

        except Exception as e:
            return str(e), 500

    return render_template_string(html_template, feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        form_data = request.form

        # Ensure all required features are present
        features = []
        for feature in feature_names:
            if feature not in form_data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            features.append(float(form_data[feature]))

        # Convert features to numpy array
        features = np.array(features).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(features)

        # Display the prediction on the same page
        return render_template_string(
            html_template + "<h2>Prediction: {{ prediction }}</h2>",
            feature_names=feature_names,
            prediction=int(prediction[0])
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the app
    app.run(debug=True)
