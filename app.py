from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
# Ensure naive_model.pkl is in the same directory
with open('naive_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define your categories here
# Change these values based on what your model predicts (e.g., 0: 'Not Purchased', 1: 'Purchased')
PREDICTION_MAP = {0: 'Category A', 1: 'Category B'}

# Embedded HTML/CSS for a sleek UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; padding: 50px; }
        .card { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); width: 350px; }
        h2 { color: #333; text-align: center; }
        input, select { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
        button { width: 100%; padding: 10px; background-color: #5c67f2; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #4a54e1; }
        .result { margin-top: 20px; padding: 15px; background: #eef2ff; border-radius: 5px; text-align: center; font-weight: bold; color: #5c67f2; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Model Predictor</h2>
        <form action="/predict" method="post">
            <select name="gender">
                <option value="0">Male (0)</option>
                <option value="1">Female (1)</option>
            </select>
            <input type="number" name="age" placeholder="Age" required>
            <input type="number" name="salary" placeholder="Estimated Salary" required>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
            <div class="result">Prediction: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    salary = int(request.form['salary'])
    
    # Create DataFrame (assuming input order matches model training)
    features = pd.DataFrame([[gender, age, salary]], columns=['Gender', 'Age', 'EstimatedSalary'])
    
    # Predict
    prediction_idx = model.predict(features)[0]
    prediction_label = PREDICTION_MAP.get(prediction_idx, "Unknown")
    
    return render_template_string(HTML_TEMPLATE, prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
