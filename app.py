from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
with open('naive_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ─── CATEGORICAL LABELS ───
# Update these names to best fit your dataset's actual target outcomes
PREDICTION_MAP = {
    0: "🔴 Unlikely to Convert (Low Propensity)",
    1: "🟢 Likely to Convert (High Propensity)"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Propensity Intelligence</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --input-bg: #334155;
            --border: #475569;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        .container {
            width: 100%;
            max-width: 450px;
            padding: 20px;
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 700;
            margin: 0 0 8px 0;
            letter-spacing: -0.5px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 14px;
            margin: 0;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input, select {
            width: 100%;
            padding: 12px 14px;
            background-color: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 15px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            background-color: rgba(99, 102, 241, 0.1);
            border: 1px dashed var(--accent);
            border-radius: 8px;
            text-align: center;
        }

        .result-title {
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 1px;
            margin-bottom: 4px;
        }

        .result-value {
            font-size: 18px;
            font-weight: 700;
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="card">
            <div class="header">
                <h1>TargetInsight AI</h1>
                <p>Predictive Customer Segmentation Dashboard</p>
            </div>
            
            <form action="/predict" method="post">
                <div class="form-group">
                    <label for="gender">Demographic Gender</label>
                    <select name="gender" id="gender">
                        <option value="0">Male</option>
                        <option value="1">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="age">Age (Years)</label>
                    <input type="number" name="age" id="age" min="1" max="120" placeholder="e.g. 35" required>
                </div>

                <div class="form-group">
                    <label for="salary">Estimated Annual Salary ($)</label>
                    <input type="number" name="salary" id="salary" min="0" placeholder="e.g. 75000" required>
                </div>

                <button type="submit">Analyze Profile</button>
            </form>

            {% if prediction %}
                <div class="result-box">
                    <div class="result-title">Classification Output</div>
                    <div class="result-value">{{ prediction }}</div>
                </div>
            {% endif %}
        </div>
    </div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    salary = int(request.form['salary'])
    
    features = pd.DataFrame([[gender, age, salary]], columns=['Gender', 'Age', 'EstimatedSalary'])
    
    prediction_idx = model.predict(features)[0]
    prediction_label = PREDICTION_MAP.get(prediction_idx, "Unknown Class")
    
    return render_template_string(HTML_TEMPLATE, prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
