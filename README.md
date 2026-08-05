https://puechased-predict-1.onrender.com


Markdown
# TargetInsight AI 🎯

**TargetInsight AI** is a lightweight machine learning web application built with Python and Flask. Powered by a Gaussian Naive Bayes classifier, it predicts customer conversion propensity based on demographic features such as **Gender**, **Age**, and **Estimated Salary**.

---

## 🚀 Features

* **Machine Learning Inference:** Real-time predictions powered by a `scikit-learn` Naive Bayes model (`naive_model.pkl`).
* **Categorical Mapping:** Automatically maps model classification predictions to human-readable categories.
* **Modern UI:** Styled with a clean, dark-mode CSS theme, responsive container layout, and clear output visualization.
* **Render-Ready:** Includes configurations required for continuous integration and deployment on Render using Gunicorn.

---

## 🛠️ Project Structure

```text
.
├── app.py                # Main Flask application with embedded templates & routes
├── naive_model.pkl       # Trained Gaussian Naive Bayes model
├── requirements.txt      # Python dependencies required for deployment
└── README.md             # Project documentation
💻 Local Installation & Setup
Follow these steps to run the application on your local machine:

Clone the Repository

Bash
git clone [https://github.com/your-username/targetinsight-ai.git](https://github.com/your-username/targetinsight-ai.git)
cd targetinsight-ai
Create a Virtual Environment

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install Dependencies

Bash
pip install -r requirements.txt
Run the Application

Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/.

🌐 Deploying to Render
This project is pre-configured for seamless deployment on Render.

Deployment Steps:
Push Code to GitHub: Ensure app.py, naive_model.pkl, and requirements.txt are committed to your GitHub repository.

Create New Web Service: Log in to Render, click New +, and select Web Service.

Connect Repository: Link your GitHub repository.

Configure Settings:

Runtime: Python 3

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

Deploy: Click Create Web Service to start the build and deployment process.

📄 Requirements
Flask: Web framework

pandas: Data manipulation for input features

numpy: Numerical operations

scikit-learn (1.6.1): Model serialization compatibility

gunicorn: Production WSGI server for Linux deployment
