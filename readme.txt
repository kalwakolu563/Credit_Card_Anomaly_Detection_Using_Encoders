Credit Card Anomaly Detection Using Encoders
Introduction
This project focuses on detecting anomalies (fraudulent transactions) in credit card datasets using machine learning techniques and encoding methods. The approach uses unsupervised and supervised models along with feature engineering and streamlit-based deployment to evaluate any new data in real-time.

Overview
The project consists of the following main components:

1️⃣ EDA (Exploratory Data Analysis)
2️⃣ Feature Engineering
3️⃣ Model Building with Encoders for Anomaly Detection
4️⃣ Streamlit Application for Real-Time File Evaluation
5️⃣ Environment Setup and Requirements Installation Guide

This pipeline provides a robust way to preprocess data, detect anomalies, and evaluate predictions with useful classification metrics like F1 Score, Recall, and Precision.

Acknowledgements
Inspired by real-world credit card fraud detection problems, this project combines data science, anomaly detection, and deployment into a single end-to-end solution. It leverages encoding methods to transform features and enhance anomaly detection accuracy.

Approach
✅ Tasks Done:
1. Exploratory Data Analysis (EDA)
Initial data inspection and visualization

Distribution analysis and class imbalance check

Correlation heatmaps and transaction patterns

2. Feature Engineering
Formatting and encoding variables

Creating time-based and amount-based features

Scaling and preparing the dataset for modeling

3. Model Building
Anomaly detection using trained models

Leveraging encoders and advanced transformations

Evaluating predictions using classification metrics

Identifying outliers based on learned patterns

4. Streamlit Deployment
Upload your dataset via UI

Get instant classification report

Visual representation of evaluation metrics (F1 Score, Precision, Recall)

Easy to use interface for business users or analysts

How to Use the Project
🛠️ Setup the Environment

Create a virtual environment
python -m venv credit_card_env

Activate the environment
Windows:
credit_card_env\Scripts\activate

macOS/Linux:
source credit_card_env/bin/activate

Install the requirements
pip install -r requirements.txt

Usage
Once the environment is set up and dependencies installed:

Run EDA:
python eda.py

Perform Feature Engineering:
python feature_engineering.py

Train and Detect Anomalies:
python model_building.py

Launch the Streamlit App:
streamlit run app.py

After launching the app, upload your CSV file to get the anomaly detection report and metrics.

Conclusion
This project demonstrates an effective way of identifying anomalous or fraudulent credit card transactions by using encoder-based feature transformations. With a clean UI and classification metrics displayed on the fly, it's built for both data scientists and end users.
