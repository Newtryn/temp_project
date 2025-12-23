# Solar Power Generation Forecasting System  
# End-to-End Machine Learning Pipeline Deployed on AWS EC2

An end-to-end machine learning project for forecasting hourly solar power generation (kW) using weather-based sensor data.  
The system is deployed as a live Streamlit dashboard on an AWS EC2 instance.

---

# Live Demo
**AWS EC2 + Streamlit Dashboard:**  
http://13.60.26.89:8501  

> Note: The EC2 instance must be running to access the dashboard.

---

# Project Overview
Solar power generation is highly dependent on weather conditions and is inherently intermittent.  
Accurate short-term forecasting is essential for maintaining grid stability, optimizing energy dispatch, and supporting energy management decisions.

This project presents a complete machine learning workflow, including:
- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Baseline machine learning modeling
- Model evaluation and interpretation
- Deployment of a real-time prediction dashboard on AWS EC2

The system uses simulated IoT weather sensor data to predict hourly solar power output.

---

# Problem Statement
Solar energy variability poses challenges for power grid stability and operational planning.

**Objective:**  
Predict hourly solar power generation (kW) using weather-related sensor features.

**Evaluation Metrics:**  
- R² Score (target: ≥ 0.85)  
- Root Mean Squared Error (RMSE)

---

# Tech Stack
- Python 3.12  
- Pandas, NumPy  
- Scikit-Learn  
- Streamlit  
- AWS EC2 (Ubuntu)  
- Linux CLI & Virtual Environments  

---

# Model Performance
- Model: Random Forest Regressor  
- R² Score: **0.81** (baseline)  
- RMSE: Computed during evaluation  

> The current model serves as a baseline solution.  
> Performance can be further improved using feature engineering techniques such as lag features and rolling averages.

---

# Deployment
The application is deployed on an **AWS EC2 (Ubuntu)** instance and served as a Streamlit web application.

Deployment steps:
1. EC2 instance setup
2. Python virtual environment configuration
3. Model training on the cloud
4. Streamlit dashboard deployment
5. Public access configuration via AWS Security Groups

The application runs on port **8501** and is executed as a background process using `nohup`.

---

# How to Run
```bash
git clone https://github.com/your-username/temp_project.git
cd temp_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 train_model.py
streamlit run dashboards/app.py
