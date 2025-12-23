# Solar Power Forecasting System with AWS IoT

![Python](https://img.shields.io/badge/Python-3.10-blue)
![AWS](https://img.shields.io/badge/AWS-EC2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project is an end-to-end IoT data science solution designed to predict hourly solar power generation (kW). By leveraging machine learning algorithms and cloud computing, the system aids grid operators in managing supply-demand balance efficiently.

**Live Demo:** [http://[SENIN_AWS_IP_ADRESIN]:8501](http://[SENIN_AWS_IP_ADRESIN]:8501)
*(Note: Ensure the AWS Instance is running)*

## Problem Statement
Solar energy is inherently intermittent. To ensure grid stability, accurate forecasting is crucial.
* **Goal:** Predict `generated_power_kw` using simulated sensor data.
* **Metric:** R² Score (> 0.85) and RMSE.
* **Tech Stack:** Python, Scikit-Learn, AWS EC2, Streamlit.

## Repository Structure
```bash
temp_project/
├── data/          # # raw/ and processed/ (never commit large raw files)
├── src/           # reusable modules (io, features, models)
├── models/        # saved artifacts (.pkl, .onnx)
├── dashboards/    # streamlit/dash apps
├── docs/          # proposal, reports, figures
├── notebooks/     # EDA, cleaning, feature engineering
└── README.md      # Project documentation
