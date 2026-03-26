# Demand Forecasting using Machine Learning

## Overview

This project implements an end-to-end demand forecasting pipeline using real-world structured data.
It covers data preprocessing, anomaly handling, feature engineering, and machine learning-based prediction.

The objective was to forecast product demand for **FY26 Q2** using historical quarterly data and business-driven features.

---

## Problem Statement

Forecast demand for multiple products using:

* Historical booking data (quarterly time series)
* Business variables (Big Deal, SCMS, VMS)
* External team forecasts with bias and accuracy history

---

## Project Pipeline

Raw Data
→ Missing Value Handling
→ Anomaly Detection
→ Anomaly Correction
→ Feature Engineering
→ XGBoost Model (Final)

---

## Dataset Structure (Column Headers Only)

Due to confidentiality, the dataset is not shared. Below are the column structures used:

### Bookings Data

Cost_Rank, Product_Name, Product_Life_Cycle,
FY23_Q2, FY23_Q3, FY23_Q4, FY24_Q1, FY24_Q2, FY24_Q3, FY24_Q4,
FY25_Q1, FY25_Q2, FY25_Q3, FY25_Q4, FY26_Q1,
Demand_Planners_Forecast, Marketing_Team_Forecast, DS_Team_Forecast

---

### Forecast Accuracy Data

Cost_Rank, Product_Name,
DP_FY26Q1_Acc, DP_FY26Q1_Bias, DP_FY25Q4_Acc, DP_FY25Q4_Bias, DP_FY25Q3_Acc, DP_FY25Q3_Bias,
Mkt_FY26Q1_Acc, Mkt_FY26Q1_Bias, Mkt_FY25Q4_Acc, Mkt_FY25Q4_Bias, Mkt_FY25Q3_Acc, Mkt_FY25Q3_Bias,
DS_FY26Q1_Acc, DS_FY26Q1_Bias, DS_FY25Q4_Acc, DS_FY25Q4_Bias, DS_FY25Q3_Acc, DS_FY25Q3_Bias

---

### SCMS Data

Cost_Rank, Product_Name, Segment,
2023Q1, 2023Q2, 2023Q3, 2023Q4,
2024Q1, 2024Q2, 2024Q3, 2024Q4,
2025Q1, 2025Q2, 2025Q3, 2025Q4, 2026Q1,
Has_Negative

---

### VMS Data

Cost_Rank, Product_Name, Vertical,
2023Q1, 2023Q2, 2023Q3, 2023Q4,
2024Q1, 2024Q2, 2024Q3, 2024Q4,
2025Q1, 2025Q2, 2025Q3, 2025Q4, 2026Q1,
Has_Negative

---

### Big Deal Data

Cost_Rank, Product_Name,
MFG_2024Q2, MFG_2024Q3, MFG_2024Q4, MFG_2025Q1, MFG_2025Q2, MFG_2025Q3, MFG_2025Q4, MFG_2026Q1,
Big_2024Q2, Big_2024Q3, Big_2024Q4, Big_2025Q1, Big_2025Q2, Big_2025Q3, Big_2025Q4, Big_2026Q1,
Avg_2024Q2, Avg_2024Q3, Avg_2024Q4, Avg_2025Q1, Avg_2025Q2, Avg_2025Q3, Avg_2025Q4, Avg_2026Q1

---

## Preprocessing

### Missing Value Handling

* Lifecycle-based strategy:

  * NPI → 0 (pre-launch)
  * Sustaining → backward fill
  * Decline → minimal modification

### SCMS & VMS Cleaning

* Gap-aware filling (leading, trailing, middle)
* Interpolation and segment-based estimation

### Anomaly Detection

* Multi-method detection:

  * IQR
  * Z-score
  * Isolation Forest
* Confirmed anomalies using multiple signals

### Anomaly Correction

* Domain-driven rules
* Big Deal validation
* Rolling mean smoothing and clipping

---

## Modeling Approach

### Final Model: XGBoost

Features used:

* Lag features (lag_1, lag_2)
* SCMS, VMS
* Big Deal
* Optional team forecast (as feature)

Strategy:

* Unified ML model instead of multiple statistical models
* Avoided excessive ensemble complexity

---

## Evaluation Metric

Model performance was evaluated using a **cost-weighted accuracy metric**, defined as:

* Accuracy = 1 − |Forecast − Actual| / Actual
* Weighted by product-level cost importance

Final score = weighted average across products.

---

## Results

| Approach                   | Accuracy   |
| -------------------------- | ---------- |
| XGBoost (Model Only)       | **83.58%** |
| XGBoost (+ Team Features)  | 78.35%     |
| Rule-Based Model Selection | 63.51%     |

---

## Key Insights

* Feature engineering had higher impact than model selection
* Over-complex ensembles reduced performance
* Team forecasts are useful as features, not as final predictions
* A unified ML model outperformed hybrid and rule-based systems

---

## Experimental Approaches

### Competition Forecasting System

* Hybrid ensemble of statistical models + team forecasts
* Result: lower accuracy due to over-blending

### Model Selection Framework

* Rule-based classification (stable, trend, intermittent)
* Underperformed compared to ML approach

---

## Disclaimer

The original dataset contained real business values and is not included due to confidentiality.
Only column structures are provided.

---

## Technologies Used

* Python
* Pandas, NumPy
* XGBoost
* Statsmodels
* Scikit-learn

---

## Author

Alisha Basa
