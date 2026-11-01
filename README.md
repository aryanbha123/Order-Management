# Order Fulfillment Time Estimator

A Streamlit app that predicts how long an order will take to fulfil, backed by a scikit-learn regression model.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

## About

This project estimates order fulfilment time (in hours) from a few operational features: item count, delivery distance, priority, warehouse load and delivery method. It includes a training script that builds a Random Forest regressor and a Streamlit front end that loads a pickled model and serves predictions through a simple form.

## Features

- **Streamlit UI** (`app.py`): five numeric inputs and an "Estimate Time" button that shows the predicted fulfilment time in hours
- **Safe model loading**: clear error messages if the model file is missing or fails to load, and predictions are disabled until a model is available
- **Training pipeline** (`model.py`):
  - EDA: summary statistics, missing-value counts, target histogram and correlation heatmap
  - Feature engineering: ordinal encoding of `Priority` (Low/Medium/High) and a `Warehouse_Load x Distance_km` interaction term
  - `StandardScaler` plus `RandomForestRegressor` tuned with `GridSearchCV` (5-fold, over `n_estimators`, `max_depth`, `min_samples_split`)
  - Evaluation with MAE, RMSE, R² and 10-fold cross-validated R², plus a feature-importance plot
  - Saves the model and scaler with joblib and runs a reload-and-predict sanity check

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas, NumPy
- Matplotlib, Seaborn
- joblib / pickle

## Project Structure

```
Order-Management/
├── app.py                       # Streamlit prediction app
├── model.py                     # EDA, training, tuning, evaluation
├── X_train.csv                  # sample feature data (120 rows)
├── fulfillment_time_model.pkl   # model loaded by app.py
├── delivery_model.pkl           # additional saved model
└── scaler.pkl                   # saved StandardScaler
```

## Data

`X_train.csv` contains the columns `Item_Count`, `Distance_km`, `Priority`, `Warehouse_Load` and `Delivery_Method`. `model.py` also expects a `Delivery_Time` target column, so add it to the CSV (or point the script at a labelled dataset) before training.

## Getting Started

### Prerequisites

- Python 3

### Install

```bash
pip install streamlit numpy pandas scikit-learn matplotlib seaborn joblib
```

### Run the app

```bash
streamlit run app.py
```

Keep `fulfillment_time_model.pkl` in the same folder as `app.py`. The app passes the five inputs to the model in this order: item count, distance (km), priority, warehouse load (%), delivery method (0/1).

### Retrain the model

```bash
python model.py
```

This writes `delivery_regressor.pkl` and `scaler.pkl`.

## Author

**Aryan Bhandari**: [GitHub](https://github.com/aryanbha123)
