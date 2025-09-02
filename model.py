"""
Delivery Time Prediction Project (Regression)
Author: Your Name
Duration: ~2 months (Data collection, cleaning, EDA, modeling, and deployment)
"""

# ==========================
# 1. IMPORTS & CONFIGURATION
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==========================
# 2. LOAD DATA
# ==========================
print("Loading dataset...")
data = pd.read_csv("X_train.csv")
print(f"Data shape: {data.shape}")
print(data.head())

# ==========================
# 3. EDA
# ==========================
print("\nPerforming Exploratory Data Analysis...")
print(data.describe())
print("\nMissing values per column:")
print(data.isnull().sum())

# Visualize target distribution (Delivery Time)
sns.histplot(data['Delivery_Time'], kde=True, bins=30)
plt.title("Target Variable Distribution (Delivery Time)")
plt.xlabel("Delivery Time")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# ==========================
# 4. FEATURE ENGINEERING
# ==========================
print("\nPerforming Feature Engineering...")

# Example: Encode priority as ordinal
priority_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
if data['Priority'].dtype == object:
    data['Priority'] = data['Priority'].map(priority_mapping)

# Example: Create interaction term
data['Load_Distance_Interaction'] = data['Warehouse_Load'] * data['Distance_km']

# ==========================
# 5. SPLIT DATA
# ==========================
X = data[['Item_Count', 'Distance_km', 'Priority', 'Warehouse_Load', 'Load_Distance_Interaction']]
y = data['Delivery_Time']   # <-- continuous target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ==========================
# 6. SCALING
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 7. MODEL TRAINING & TUNING
# ==========================
print("\nTraining RandomForestRegressor with GridSearchCV...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 15],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestRegressor(random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# ==========================
# 8. EVALUATION
# ==========================
y_pred = best_model.predict(X_test_scaled)
print("\nEvaluation Metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))

# Cross-validation for robustness
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==========================
# 9. FEATURE IMPORTANCE
# ==========================
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title="Feature Importance")
plt.show()

# ==========================
# 10. SAVE MODEL & SCALER
# ==========================
joblib.dump(best_model, "delivery_regressor.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and scaler saved successfully!")

# ==========================
# 11. DEPLOYMENT TEST
# ==========================
print("\nTesting saved regression model...")
loaded_model = joblib.load("delivery_regressor.pkl")
sample_input = X_test.iloc[0:1]
scaled_input = joblib.load("scaler.pkl").transform(sample_input)
print("Predicted delivery time:", loaded_model.predict(scaled_input)[0])
