# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:19:56 2024

@author: Carla and Luciana
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np

data = pd.read_csv("../dat/cleaned/data_for_regression.csv")
feature_names = ["Corruption", "GDP", "Generosity", "Freedom of Choice", "Social Support", "Suicide Rates",
                 "Schizophrenia", "Depression", "Anxiety", "Bipolar Disorder", "Eating Disorder", "Drug Abuse Disorder",
                 "Alcohol Abuse Disorder", "Random Data"]

# Split the dataset into features (X) and the target variable (y)
X = data.drop("Happiness Score", axis=1)
y = data["Happiness Score"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =============================================================================
# APPLY REGRESSION MODEL AND DISPLAY RESULTS
# =============================================================================
# Create a Random Forest Regressor
rf_model = RandomForestRegressor(max_depth=None, n_estimators=3000, max_features='sqrt', min_samples_leaf=1,
                                 min_samples_split=2)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")

# Inspect feature importance
feature_importance = rf_model.feature_importances_
print("Feature Importance:")
for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
    print(f"{feature}: {importance:.3f}")


# =============================================================================
# L1 Regularization - Lasso
# =============================================================================

scaler = StandardScaler()
# Extract features and target variable
X = data.drop(columns=['Happiness Score'])
X_scaled = scaler.fit_transform(X)  # standardize the features
y = data['Happiness Score']

# Create a Lasso model
lasso_model = Lasso(alpha=0.1)  # Adjust the alpha parameter for regularization strength

# Fit the model to the data
lasso_model.fit(X_scaled, y)

# Print the coefficients of the model
coefficients = pd.Series(lasso_model.coef_, index=X.columns)
print("\n\n****  Lasso Coefficients  ****")


for feature, coef in list(zip(feature_names, lasso_model.coef_)):
    print(f"{feature}: {coef:.15f}")

# =============================================================================
# L2 Regularization - Ridge regression
# =============================================================================

# Create a Ridge model
ridge_model = Ridge(alpha=1)  # Adjust the alpha parameter for regularization strength

# Fit the model to the standardized data
ridge_model.fit(X_scaled, y)

# Print the coefficients of the Ridge model
coefficients_ridge = pd.Series(ridge_model.coef_, index=X.columns)
print("\n\n****  Ridge Coefficients  ****")


for feature, coef in list(zip(feature_names, ridge_model.coef_)):
    print(f"{feature}: {coef:.15f}")
