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
import argparse

def regression(filename="data_for_regression.csv", alpha=0.1):
    data = pd.read_csv("../dat/cleaned/"+filename)
    data.drop(columns=['Unnamed: 0'], inplace=True)
    feature_names = ["Corruption", "GDP", "Generosity", "Freedom of Choice", "Social Support", "Suicide Rates",
                    "Schizophrenia", "Depression", "Anxiety", "Bipolar Disorder", "Eating Disorder", "Drug Abuse Disorder",
                    "Alcohol Abuse Disorder", "Random Data"]

    # Split the dataset into features (X) and the target variable (y)
    X = data.drop("Happiness Score", axis=1)
    y = data["Happiness Score"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

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
    mse_RF = mean_squared_error(y_test, y_pred)
    
    # Inspect feature importance
    feature_importance_RF = rf_model.feature_importances_

    # =============================================================================
    # L1 Regularization - Lasso
    # =============================================================================
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)
    # Create a Lasso model
    lasso_model = Lasso(alpha=alpha)  # Adjust the alpha parameter for regularization strength

    # Fit the model to the data
    lasso_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = lasso_model.predict(X_test)

    # Evaluate the model
    mse_LS = mean_squared_error(y_test, y_pred)

    # Print the coefficients of the model
    feature_importance_LS = lasso_model.coef_

    # =============================================================================
    # L2 Regularization - Ridge regression
    # =============================================================================

    # Create a Ridge model
    ridge_model = Ridge(alpha=alpha)  # Adjust the alpha parameter for regularization strength

    # Fit the model to the data
    ridge_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = ridge_model.predict(X_test)

    # Evaluate the model
    mse_RM = mean_squared_error(y_test, y_pred)

    # Print the coefficients of the model
    feature_importance_RM = ridge_model.coef_

    dict_for_df = {'Feature Names': feature_names+['MSE'],
          'Random Forest': np.hstack((feature_importance_RF,mse_RF.reshape(1,))), 
          'Lasso': np.hstack((feature_importance_LS,mse_LS.reshape(1,))), 
          'Ridge': np.hstack((feature_importance_RM,mse_RM.reshape(1,)))}
    df = pd.DataFrame(data=dict_for_df)
    df.set_index('Feature Names', inplace=True)
    df.to_csv('../dat/cleaned/'+str(alpha)+'_'+filename[9:-4]+'_features.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regression on the happiness and mental health datasets')
    parser.add_argument('--filename', type=str, default='data_for_regression.csv',
                        help='filename of the csv data for regression, see datacleaning.py')
    parser.add_argument('--alpha', type=int, default=0.1,
                        help='alpha for lasso and ridge regression')
    args = parser.parse_args()
    regression(args.filename, args.alpha)
