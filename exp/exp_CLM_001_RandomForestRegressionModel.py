# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:56:34 2024

@author: carla
"""

# Explanation of the code:
# To use this model, the only code you would need to modify is the read files, and ensure
# that the initial datasets have the same size (rows and columns). The explanation
# for each step is written before the relevant code.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import GridSearchCV

# Read the CSV files into a pandas DataFrame
df_suicide_rates = pd.read_csv('Cleaned-Both-Sexes-Age-Standardized-Suicide-Rates.csv')
df_schizo = pd.read_csv('Cleaned_Schizophrenia_Prevalence.csv')
df_depression = pd.read_csv('Cleaned_Depression_Prevalence.csv')
df_anxiety = pd.read_csv('Cleaned_Anxiety_Prevalence.csv')
df_bipolar = pd.read_csv('Cleaned_Bipolar_Prevalence.csv')
df_eating = pd.read_csv('Cleaned_Eating_Disorder_Prevalence.csv')
df_drug = pd.read_csv('Cleaned_Drug_Use_Disorder_Prevalence.csv')
df_alcohol =pd.read_csv('Cleaned_Alcohol_Use_Disorder_Prevalence.csv')

# Merge with the country_names_df to filter by common country names to have datasets of the same size
df_suicide_rates = pd.merge(df_schizo, df_suicide_rates, on='Location').iloc[:, [1] + list(range(22, 42))]

# Convert DataFrames to NumPy arrays
schizo_data = df_schizo.values[:,2:].astype(float);
depression_data = df_depression.values[:,2:].astype(float);
anxiety_data = df_anxiety.values[:,2:].astype(float);
bipolar_data = df_bipolar.values[:,2:].astype(float);
eating_data = df_eating.values[:,2:].astype(float);
drug_data = df_drug.values[:,2:].astype(float);
alcohol_data = df_alcohol.values[:,2:].astype(float);
suicide_data = df_suicide_rates.values;


# Flatten the data to get the regression model input shape
suicide_rates = suicide_data[:,1:].astype(float).flatten()
independent_data = [schizo_data.flatten(), depression_data.flatten(), anxiety_data.flatten(), bipolar_data.flatten(), eating_data.flatten(), drug_data.flatten(), alcohol_data.flatten()]

# Create a DataFrame with all the data
data = pd.DataFrame({"SuicideRates": suicide_rates})
for i, data_flat in enumerate(independent_data, 1):
    data[f"Feature{i}"] = data_flat

# Split the dataset into features (X) and the target variable (y)
X = data.drop("SuicideRates", axis=1)
y = data["SuicideRates"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None]
# }

# # Create the grid search model
# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search model to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# Use the best parameters to create the final model
# rf_model = RandomForestRegressor(**best_params, random_state=42)
# rf_model.fit(X_train, y_train)

# Create a Random Forest Regressor
rf_model = RandomForestRegressor(max_depth= None,n_estimators=3000, max_features='sqrt',min_samples_leaf=1,min_samples_split=2)

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
for i, importance in enumerate(feature_importance):
    print(f"Feature {i+1}: {importance:.3f}")