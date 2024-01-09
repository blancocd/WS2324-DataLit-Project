# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:37:57 2024

@author: carla
"""

import pandas as pd

# Load the suicide rate dataset
data = pd.read_csv('Age-Standardized-Suicide-Rate-Data.csv')

# Create a new dataset with countries as rows, years as columns, and suicide rates as values
# There will be three datasets, one for suicide rates of both sexes combined, one for suicide rates of males and another one for females

new_dataset_both = data[data['Dim1'] == 'Both sexes'].pivot_table(index='Location', columns='Period', values='FactValueNumeric')
new_dataset_male = data[data['Dim1'] == 'Male'].pivot_table(index='Location', columns='Period', values='FactValueNumeric')
new_dataset_female = data[data['Dim1'] == 'Female'].pivot_table(index='Location', columns='Period', values='FactValueNumeric')

# Display the new dataset
print('Cleaned dataset of suicide rates for both sexes \n')
print(new_dataset_both)
print('Cleaned dataset of suicide rates for males \n')
print(new_dataset_male)
print('Cleaned dataset of suicide rates for females \n')
print(new_dataset_female)