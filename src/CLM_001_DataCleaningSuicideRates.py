# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:37:57 2024

@author: carla
"""

import pandas as pd

def cleaningSuicideRate():
    # Load the suicide rate dataset
    data = pd.read_csv('./dat/Age-Standardized-Suicide-Rate-Data.csv')

    # Create a new dataset with countries as rows, years as columns, and suicide rates as values
    # There will be three datasets, one for suicide rates of both sexes combined, one for suicide rates of males and another one for females

    new_dataset_both = data[data['Dim1'] == 'Both sexes'].pivot_table(index='Location', columns='Period', values='FactValueNumeric')
    new_dataset_male = data[data['Dim1'] == 'Male'].pivot_table(index='Location', columns='Period', values='FactValueNumeric')
    new_dataset_female = data[data['Dim1'] == 'Female'].pivot_table(index='Location', columns='Period', values='FactValueNumeric')

    new_dataset_both.to_csv('./dat/cleaned/Both-Sexes-Age-Standardized-Suicide-Rates.csv')
