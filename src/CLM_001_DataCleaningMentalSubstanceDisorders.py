# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:07:20 2024

@author: carla
"""

import pandas as pd

def cleaningMentalSubstanceDisorders():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('./dat/prevalence-by-mental-and-substance-use-disorder.csv')
    df_suiciderates = pd.read_csv('./dat/cleaned/Both-Sexes-Age-Standardized-Suicide-Rates.csv')

    # Extract the headers for the illnesses from the first row, starting from the 4th column
    illness_columns = df.columns[3:].tolist()

    # Filter the DataFrame to include only the years from 2000 to 2019
    df_filtered = df[(df['Year'] >= 2000) & (df['Year'] <= 2019)]

    # Create an empty dictionary to store DataFrames for each illness
    illness_dataframes = {}

    # Loop through each illness column and create a separate DataFrame for each illness
    for illness in illness_columns:

        # Extract relevant columns (Country, Year, and the current illness)
        subset_df = df_filtered[['Entity', 'Year', illness]]

        # Reshape the DataFrame using pivot to have countries as rows, years as columns, and illness data as values
        reshaped_df = subset_df.pivot(index='Entity', columns='Year', values=illness)

        # Store the reshaped DataFrame in the dictionary
        illness_dataframes[illness] = reshaped_df

    # Now we have five DataFrames, one for each illness, accessible using the keys in the dictionary
    schizo_df = illness_dataframes[illness_columns[0]]
    bipolar_df = illness_dataframes[illness_columns[1]]
    eating_df = illness_dataframes[illness_columns[2]]
    anxiety_df = illness_dataframes[illness_columns[3]]
    drug_df = illness_dataframes[illness_columns[4]]
    depression_df = illness_dataframes[illness_columns[5]]
    alcohol_df = illness_dataframes[illness_columns[6]]

    # Merge with the country_names_df to filter by common country names
    schizo_df = pd.merge(df_suiciderates, schizo_df, left_on='Location', right_on='Entity').iloc[:, [0] + list(range(21, 41))]
    depression_df = pd.merge(df_suiciderates, depression_df, left_on='Location', right_on='Entity').iloc[:, [0] + list(range(21, 41))]
    anxiety_df = pd.merge(df_suiciderates, anxiety_df, left_on='Location', right_on='Entity').iloc[:, [0] + list(range(21, 41))]
    bipolar_df = pd.merge(df_suiciderates, bipolar_df, left_on='Location', right_on='Entity').iloc[:, [0] + list(range(21, 41))]
    eating_df = pd.merge(df_suiciderates, eating_df, left_on='Location', right_on='Entity').iloc[:, [0] + list(range(21, 41))]
    drug_df = pd.merge(df_suiciderates, drug_df, left_on='Location', right_on='Entity').iloc[:, [0] + list(range(21, 41))]
    alcohol_df = pd.merge(df_suiciderates, alcohol_df, left_on='Location', right_on='Entity').iloc[:, [0] + list(range(21, 41))]

    schizo_df.to_csv('./dat/cleaned/Schizophrenia_Prevalence.csv')
    depression_df.to_csv('./dat/cleaned/Depression_Prevalence.csv')
    anxiety_df.to_csv('./dat/cleaned/Anxiety_Prevalence.csv')
    bipolar_df.to_csv('./dat/cleaned/Bipolar_Prevalence.csv')
    eating_df.to_csv('./dat/cleaned/Eating_Disorder_Prevalence.csv')
    drug_df.to_csv('./dat/cleaned/Drug_Use_Disorder_Prevalence.csv')
    alcohol_df.to_csv('./dat/cleaned/Alcohol_Use_Disorder_Prevalence.csv')
