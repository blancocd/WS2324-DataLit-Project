# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:45:29 2024

@author: carla
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tueplots import bundles

data_general = pd.read_csv("../dat/general_regression_output.csv").T
data_europe = pd.read_csv("../dat/europe_regression_output.csv").T
data_developed = pd.read_csv("../dat/developed_regression_output.csv").T
data_centraleurope = pd.read_csv("../dat/centraleurope_regression_output.csv").T
data_germany = pd.read_csv("../dat/germany_regression_output.csv").T

feature_names = ["Corruption", "GDP", "Generosity", "Freedom of Choice", "Social Support", "Suicide Rates",
                 "Schizophrenia", "Depression", "Anxiety", "Bipolar Disorder", "Eating Disorder", "Drug Abuse Disorder",
                 "Alcohol Abuse Disorder", "Random Data"]

# Create a DataFrame for feature_names
feature_names_df = pd.DataFrame({"Feature Names": feature_names})

# Reset index for each DataFrame and set the custom column name
data_general = data_general.reset_index().rename(columns={"index": "World"})
data_europe = data_europe.reset_index().rename(columns={"index": "Europe"})
data_developed = data_developed.reset_index().rename(columns={"index": "Developed Europe"})
data_centraleurope = data_centraleurope.reset_index().rename(columns={"index": "Central Europe"})
data_germany = data_germany.reset_index().rename(columns={"index": "Germany"})

# Concatenate the DataFrames along columns
all_data = pd.concat([feature_names_df, data_general, data_europe, data_developed, data_centraleurope, data_germany], axis=1)

# Select the specific columns for the bar chart
selected_columns = ["Feature Names", "World", "Europe", "Developed Europe", "Central Europe", "Germany"]
selected_rows = ["Corruption","GDP","Eating Disorder", "Drug Abuse Disorder"]
# Filter the DataFrame to include only selected columns and rows corresponding to feature names
selected_data = all_data[selected_columns].loc[all_data['Feature Names'].isin(selected_rows)]

# Convert values to numeric
selected_data.iloc[:, 1:] = selected_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Set "Feature Names" as the index for better visualization
selected_data.set_index("Feature Names", inplace=True)

# Define a function to format y-axis tick labels as percentages
def percentage_formatter(x, pos):
    return f'{x:.0%}'

with plt.rc_context({**bundles.icml2022()}):
    fig, ax = plt.subplots(figsize = (7,3))
    selected_data.T.plot(kind="bar", stacked=False, ax=ax)
    ax.set_title("Impact of Selected Features on the Happiness Score")
    ax.set_ylabel("Explained Percentage")
    ax.legend(title="Features", loc = 'upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=ax.yaxis.label.get_fontsize())
    # Set x-tick labels with line breaks
    column_labels = [col.replace(' ', '\n') if ' ' in col else col for col in selected_data.columns]
    plt.xticks(range(len(selected_data.columns)), column_labels)
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
   
    # Save the plot as PDF
    plt.savefig("FeatureImportanceComparison.pdf")
    # Show the plot
    plt.show()