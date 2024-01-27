# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:45:29 2024

@author: carla
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tueplots import bundles
from tueplots.constants.color import palettes
import numpy as np

# =============================================================================
# FIRST PLOT: SELECTED FEATURE COMPARISON
# =============================================================================
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
    selected_data.T.plot(kind="bar", stacked=False, ax=ax, color = palettes.tue_plot)
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
    
    
# =============================================================================
# SECOND PLOT: REGRESSION MODEL RESULTS COMPARED
# =============================================================================
developed_europe = ['Switzerland','Ireland','Iceland','Germany','Sweden','Australia','Netherlands','Denmark','Singapore','Finland',
             'Belgium','New Zealand','Canada','Austria','Japan','Israel','Slovenia','Luxembourg','Spain','France']

central_europe = ['Austria', 'Croatia', 'Czechia', 'Germany', 'Hungary', 'Lithuania', 'Poland', 'Slovakia', 'Slovenia', 'Switzerland']

europe = ['Luxembourg', 'Portugal', 'Ireland', 'Poland', 'Spain', 'Serbia', 'Austria', 'Denmark', 'Hungary', 'Bulgaria', 'Sweden',
           'Ukraine', 'Greece', 'Slovakia', 'Romania', 'Finland', 'Latvia', 'Malta', 'Lithuania', 'Norway', 'Netherlands', 'Albania',
            'Germany', 'Italy', 'Croatia', 'Bosnia and Herzegovina', 'France', 'Slovenia', 'Montenegro', 'Estonia', 'Belarus', 
            'Iceland', 'Switzerland', 'Czechia', 'Belgium']

happiness_score = pd.read_csv("../dat/cleaned/Life_Ladder.csv")
happiness_score.set_index('Unnamed: 0', inplace=True)

# Function to calculate the mean score for a given set of countries in 2019
def calculate_mean_score(df, countries):
    if "World" in countries:
        return df["2019"].mean()  # Calculate mean score for all countries in 2019
    else:
        filtered_df = df.loc[countries, "2019"]
        return filtered_df.mean(skipna=True)

# List of sets of countries
sets_of_countries = {"World": ["World"], "Europe": europe, "Developed Europe": developed_europe, "Central Europe": central_europe, "Germany": ["Germany"]}
mean_score = []

# Calculate the mean score for each set of countries in 2019 and store in a dictionary with custom labels
mean_scores = {}
for label, countries in sets_of_countries.items():
    mean_score = calculate_mean_score(happiness_score, countries)
    mean_scores[label] = mean_score

# Convert the mean_scores dictionary to a DataFrame
mean_scores_df = pd.DataFrame.from_dict(mean_scores, orient='index', columns=['Happiness Score'])

print("DataFrame of mean scores for each set of countries:")
print(mean_scores_df)

all_data.set_index('Feature Names', inplace=True)

# Combine the two DataFrames
combined_df = all_data.T.merge(mean_scores_df, left_index=True, right_index=True)
# Round the values in the combined_df DataFrame to 4 decimal places
combined_df = combined_df.round(4)

# Plotting with rc_context
with plt.rc_context({**bundles.icml2022()}):
    fig, ax = plt.subplots(figsize=(7, 3))

    for i, region in enumerate(combined_df.index):
        feature_importances = combined_df.loc[region, combined_df.columns[:-1]].astype(float)  # Convert to float
        happiness_score = combined_df.loc[region, 'Happiness Score']
        left_positions = np.cumsum([0] + feature_importances.tolist())[:-1]  # Calculate left positions for each feature
        ax.barh(region, happiness_score, color='lightblue', zorder=2)
        ax.barh(region, feature_importances, left=left_positions, color=palettes.tue_plot, zorder=1)

    # Adding labels and legend
    ax.set_title("Impact of Selected Features on the Happiness Score")
    ax.set_xlabel("Happiness Score")

    # Create a custom legend without including "Happiness Score"
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:], labels[1:], title="Features", loc='upper right')

    # Save the plot as PDF
    plt.savefig("StackedFeatureImportance.pdf")

    # Show the plot
    plt.show()
