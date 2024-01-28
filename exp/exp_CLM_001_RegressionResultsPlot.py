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
data_general = pd.read_csv("../dat/features/regression.csv")["Random Forest"][:-1]
data_europe = pd.read_csv("../dat/features/regression_europe.csv")["Random Forest"][:-1]
data_developed = pd.read_csv("../dat/features/regression_developed.csv")["Random Forest"][:-1]
data_centraleurope = pd.read_csv("../dat/features/regression_central_europe.csv")["Random Forest"][:-1]
data_germany = pd.read_csv("../dat/features/regression_germany.csv")["Random Forest"][:-1]

# Will have the corresponding colors to the palettes.tue_plot array, so that the last 4 are the same as the first 4 
# But the last 4 are not used with the first 4 in the first plot!
sorted_mental = ["Eating Disorder", "Drug Abuse Disorder", "Bipolar Disorder", "Schizophrenia", "Anxiety", "Alcohol Abuse Disorder", "Depression", "Suicide Rates", "Random Data"]
sorted_whr = ["Log GDP per capita", "Social Support" , "Freedom of Choice", "Corruption", "Generosity"]
feature_names = sorted_whr+sorted_mental

# Create a DataFrame for feature_names
feature_names_df = pd.DataFrame({"Feature Names": feature_names})

# Reset index for each DataFrame and set the custom column name
data_general = data_general.reset_index().rename(columns={"Random Forest": "World"})
data_developed = data_developed.reset_index().rename(columns={"Random Forest": "Developed"})
data_europe = data_europe.reset_index().rename(columns={"Random Forest": "Europe"})
data_centraleurope = data_centraleurope.reset_index().rename(columns={"Random Forest": "Central Europe"})
data_germany = data_germany.reset_index().rename(columns={"Random Forest": "Germany"})

# Concatenate the DataFrames along columns
all_data = pd.concat([feature_names_df, data_general, data_developed, data_europe, data_centraleurope, data_germany], axis=1)

# Select the specific columns for the bar chart
selected_columns = ["Feature Names", "World", "Developed", "Europe", "Central Europe", "Germany"]
selected_rows = ["Log GDP per capita","Corruption","Eating Disorder", "Drug Abuse Disorder"]
# Filter the DataFrame to include only selected columns and rows corresponding to feature names
selected_data = all_data[selected_columns].loc[all_data['Feature Names'].isin(selected_rows)]

# Convert values to numeric
selected_data.iloc[:, 1:] = selected_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Set "Feature Names" as the index for better visualization
selected_data.set_index("Feature Names", inplace=True)

# Define a function to format y-axis tick labels as percentages
def percentage_formatter(x, pos):
    return f'{x:.0%}'

mypalette = []
for feature in selected_rows:
    index = feature_names.index(feature)
    mypalette.append(palettes.tue_plot[index])

with plt.rc_context({**bundles.icml2022()}):
    fig, ax = plt.subplots(figsize = (7,3))
    selected_data.T.plot(kind="bar", stacked=False, ax=ax, color = mypalette)
    ax.set_title("Impact of Selected Features on the Happiness Score")
    ax.set_ylabel("Explained Percentage")
    ax.legend(title="Features", loc = 'upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=ax.yaxis.label.get_fontsize())
    # # Set x-tick labels with line breaks
    # column_labels = [col.replace(' ', '\n') if ' ' in col else col for col in selected_data.columns]
    # plt.xticks(range(len(selected_data.columns)), column_labels)
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
   
    # Save the plot as PDF
    plt.savefig("FeatureImportanceComparison.pdf")
    
# =============================================================================
# SECOND PLOT: REGRESSION MODEL RESULTS COMPARED
# =============================================================================
developed = ['Switzerland','Ireland','Iceland','Germany','Sweden','Australia','Netherlands','Denmark','Singapore','Finland',
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
sets_of_countries = {"World": ["World"], "Developed": developed, "Europe": europe, "Central Europe": central_europe, "Germany": ["Germany"]}
mean_score = []

# Calculate the mean score for each set of countries in 2019 and store in a dictionary with custom labels
mean_scores = {}
for label, countries in sets_of_countries.items():
    mean_score = calculate_mean_score(happiness_score, countries)
    mean_scores[label] = mean_score

# Convert the mean_scores dictionary to a DataFrame
mean_scores_df = pd.DataFrame.from_dict(mean_scores, orient='index', columns=['Happiness Score'])

# print("DataFrame of mean scores for each set of countries:")
# print(mean_scores_df)

all_data.set_index('Feature Names', inplace=True)

# Combine the two DataFrames
combined_df = all_data.T.merge(mean_scores_df, left_index=True, right_index=True)
# Round the values in the combined_df DataFrame to 4 decimal places
combined_df = combined_df.round(4)

# Plotting with rc_context
regions = ["Germany", "Central Europe", "Europe", "Developed", "World"]
with plt.rc_context({**bundles.icml2022()}):
    fig, ax = plt.subplots(figsize=(7, 3))

    for i in range(len(regions)):
        region = regions[i]
        feature_importances = combined_df.loc[region, combined_df.columns[:-1]].astype(float)  # Convert to float
        happiness_score = combined_df.loc[region, 'Happiness Score']
        left_positions = np.cumsum([0] + feature_importances.tolist())[:-1]  # Calculate left positions for each feature
        if (i==0):
            ax.barh(region, feature_importances*happiness_score, 
                left=left_positions*happiness_score, color=palettes.tue_plot, label=feature_names)
        else:
            ax.barh(region, feature_importances*happiness_score, 
                left=left_positions*happiness_score, color=palettes.tue_plot)

    # Adding labels and legend
    ax.set_title("Impact of Selected Features on the Happiness Score")
    ax.set_xlabel("Happiness Score")

    # Create a custom legend without including "Happiness Score"
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Features", loc='upper center', 
              bbox_to_anchor=(0.5, -0.15),ncol=7, fancybox=True, shadow=True)

    # Save the plot as PDF
    plt.savefig("StackedFeatureImportance.pdf")
