# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:45:29 2024

@author: carla
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tueplots import bundles
# from tueplots.constants.color import palettes
import numpy as np

# =============================================================================
# FIRST PLOT: SELECTED FEATURE COMPARISON
# =============================================================================

def bar_plots(regression_model):
        data_general = abs(pd.read_csv("../dat/features/regression.csv")[regression_model][:-1])
        data_europe = abs(pd.read_csv("../dat/features/regression_europe.csv")[regression_model][:-1])
        data_developed = abs(pd.read_csv("../dat/features/regression_developed.csv")[regression_model][:-1])
        data_centraleurope = abs(pd.read_csv("../dat/features/regression_central_europe.csv")[regression_model][:-1])
        data_germany = abs(pd.read_csv("../dat/features/regression_germany.csv")[regression_model][:-1])

        # Will have the corresponding colors to the palettes.tue_plot array, so that the last 4 are the same as the first 4 
        # But the last 4 are not used with the first 4 in the first plot!
        sorted_mental = ["Eating Disorder", "Drug Abuse Disorder", "Bipolar Disorder", "Schizophrenia", "Anxiety", "Alcohol Abuse Disorder", "Depression", "Suicide Rates", "Random Data"]
        sorted_whr = ["Log GDP per capita", "Social Support" , "Freedom of Choice", "Corruption", "Generosity"]
        all_feature_names = sorted_whr+sorted_mental

        idxs = (data_general>0.05)+(data_europe>0.05)+(data_developed>0.05)+(data_centraleurope>0.05)+(data_germany>0.05)
        feature_names = np.array(all_feature_names)[idxs]

        if regression_model != "Lasso":
            feature_names = all_feature_names

        # Create a DataFrame for feature_names
        feature_names_df = pd.DataFrame({"Feature Names": feature_names})

        # Reset index for each DataFrame and set the custom column name
        data_general = data_general.reset_index().rename(columns={regression_model: "World"})
        data_developed = data_developed.reset_index().rename(columns={regression_model: "Developed"})
        data_europe = data_europe.reset_index().rename(columns={regression_model: "Europe"})
        data_centraleurope = data_centraleurope.reset_index().rename(columns={regression_model: "Central Europe"})
        data_germany = data_germany.reset_index().rename(columns={regression_model: "Germany"})

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
        # Define a list of distinct colors manually
        custom_palette = [
            "#17becf", "#aec7e8", "#ffbb78","#ff9896", "#98df8a", 
            "#9467bd", "#ff7f0e", "#2ca02c", "#d62728",
            "#1f77b4", "#8c564b", "#bcbd22", "#e377c2", "#7f7f7f"
        ]

        for feature in selected_rows:
            index = all_feature_names.index(feature)
            mypalette.append(custom_palette[index])

        with plt.rc_context({**bundles.icml2022(), 'font.size': 12}):
            fig, ax = plt.subplots(figsize = (7,3))
            selected_data.T.plot(kind="bar", stacked=False, ax=ax, color = mypalette)
            ax.set_ylabel("Explained Percentage (\%)", fontsize= 15)
            ax.legend(title="Features", loc = 'upper right', fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=ax.yaxis.label.get_fontsize())
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            # Setting y-axis range
            ax.set_ylim(0, 0.50)
            # Adding horizontal grid lines
            ax.grid(axis='y')
            # Ensure bars are drawn on top of grid lines
            ax.set_axisbelow(True)
            

            # Save the plot as PDF
            plt.savefig("FeatureImportanceComparison_"+regression_model.replace(" ", '')+".pdf")
            
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

        all_data.set_index('Feature Names', inplace=True)

        # Combine the two DataFrames
        combined_df = all_data.T.merge(mean_scores_df, left_index=True, right_index=True)

        # Plotting with rc_context
        regions = ["Germany", "Central Europe", "Europe", "Developed", "World"]
        cutidx, ncols = 1, 5
        if regression_model == "Lasso":
            cutidx, ncols = 5, 5
        with plt.rc_context({**bundles.icml2022()}):
            fig, ax = plt.subplots(figsize=(11, 5))
            for i in range(len(regions)):
                region = regions[i]
                feature_importances = combined_df.loc[region, combined_df.columns[:-cutidx]].astype(float)  # Convert to float

                feature_importances = feature_importances/sum(feature_importances)
                happiness_score = combined_df.loc[region, 'Happiness Score']
                left_positions = np.cumsum([0] + feature_importances.tolist())[:-1]  # Calculate left positions for each feature

                if (i==0):
                    ax.barh(region, feature_importances*happiness_score, 
                        left=left_positions*happiness_score, color=custom_palette, label=feature_names, height=0.5)
                else:
                    ax.barh(region, feature_importances*happiness_score, 
                        left=left_positions*happiness_score, color=custom_palette, height=0.5)
            
            # Adding vertical grid lines
            ax.grid(axis='x')
            # Ensure bars are drawn on top of grid lines
            ax.set_axisbelow(True)
            # Adding labels and legend
            ax.set_xlabel("Happiness Score", fontsize= 15)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=ax.xaxis.label.get_fontsize())
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
            # Create a custom legend without including "Happiness Score"
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center', 
                    bbox_to_anchor=(0.5, -0.15),ncol=ncols, fancybox=True, shadow=True, fontsize = 13)
            plt.xlim([0,7.5])
            # Save the plot as PDF
            plt.savefig("StackedFeatureImportance_"+regression_model.replace(" ", '')+".pdf")

if __name__ == "__main__":
    bar_plots("Random Forest")
    bar_plots("Lasso")
    bar_plots("Ridge")