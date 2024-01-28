import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

import numpy as np
#from sklearn.model_selection import GridSearchCV

# =============================================================================
# LOAD THE CSV FILES AS DATAFRAMES
# =============================================================================

# listCountries contains the names of the countries that we wish to do the regression on
# as a list of strings
# default is None which includes all countries
def dataForRegression(filename=None, listCountries = None):
        # Read the CSV files into a pandas DataFrame
        df_suicide_rates = pd.read_csv('./dat/cleaned/Both-Sexes-Age-Standardized-Suicide-Rates.csv')
        df_schizo = pd.read_csv('./dat/cleaned/Schizophrenia_Prevalence.csv')
        df_depression = pd.read_csv('./dat/cleaned/Depression_Prevalence.csv')
        df_anxiety = pd.read_csv('./dat/cleaned/Anxiety_Prevalence.csv')
        df_bipolar = pd.read_csv('./dat/cleaned/Bipolar_Disorder_Prevalence.csv')
        df_eating = pd.read_csv('./dat/cleaned/Eating_Disorder_Prevalence.csv')
        df_drug = pd.read_csv('./dat/cleaned/Drug_Use_Disorder_Prevalence.csv')
        df_alcohol = pd.read_csv('./dat/cleaned/Alcohol_Use_Disorder_Prevalence.csv')

        df_corruption = pd.read_csv('./dat/cleaned/Corruption.csv')
        df_GDP = pd.read_csv('./dat/cleaned/GDP.csv')
        df_generosity = pd.read_csv('./dat/cleaned/Generosity.csv')
        df_lifechoices = pd.read_csv('./dat/cleaned/Life_Choices.csv')
        df_lifeladder = pd.read_csv('./dat/cleaned/Life_Ladder.csv')
        df_socialsup = pd.read_csv('./dat/cleaned/Social_Support.csv')

        df_happiness = [df_lifeladder, df_GDP, df_socialsup, df_lifechoices, df_corruption, df_generosity]
        df_mental = [df_eating.iloc[:, 1:], df_drug.iloc[:, 1:], df_bipolar.iloc[:, 1:], df_schizo.iloc[:, 1:], df_anxiety.iloc[:, 1:], df_alcohol.iloc[:, 1:], df_depression.iloc[:, 1:], df_suicide_rates]

        # =============================================================================
        # FILL NAN VALUES USING AN IMPUTER IN HAPPINESS SCORE DATAFRAMES
        # =============================================================================
        # Create a list to store the modified DataFrames
        df_happiness_filled = []

        for df in df_happiness:
            # Transpose the DataFrame
            df_transposed = df.T

            # Drop columns with more than x NaN values
            threshold = 10
            df_transposed = df_transposed.dropna(axis=1, thresh=df_transposed.shape[0] - threshold + 1)

            # Identify the rows with numeric values (excluding the first row)
            numeric_rows = df_transposed.iloc[1:]

            # Convert the numeric values to numeric type
            numeric_rows = numeric_rows.astype(np.float64)

            df_interpolated = numeric_rows.interpolate(axis=0)

            # Combine the first row (country names) with the converted numeric values
            df_interpolated = pd.concat([df_transposed.iloc[:1], df_interpolated])

            # Instantiate the SimpleImputer with the desired strategy (e.g., 'mean')
            imputer = SimpleImputer(strategy='mean')

            # Apply imputation to the transposed DataFrame, starting from the specified row
            df_transposed_imputed = pd.DataFrame(imputer.fit_transform(df_interpolated.iloc[1:]),
                                                columns=df_interpolated.columns)

            # Concatenate the rows that were not imputed with the imputed rows
            df_imputed = pd.concat([df_interpolated.iloc[:1], df_transposed_imputed], axis=0)

            # Transpose back
            df_imputed = df_imputed.T

            # Append the modified DataFrame to the list
            df_happiness_filled.append(df_imputed)
        # =============================================================================
        # CLEAN AND FILTER DATAFRAMES SO ALL HAVE THE SAME COUNTRIES AND YEARS
        # =============================================================================

        ## FOR THE HAPPINESS SCORE DATAFRAMES

        # Extract country names from the first column of each DataFrame
        country_sets = [set(df['Unnamed: 0']) for df in df_happiness_filled]

        # Find the common set of country names
        common_countries = set.intersection(*country_sets)

        # Filter each DataFrame to keep only rows corresponding to common countries
        df_happiness_filtered = [df[df['Unnamed: 0'].isin(common_countries)] for df in df_happiness_filled]

        new_headers = list(range(2005, 2023))  

        # Iterate over each DataFrame and set the new headers starting from column 1
        for df in df_happiness_filtered:
            df.columns = [df.columns[0]] + new_headers

        ## FOR THE MENTAL ILLNESS DATAFRAMES

        # Extract country names from the first column of each DataFrame
        country_sets = [set(df['Location']) for df in df_mental]

        # Find the common set of country names
        common_countries = set.intersection(*country_sets)

        # Filter each DataFrame to keep only rows corresponding to common countries
        df_mental_filtered = [df[df['Location'].isin(common_countries)] for df in df_mental]


        ## FOR BOTH COMBINED
        # Identify the common set of years
        common_years = sorted(list(set.intersection(set(range(2005, 2023)), set(range(2000, 2020)))))

        # Identify the common set of countries based on the columns "Unnamed 0" and "Location"
        common_countries_1 = set(df_happiness_filtered[0]["Unnamed: 0"])
        common_countries_2 = set(df_mental_filtered[0]["Location"])

        # Find the intersection of common countries
        common_countries = sorted(list(common_countries_1.intersection(common_countries_2)))
        
        if listCountries != None:
            common_countries = listCountries

        # Filter DataFrames in the first list based on common years and countries
        final_df_happiness = [df[df["Unnamed: 0"].isin(common_countries)][common_years] for df in df_happiness_filtered]

        # Convert common_years to strings
        common_years_str = [str(year) for year in common_years]

        # Filter DataFrames in the second list based on common years and countries
        final_df_mental = [df[df["Location"].isin(common_countries)][common_years_str] for df in df_mental_filtered]

        # =============================================================================
        # CONVERT DATAFRAMES INTO NUMPY ARRAYS
        # =============================================================================

        df_data = []
        for df in final_df_happiness:
            df = df.values.astype(float);
            df_data.append(df)

        for df in final_df_mental:
            df = df.values.astype(float);
            df_data.append(df)

        # Set a random seed for reproducibility (optional)
        np.random.seed(42)

        # Create a 162x20 array of random numbers between 0 and 1
        random_array = np.random.rand(len(common_countries), 15)

        # =============================================================================
        # PREPARE DATA FOR THE REGRESSION MODEL FORMAT
        # =============================================================================
        target = df_data[0].flatten()
        independent_data = []

        for df in df_data[1:]:
            independent_data.append(df.flatten())

        independent_data.append(random_array.flatten())


        sorted_mental = ["Eating Disorder", "Drug Abuse Disorder", "Bipolar Disorder", "Schizophrenia", "Anxiety", "Alcohol Abuse Disorder", "Depression", "Suicide Rates", "Random Data"]
        sorted_whr = ["Log GDP per capita", "Social Support" , "Freedom of Choice", "Corruption", "Generosity"]
        feature_names = sorted_whr+sorted_mental

        # Create a DataFrame with all the data
        data = pd.DataFrame({"Happiness Score": target})

        for i, data_flat in enumerate(independent_data, 1):
            data[feature_names[i-1]] = data_flat

        identifier = ""
        if listCountries != None:
            identifier = "_"
            for country in listCountries[:min(3,len(listCountries))]:
                identifier = identifier+country[:2]

        if filename == None:
            data.to_csv('./dat/input_regression/data_for_regression'+identifier+'.csv')
        else:
            data.to_csv('./dat/input_regression/'+filename)
