import pandas as pd
import os

df = pd.read_csv('./dat/DataForTable2.1WHR2023.csv')

# Make a list of countries in the data base
countries_col = df["Country name"]
countries = []

for country in countries_col:
  if country not in countries:
     countries.append(country)

# Get all years in the data base
years_col = df["year"]
years = []

for year in years_col:
  if year not in years:
    years.append(year)
years.sort()

# Make an empty data base for each feature (countries as rows and years as col)
ladder_df = pd.DataFrame(index=countries, columns=years)
gdp_df = pd.DataFrame(index=countries, columns=years)
social_sup_df = pd.DataFrame(index=countries, columns=years)
corruption_df = pd.DataFrame(index=countries, columns=years)
life_choices_df = pd.DataFrame(index=countries, columns=years)
generosity_df = pd.DataFrame(index=countries, columns=years)

# Make a feature dict for each country (year:value)
for country in countries:
    df_country = df[df["Country name"] == country]
    ladder = {row["year"]: row["Life Ladder"] for index, row in df_country.iterrows()}
    gdp = {row["year"]: row["Log GDP per capita"] for index, row in df_country.iterrows()}
    social_sup = {row["year"]: row["Social support"] for index, row in df_country.iterrows()}
    corruption = {row["year"]: row["Perceptions of corruption"] for index, row in df_country.iterrows()}
    life_choices = {row["year"]: row["Freedom to make life choices"] for index, row in df_country.iterrows()}
    generosity = {row["year"]: row["Generosity"] for index, row in df_country.iterrows()}

    # Fill each feature data frame
    for i in ladder:
      value = ladder[i]
      ladder_df.at[country,i] = value

    for i in gdp:
      value = gdp[i]
      gdp_df.at[country,i] = value

    for i in social_sup:
      value = social_sup[i]
      social_sup_df.at[country,i] = value

    for i in corruption:
      value = corruption[i]
      corruption_df.at[country,i] = value

    for i in life_choices:
      value = life_choices[i]
      life_choices_df.at[country,i] = value

    for i in generosity:
      value = generosity[i]
      generosity_df.at[country,i] = value

os.chdir("./dat/cleaned/")

ladder_df.to_csv('Life_Ladder.csv', index=True)
gdp_df.to_csv('GDP.csv', index=True)
social_sup_df.to_csv('Social_Support.csv', index=True)
corruption_df.to_csv('Corruption.csv', index=True)
life_choices_df.to_csv('Life_Choices.csv', index=True)
generosity_df.to_csv('Generosity.csv', index=True) 

os.chdir("./../..")
