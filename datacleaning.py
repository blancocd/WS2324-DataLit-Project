# We clean the data with the next 3 following scripts. Feel free to check out the cleaning process in the corresponding files in the src directory.
import os
# Make sure we are in the directory is the positivevibes directory where the exp, dat, and src directories are
# The following command assumes python starts in the directory where the file is: positivevibes/exp/

if not (os.path.exists("./dat/cleaned/")):
    os.mkdir("./dat/cleaned")

from src.CLM_001_DataCleaningSuicideRates import cleaningSuicideRate
from src.CLM_001_DataCleaningMentalSubstanceDisorders import cleaningMentalSubstanceDisorders
from src.LVV_001_DataCleaningFeaturesWHR import cleaningFeaturesWHR
from src.CLM_CDB_002_DataForRegression import dataForRegression

cleaningSuicideRate()
cleaningMentalSubstanceDisorders()
cleaningFeaturesWHR()

developed = ['Switzerland','Ireland','Iceland','Germany','Sweden','Australia','Netherlands','Denmark','Singapore','Finland',
             'Belgium','New Zealand','Canada','Austria','Japan','Israel','Slovenia','Luxembourg','Spain','France']

central_europe = ['Austria', 'Croatia', 'Czechia', 'Germany', 'Hungary', 'Lithuania', 'Poland', 'Slovakia', 'Slovenia', 'Switzerland']

europe = ['Luxembourg', 'Portugal', 'Ireland', 'Poland', 'Spain', 'Serbia', 'Austria', 'Denmark', 'Hungary', 'Bulgaria', 'Sweden',
           'Ukraine', 'Greece', 'Slovakia', 'Romania', 'Finland', 'Latvia', 'Malta', 'Lithuania', 'Norway', 'Netherlands', 'Albania',
            'Germany', 'Italy', 'Croatia', 'Bosnia and Herzegovina', 'France', 'Slovenia', 'Montenegro', 'Estonia', 'Belarus', 
            'Iceland', 'Switzerland', 'Czechia', 'Belgium']

# Here we mix the datapoints and select the subset of countries that are to be included
# The default set is all countries (that were surveyed)
dataForRegression()
dataForRegression("data_for_regression_germany.csv", ["Germany"])
dataForRegression("data_for_regression_developed.csv", developed)
dataForRegression("data_for_regression_central_europe.csv", central_europe)
dataForRegression("data_for_regression_europe.csv", europe)
