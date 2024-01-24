# We clean the data with the next 3 following scripts. Feel free to check out the cleaning process in the corresponding files in the src directory.
import os
# Make sure we are in the directory is the positivevibes directory where the exp, dat, and src directories are
# The following command assumes python starts in the directory where the file is: positivevibes/exp/

if not (os.path.exists("./dat/cleaned/")):
    os.mkdir("./dat/cleaned")

from src.CLM_001_DataCleaningSuicideRates import cleaningSuicideRate
from src.CLM_001_DataCleaningMentalSubstanceDisorders import cleaningMentalSubstanceDisorders
from src.LVV_001_DataCleaningFeaturesWHR import cleaningFeaturesWHR

cleaningSuicideRate ()
cleaningMentalSubstanceDisorders()
cleaningFeaturesWHR()
