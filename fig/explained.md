# Explaining the figures
In this short document we explain the reasoning behind our 3 main types of plots: scatterplot, vertical bar, and horizontal bar


## Scatterplot figures
### File description
They start with the `scatter_happiness_gdp` and finish with the corresponding mental health illness/disorder that is included as the size of the markers in the scatterplot. 

### Plot description
In all of these the y-axis is the happiness score and the x-axis is the log GDP per capita. The markers have also different color according to the continent they're in. Germany is in black as to differentiate it from all other countries and make it the focus of our report.

### Design choices
It is plotted this way to show the high correlation between the log GDP per capita and the happiness score while also showing how different mental illnesses/disorders impact the happier and wealthier countries than poorer and unhappier countries and viceversa.

### Possible titles
<li>Scatterplot of log GDP per capita vs Happiness Score with marker size relative to prevalence of [add disorder here] disorder</li>
<li>[add disorder here] across log GDP per capita and happiness score</li>



## Feature Importance Comparison figures
### File description
They start with the `FeatureImportanceComparison_` and finish with the corresponding regression model. 

### Plot description
In all of these the y-axis is the explained percentage according to the regression model calculated by taking the sum of all features and normalizing it to add to 1 to get percentages. The x-axis is 5 different subsets going from biggest to smallest size from left to right of which Germany is part of. 

### Design choices
Features are "log GDP per capita", "Corruption", "Eating Disorder", and "Drug Abuse Disorder". We selected the highest feature from the World Happiness Report (WHR) and one of the lowest to put it in contrast with the highest feature from the mental health datasets: "Eating Disorder Prevalence". In some subsets of the country including "World" the "Depression" importance is lower than the "Eating Disorder" proving our point that our question is relevant.

### Possible titles
<li>Comparison of economical and mental features importance across regions</li>



## Stacked Feature Importance Comparison figures
### File description
They start with the `StackedFeatureImportanceComparison_` and finish with the corresponding regression model. 

### Plot description
In all of these the y-axis is 5 different subsets going from biggest to smallest size from top to bottom of which Germany is part of. The x-axis is the happiness score.

### Design choices
Included features are those with 5% or higher percentage in the regression models coefficients/features. There are two subsets of features: 5 economical features from the World Happiness Report and 9 from the mental health dataset. 
These subsets are sorted separately with respect to the "World" dataset so that the first 5 features from left to right are all economical and the next 9 all mental.

The legend should be read column by column, top to bottom. This order corresponds to how they show in the barplot from left to right.


### Possible titles
<li>Comparison of happiness score across regions with economical and mental features importance</li>