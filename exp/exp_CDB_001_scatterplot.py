import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from tueplots import bundles
import matplotlib.pyplot as plt
from tueplots.constants.color import rgb


bundles.icml2022(family="sans-serif", usetex=False, column="full", nrows=2)
# Plug any of those into either the rcParams or into an rc_context:
plt.rcParams.update(bundles.icml2022())
with plt.rc_context(bundles.icml2022()):
    pass

import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.axisbelow'] = True

# Scatterplot with the x-axis as the log GDP per capita, the y-axis as the happiness score
# and the size of the markers as the selected prevalence, default is eating disorder
# as it has the highest coefficient in our regression analysis
# more_countries gives the option to show the name of some countries which may be interesting
def scatter(prevalence="Eating_Disorder", more_countries = False):
    whr_df = pd.read_csv('../dat/DataForTable2.1WHR2023.csv')
    whr_df = whr_df[whr_df['year'] >= 2007]

    # We extract a list of countries as these don't have information for every year so we need to query them individually
    all_countries = ['Afghanistan', 'Albania', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bangladesh', 'Belarus', 
                    'Belgium', 'Benin', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Cambodia', 'Cameroon', 
                    'Canada', 'Chad', 'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Cyprus', 'Denmark', 'Dominican Republic', 
                    'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Ethiopia', 'Finland', 'France', 'Gabon', 'Georgia', 'Germany', 'Ghana', 
                    'Greece', 'Guatemala', 'Guinea', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 
                    'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Liberia', 'Lithuania', 'Luxembourg', 
                    'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Mongolia', 'Montenegro', 'Morocco', 
                    'Mozambique', 'Myanmar', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Pakistan', 'Panama',
                    'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Rwanda', 'Senegal', 'Serbia', 'Sierra Leone', 'Singapore', 
                    'Slovakia', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand', 'Togo', 
                    'Tunisia', 'Uganda', 'Ukraine', 'Uruguay', 'Uzbekistan', 'Yemen', 'Zambia', 'Zimbabwe']

    import json
    # We make scatter plot of the happiness score and the log GDP per capita 
    disrd_df = pd.read_csv("../dat/cleaned/"+prevalence+"_Prevalence.csv")
    cont_json = open('continents.json')
    cont_dict = json.load(cont_json)
    values_dict = {"Asia":0,"Europe":1,"Africa":2,"North America":3,"South America":4, "Oceania":5, "Germany":6}

    x, y, s, v, yr = [], [], [], [], []
    for i in range(len(all_countries)):
        country = all_countries[i]
        hdf = whr_df[(whr_df['Country name'] == country)][(whr_df['year'] == 2019)]
        mhe = disrd_df[(disrd_df['Location'] == country)]["2019"]

        x.append(hdf["Log GDP per capita"].values[0])
        y.append(hdf["Life Ladder"].values[0])
        s.append(mhe.values[0])
        v.append(values_dict[cont_dict[country]])

    x, y, s = np.array(x), np.array(y), np.array(s)

    # z is list of elements to map
    # z minimum(max) value is mapped to min(max)
    def linear_inc(z, min_rad=1, max_rad=15):
        z = np.array(z)
        slope = (max_rad - min_rad)/(max(z)-min(z))
        bias = min_rad - min(z)*slope
        return z*slope + bias, slope, bias

    from matplotlib.colors import ListedColormap
    mpl.rcParams.update({'font.size': 5})

    classes = ["Asia","Europe","Africa","North America","South America","Oceania", "Germany"]
    colors = ListedColormap(["tab:gray","tab:orange","tab:green","tab:purple","tab:cyan", "tab:blue", "w"])

    fig, ax = plt.subplots()
    s, slope, bias =linear_inc(s,5,40)

    plt.grid()
    scatter = ax.scatter(x,y,s=s, c=v,cmap=colors, linewidth=0.25, edgecolor='k')
    plt.xlabel("Log GDP per capita")
    plt.ylabel("Happiness Score")

    if more_countries:
        int_count = ["Germany", "Costa Rica", "Singapore", 
                    "Botswana", "India", "Afghanistan", 
                    "Finland", "Morocco", "Israel",
                    "Ethiopia", "Japan", "El Salvador"]
        offsets = [[0.045,-0.2],[-0.25,0.18],[-0.22,-0.25],
                [-0.21,-0.29],[-0.1,-0.3],[0,-0.18],
                [-0.45,0.0],[-0.22,-0.26],[-0.3,0],
                [0,-0.33],[0.05,-0.2],[-0.3,0.2]]
        for i in range(len(all_countries)):
            country = all_countries[i]
            if country in int_count:
                index = int_count.index(country)
                plt.text(x[i]+offsets[index][0], y[i]+offsets[index][1], country)
    else:
        country = "Germany"
        i = all_countries.index(country)
        plt.text(x[i]+0.045, y[i]-0.2, country)
                
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(handles=scatter.legend_elements()[0][:-1], labels=classes, loc="lower right")
    ax.add_artist(legend1)

    # produce a legend with a cross section of sizes from the scatter
    kw = dict(prop="sizes", num=3, alpha=0.6, fmt="{x:.2f}\%", func=lambda s: (s-bias)/slope)
    legend2 = ax.legend(*scatter.legend_elements(**kw),loc="upper left", title="Prevalence of\n"+prevalence.replace('_',' '))

    plt.savefig("scatter_happiness_gdp_"+prevalence+"_"+str(int(more_countries))+".pdf")
