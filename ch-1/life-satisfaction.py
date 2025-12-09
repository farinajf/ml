import matplotlib.pyplot as pyplot
import numpy as numpy
import pandas as pandas
import sklearn.linear_model

def prepare_country_stats(oecb_bli, gdp_per_capita):
    oecb_bli = oecb_bli[oecb_bli["INEQUALITY"] == "TOT"]
    oecb_bli = oecb_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    
    full_country_stats = pandas.merge(left=oecb_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)

    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(len(full_country_stats))) - set(remove_indices))

    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[keep_indices]

# Load the data
oecb_bli = pandas.read_csv("../datasets/lifesat/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pandas.read_csv("../datasets/lifesat/gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecb_bli, gdp_per_capita)
x = numpy.c_[country_stats["GDP per capita"]]
y = numpy.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
pyplot.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(x, y)

# Make a prediction for Cyprus (GDP per capita = 22587)
X_new = [[22587]]
print(model.predict(X_new))  # Output the prediction
