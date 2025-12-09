import os
import tarfile
import urllib
import urllib.request
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH  = os.path.join("/home/fran/PythonProjects/ml/datasets", "housing")
HOUSING_URL   = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

################################################################
# Descarga el dataset de housing y lo descomprime              #
# housing_url: URL del dataset de housing                      #
# housing_path: Ruta donde se guardará el dataset              #
################################################################
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    
    urllib.request.urlretrieve(housing_url, tgz_path)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


################################################################
# Carga el dataset de housing desde un archivo CSV             #
# housing_path: Ruta donde se encuentra el dataset de housing  #
# Returns: DataFrame con los datos del dataset de housing      #
################################################################
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


################################################################
# Describe el dataset de housing, mostrando las primeras filas,#
# información general, estadísticas descriptivas y conteo de   #
# categorías en la columna "ocean_proximity"                   #
# housing: DataFrame con los datos del dataset de housing      #
################################################################
def describe(data):
    print(data.head())
    print("---------------------------------------------")
    data.info()
    print("---------------------------------------------")
    print(data.describe())
    print("---------------------------------------------")
    print(data["ocean_proximity"].value_counts())

    data.hist(bins=50, figsize=(20, 15))
    plot.show()


################################################################
# Construye el DataSet manteniendo el porcentaje de categorias #
# definidas en la columna "income_cat" que hay en el conjunto  #
# original                                                     #
################################################################
def split_train_set(data):
    data["income_cat"] = pd.cut(data["median_income"],
                                bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set  = data.loc[test_index]

    #Remove the income_cat attribute
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    return strat_train_set, strat_test_set


################################################################
#                                                              #
################################################################
def preprocess_data(data):
    data["rooms_per_household"]      = data["total_rooms"]    / data["households"]
    data["bedrooms_per_room"]        = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"]     / data["households"]
    return data


################################################################
#                                                              #
################################################################
def plot_data(data):
    data.plot(kind="scatter",
              x="longitude",
              y="latitude",
              alpha=0.4,
              s=housing["population"]/100,  #radius represents the district's population
              label="population",
              figsize=(20,14),
              c="median_house_value",       #color represents the price
              cmap=plot.get_cmap("jet"),    #color map
              colorbar=True)
    plot.legend()
    plot.show()


################################################################
#                                                              #
################################################################
def correlation_data(data, plotData=False):
    y = data.corr()
    print(y["median_house_value"].sort_values(ascending=False))

    if plotData == True:
        data.plot(kind="scatter",
                  x="median_income",
                  y="median_house_value",
                alpha=0.1)
        plot.show()


################################################################
#                           MAIN                               #
################################################################

#fetch_housing_data()
housing = load_housing_data()
#describe(housing)

housing = preprocess_data(housing)

#Stratified sampling based on the income category
housing, housing_test = split_train_set(housing)
#plot_data(housing)
#correlation_data(housing)
