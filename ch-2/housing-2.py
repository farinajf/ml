import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


HOUSING_PATH  = os.path.join("datasets", "housing")

################################################################
#                                                              #
################################################################
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


################################################################
#                                                              #
################################################################
def preprocess_data(data):
    data["rooms_per_household"]      = data["total_rooms"]    / data["households"]
    data["bedrooms_per_room"]        = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"]     / data["households"]
    return data


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
#                           MAIN                               #
################################################################
housing = load_housing_data()
housing = preprocess_data(housing)

#Stratified sampling based on the income category
housing, housing_test = split_train_set(housing)
