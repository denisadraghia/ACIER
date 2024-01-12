import pandas as pd
import numpy as np
from src.functions.basic_functions import coordinates_change




def weigths_continent(asset_database,continent):
    '''Returns the weights of each plant proportional to its capacity on a
    specified continent'''
    if continent=="world":
        continental_database=asset_database
    else:
        continental_database=asset_database.loc[asset_database["Region"] == continent]
    continental_database=continental_database['Capacity']
    total_cap=continental_database.sum()
    my_weights=np.array(continental_database/total_cap)
    return my_weights



def weighted_plants_dataset(asset_database,continent):
    '''Returns a dataset where the a plant appears a number of times proportional to 
    its capacity'''
    if continent=="world":
        continental_database=asset_database
    else:
        continental_database=asset_database.loc[asset_database["Region"] == continent]
    expanded_data = []
    my_weights=weigths_continent(continent)
    for i, weight in enumerate(my_weights):
        expanded_data.extend([coordinates_change(continental_database)[i]] 
                         * (int(weight/min(my_weights))-1))
    expanded_data_array=np.array(expanded_data)                                                                                            
    weighted_plants=np.concatenate((coordinates_change(continental_database),
                                       expanded_data_array),axis=0)
    return weighted_plants