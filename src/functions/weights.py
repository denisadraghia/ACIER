import pandas as pd
import numpy as np
from src.functions.basic_functions import coordinates_change,globus
df_gspt=pd.read_csv("data\clean_data\steel_factories_dataset.csv")

def weigths_continent(continent):
    '''Returns the weights of each steel plant proportional to its capacity on a
    specified continent'''
    if continent=="world":
        steel_plants=df_gspt
    else:
        steel_plants=df_gspt.loc[df_gspt["Region"] == continent]
    steel_capacity=steel_plants['Nominal crude steel capacity (ttpa)']
    total_cap=steel_capacity.sum()
    my_weights=np.array(steel_capacity/total_cap)
    return my_weights


def weighted_plants_dataset(continent):
    ''''''
    if continent=="world":
        steel_plants=df_gspt
    else:
        steel_plants=df_gspt.loc[df_gspt["Region"] == continent]
    expanded_data = []
    my_weights=weigths_continent(continent)
    for i, weight in enumerate(my_weights):
        expanded_data.extend([coordinates_change(steel_plants)[i]] 
                         * (int(weight/min(my_weights))-1))
    expanded_data_array=np.array(expanded_data)                                                                                            
    weighted_plants=np.concatenate((coordinates_change(steel_plants),
                                       expanded_data_array),axis=0)
    return weighted_plants