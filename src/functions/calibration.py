import numpy as np 
import pandas as pd
import itertools
from src.functions.basic_functions import globus, coordinates_change
from src.functions.grid import construct_grid
from src.functions.weights import weigths_continent, weighted_plants_dataset
from sklearn.neighbors import KernelDensity
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from src.GSPTDataset import GSPTDataset
path="P:\\Projets Internes\\PLADIFES\\PLADIFES DATA CREATION\\Sectorial wealth\\"
df_gspt=pd.read_csv(path+"data\\clean_data\\steel_factories_dataset_reg.csv")

def capacity_estimation(h,noyau):
    kde = KernelDensity(bandwidth=h, metric='haversine',kernel=noyau)
    kde.fit(np.radians(weighted_plants_dataset('world')))
    list_cont=['Europe','Africa','Central & South America','North America','Australia','Asia']
    dict={}
    for cont in list_cont:
        print(cont)
        xgrid,ygrid=construct_grid(cont,0.1)
        X, Y = np.meshgrid(xgrid, ygrid)
        land_mask = np.load(path+"data\intermediary_data\land_mask\land_mask_"+cont +".npy")
        xy = np.vstack([Y.ravel(), X.ravel()]).T
        xy = np.radians(xy[land_mask])
        Z = np.full(land_mask.shape[0], -9999,dtype=np.float64)
        Z[land_mask] = np.exp(kde.score_samples(xy))
        dict[cont]=Z[land_mask].sum()
    summ=sum(dict.values())
    for key in dict:
        dict[key] = dict[key] /summ
    return dict, summ

def distance(capacity):
    '''Computes performance metrics'''
    estimation = np.array(list(capacity.values()))
    true = np.array(list(capacity_continent.values()))

    mean_abs_err = mean_absolute_error(estimation, true)
    mean_sq_err = mean_squared_error(estimation, true)
    r2 = r2_score(estimation, true)

    d = {
        'mean_absolute_error': mean_abs_err,
        'mean_squared_error': mean_sq_err,
        'r2_score': r2
    }

    return d