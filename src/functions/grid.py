import numpy as np 
import pandas as pd
import itertools
from src.functions.basic_functions import globus, coordinates_change
from src.functions.weights import weigths_continent
from sklearn.neighbors import KernelDensity
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from src.GSPTDataset import GSPTDataset
path="P:\\Projets Internes\\PLADIFES\\PLADIFES DATA CREATION\\Sectorial wealth\\"
df_gspt=pd.read_csv(path+"data\\clean_data\\steel_factories_dataset_reg.csv")

def land_reference_subunit(long_min, long_max, lat_min, lat_max, step):
    '''Takes the limit coordinates of an area and creates a grid that specifies
      whether the points is in land or not'''
    xgrid=np.arange(long_min, long_max,step) #longitude
    ygrid=np.arange(lat_min, lat_max, step) #latitude
    tuples_coordinates = list(itertools.product(ygrid,xgrid))
    number_of_tuples=int((long_max-long_min)/step)
    latitude_lines = [tuples_coordinates[i:i + number_of_tuples] for i in range(0, len(tuples_coordinates), number_of_tuples)]
    latitude_lines=np.array(latitude_lines)
    Z1=[]

    for i in range(latitude_lines.shape[0]):
        Z1.append(np.array(list(map(globus, latitude_lines[i]))))

    Z1_array = np.array(Z1)
    land_reference= Z1_array.astype(int)
    land_reference_99 = [[-9999 if x == 0 else x for x in row] for row in land_reference] # -9999 corresponds to oceans
    return np.array(land_reference_99)



def construct_grid(continent, step):
    '''Constructs grids for a certain continent with a resolution=step (measured in
     degrees). Not all continents included yet...'''
    continent_coordinates = {'Europe':(-10,42,31,70), 'Africa': (-18,51,-35,38),
                             'Central & South America':(-95,-34,-56,24),
                             'North America':(-180,-45,14,90),
                             'Australia':(112,180,-47,-10),
                             'Asia':(32,180,-10,90),
                             'world':(-180,180,-90,90)
                             }
    coordinates= continent_coordinates[continent]
    long_min, long_max, lat_min, lat_max=coordinates
    xgrid=np.arange(long_min, long_max,step)
    ygrid=np.arange(lat_min, lat_max, step)
    return xgrid, ygrid

def estimate_kernel_density_database(database,noyau,h):
    '''Takes as argument a LitPop-style database and it adds a column with a kernel
     density of parameters (noyau, h) that has been fitted on all steel plants in the
     world (weighted). It also normalizes the LitPop values.  '''
    
    kde = KernelDensity(bandwidth=h, metric='haversine',kernel=noyau).set_fit_request(sample_weight=True)
    kde.fit(np.radians(coordinates_change(df_gspt)),sample_weight=weigths_continent("world"))
    coordinates=database[['latitude','longitude']].values
    coordinates=np.radians(coordinates)
    f_kernel_world=np.exp(kde.score_samples(coordinates))
    f_kernel=f_kernel_world/f_kernel_world.sum()
    database['kernel_density']=pd.DataFrame(f_kernel)
    database["value"]=database["value"]/database["value"].sum()
    database=database.rename(columns={'value':'litpop_density'})
    return database


