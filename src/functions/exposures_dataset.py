import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from src.functions.basic_functions import coordinates_change
from src.functions.weights import weigths_continent,weigths_continent2
import sklearn
sklearn.set_config(enable_metadata_routing=True)

steel_plants=pd.read_csv("data\\clean_data\\steel_plants_continent_dataset.csv")

country_codes={'india':'IND','japan':'JPN','australia':'AUS','mexico':'MEX','usa':'USA'}
country_continent={'india':'Asia Pacific','japan':'Asia Pacific','australia':'Australia',
                   'mexico':'North America','usa':'North America'}
country_output={'india':1.099165*(10**11),'japan':2.125171*(10**11), 
                'australia':1.351573*(10**10),'mexico':2.066791*(10**10),
                'usa':1.301384*(10**11)}

def database_exposure_country(country, h, noyau):
    '''Creates a database exposure (adapted to use with Climada) for a country using 
    a kernel density estimation with paramters h and noyau'''
    
    data=pd.read_hdf("P:\\Projets Internes\\PLADIFES\\PLADIFES DATA CREATION\\Sectorial wealth\\data\\5 minutes\\LitPop_pc_300_arcsec_"+country_codes[country]+"_v1.hdf5")
    coordinates=data[['latitude','longitude']].values
    coordinates=np.radians(coordinates)
    kde = KernelDensity(bandwidth=h, metric='haversine',kernel=noyau).set_fit_request(sample_weight=True)
    kde.fit(np.radians(coordinates_change(steel_plants.loc[steel_plants["Region"]==country_continent[country]])),sample_weight=weigths_continent2(country_continent[country]))
    f_kernel=np.exp(kde.score_samples(coordinates))
    f_kernel=f_kernel/f_kernel.sum()
    f_kernel_output=f_kernel*country_output[country]
    data['value']=pd.DataFrame(f_kernel_output)
    data=data.rename(columns={"impf_":"impf_TC"})
    data=data[['value','latitude','longitude','impf_TC']]
    data.to_csv("new bases exposure\\"+country+"_steel.csv",index=False)
    return data
