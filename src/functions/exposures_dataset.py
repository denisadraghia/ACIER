import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs
from geopy.distance import  great_circle
from sklearn.neighbors import KernelDensity
from src.functions.basic_functions import coordinates_change
from src.functions.weights import weigths_continent
import sklearn
sklearn.set_config(enable_metadata_routing=True)

country_codes={'india':'IND','japan':'JPN','australia':'AUS','mexico':'MEX','usa':'USA'}
country_continent={'india':'Asia Pacific','japan':'Asia Pacific','australia':'Australia',
                   'mexico':'North America','usa':'North America'}
country_output={'india':1.099165*(10**11),'japan':2.125171*(10**11), 
                'australia':1.351573*(10**10),'mexico':2.066791*(10**10),
                'usa':1.301384*(10**11)}

def haversine_distance(coord1, coord2):
    return great_circle(coord1, coord2).kilometers

def plot_circle_proportion_in_land(point_coords,h):
    '''Plots a map with a circle centered in point_coords with a h radius (in km)
    and computes the percentage of it that is on land.
    Inputs:
    point_coords=(longitude, latitude)
    h (int) : radius in km
    
    Output: 
    land_circle.boundary(matplotlib.axes._subplots.AxesSubplot)'''

    # crs="EPSG:4326" means coordinates in the form (lat,lon)
    gdf_point = gpd.GeoDataFrame(geometry=[Point(point_coords)], crs="EPSG:4326") 
    buffer_radius_deg = h / 111  # transforms km to degrees
    gdf_circle = gdf_point.copy()
    gdf_circle['geometry'] = gdf_circle.buffer(buffer_radius_deg) # creates circle
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).to_crs("EPSG:4326")

    land_circle = gpd.overlay(gdf_circle, world[world['continent'] != 'Antarctica'], how='intersection')
    
    circle_area = gdf_circle.iloc[0].geometry.area 
    proportion_land = land_circle.area.sum() / circle_area
    # Plot the results
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    world.boundary.plot(ax=ax, linewidth=1)
    gdf_point.plot(ax=ax, color='red', markersize=50)
    land_circle.boundary.plot(ax=ax, color='green', linewidth=1)
    plt.title(f'{h} km Radius around the Point ({point_coords[0]}, {point_coords[1]})')
    plt.show()
    print(f"Proportion of the circle on land: {proportion_land:.3%}")



def correction_coast_factor(point_coords,h):
    '''Computes the factor by which the points that are in a h radius from
    a plant that is near the coast will be multiplied.

    Inputs:
    point_coords (float, float)=(longitude, latitude)
    h (float) : radius in km
    
    Outputs:
    1/proportion_land (float): the factor by which the density of the points which are 
    within h km from the point_coords will be multiplied.'''

    gdf_point = gpd.GeoDataFrame(geometry=[Point(point_coords)], crs="EPSG:4326")
    
    buffer_radius_deg = h / 111  #transform to degrees of latitude and longitude
    gdf_circle = gdf_point.copy()
    gdf_circle['geometry'] = gdf_circle.buffer(buffer_radius_deg)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).to_crs("EPSG:4326")


    land_circle = gpd.overlay(gdf_circle, world[world['continent'] != 'Antarctica'], how='intersection')
    circle_area = gdf_circle.iloc[0].geometry.area 
    proportion_land = land_circle.area.sum() / circle_area
    return 1/proportion_land


def database_exposure_country(input_path,country, h, noyau,asset_database, output_path):
    '''Creates a database exposure (adapted to use with Climada) for a country using 
    a kernel density estimation with paramters h and noyau.
    h is measured is km.'''
    h_kernel=round(h/6371,4) #transform in radians
    data=pd.read_hdf(input_path)
    coordinates=data[['latitude','longitude']].values
    coordinates=np.radians(coordinates)
    kde = KernelDensity(bandwidth=h_kernel, metric='haversine',kernel=noyau).set_fit_request(sample_weight=True)
    kde.fit(np.radians(coordinates_change(asset_database.loc[asset_database["Region"]==country_continent[country]])),sample_weight=weigths_continent(country_continent[country]))
    f_kernel=np.exp(kde.score_samples(coordinates))
    f_kernel=f_kernel/f_kernel.sum()
    f_kernel_output=f_kernel*country_output[country]
    data['value']=pd.DataFrame(f_kernel_output)
    data=data.rename(columns={"impf_":"impf_TC"})
    data=data[['value','latitude','longitude','impf_TC']]
    data.to_csv(output_path+country+'.csv',index=False)
    return data


def database_exposure_country_coast_corrected(input_path,asset_database,country, h, noyau,rayon, output_path):
    '''Creates a database exposure (adapted to use with Climada) for a country using 
    a kernel density estimation corrected for the plants that are close to
    the coastline.'''
    
    data=pd.read_hdf(input_path)
    coordinates=data[['latitude','longitude']].values
    coordinates=np.radians(coordinates)
    h=round(h/6371,4) #transform in radians
    kde = KernelDensity(bandwidth=h, metric='haversine',kernel=noyau).set_fit_request(sample_weight=True)
    kde.fit(np.radians(coordinates_change(asset_database.loc[asset_database["Region"]==country_continent[country]])),sample_weight=weigths_continent(country_continent[country]))
    f_kernel=np.exp(kde.score_samples(coordinates))
    data['f_kernel']=pd.DataFrame(f_kernel)
    asset_plants_country=asset_database.loc[(asset_database['Country']==country)&(asset_database['more_than_50km']==False)]
    for index,row in asset_plants_country.iterrows():
        lat,lon=row['latitude'],row['longitude']
        correction_factor=correction_coast_factor((lon,lat),h)
        data['distance'] = data.apply(lambda x: haversine_distance((lat,lon), (x['latitude'], x['longitude'])), axis=1)
        data['f_kernel'] = data.apply(lambda x: x['f_kernel'] * correction_factor if x['distance'] < rayon*h else x['f_kernel'], axis=1)
    
    f_kernel=f_kernel/f_kernel.sum()
    f_kernel_output=f_kernel*country_output[country]
    data['value']=pd.DataFrame(f_kernel_output)
    data=data.rename(columns={"impf_":"impf_TC"})
    data=data[['value','latitude','longitude','impf_TC']]
    data.to_csv(output_path+country+'.csv',index=False)
    return data