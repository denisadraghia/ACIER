import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.functions.grid import construct_grid,land_reference_subunit
from src.functions.basic_functions import coordinates_change
from src.functions.weights import weighted_plants_dataset, weigths_continent
from mpl_toolkits.basemap import Basemap
import sklearn
sklearn.set_config(enable_metadata_routing=True)
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

df_gspt=pd.read_csv("data\clean_data\steel_factories_dataset.csv")

def choose_bandwidth(continent,noyau):
    param_grid = {'bandwidth': np.linspace(0.001, 1, 100)}
    kde = KernelDensity(metric='haversine',kernel=noyau)
    df_gspt_europe=df_gspt.loc[df_gspt["Region"] == continent]
    X=np.radians(coordinates_change(df_gspt_europe))
    grid_search = GridSearchCV(kde, param_grid, cv=10)
    grid_search.fit(X)  
    best_bandwidth = grid_search.best_params_['bandwidth']
    return best_bandwidth



def unweighted_plot_density(continent,noyau, h):
    if continent =="world":
        steel_dataset=df_gspt
    else:
        steel_dataset=df_gspt.loc[df_gspt["Region"]==continent]
    xgrid,ygrid=construct_grid(continent,0.1)
    X, Y = np.meshgrid(xgrid, ygrid)
    land_mask = np.load("data\intermediary_data\land_mask\land_mask_"+continent +".npy")
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy = np.radians(xy[land_mask])
    fig, ax = plt.subplots(1, 1)
    m=Basemap(projection='cyl', llcrnrlat=Y.min(),urcrnrlat=Y.max(), llcrnrlon=X.min(),
                urcrnrlon=X.max(), resolution='c')
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="#87139c")
    kde = KernelDensity(bandwidth=h, metric='haversine',kernel=noyau)
    kde.fit(np.radians(coordinates_change(steel_dataset)))

    # evaluate only on the land: -9999 indicates ocean
    Z = np.full(land_mask.shape[0], -9999)
    Z[land_mask] = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    # plot contours of the density
    levels = np.linspace(0, Z.max(),100)
    ax.contourf(X, Y, Z,levels=levels,cmap='viridis')



def weighted_plants_plot_density(continent, noyau, h):
    xgrid,ygrid=construct_grid(continent,0.1)
    X, Y = np.meshgrid(xgrid, ygrid)
    land_mask = np.load("data\intermediary_data\land_mask\land_mask_"+continent +".npy")
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy = np.radians(xy[land_mask])
    fig, ax = plt.subplots(1, 1)
    m=Basemap(projection='cyl', llcrnrlat=Y.min(),urcrnrlat=Y.max(), llcrnrlon=X.min(),
                urcrnrlon=X.max(), resolution='c')
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="#87139c")
    kde = KernelDensity(bandwidth=h, metric='haversine',kernel=noyau)
    kde.fit(np.radians(weighted_plants_dataset(continent)))
    # evaluate only on the land: -9999 indicates ocean
    Z = np.full(land_mask.shape[0], -9999)
    Z[land_mask] = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    # plot contours of the density
    levels = np.linspace(0, Z.max(),100)
    ax.contourf(X, Y, Z,levels=levels,cmap='viridis')


def weighted_plot_density(continent, noyau, h):
    if continent =="world":
        steel_dataset=df_gspt
    else:
        steel_dataset=df_gspt.loc[df_gspt["Region"]==continent]
    xgrid,ygrid=construct_grid(continent,0.1)
    X, Y = np.meshgrid(xgrid, ygrid)
    land_mask = np.load("data\intermediary_data\land_mask\land_mask_"+continent +".npy")
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy = np.radians(xy[land_mask])
    fig, ax = plt.subplots(1, 1)
    m=Basemap(projection='cyl', llcrnrlat=Y.min(),urcrnrlat=Y.max(), llcrnrlon=X.min(),
                urcrnrlon=X.max(), resolution='c')
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="#87139c")
    weights=weigths_continent(continent)
    kde = KernelDensity(bandwidth=h, metric='haversine',kernel=noyau).set_fit_request(sample_weight=True)
    kde.fit(np.radians(coordinates_change(steel_dataset)),
            sample_weight=weights)

    # evaluate only on the land: -9999 indicates ocean
    Z = np.full(land_mask.shape[0], -9999)
    Z[land_mask] = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    # plot contours of the density
    levels = np.linspace(0, Z.max(),100)
    ax.contourf(X, Y, Z,levels=levels,cmap='viridis')


def format_database(litpop_database):
    work=litpop_database.copy()
    work['latitude']=work['latitude'].round(1)
    work['longitude']=work['longitude'].round(1)
    work=work.rename(columns={'value':'litpop_density'})
    work=work[['latitude','longitude','litpop_density','kernel_density']]
    work = work.groupby(['latitude', 'longitude']).agg({'litpop_density': 'sum',
                                                        'kernel_density': 'sum'}).reset_index()
    work=work.sort_values(by=['latitude', 'longitude'])
    return work

def create_linear_combination_column(database,alpha):
    database["combination"]=alpha*database["litpop_density"]+(1-alpha)*database["kernel_density"]
    return database


def plot_density_combination(litpop_database,alpha):
    formatted_data=format_database(litpop_database)
    work=create_linear_combination_column(formatted_data,alpha)
    xgrid,ygrid=construct_grid('world',0.1)
    X, Y = np.meshgrid(xgrid, ygrid)
    land_mask = np.load("data\intermediary_data\land_mask\land_mask_world.npy")
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy = xy[land_mask]
    mapping = {}
    for elem in xy:
        _key = tuple([round(elem[1],1), round(elem[0],1)])
        mapping[_key] = 0
    for idx, row in work.iterrows():
        key = (round(row.longitude, 1), round(row.latitude, 1))
        if key in mapping:
            mapping[key] = row.combination
    c=list(mapping.values())
    c=np.array(c)
    c=c*1000000
    fig, ax = plt.subplots(1, 1)
    m=Basemap(projection='cyl', llcrnrlat=Y.min(),urcrnrlat=Y.max(), llcrnrlon=X.min(),
                urcrnrlon=X.max(), resolution='c')
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="#87139c")

    Z = np.full(land_mask.shape[0], -9999)
    Z[land_mask] = c
    Z = Z.reshape(X.shape)

    # plot contours of the density
    levels = np.linspace(0, Z.max(),100)
    ax.contourf(X, Y, Z,levels=levels,cmap='viridis')
   