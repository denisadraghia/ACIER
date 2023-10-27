import numpy as np
from math import radians, sin, cos, sqrt
from numpy import arctan2
from scipy.stats import vonmises_fisher
from sklearn.metrics.pairwise import haversine_distances

from geopy.distance import geodesic, great_circle
from global_land_mask import globe
def get_cartesian(array):
    '''Takes a (latitude,longitude) vector and returns (x,y,z) cartesian coordinates'''
    lat=array[0]
    lon=array[1]
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 1 # radius of unit sphere
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z

def coordinates_change(data):
    ''' Mettre en forme les données de la base sur l'acier pour appliquer la distance haversine'''
    data_copy = data.copy()  # Create a copy of the DataFrame
    data_copy[['latitude', 'longitude']] = data_copy['Coordinates'].str.split(',', expand=True)
    data_copy['latitude'] = data_copy['latitude'].astype(float)
    data_copy['longitude'] = data_copy['longitude'].astype(float)
    return data_copy[['latitude', 'longitude']].values



def transform_coordinates_to_lat_long(vector):
    '''Tranformer des coordonnées cartésiennes en latitude et longitude'''
    lat = np.arcsin(vector[0][2])*(180/np.pi)
    if (vector[0][0]> 0): 
        lon=np.arctan(vector[0][1]/vector[0][0])*(180/np.pi)
    elif (vector[0][1] > 0) :
        lon=round(np.arctan(vector[0][1]/vector[0][0])*(180/np.pi) + 180,6)
    else :
        lon=round(np.arctan(vector[0][1]/vector[0][0])*(180/np.pi) - 180,6) 
    return np.array([round(lat,6),round(lon,6)])


def f_for_integral_simulation(coordinates_tirage, steel_coordinates,h):
    '''Kernel exponential function'''
    coordinates_lat_lon_tirage=transform_coordinates_to_lat_long(coordinates_tirage)
    def distance(a):
        return haversine_distances([a,coordinates_lat_lon_tirage])[0][1]
    distances=list(map(distance,steel_coordinates))
    return np.exp(-sum(distances)/h)

def f(coordinate, steel_coordinates, h):
    def distance(a):
        return haversine_distances([a,coordinate])[0][1]
    distances=list(map(distance,steel_coordinates))
    return -sum(distances)/h
    

def von_mises_mu(data):
    '''Fits a von Mises distribution on a dataset and estimate a directional vector'''
    coordinates=coordinates_change(data)
    samples=[]
    for i in range(len(data)):
        samples.append(list(get_cartesian((data['latitude'].iloc[i],data['longitude'].iloc[i]))))
    mu_fit, kappa_fit = vonmises_fisher.fit(samples)
    return mu_fit


def haversine(lat1, lon1, lat2, lon2):
    '''Computes haversine distance between 2 coordinates'''
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
    return c


def geo(lat,lon,h):
    '''Computes bayesian update factor for an exponential kernel with a paramter h using
    geodesic distance- quite slow because this distance is very precise'''
    coordinate=np.array([lat,lon])
    def distance(tuple_steel_coordinate):
        return geodesic(coordinate,tuple_steel_coordinate).kilometers /6371 # rayon de la terre
    distances=list(map(distance,tuple_steel))
    return -sum(distances)/h

def geo_great_circle(lat_lon,h,tuple_steel):
    '''Computes bayesian update factor for an exponential kernel with a paramter h using
    great cirle distance'''
    lat, lon = lat_lon
    coordinate=np.array([lat,lon])
    def distance(tuple_steel_coordinate):
        return great_circle(coordinate,tuple_steel_coordinate).kilometers /6371 # rayon de la terre
    distances=list(map(distance,tuple_steel))
    return -sum(distances)/h

def geo_great_circle_gaussian_kernel(lat_lon,h,tuple_steel):
    '''Computes bayesian update factor for a gaussian kernel with a paramter h using
    great cirle distance-much more rapid than geodesic distance'''
    lat, lon = lat_lon
    coordinate=np.array([lat,lon])
    def distance(tuple_steel_coordinate):
        return (great_circle(coordinate,tuple_steel_coordinate).kilometers /6371)**2 # rayon de la terre
    distances=list(map(distance,tuple_steel))
    return -sum(distances)/(2*h**2)


def globus(simulated_coordiante):
  '''Returns whether a coordinate is inland or not'''
  return globe.is_land(*simulated_coordiante)

def list_coordinates(data):
    '''Mettre en forme les coordonnées d'un dataset'''
    coordinates=data[['latitude','longitude']]
    list_coordinates=[tuple(row) for row in coordinates.to_records(index=False)]
    return list_coordinates