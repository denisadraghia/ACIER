import pycountry
import pymrio
import pandas as pd
import matplotlib.pyplot as plt
import country_converter as coco

def country_list_2letters():
    countries = {}
    for country in pycountry.countries:
        countries[country.name] = country.alpha_2
    return countries

def read_matrix(year):
    '''Read input-output matrix for a given year'''
    exio3=pymrio.parse_exiobase3('IOT_'+str(year)+'_pxp.zip')
    exio3.calc_all()
    return exio3

def aggregate_sectors(sector,specific_sector,rebuilding):
    '''Agregates the sectors'''
    dict_sect={'steel':['Basic iron and steel and of ferro-alloys and first products thereof',
                    'Secondary steel for treatment, Re-processing of secondary steel into new steel',
                    'Oxygen Steel Furnace Gas'],
                'cement':['Cement, lime and plaster'],
                'electricity':['Electricity by coal','Electricity by gas','Electricity by nuclear',
                               'Electricity by hydro','Electricity by wind','Electricity by petroleum and other oil derivatives',
                               'Electricity by biomass and waste','Electricity by solar photovoltaic',
                               'Electricity by solar thermal','Electricity by tide, wave, ocean',
                               'Electricity by Geothermal','Electricity nec']
                }
    if rebuilding==1:
        if sector in ['Bricks, tiles and construction products, in baked clay',
                      'Construction work (45)',
                      'Secondary construction material for treatment, Re-processing of secondary construction material into aggregates']:
            return 'construction'
    listt=dict_sect[specific_sector]
    if sector in listt:
        return specific_sector
    else:
        return 'other'


def aggregation(base,country,specific_sector, rebuilding):
    '''It will aggregate the Exiobase for the year_base in one country and the rest 
    of world (ROW) and one sector (among steel, cement, electricity generation for the 
    moment) and construction (if rebuilding) + the other sectors.'''
    
    exio=base.copy()
    countries=country_list_2letters()
    reg_agg_coco = coco.agg_conc(original_countries=exio.get_regions(),
                                 aggregates={countries[country]:country},
                                 missing_countries="ROW",)
    sect_ag=pd.DataFrame()
    sect_ag['original']=pd.DataFrame(exio.get_sectors())
    sect_ag['aggregated'] = sect_ag['original'].apply(lambda x: aggregate_sectors(x,specific_sector=specific_sector,rebuilding=rebuilding))
    exio.aggregate(region_agg=reg_agg_coco,sector_agg=sect_ag)
    exio.calc_all()
    return exio

def compute_total_damage(df):
    '''Computes the total damages for each sector and country over the specified period
    of time.'''
    df=df-df.loc[0]
    sum_by_columns = df.sum(axis=0)
    return sum_by_columns


def plot(df,country):
    '''Plots the evolution of the realised production. 
    It takes as argument a dataframe with the results of ARIO model's simulation and
    it plots them for a specific country '''

    df = df / df.loc[0]*100
    df.loc[:, (country, slice(None))].plot()
    plt.xlabel('Days after a disaster')
    plt.ylabel('Percentage of production')
    plt.title('Evolution of production')
