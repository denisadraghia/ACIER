import pycountry
import pymrio
import pandas as pd
import matplotlib.pyplot as plt
import country_converter as coco
import json

with open("countries_sectors.json", 'r') as file:
    countries = json.load(file).get('countries', {})

def load_exiobase(year):
    '''Read input-output matrix for a given year'''
    exio3=pymrio.parse_exiobase3('IOT_'+str(year)+'_ixi.zip')
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


def aggregation_sector(base,country,specific_sector, rebuilding):
    '''It will aggregate the Exiobase in one country and the rest 
    of world (ROW) and one sector (among steel, cement, electricity generation for the 
    moment) and construction (if rebuilding) + the other sectors.'''
    
    exio=base.copy()
    
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


def plot_production_evolution(df,country):
    '''Plots the evolution of the realised production. 
    It takes as argument a dataframe with the results of ARIO model's simulation and
    it plots them for a specific country '''

    df = df / df.loc[0]*100
    df.loc[:, (country, slice(None))].plot()
    plt.xlabel('Days after a disaster')
    plt.ylabel('Percentage of production')
    plt.title('Evolution of production in '+country)


def aggregation_from_excel(sector_excel_file,exio_database,country):
    '''Returns an aggregation of sectors as indicated in the sector_excel_file for the indicated
    country and the rest of the world.
    Inputs:
    sector_excel_file: excel file with the list of economic sectors and 
    exio_database: MRIO EXIOBASE3
    country(str): the country for which we do the aggregation
    
    Outputs: An aggregated database (EXIOBASE3 type)'''


    df1 = pd.read_excel(sector_excel_file, sheet_name='aggreg_input')
    df2 = pd.read_excel(sector_excel_file, sheet_name='name_input')
    merged_df = pd.merge(df1, df2, left_on='group', right_on='group_id', how='left')
    merged_df = merged_df.drop(['group', 'group_id'], axis=1)

    exio=exio_database.copy()
    
    reg_agg_coco = coco.agg_conc(original_countries=exio.get_regions(),
                                    aggregates={countries[country]:country},
                                    missing_countries="ROW",)
    sect_ag=pd.DataFrame()

    exio.aggregate(region_agg=reg_agg_coco,sector_agg=merged_df)
    exio.calc_all()
    return exio