import pycountry
import pymrio
import pandas as pd
import matplotlib.pyplot as plt
import country_converter as coco
from boario.event import EventKapitalRebuild,EventKapitalRecover,EventArbitraryProd
from boario.simulation import Simulation  # Simulation wraps the model
from boario.extended_models import ARIOPsiModel  # The core of the model
from utils import country_list_2letters,read_matrix,aggregate_sectors, aggregation


def propagation_recovery(base,value_impact,country,sector,recovery_type,recovery_time,psi=0.8, simulation_time=730):
    '''Creates a dataframe with the evolution of production.
    We gave as argument the proportion of production capacity that has been made unfonctional
    for a time=recovery time. We assume an exogenous recovery function that can take 
    different forms(linear, concave, convexe).

    Need to add VA dictionary because the ratio capital to VA is assumed to be 
    equal to 4. (Hallegate(2008)) i.e productive_capital_to_VA_dict
    
    List of implicit parameters that can be change in ARIOPsiModel(next line) and that
    are the standard parameters used in the literature (Hallegate(2013))
    alpha_base=100% (Base overproduction capacity)
    alpha max=125%(Maximum overproduction capacity)
    alpha tau=1 year (Overproduction increase characteristic time)
    main_inv_dur= 90 days (Initial inventory size)
    inventory_restoration_tau= 60 days
    monetary_factor= 10**6 (all flows in Exiobase are in millions of $)

    psi=0.8- Inventories heterogeneity parameter. For values lower
    than 0.7 there is no forward propagation.(no shortages)
    recovery_time- nb of days needed for the exogenous recovery. Most used 
    values: 90, 180 days. (Robustness article (2023))'''
    model = ARIOPsiModel(base,psi_param=psi)
    sim=Simulation(model,n_temporal_units_to_sim=simulation_time)
    impact = pd.Series(
        data=[value_impact],
        index=pd.MultiIndex.from_product(
            [[country], [sector]], names=["region", "sector"]
        ),
    )
    ev = EventKapitalRecover.from_series(
        impact=impact,
        duration=7,  # we assume that it takes 7 days for the industry to start its recovery
        occurrence=1, # we consider 1 event
        recovery_function=recovery_type,
        recovery_time=recovery_time)
    
    #the step of the simulation is 1 day.
    sim.add_event(ev)
    sim.loop(progress=False)

    df = sim.production_realised
    return df



def propagation_prop(base,proportion,country,sector,recovery_type,recovery_time=150,psi=0.8,simulation_time=730):
    '''Creates a dataframe with the evolution of production.
    We gave as argument the proportion of production capacity that has been made unfonctional
    for a time=recovery time. We assume an exogenous recovery function that can take 
    different forms(linear, concave, convexe).'''

    model = ARIOPsiModel(base,psi_param=psi)
    sim=Simulation(model,n_temporal_units_to_sim=simulation_time)
    impact = pd.Series(
        data=[proportion],
        index=pd.MultiIndex.from_product(
            [[country], [sector]], names=["region", "sector"]
        ),
    )

    ev = EventArbitraryProd.from_series(
    impact=impact,
    occurrence=1, # we consider 1 event
    duration=7, # we assume that it takes 7 days for the industry to start its recovery
    recovery_function=recovery_type,
    recovery_time=recovery_time,)
 
    sim.add_event(ev)
    sim.loop(progress=False)

    df = sim.production_realised
    return df


def propagation_rebuilding(base,value_impact,country,sector,rebuilding_time=180,simulation_time=730,psi=0.8):
    '''Need to add VA dictionary
    rebuilding_time=180 (6 months) -from Robustness article (2023). 3 months also a 
    posible value.'''
    model = ARIOPsiModel(base,psi_param=psi)
    sim=Simulation(model,n_temporal_units_to_sim=simulation_time)
    impact = pd.Series(
        data=[value_impact],
        index=pd.MultiIndex.from_product(
            [[country], [sector]], names=["region", "sector"]
        ),
    )
    ev = EventKapitalRebuild.from_series(
    impact=impact,
    rebuild_tau=rebuilding_time,
    rebuilding_sectors={"construction": 1},
    rebuilding_factor=0.9,
    occurrence=1,
    duration=7,)
    
    sim.add_event(ev)
    sim.loop(progress=False)

    df = sim.production_realised
    
    return df




    