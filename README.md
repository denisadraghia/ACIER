# Pladifes Physical Risk Module
## Future economic damages estimations tied to physical disasters based on sectoral wealth densities


The objective of this project is to construct a tool that can assess the sector-specific ripple effects caused by future natural disasters, such as tropical cyclones, under various climate change scenarios.

While the direct damages of natural catastrophes carry a certain degree of uncertainty, the economic repercussions, particularly indirect damages, are even more unpredictable. These arise from disruptions in the regular operation of the economic system, such as delayed production, supply bottlenecks, and other related factors. They are signficant, amounting to more than 50% of direct damages in certain cases.

In order to fully comprehend the future physical risk in the context of climate change (that will substantially change the freaquency and intensity of tropical cyclones), we need to account for these indirect damages.

First, we estimate sectoral densities based on the physical assets for heavy industries as Steel, Cement,...




This densities will be used to estimate tropical cyclone damages using the Python module Catherina.
Once the damages computed, we'll analyse how they propagate in the economy, from one sector to another using the ARIO model and BoArio python package.

## <a id="pladifes"></a> Pladifes

Pladifes is a research program aiming at easing the research in green and sustainable finance, as well as traditional one. They are the main authors of <b>CGEE</b>. Learn more about Pladifes [here](https://www.institutlouisbachelier.org/en/pladifes-a-large-financial-and-extra-financial-database-project-2/).

Databases produced in the context of this project are available [here](https://pladifes.institutlouisbachelier.org/data/#ghg-estimations). Other Paldifes databases can be access [here (only ESG)](https://pladifes.institutlouisbachelier.org/data/) and [here (financial and soon ESG)](https://www.eurofidai.org/).


# <a id="refs"></a> References

Our approch is highly inspired from the following publications and discussion with some of the main authors:

-[`Le Guenedal, T., Drobinski, P., and Tankov, P.: Cyclone generation Algorithm including a THERmodynamic module for Integrated National damage Assessment (CATHERINA 1.0) compatible with Coupled Model Intercomparison Project (CMIP) climate data, Geosci. Model Dev., 15, 8001–8039,`](https://doi.org/10.5194/gmd-15-8001-2022)

-[`S.Hallegate,Modeling the Role of Inventories and Heterogeneity in the Assessment of the Economic Costs of Natural Disasters`](https://pubmed.ncbi.nlm.nih.gov/23834029)

-[`Samuel Juhel, Adrien Delahais, Vincent Viguie. Robustness of the evaluation of indirect costs of natural disasters: example of the ARIO model.`](https://hal.science/hal-04196749/document)