# Pladifes Physical Risk Module
## Future economic damages estimations tied to physical disasters based on sectoral wealth densities


The objective of this project is to construct a tool that can assess the sector-specific ripple effects caused by future natural disasters, such as tropical cyclones (TC), under various climate change scenarios.

While the direct damages of natural catastrophes carry a certain degree of uncertainty, the economic repercussions, particularly indirect damages, are even more unpredictable. These arise from disruptions in the regular operation of the economic system, such as:
- delayed production,
- supply bottlenecks,
- and other related factors.
They are signficant, amounting to more than 50% of direct damages in certain cases.

In order to fully comprehend the future physical risk in the context of climate change (that will substantially change the freaquency and intensity of tropical cyclones), we need to account for these indirect damages.

First, we estimate sectoral densities based on the physical assets for heavy industries as Steel, Cement,.. For the sectors with unavailable data, we use the LitPop wealth density of physical assets (refer to the first reference) multiplied by a proportional factor.


The densities estimated serve as the basis for estimating sector-specific damages caused by tropical cyclones, employing the Python module <b>Catherina<b>. This module generates tropical cyclones under various climate scenarios (such as RCP 4.5, 8.5, etc.) and evaluates the resulting damages on a grid derived from the LitPop database (refer to the first reference).

After calculating sector-specific damages, we examine their propagation within the economy, transitioning from one sector to another. This analysis is conducted through the ARIO model, utilizing its Python implementation, the BoARIO package.

We use EXIOBASE3 (a multi-regional input-output matrix) to represent the transactions flows between different sectors of the economy.

## <a id="pladifes"></a> Pladifes

Pladifes is a research program aiming at easing the research in green and sustainable finance, as well as traditional one. They are the main authors of <b>CGEE</b>. Learn more about Pladifes [here](https://www.institutlouisbachelier.org/en/pladifes-a-large-financial-and-extra-financial-database-project-2/).

Databases produced in the context of this project are available [here](https://pladifes.institutlouisbachelier.org/data/#ghg-estimations). Other Paldifes databases can be access [here (only ESG)](https://pladifes.institutlouisbachelier.org/data/) and [here (financial and soon ESG)](https://www.eurofidai.org/).

# <a id="quickstart"></a> Quickstart 🚀

## <a id="installation"></a> Installation

### <a id="get"></a> Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    https://github.com/Pladifes/Sectoral-damages/

### <a id="dependencies"></a> Dependencies

You'll need a working Python environment to run the code.
The required dependencies are specified in the file `Pipfile`.
We use `pipenv` virtual environments to manage the project dependencies in
isolation.

Thus, you can install our dependencies without causing conflicts with your
setup (for a Python >= 3.9).
Run the following command in the repository folder (where `Pipfile` and `Pipfile.lock`
is located) to create a separate environment.
Make sure pipenv is installed on your system. You can install it using pip by running:

    pip install pipenv

to install all required dependencies in it:

    pipenv install

Once the installation is complete, activate the virtual environment by running:

    pipenv shell

This will activate the virtual environment and you can start working with your installed dependencies.


# <a id="refs"></a> References

Our approach is highly inspired by the following publications and discussions with some of the main authors:

-[`Eberenz, S., Stocker, D., Röösli, T., and Bresch, D. N.: Asset exposure data for global physical risk assessment, Earth Syst. Sci. Data, 12, 817–833`](https://doi.org/10.5194/essd-12-817-2020)

-[`Le Guenedal, T., Drobinski, P., and Tankov, P.: Cyclone generation Algorithm including a THERmodynamic module for Integrated National damage Assessment (CATHERINA 1.0) compatible with Coupled Model Intercomparison Project (CMIP) climate data, Geosci. Model Dev., 15, 8001–8039,`](https://doi.org/10.5194/gmd-15-8001-2022)

-[`S.Hallegate,Modeling the Role of Inventories and Heterogeneity in the Assessment of the Economic Costs of Natural Disasters`](https://pubmed.ncbi.nlm.nih.gov/23834029)

-[`Samuel Juhel, Adrien Delahais, Vincent Viguie. Robustness of the evaluation of indirect costs of natural disasters: example of the ARIO model.`](https://hal.science/hal-04196749/document)

# <a id="contributing"></a> Contributing 🤝

We are hoping that the open-source community will help us edit the code and improve it!

You are welcome to open issues, even suggest solutions and better still contribute the fix/improvement! We can guide you if you're not sure where to start but want to help us out 🥇

In order to contribute a change to our code base, you can submit a pull request (PR) via GitLab and someone from our team will go over it. Yet, we usually prefer that you reach out to us before doing so at [contact](mailto:pladifes@institutlouisbachelier.org).


