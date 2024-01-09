# Pladifes Physical Risk Module
## Future economic damages estimations tied to physical disasters based on sectoral wealth densities


The objective of this project is to construct a tool that can assess the sector-specific ripple effects caused by future natural disasters, such as tropical cyclones, under various climate change scenarios.

While the direct damages of natural catastrophes carry a certain degree of uncertainty, the economic repercussions, particularly indirect damages, are even more unpredictable. These arise from disruptions in the regular operation of the economic system, such as delayed production, supply bottlenecks, and other related factors. They are signficant, amounting to more than 50% of direct damages in certain cases.

In order to fully comprehend the future physical risk in the context of climate change (that will substantially change the freaquency and intensity of tropical cyclones), we need to account for these indirect damages.

First, we estimate sectoral densities based on the physical assets for heavy industries as Steel, Cement,...




This densities will be used to estimate tropical cyclone damages using the Python module Catherina.
Once the damages computed, we'll analyse how they propagate in the economy, from one sector to another using the ARIO model and BoArio python package.

