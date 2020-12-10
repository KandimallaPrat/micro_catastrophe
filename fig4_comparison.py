#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:14:02 2020

@author: pratyush
"""

# Importing the packages
from cat_analysis.figure_plotter import *
from cat_analysis.modeling import *
from cat_analysis.data_cleanup import *

from bokeh.io import export_png
import pandas as pd

# Reading in the tidy dataframe 
df_tidy = tidy_reader("data/tidy_mt_catastrophe.xlsx")

# Extracting the data for 12uM conc
data_12 = (df_tidy.loc[df_tidy["Concentration (uM)"] == 12, "Time to Catastrophe (s)"]).values


# Calculating the MLEs for the Parameters modeled as Gamma Distribution
parameters_gamma = mle_iid_gamma(data_12)

# Extracing the alpha and beta
alpha_mle = parameters_gamma[0]
beta_mle = parameters_gamma[1]

# Calculating the MLEs for the Parameters modeled as Gamma Distribution
parameters_story = mle_model(data_12)

# Extracing the alpha and beta
beta1_mle = parameters_story[0]
beta2_mle = parameters_story[1]

# Making the plot 
comparison_plot = single_data_gs_plotter(data_12,
                                         alpha_mle,
                                         beta_mle,
                                         beta1_mle,
                                         beta2_mle)

# Saving the figure
export_png(comparison_plot, filename = "figure_4.png")