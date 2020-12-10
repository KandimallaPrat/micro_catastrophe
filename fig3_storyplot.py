#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:11:22 2020

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
parameters = mle_model(data_12)

# Extracing the alpha and beta
beta1_mle = parameters[0]
beta2_mle = parameters[1]

# Making the plot 
story_plot = single_data_story_plotter(data_12, 
                                       beta1_mle,
                                       beta2_mle)

# Saving the figure
export_png(story_plot, filename = "figure_3.png")