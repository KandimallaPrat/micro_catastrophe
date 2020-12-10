#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:27:41 2020

@author: pratyush (@KandimallaPrat)
"""

# Importing the packages
from cat_analysis.figure_plotter import *
from cat_analysis.modeling import *
from cat_analysis.data_cleanup import *

from bokeh.io import export_png

### Specifying the data path for the untidy dataset
data_path = "data/gardner_mt_catastrophe_only_tubulin.csv"

# Cleaning up the data using the data cleanup
df_tidy = data_cleanup(data_path, "data/tidy_mt_catastrophe")

# Plotting the ECDF
ecdf_fig = exploratory_ecdf_plotter(df_tidy)

# Saving the figure
export_png(ecdf_fig, filename = "figure_1.png")