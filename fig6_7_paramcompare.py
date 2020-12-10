#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:10:48 2020

@author: pratyush
"""

# Importing the packages
from cat_analysis.figure_plotter import *
from cat_analysis.modeling import *
from cat_analysis.data_cleanup import *

from bokeh.io import export_png
import pandas as pd
from bokeh.models.annotations import Title
from bokeh.models import Legend


# Reading in the tidy dataframe 
df_tidy = tidy_reader("data/tidy_mt_catastrophe.xlsx")

# Gamma Comparison for every conc
df_conc = compare_concentrations(
    mle_iid_gamma, 
    df_tidy, 
    size = 10000, 
    progress_bar = True
)

# Plotting the ECDF
conc_plot = conc_param_gamma_plotter(df_conc)

# Save the figure
export_png(conc_plot, filename = "figure_6.png")


# Plotting alpha vs beta 
ab_plot = alpha_beta_plotter(df_conc)

# Save figure
export_png(ab_plot, filename = "figure_7.png", height = 750, width = 1000)
