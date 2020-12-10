#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:24:37 2020

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

# AIC with bootstrap samples
# Gamma
df_gamma = bootstrap_aic(
    mle_iid_gamma,
    data_12,
    size = 10000
    )

# Story
df_model = bootstrap_aic(
    mle_model,
    data_12,
    size = 10000
    )

# Concatnating
df_aic = pd.concat([df_gamma, df_model], ignore_index = True)

# Plotting the figure 
aic_ecdf = aic_ecdf_plotter(df_aic, "AIC Values for Gamma and Story Distribution")


# Save the figure
export_png(aic_ecdf, filename = "figure_5.png")