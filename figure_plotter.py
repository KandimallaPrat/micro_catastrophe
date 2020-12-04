#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:47:49 2020

@author: pratyush
"""

# Importing required packages
import numpy as np

import iqplot
import pandas as pd

import bebi103 


import warnings
import tqdm
import scipy
import iqplot
import math

import holoviews as hv
import bokeh

hv.extension('bokeh')
bokeh.io.output_notebook()

import os


def exploratory_ecdf(df_tidy, conf_int = True):
    """
    Function to generate the exploratory ECDFs for the data.

    Parameters
    ----------
    df_tidy : pandas DataFrame
        Tidy DataFrame for the microtubule time to catastrophe as a function 
        of tubulin concentration.
        
    conf_int : boolean
        True/False whether or not to plot the confidence intervals.

    Returns
    -------
    ecdf_catastrophe : Figure
        bokeh ecdf figure.

    """
    # Using iqplot
    ecdf_catastrophe = iqplot.ecdf(
        # Loading the data
        data = df_tidy, 
        
        # Concentration ECDFs plotted
        q = "Time to Catastrophe (s)",
        
        # Group by concentrations
        cats = "Concentration (uM)",
        
        # Plot Title
        title = "Microtubule Catastrophe Time as a Function of Tubulin Concentration",
        
        # Staircase
        style = "staircase",
        
        # Plotting Confidence intervals 
        conf_int = conf_int, 
        
        # Figure size
        height = 500,
        width = 750,
        
        # Marker alpha
        marker_kwargs = dict(alpha = 0.3),
    )

    # Setting the legend labels
    ecdf_catastrophe.legend.title = "Tubulin Conc. (uM)"
    
    return ecdf_catastrophe


def cdf_model_with_params(beta1_mle, beta2_mle, t):
    """
    Function to plot the theoretical model of time to catastrophe
    
    Parameters
    ----------
    t : array
        Array containing time values to calulate the function value for
    
    beta1_mle : float
        MLE derived parameter value for beta1
        
    beta2_mle : float
        MLE derived parameter value for beta2
        
    Returns
    -------
    y : array
        Array containging the values of the function at provided time points.
    """
    
    # Calculating the terms 
    scaling = (beta1_mle * beta2_mle) / (beta2_mle - beta1_mle)
    
    # Calculation for the first arrival rate
    term1 = (1 / beta1_mle) * (1 - np.exp(-beta1_mle * t))
    
    # Calculation for the second arrival rate 
    term2 = (1 / beta2_mle) * (1 - np.exp(-beta2_mle * t))
    
    # Compiling these calculation bits 
    y = scaling * (term1 - term2)
       
    return y


def plot_model_data(data, beta1_mle, beta2_mle):
    """
    Function to plot a the ECDF of the data and compare it to the model
    CDF.

    Parameters
    ----------
    data : array
        1D array containing the data.
    
    
    beta1_mle : float
        MLE derived parameter value for beta1
        
    beta2_mle : float
        MLE derived parameter value for beta2   

    Returns
    -------
    model_v_data : Figure
        bokeh ecdf figure.

    """
    # Plotting the ECDF
    model_v_data = iqplot.ecdf(
        data = data,
        title = "Microtubule Time to Catastrophe"
        )
    
    # Changing the x-axis label 
    model_v_data.xaxis.axis_label = "Time to Catastrophe (s)"
    
    # Determining the maximum value in the data
    data_max_real = np.max(data)
    # Rounding to nearest 100
    data_max = math.ceil(data_max_real)
    
    # Timeline for creating the model CDF
    t = np.linspace(0, data_max + 100, data_max + 100)

    # Function values 
    values = cdf_model_with_params(beta1_mle, beta2_mle, t)
    
    # Overlaying 
    model_v_data.line(t, values, color = "red")

    # Adding legend
    legend = bokeh.models.Legend(
            items=[("Data", [model_v_data.circle(color = "blue")]),
                   ("Model", [model_v_data.circle(color = "red")])
                  ],
            location='center')

    model_v_data.add_layout(legend, 'right')
    
    return model_v_data





