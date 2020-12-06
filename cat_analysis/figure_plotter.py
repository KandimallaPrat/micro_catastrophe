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


def exploratory_ecdf_plotter(df_tidy, conf_int = True):
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



def single_data_gamma_plotter(data, alpha, beta):
    """
    Function to create a plot with the ECDF of the data and the Gamma Distribution 
    with the provided parameters.
    
    Parameters
    ----------
    data : 1D numpy array
        Array containing the data.
    
    alpha : float
        Gamma Distribution Alpha
    
    beta : float 
        Gamma Distribution Beta
    
    Returns
    -------
    single_data_param_plot : figure
        Bokeh figure containing the ECDF and CDF
    
    """
    
    # Plotting the ECDF
    single_data_gamma_plot = iqplot.ecdf(
        data = data,
        title = "Microtubule Time to Catastrophe"
    )
    
    # Determining the maximum value in the data
    data_max_real = np.max(data)
    # Rounding to nearest 100
    data_max = math.ceil(data_max_real)
    
    # Timeline for creating the model CDF
    t = np.linspace(0, data_max + 100, data_max + 100)
    
    # Overlapping 
    single_data_gamma_plot.line(t,
                               scipy.stats.gamma.cdf(t,
                                                    a = alpha, 
                                                    scale = (1 / beta)
                                                    ), 
                                color = "red"
                               )
    
    
    # Setting the x_label 
    single_data_gamma_plot.xaxis.axis_label = "Time to Catastrophe (s)"
    
    # Adding legend
    legend = bokeh.models.Legend(
            items=[("Data", [single_data_gamma_plot.circle(color = "blue")]),
                   ("Gamma Model", [single_data_gamma_plot.circle(color = "red")])
                  ],
            location='center')
    
    # Legend
    single_data_gamma_plot.add_layout(legend, "right")
    
    return single_data_gamma_plot



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


def single_data_story_plotter(data, beta1_mle, beta2_mle):
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
    single_data_story_plot : Figure
        bokeh ecdf figure.

    """
    # Plotting the ECDF
    single_data_story_plot = iqplot.ecdf(
        data = data,
        title = "Microtubule Time to Catastrophe"
        )
    
    # Changing the x-axis label 
    single_data_story_plot.xaxis.axis_label = "Time to Catastrophe (s)"
    
    
    # MODEL STORY
    # Determining the maximum value in the data
    data_max_real = np.max(data)
    # Rounding to nearest 100
    data_max = math.ceil(data_max_real)
    
    # Timeline for creating the model CDF
    t = np.linspace(0, data_max + 100, data_max + 100)

    # Function values for model CDF
    values = cdf_model_with_params(beta1_mle, beta2_mle, t)
    
    # Overlaying model CDF
    single_data_story_plot.line(t, values, color = "red")

    # Adding legend
    legend = bokeh.models.Legend(
            items=[("Data", [single_data_story_plot.circle(color = "blue")]),
                   ("Story Model", [single_data_story_plot.circle(color = "red")])
                  ],
            location='center')

    single_data_story_plot.add_layout(legend, 'right')
    
    return single_data_story_plot



def single_data_gs_plotter(data, alpha, beta, beta1_mle, beta2_mle):
    """
    Function to plot a the ECDF of the data and compare it to the model
    CDF and Gamma CDF.

    Parameters
    ----------
    data : array
        1D array containing the data.
    
    alpha : float
        Gamma Distribution Alpha
    
    beta : float 
        Gamma Distribution Beta
    
    beta1_mle : float
        MLE derived parameter value for beta1
        
    beta2_mle : float
        MLE derived parameter value for beta2   

    Returns
    -------
    single_data_gs_plot : Figure
        bokeh ecdf figure.

    """
    # Plotting the ECDF
    single_data_gs_plot = iqplot.ecdf(
        data = data,
        title = "Microtubule Time to Catastrophe"
        )
    
    # Changing the x-axis label 
    single_data_gs_plot.xaxis.axis_label = "Time to Catastrophe (s)"
    
    
    # MODEL STORY
    # Determining the maximum value in the data
    data_max_real = np.max(data)
    # Rounding to nearest 100
    data_max = math.ceil(data_max_real)
    
    # Timeline for creating the model CDF
    t = np.linspace(0, data_max + 100, data_max + 100)

    # Function values for model CDF
    values = cdf_model_with_params(beta1_mle, beta2_mle, t)
    
    # Overlaying model CDF
    single_data_gs_plot.line(t, values, color = "red", line_width = 2)
    
    
    # GAMMA DISTRIBUTION
    single_data_gs_plot.line(t,
                               scipy.stats.gamma.cdf(t,
                                                    a = alpha, 
                                                    scale = (1 / beta)
                                                    ), 
                                color = "orange", 
                             line_width = 2
                               )

    # Adding legend
    legend = bokeh.models.Legend(
            items=[("Data", [single_data_gs_plot.circle(color = "blue")]),
                   ("Story Model", [single_data_gs_plot.circle(color = "red")]),
                   ("Gamma Model", [single_data_gs_plot.circle(color = "orange")])
                  ],
            location='center')

    single_data_gs_plot.add_layout(legend, 'right')
    
    return single_data_gs_plot



def aic_ecdf_plotter(df_aic, title, conf_int = True):
    """
    Function to plot the ECDF of the AIC values for multiple models

    Parameters
    ----------
    df_aic : pandas DataFrame
        DataFrame containing the bootstrapped sample parameters, AIC values, 
        and the name of the model. 
        
    title : String
        The title you want to give the plot
        
    conf_int : Boolean
        Whether or not to plot the confidence intervals. The default is True.

    Returns
    -------
    eic_ecdf_plot : Figure
        bokeh ecdf figure.
    
    """
    
    # Creating the ECDF
    aic_ecdf_plot = iqplot.ecdf(
        data = df_aic, 
        
        # Plotting the AIC values 
        q = "AIC Value",
        
        # Groupby the MLE function/model
        cats = "MLE Function",
        
        # Title 
        title = title,
        
        # Staircase
        style = "staircase",
        
        # Confidence interval
        conf_int = conf_int,
        
        height = 400,
        width = 600,
        marker_kwargs = dict(alpha = 0.5)
    )

    # Setting the legend title
    aic_ecdf_plot.legend.title = "Model"
    
    return aic_ecdf_plot


def conc_param_gamma_plotter(df_conc, conf_int = True):
    """
    Function to plot the ECDF for the alpha and beta parameters of the
    bootstrapped samples - grouped by the concentrations.
    
    Parameters
    ----------
    df_conc : pandas DataFrame
        DataFrame containing the bootstrapped parameter values.
    
    conf_int : Boolean
        Whether or not to plot the confidence intervals.
        
    Returns
    -------
    param_ecdf : figure
        bokeh figure showing the ECDF of the parameters.
    
    """
    
    # ECDF for alpha
    alpha_ecdf = iqplot.ecdf(
        data = df_conc, 
        q = "Alpha_MLE",
        cats = "Concentration (uM)",
        title = "Distribution of Alpha Values",
        style = "staircase",
        conf_int = conf_int,
        height = 400,
        width = 600,
        marker_kwargs = dict(alpha = 0.5)
        )

    # Setting the legend title
    alpha_ecdf.legend.title = "Conc (uM)"
    
    # ECDF for beta
    beta_ecdf = iqplot.ecdf(
        data = df_conc, 
        q = "Beta_MLE",
        cats = "Concentration (uM)",
        title = "Distribution of Beta Values",
        style = "staircase",
        conf_int = conf_int,
        height = 400,
        width = 600,
        marker_kwargs = dict(alpha = 0.5)
        )

    # Setting the legend title
    beta_ecdf.legend.title = "Conc (uM)"
    
    # Compiling to a single horizontal figure
    conc_param_gamma_plot = bokeh.layouts.gridplot([alpha_ecdf, beta_ecdf], ncols = 2)
    
    return conc_param_gamma_plot


def alpha_beta_plotter(df_conc, dot_alpha = 0.5, dot_size = 1.5):
    """
    Function to plot the alpha vs beta parameters for bootstrapped samples
    for all Tubulin concentration values.
    
    Parameters
    ----------
    df_conc : pandas DataFrame
        DataFrame containing the bootstrapped parameter values.
    
    dot_alpha : float
        Alpha value for the plot dots
        
    dot_size : float
        Size of the glyphs.
        
    Returns
    -------
    alpha_beta_plot : figure
        bokeh figure of alpha-beta scatter plots.
    
    """
    
    # Creating scatterplot
    alpha_beta = hv.Scatter(
        data = df_conc, 
        kdims = ["Alpha_MLE"],
        vdims = ["Beta_MLE", "Concentration (uM)"]
    ).opts(
        color = "Concentration (uM)",
        title = "Alpha_MLE vs Beta_MLE",
        size = dot_size,
        alpha = dot_alpha
    )

    # Rendered
    alpha_beta_plot = hv.render(alpha_beta)

    # Aligning the title
    alpha_beta_plot.title.align = "center"

    # Setting the legend labels
    alpha_beta_plot.legend.title = "Concentration (uM)"
    
    return alpha_beta_plot

