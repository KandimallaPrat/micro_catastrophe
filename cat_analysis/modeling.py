#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:44:03 2020

@author: pratyush
"""

# Importing required packages
import numpy as np

import iqplot
import pandas as pd

import bebi103 

import seaborn as sns # Color and Style for Plotting Library
#sns.set_style("darkgrid")

import warnings
import tqdm
import scipy

import holoviews as hv
import bokeh

import os


    
     
def draw_bs_sample(data):
    """
    Draw a bootstrap sample from a 1D data set.
    
    Parameters
    ----------
    data : array
        1D array containing the data.
    
    Returns
    -------
    bs : array
        1D array containing the bootstrapped data.
    """
    
    # Specifying random number generator
    rg = np.random.default_rng(3252)
    
    # Drawing a bootstrap replicate
    bs = rg.choice(data, size = len(data))
    
    return bs


def draw_bs_reps_mle(mle_fun, data, args=(), size = 1, progress_bar = False):
    """
    Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    
    data : one-dimemsional Numpy array
        Array of measurements
    
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    
    size : int, default 1
        Number of bootstrap replicates to draw.
    
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    
    # Whether or not to display the progress bar
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)
        
    rep_mles = np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])

    return res_mles



def mle_iid_gamma(data):
    """
    Function to calculate the MLE values (and log likelihood) for the parameter. 
    
    Parameters
    ----------
    data : array
        Array containing the data.
    
    Returns
    -------
    return_array : array 
        Array containing [0, 3] arrays of the MLE values for the parameters from each 
        bootstrap sample and the log likelihood.
        
    """
    # Warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # scipy minimize function on the negative log likelihood
        # which we previously defined
        res = scipy.optimize.minimize(
            fun = lambda params, data: -log_likelihood_gamma(data, params),

            # Guess values
            x0 = np.array([2.5, 0.01]),
            args = (data),
            method = 'Powell'
        )

    # If it converges
    if res.success:
        alpha_mle, beta_mle = res.x
        log_likelihood = -res.fun
        
        
        return_array = alpha_mle, beta_mle, log_likelihood
        
        return return_array

    # If it does not converge
    else:
        raise RuntimeError('Convergence failed with message', res.message)   





def model_log_likelihood(params, data):
    """
    Function to determine the log likelihood of the data given the parameters of the model.
    
    Parameters
    ----------
    params : tuple of floats 
        Format (alpha, beta)
        Tuple containing the parameter values
    
    data : array 
        numpy array containing the data values
    
    
    Returns 
    -------
    model_log_likelihood : float
        Value of the log likelihood for the model given the parameter values.
        
    """
    
    # First extracting the individual parameter values
    beta1, beta2 = params   
    
    # Setting constrains on the alpha and beta 
    # They cannot be zero
    # They cannot be equal to each other (due to subtraction in the denominator)
    if beta1 <= 0 or beta2 <= 0 or beta1 == beta2:
        
        # return negative infinity if either is zero
        return -np.inf
    
    # Writing out the terms of the model
    term_1 = (beta1 * beta2) / (beta2 - beta1)
    
    term_2 = np.exp(-beta1 * data) - np.exp(-beta2 * data)
    
    # Calculating the log of the model
    log_terms = np.log(term_1 * term_2)
    
    # Calculating the log likelihood
    model_log_likelihood = np.sum(log_terms)
    
    return model_log_likelihood


def mle_model(data):
    """
    Function to calculate the MLE values for the parameter. 
    
    Parameters
    ----------
    data : array
        Array containing the data.
    
    Returns
    -------
    return_array : array 
        Array containing [0, 3] arrays of the MLE values for the parameters from each 
        bootstrap sample and the log likelihood.
        
    """
    # Warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # scipy minimize function on the negative log likelihood
        # which we previously defined
        res = scipy.optimize.minimize(
            fun = lambda params, data: -model_log_likelihood(params, data),

            # Guess values
            x0 = np.array([0.005, 0.004]),
            args = (data),
            method = 'Powell'
        )

    # If it converges
    if res.success:
        beta1_mle, beta2_mle = res.x
        log_likelihood = -res.fun
        
        
        return_array = [beta1_mle, beta2_mle, log_likelihood]
        
        return return_array

    # If it does not converge
    else:
        raise RuntimeError('Convergence failed with message', res.message)   



def akaike_information_criterion(log_likelihood, num_params):
    """
    Calculate the Akaike Information Criterion for a log-likelihood for a given number of parameters.
    
    Parameters
    ----------
    log_likelihood : float 
        log-likelihood evaluated for a particular set of parameter values
        
    num_params : int
        Number of parameters in the model.
    
    Returns
    -------
    akaike_information_criterion : float
        The Akaike Information Criterion
        
        Calculated as:
        
        ((log-likelihood) * (- 2)) + (2 * num_params)
    """
    
    akaike_information_criterion = ((log_likelihood) * (- 2)) + (2 * num_params)
    
    return akaike_information_criterion


      
def bootstrap_aic(mle_function, data, size, progress_bar = True):
    """
    Function to calculate the MLE parameters for bootstrapped data and calculate the AIC.
    
    Parameters
    -----------
    mle_function : function
        Function to use to calculate the MLE of the Data
        
    data : array
        Array containing the data.
    
    size : int
        The number of bootstrap samples to draw
    
    progress_bar : Boolean
        Whether or not to show the progress bar.
        Default : True
        
    Returns
    -------
    df_mle : pandas DataFrame
        DataFrame containing the parameters, log-likelihood, and AIC for 
        every bootstrapped sample.
        
    """
    
    # Function name
    function_name = mle_function.__name__
    
    # Get all the MLE information
    bs_reps = draw_bs_reps_mle(
        mle_function, 
        data,
        size = size, 
        progress_bar = progress_bar
    )
    
    
    # Creating a DataFrame to store all these values
    df_mle = pd.DataFrame(bs_reps[0])
    df_mle.columns = ["Param1_MLE", "Param2_MLE", "Log-Likelihood"]
    
    # Extracting the log likelihood values 
    log_likelihood = df_mle["Log-Likelihood"].values
    
    # Calculating the AIC values for all these bootstrapped log likelihoods
    aic_values = akaike_information_criterion(log_likelihood, 2)
    
    # Adding this to the DataFrame
    df_mle["AIC Value"] = aic_values
    
    # Just for good measure we will add the name of the MLE Function to the DataFrame as well
    df_mle["MLE Function"] = [function_name] * size
    
    # Moving the position of the MLE Function Column 
    col = df_mle.pop("MLE Function")
    df_mle.insert(0, col.name, col)
    
    return df_mle
        

def compare_concentrations(mle_function, df_tidy, size, progress_bar = True):
    """
    Function to calculate the MLE parameters for bootstrapped data and calculate the AIC.
    Does so for every concentration value provided in the tidy DataFrame. 
    User also has the option to select the MLE function / model to test.
    
    Parameters
    ---------
    mle_function : function
        Function to use to calculate the MLE of the Data
        
    df_tidy : pandas DataFrame
        DataFrame containing the concentration vs time to catastrophe data.
    
    size : int
        The number of bootstrap samples to draw
    
    progress_bar : Boolean
        Whether or not to show the progress bar.
        Default : True
        
    Returns
    -------
    df_mle : pandas DataFrame
        DataFrame containing the parameters, log-likelihood, and AIC for 
        every bootstrapped sample for every concentration.
        
    """
    
    # Finding the number of unique concentrations
    unique_conc = (np.unique(df_tidy["Concentration (uM)"])).tolist()
    
    # Creating a large DataFrame to save all the values
    df_mle = pd.DataFrame()
    
    # Looping over every concentration
    for i, j in enumerate(unique_conc):
        
        # Getting all the data values for said concentration
        conc_data = (
            df_tidy.loc[df_tidy["Concentration (uM)"] == j, "Time to Catastrophe (s)"]
            ).values
        
        # Drawing bootstrap replicates and calculating MLEs for parameters    
        bs_reps = draw_bs_reps_mle(
            mle_function, 
            conc_data,
            size = size, 
            progress_bar = True
        )
        
        # Creating a DataFrame to store all these values
        df_rep = pd.DataFrame(bs_reps[0])
        
        # If it is our Gamma Distribution
        if mle_function.__name__ == mle_iid_gamma.__name__:
            # We use Alpha and Beta as parameter names in the df_mle
            df_rep.columns = ["Alpha_MLE", "Beta_MLE", "Log-Likelihood"]
            
        else:
            # Generic Column names
            df_rep.columns = ["Param1_MLE", "Param2_MLE", "Log-Likelihood"]
        
        # Just for good measure we will add the name of the MLE Function to the DataFrame as well
        df_rep["MLE Function"] = [mle_function.__name__] * size

        # Moving the position of the MLE Function Column 
        col = df_rep.pop("MLE Function")
        df_rep.insert(0, col.name, col)
        
        # Adding the conentration value 
        df_rep["Concentration (uM)"] = [j] * size
        col_c = df_rep.pop("Concentration (uM)")
        df_rep.insert(0, col_c.name, col_c)
            
        # Concatnating this to the Large DataFrame
        df_mle = pd.concat([df_mle, df_rep], ignore_index = True)
    
    return df_mle



