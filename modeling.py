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
    Function to calculate the MLE values for the parameter. 
    
    Parameters
    ----------
    data : array
        Array containing the data.
    
    Returns
    -------
    res.x : array 
        Array containing [0, 2] arrays of the MLE values for the parameters from each 
        bootstrap sample.
        
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
        return res.x

    # If it does not converge
    else:
        raise RuntimeError('Convergence failed with message', res.message)  



def model_log_likelihood(data, params):
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
    res.x : array 
        Array containing [0, 2] arrays of the MLE values for the parameters from each 
        bootstrap sample.
        
    """
    # Warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # scipy minimize function on the negative log likelihood
        # which we previously defined
        res = scipy.optimize.minimize(
            fun = lambda params, data: -model_log_likelihood(data, params),

            # Guess values
            x0 = np.array([0.005, 0.004]),
            args = (data),
            method = 'Powell'
        )

    # If it converges
    if res.success:
        beta1_mle, beta2_mle = res.x
        return res.x

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



def compare_models(data):
    """
    Function to calculate and display the AIC values for the two models.
    
    Parameters
    ----------
    data : array
        Array containing the data.

    Returns
    -------
    gamma_aic : float
        Akaike Information Criterion value for Gamma Distribution 
        
    model_aic : float
        Akaike Information Criterion value for the Model

    """

    # MLEs for Gamma Distribution Parameters
    alpha_mle, beta_mle = mle_iid_gamma(data)
    
    # MLEs for Model
    beta1_mle, beta2_mle = mle_model(data)
    
    # Log Likelihood for Gamma
    gamma_ll = log_likelihood_gamma(data, (alpha_mle, beta_mle))
    
    # Log Likelihood for Model
    model_ll = model_log_likelihood(data, (beta1_mle, beta2_mle))

    # AIC for Gamma
    gamma_aic = akaike_information_criterion(gamma_ll, 2)
    
    # AIC for Model
    model_aic = akaike_information_criterion(model_ll, 2)


    return gamma_aic, model_aic




