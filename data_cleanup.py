#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:43:29 2020

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



def column_names(df):
    """
    Function to extract the column names (concentrations) and organize them 
    to be tidied up.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to determine the column names from.

    Returns
    -------
    col_dict : Dictionary
        {current column name : "value", "unit"}

    """
    
    # Column list
    col_list = list(df.columns)
    
    # Creating a dictionary to save the values
    col_dict = {}
    
    # Iterate over the column names
    for i in range(len(col_list)):
        
        # Get the ith element
        col_name = col_list[i]
        
        # Split it by the space
        conc_value = col_name.split(" ")[0]
        conc_unit = col_name.split(" ")[1]
        
        # Make a dictionary 
        new_pair = {col_name : conc_value, conc_unit}
        
        # Adding this to the main dictionary 
        col_dict.update(new_pair)
        
    return col_dict
    

# The data output of the Gardner et al., measurements is not tidy.
def data_cleanup(data_path, filetype = "csv", save_path):
    """
    Function to create a tidy DataFrame from the input csv/excel file and
    and save it as an excel file for future use.

    Parameters
    ----------
    data_path : STRING
        Path to the orginal output file
        
    filetype : STRING
        Format of the input file. csv or xlsx
        Default : csv
        Alternative : xlsx
        
    save_path : STRING 
        Path to save the tidy dataframe as excel.

    Returns
    -------
    tidy_df : Pandas DataFrame
        Tidy DataFrame containing the data from the experiments

    """
    
    # specifying the filetype reader 
    if filetype == "csv":
        # Read a csv file
        df = pd.read_csv(data_path, skiprows = range(9))
        
    # The alternative would be an xlsx file
    elif filetype == "xlsx":
        # Read a excel file
        df = pd.read_xlsx(data_path, skiprows = range(9))
    
    # If inappropriate filetype added 
    else:
        error = "Please enter a valid filetype."
        return print(error)
    
    # Column names
    col_names = column_names(df)
    
    # Renaming the columns 
    df = df.rename(columns = col_names)
    
    # MultiIndexing
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, 
        names = ("Concentration (uM)", "Units")
        )
    
    # Melting the DataFrame
    df_tidy = pd.melt(df, value_name = "Time to Catastrophe (s)")
    
    # Removing the redundant units column and NaN
    df_tidy = df_tidy.drop("Units", axis = 1)
    df_tidy = df_tidy.dropna()
    
    # Save DataFrame to Excel 
    df_tidy.to_excel(save_path + ".xlsx",
               index = False)
    
    return df_tidy