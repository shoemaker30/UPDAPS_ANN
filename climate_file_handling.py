# ================================================
# MATLAB Climate File Handling Script
# ------------------------------------------------
# Description:
# ------------
# This script provides methods for reading 
# MATLAB (saved as .hcm) files containing 
# chronological data about the climate at certain 
# locations.
#
# Its main purpose is to save the data from 
# the files located at https://www.egr.msu.edu/~kutay/HCM
# into a pkl file for quick data access.
# ------------------------------------------------
# Author/Contacts:
# ---------------
# Eric Shoemaker
# Email: shoemaker30@marshall.edu
# Github: shoemaker30
# ================================================

# Python Libraries
import os
import scipy.io
import numpy as np
import re
import matplotlib.pyplot as plt
from math import log
import json
import pandas as pd

# Custom Imports
from console_loading_bar import ConsoleLoadingBar

#CLIMATE_DATA = pd.read_pickle('climate_dataset.pkl')

# This function reads a directory of MATLAB files and saves the data into a pickle file 
def convert_MATLAB_files_to_pkl(input_dir, output_file_name, yr_begin, yr_end):

    climate_dataset = {}                        # dictionary to hold data from all MATLAB files
    naming_pattern = re.compile("CFM-[0-9]+")   # naming pattern for hcm files : CFM-<id_num>.hcm

    # Read all data from the MATLAB files in the input directory
    loading_bar = ConsoleLoadingBar(
        len(list(os.scandir(input_dir))), 
        'Reading MATLAB files from ' + input_dir
    )
    for file in os.scandir(input_dir):
        file_name = os.path.splitext(file.name)
        if naming_pattern.match(file_name[0]) and file_name[1] == '.hcm':
            try:
                climate_dataset[file.name] = read_matlab_file(file.path, yr_begin, yr_end)
            except(Exception):pass
        loading_bar.increment()
      
    # Write the data to the output file
    pd.to_pickle(climate_dataset, output_file_name)

# This function reads a given MATLAB file and returns data 
# from it in a numpy array (the data is reduced to monthly 
# averages if specified).
def read_matlab_file(file_path, yr_begin, yr_end, month=1, averages=False):

    try:
        # load the climate file
        # Climate files contain hourly data for 6 columns:
        #    0. Timestamp (YYYYMMDDHH.)
        #    1. Temperature (F)
        #    2. Wind Speed (mph)
        #    3. Sunshine (%)
        #    4. Precipitation (in)
        #    5. Humidity (%)
        climate = scipy.io.loadmat(file_path)['A'][:,:6]
        
        # reduce data to the timeframe specified
        # january of yr_begin to december of yr_end
        yr_begin = yr_begin * 10**6
        yr_end = yr_end * 10**6
        climate = np.delete(climate, np.where(
            (climate[:, 0] < yr_begin + month * 10**4) | (climate[:, 0] >= yr_end + month * 10**4))[0], axis=0)     

        if averages:
            # increment through the data day-by-day to find the row indicies where each new month starts
            curr_yr = yr_begin      # keep track of year currently at
            curr_mnth = 8           # keep track of month currently at
            month_indicies = [0]     # indicies where each month begins
            for i in range(0, len(climate), 24):
                t = climate[i][0]
                if t > (curr_yr + 1 * 10**6):   # if a new year was reached
                    curr_yr += 1 * 10**6
                    curr_mnth = 1
                    month_indicies.append(i)
                elif t // 10 ** (int(log(t, 10)) - 6 + 1) - curr_yr / 10**4 > curr_mnth:    # if a new month was reached
                    curr_mnth += 1
                    month_indicies.append(i)
            month_indicies.append(len(climate))
            
            # sum up each column for each month
            monthly_avgs = []    
            for i in range(len(month_indicies)):
                if i != 0:
                    current_month_avgs = list(np.mean(climate[month_indicies[i-1]:month_indicies[i]], axis=0))
                    current_month_avgs[0] = i-1 # set first item in row to be the month 
                    monthly_avgs.append(current_month_avgs)
            return monthly_avgs
        return climate
    
    except(FileNotFoundError):
        print('MATLAB file not found.', end='')
    except(KeyError):
        print('Input Json does not contain a key "fnameAirTemp".', end='')

# function for retrieving the nth digit of a given number
def get_digit(number, n):
    return number // 10**n % 10

