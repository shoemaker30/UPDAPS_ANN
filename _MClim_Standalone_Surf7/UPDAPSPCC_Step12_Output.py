# Title: Prepare the outputs.
#
# Description: At the begining of this code, .
#
#
#
# This code was initially developed by Prof. M. Emin Kutay (kutay@msu.edu) in MATLAB environment. It is later converted
#   to Python by Farhad Abdollahi (abdolla4@msu.edu).
#
# Date: 06/22/2021
# ======================================================================================================================

# Importing the required libraries.
import json
import os
import pickle
import warnings
import numpy as np
# import matlab.engine
from time import perf_counter

def appendtxt(txtfld, txtfile, txttoadd):
    with open(txtfld + '/' + txtfile, 'a') as outfile:
        outfile.write(txttoadd)

def save_outputs_pkl(fld_out, jsonfilenoext, JsonData):
    outpkl= os.path.join(fld_out, jsonfilenoext + "-Python-output.pkl")
    print(f' --> Saving the PKL output: {outpkl}')
    with open(outpkl, 'wb') as file:
        pickle.dump(JsonData, file)

def save_outputs_json(fld_out, jsonfilenoext, JsonDataDistressesOnly):

    outjson = os.path.join(fld_out, jsonfilenoext + "-Python-output.json")
    print(f' --> Saving the JSON output: {outjson}')
    with open(outjson, 'w') as outfile:
        json.dump(JsonDataDistressesOnly, outfile)


def save_dmgeshares_json(fld_out, jsonfilenoext, JsonDataDistressesOnly):

    outjson = os.path.join(fld_out, jsonfilenoext + "-dmgeshares.json")
    print(f' --> Saving the JSON damage shares: {outjson}')
    with open(outjson, 'w') as outfile:
        json.dump(JsonDataDistressesOnly, outfile)


def Numpy2List4JsonSave(Data):
    """
    This function converts all the numpy data types into the list for a given dictionary. This is because the "json"
        module can not handle the numpy data types.
    NOTE: This function only supports the inputs of the "Dictionary" and "list" types.
    :param Data: The input dictionary to be treated.
    :return: The updated Data dictionary.
    """
    if type(Data) == dict:
        for key in Data.keys():
            # print(f' - {key}')
            if type(Data[key]) == np.ndarray:
                Data[key] = Data[key].tolist()
            else:
                Data[key] = Numpy2List4JsonSave(Data[key])
    elif type(Data) == list:
        for i in range(len(Data)):
            if type(Data[i]) == np.ndarray:
                Data[i] = Data[i].tolist()
            elif type(Data[i]) in [int, float, np.float64, np.int64, np.float32, np.int32, str]:
                pass
            else:
                Data[i] = Numpy2List4JsonSave(Data[i])
    elif type(Data) in [int, float, np.float64, np.int64, np.float32, np.int32, str]:
        pass
    else:

        raise ValueError(f'The variable input format is <{type(Data)}>, which is NOT supported. This function only '
                         f'handle "dict" or "list".')
    return Data
