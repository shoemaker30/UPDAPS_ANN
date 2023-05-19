# Title: This function is only to run the Climatic Model of the UPDAPS-ACPCC.
#
# Author: Farhad Abdollahi (abdolla4@msu.edu) under supervision of Prof. Kutay (kutay@msu.edu)
# Date: 09/27/2022
# ======================================================================================================================

# Importing the required libraries.
import pickle
from _MClim_Standalone_Surf7.UPDAPSPCC_Step3_ClimateModel_ACPCC import RunClimaticModel
import json

PROCESSED_CLIMATE_DIR = 'C:/Users/ericj/Desktop/processed_climate'



# Function to run climatic model.
def RunOneInputFile(inppath):
    # Read the Json input file.
    with open(inppath, 'r') as file:
        sect_json = json.load(file)
    # Call the MClim model.
    sect_json = RunClimaticModel(sect_json)
    return sect_json['MClim']

def RunOneFromDict(inps):
    print('entering RunOneFromDict')
    print('calling RunClimaticModel')
    sect_dict = RunClimaticModel(inps)
    print('finsihed RunClimaticModel')
    return sect_dict['MClim']


def surf7_clim_main(inps):
    print('entering surf7_clim_main')
    # Call the function.
    print('calling RunOneFromDict')
    Res = RunOneFromDict(inps)
    print('finished RunOneFromDict')
    # variable "Res" includes the results of the climatic model.
    outputFile = PROCESSED_CLIMATE_DIR + '/' + inps['projectname'] + '_MCLIM-output-summary.pkl'
    pickle.dump(Res, open(outputFile, 'wb'))
    print('file created')

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


if __name__ == '__main__':
    # Directory of the input file.
    #   To Eric: Please provide the directory of the input JSON file here. If you want to run many files, you can modify
    #       this part and put a for loop with/without the multiprocessing. Here, I'm using an example file attached.
    inppath = 'AL1696_S7B2_st1fs1.json'

    # Call the function.
    Res = RunOneInputFile(inppath)
    # variable "Res" includes the results of the climatic model.
    outputFile = 'AL1696_S7B2_st1fs1-out.pkl'
    pickle.dump(Res, open(outputFile, 'wb'))
