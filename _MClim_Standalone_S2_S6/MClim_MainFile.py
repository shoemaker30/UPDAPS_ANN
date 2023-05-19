# Title: This is the main file for running the climatic model.
#
# How to Use this Code: You can call this Python code from the Shell or run it from IDE. The only inputs to the code are
#   the input JSON file and the output directory. Edit lines 12 and 13 and run the code.
#
# Author: Farhad Abdollahi (abdolla4@msu.edu) under supervision of Prof. Kutay (kutay@msu.edu)
# Date: 06/14/2022

# edit; change json input to dictionary input
# ======================================================================================================================
# Importing the required libraries.
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from _MClim_Standalone_S2_S6.f_sublayer_MEPDGmethod import ff_sublayer_MEPDGmethod
from _MClim_Standalone_S2_S6.MCLIM_FiniteDiffDempsey import FiniteDiffDempsey
import pickle


# ======================================================================================================================
# User defined variables.
InputJsonPath = '/Users/kutay/My Drive/_Docs/_UPDAPS_Project/MClim_4Mike_June14_2022/io/NM659_S2B2_st35fs1.json'
PROCESSED_CLIMATE_DIR = 'C:/Users/ericj/Desktop/processed_climate'#'E:/processed_climate'
OutPutFolder  = PROCESSED_CLIMATE_DIR + '/'
RawDataNeeded = False    # The CSV file for raw data is usually very heavy. You can ignore it if you don't need it.
# ======================================================================================================================





def MClimRunOneFile(inps, OutputPath):

    """
    This is the main function for preparing and calling the MClim model.
    :param InpPath: The path to the input JSON file.
    :return: Nothing (the results will be saved as a CSV file).
    """
    StartTime = time.perf_counter()


    # Load the input JSON file.
    #inps = json.load(open(InpPath, 'r'))
    InpPath = PROCESSED_CLIMATE_DIR +'/'+ inps['projectname']

    # Read the required data from the JSON file.
    layerType   = None
    layerTH     = None
    layerTHcum  = None
    layerGD     = None
    layerK      = None
    layerC      = None
    tDesign     = None       
    
    if inps['HPMS_surface_type']==7:
        layerType   = np.array(inps['layerType'])
        layerTH     = np.array(inps['ACOverlayInps']['TH_in'])
        layerTHcum  = np.cumsum(layerTH)
        layerGD     = np.array(inps['ACOverlayInps']['gd'])
        layerK      = np.array(inps['ACOverlayInps']['K'])
        layerC      = np.array(inps['ACOverlayInps']['C'])
        tDesign     = int(inps['tDesign'])
        if type(inps['ACOverlayInps']['absorp']) == list:
            layerAbsorp = inps['ACOverlayInps']['labsorp'][0]
        else:
            layerAbsorp = inps['ACOverlayInps']['absorp']
    else:
        # Read the required data from the JSON file.

        layerType   = np.array(inps['layerType'])
        layerTH     = np.array(inps['layerTH'])
        layerTHcum  = np.cumsum(layerTH)
        layerGD     = np.array(inps['layerGD'])
        layerK      = np.array(inps['layerK'])
        layerC      = np.array(inps['layerC'])
        tDesign     = int(inps['tDesign'])
        if type(inps['layerAbsorp']) == list:
            layerAbsorp = inps['layerAbsorp'][0]
        else:
            layerAbsorp = inps['layerAbsorp']
    Latitude    = inps['latitude']
    Longitude   = inps['longitude']
    DayLightSave= inps['daylightSavingSelected']
    UTC         = inps['utc']
    Model       = inps['climateModel']
    Type        = inps['climateType']
    HCM_name    = inps['fnameAirTemp']
    HCM_dir     = 'C:/Users/ericj/Documents/UPDAPS/ANN/climate_data/www.egr.msu.edu/~kutay/HCM'#'E:/raw_climate'#

    if 'monthText' in inps:
        startingMonth = inps['monthText']
    else:
        startingMonth = 'AUGUST'
    # Sublayer the pavement layers using the MEPDG method.
    TH_sub, Edum_sub, LayNo_sub = ff_sublayer_MEPDGmethod(layerTH, layerType)
    zi_sub_in = np.cumsum(TH_sub) - TH_sub / 2  # Depth from the surface to the center of each sublayer
    # Printing the Message to User.
    print(f' -->>> MClim Inputs:')
    print(f'        ** Climatic Model: {Model};')
    print(f'        ** Climatic input type: {Type};')
    print(f'        ** Climatic file directory: {os.path.join(HCM_dir, HCM_name)};')
    print(f'        ** Analysis period: {tDesign} years;')
    print(f'        ** Surface absorption coeff: {layerAbsorp};')
    print(f'        ** Project location: lat={Latitude:.4f}; long={Longitude:.4f};')
    print(f'        ** Project time props: UTC={UTC}; Day light saving model: {DayLightSave};')
    print(f'        ** Climatic Model: {Model};')
    print(f' -->>> Pavement Structure Properties:')
    print(f'           Thickness      Type  Unit Weight    K           C  ')
    for i in range(len(layerTH)):
        print(f'           {layerTH[i]:.4e}     {layerType[i]}\t{layerGD[i]:.4e}     {layerK[i]:.4e}  {layerC[i]:.4e}')
    print(f'             Semi-inf     {layerType[-1]}\t{layerGD[-1]:.4e}     {layerK[-1]:.4e}  {layerC[-1]:.4e}')

    # Running the Climatic model (MClim).
    print(f' -->>> Running MClim ...')
    Tsubdum_M, MAAT_F, Yr, Mo, Tpave_mek_F, Tsurf_mek_F, Prec_in, FreezeIndex, Prec_in_an_avg, yrstart, Dy, Hr = \
        FiniteDiffDempsey(HCM_name, HCM_dir, Latitude, Longitude, DayLightSave, UTC, startingMonth, tDesign,
                          layerGD, layerK, layerC, layerTHcum, zi_sub_in, layerAbsorp, Type, Model)
    print(f' -->>> MClim done! [run time = {time.perf_counter() - StartTime:.2f} sec.]')
    print(f'        ** Freezing Index: {FreezeIndex} F-day;')
    print(f'        ** Average surface temp (F): {Tsurf_mek_F.mean()} F;')

    # Saving the required data.
    print(f' -->>> Saving the results as CSV files ...')
    # Save the raw data.
    if RawDataNeeded:
        Res = {'Year'               : list(Yr),
               'Month'              : list(Mo),
               'Day'                : list(Dy),
               'Hour'               : list(Hr),
               'Precipitation_(in)' : list(Prec_in),
               'Surface_Temp_(F)'   : list(Tsurf_mek_F)}
        for i in range(Tpave_mek_F.shape[1]):
            Res[f'Temp_(F)_at_zi={zi_sub_in[i]:.4f}in'] = list(Tpave_mek_F[:, i])
        Res = pd.DataFrame(Res)
        Res.to_csv(os.path.join(OutputPath, f'{os.path.splitext(os.path.basename(InpPath))[0]}_RawData.csv'), index=False)
    # Save the quantile data.
    '''
    for i in range(len(zi_sub_in)):
        Res = {'Month_Number'       : list(np.arange(Tsubdum_M.shape[0]) + 1),
               'Year'               : list(Tsubdum_M[:, 1, 1])}
        for j in range(5):
            Res[f'Temp(F)_Quantile_{j+1}'] = list(Tsubdum_M[:, i, j + 2])
        Res = pd.DataFrame(Res)
        Res.to_csv(os.path.join(OutputPath, f'{os.path.splitext(os.path.basename(InpPath))[0]}_QuantileTemp_zi={zi_sub_in[i]:.3f}in.csv'), index=False)
    '''
    # you can edit this dictionary if you dont want to save all the large arrays (e.g., PvtTemp_mek_F, Tsurf_mek_F, Precipitation etc.)
    mclimdataout = {'PvtTemp_mek_F': Tpave_mek_F,
                    'Tsurf_mek_F': Tsurf_mek_F,
                    'Precipitation': Prec_in,
                    'Prec_in_an_avg': Prec_in_an_avg,
                    'MeanAnnualAirTemp_F': MAAT_F,
                    'FreezingIndx': FreezeIndex,
                    'YearList': Yr,
                    'MonthList': Mo,
                    'DayList': Dy,
                    'HourList': Hr,
                    'ZCoord': zi_sub_in}

    RemotePklPath = os.path.join(OutPutFolder, inps['projectname']+'_MCLIM-output-summary.pkl')
    with open(RemotePklPath, 'wb') as file:
        pickle.dump(mclimdataout, file)

    print(f' -->>> MClim run is successfully Done! [Total run time {time.perf_counter() - StartTime:.2f} sec]')



    # Returning Nothing.
    return
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

'''
if __name__ == '__main__':
    # Starter Message.
    print(f'==============================================================================')
    print(f'================= Pavement Strcutre Climatic Model (MClim) ===================')
    print(f'===================== Michigan State University 2022 =========================')
    print(f'==============================================================================')
    if len(sys.argv) == 1:
        print(f' -->>> Input JSON file at {InputJsonPath}')
        print(f' -->>> Output directory is {OutPutFolder}')
        MClimRunOneFile(InputJsonPath, OutPutFolder)

    elif len(sys.argv) == 3:
        print(f' -->>> Input JSON file at {sys.argv[1]}')
        print(f' -->>> Output directory is {sys.argv[2]}')
        MClimRunOneFile(sys.argv[1], sys.argv[2])

    else:
        raise Exception(f'This code requires exactly two inputs (directory of JSON file and output folder), '
                        f'while <<<{len(sys.argv) - 1}>>> inputs were entered.')
'''