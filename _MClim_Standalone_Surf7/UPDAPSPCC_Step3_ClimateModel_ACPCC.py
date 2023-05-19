# Title: Perform the climatic model analysis.
#
# Description: This code runs the climatic model for the given pavement structure. The Finite Difference Dempsey method
#   is mainly used in this regard. The climatic input is coming from an hcm file. Details of this process can be found
#   in the UPDAPS technical report.
#
# NOTE: the climatic model should be run on the original model. Do NOT use the equivalent pavement structures (which are
#   calculated in Step 2).
#
#
# This code was initially developed by Prof. M. Emin Kutay (kutay@msu.edu) in MATLAB environment. It is later converted
#   to Python by Farhad Abdollahi (abdolla4@msu.edu).
#
# Date: 06/05/2021
# ======================================================================================================================

# Importing required libraries.
import json
import os
import pickle
import configuration as cf
import numpy as np
from time import perf_counter
from _MClim_Standalone_Surf7.MClim_FiniteDiffDempsey import FiniteDiffDempsey
from _MClim_Standalone_Surf7.UPDAPSPCC_Step12_Output import save_outputs_json, save_outputs_pkl, Numpy2List4JsonSave, appendtxt


def get_clim_inputs(JsonData):
    thick = np.array(JsonData['PCCStructInps']['TH_in'])  # Layer thicknesses (in).
    thickCum = np.cumsum(thick)
    unitWeight = np.array(JsonData['PCCStructInps']['gd'])  # Layer unit weights (pcf).
    thermalConduct = np.array(JsonData['PCCStructInps']['K'])  # Layer thermal conductivity (Btu/hr/ft/F).
    heatCapacity = np.array(JsonData['PCCStructInps']['C'])  # Layer heat capacity (Btu/lb/F).
    shortWaveAbsorb = JsonData['PCCStructInps']['absorp'][0]  # Short wave absorption of the surface layer.
    airTempFile = JsonData['fnameAirTemp']  # The name of the temperature hcm file.
    airTempFolder = cf.climate_data_dir #JsonData['fldAirTemp']  # The directory of the temperature hcm files.
    latitude = JsonData['latitude']  # The latitude of the weather station.
    longitude = JsonData['longitude']  # The longitude of the weather station.
    UTC = JsonData['utc']  # The UTC of the weather station.
    daylightSaving = JsonData['daylightSavingSelected']  # Status of the daylight saving.
    openMonth = JsonData['monthText']  # The opening month of the traffic.
    designLife = JsonData['tDesign']  # Design life (year).
    climateModel = JsonData['climateModel']  # The analysis climatic model (original).
    climateType = JsonData['climateType']  # Type of the climatic data (MERRA2).

    # Specify the Z-coordinates at which the temperature profile is calculated.
    ZCoord_Pvt = np.linspace(0, thickCum[0], num=11)  # 11 points in the PCC layer.

    return airTempFile, airTempFolder, latitude, longitude, daylightSaving, UTC, openMonth, designLife, unitWeight, \
           thermalConduct, heatCapacity, thickCum, ZCoord_Pvt, shortWaveAbsorb, climateType, climateModel


def get_clim_inputs_AC_PCC(JsonData):
    thick = np.array(JsonData['PCCStructInps']['TH_in'])  # Layer thicknesses (in).
    unitWeight = np.array(JsonData['PCCStructInps']['gd'])  # Layer unit weights (pcf).
    thermalConduct = np.array(JsonData['PCCStructInps']['K'])  # Layer thermal conductivity (Btu/hr/ft/F).
    heatCapacity = np.array(JsonData['PCCStructInps']['C'])  # Layer heat capacity (Btu/lb/F).
    airTempFile = JsonData['fnameAirTemp']  # The name of the temperature hcm file.
    airTempFolder = cf.climate_data_dir #JsonData['fldAirTemp']  # The directory of the temperature hcm files.
    latitude = JsonData['latitude']  # The latitude of the weather station.
    longitude = JsonData['longitude']  # The longitude of the weather station.
    UTC = JsonData['utc']  # The UTC of the weather station.
    daylightSaving = JsonData['daylightSavingSelected']  # Status of the daylight saving.
    openMonth = JsonData['monthText']  # The opening month of the traffic.
    designLife = JsonData['tDesign']  # Design life (year).
    climateModel = JsonData['climateModel']  # The analysis climatic model (original).
    climateType = JsonData['climateType']  # Type of the climatic data (MERRA2).

    thick[0] = JsonData['ACOverlayInps']['TH_in']
    thickCum = np.cumsum(thick)
    unitWeight[0] = JsonData['ACOverlayInps']['gd']
    thermalConduct[0] = JsonData['ACOverlayInps']['K']
    heatCapacity[0] = JsonData['ACOverlayInps']['C']

    shortWaveAbsorb = JsonData['ACOverlayInps']['absorp']  # Short wave absorption of the surface layer.

    # Specify the Z-coordinates at which the temperature profile is calculated.
    ZCoord_Pvt = np.hstack((np.linspace(0, thickCum[0], num=7), np.linspace(thickCum[0], thickCum[1], num=11)))
    ZCoord_Pvt = np.unique(ZCoord_Pvt)
    return airTempFile, airTempFolder, latitude, longitude, daylightSaving, UTC, openMonth, designLife, unitWeight, \
           thermalConduct, heatCapacity, thickCum, ZCoord_Pvt, shortWaveAbsorb, climateType, climateModel


def RunClimaticModel(JsonData):
    """
    This function runs the climatic model (MClim) for the current pavement.
    :param JsonData: A dictionary of all inputs and results of previous steps.
    :return: JsonData updated dictionary.
    """
    TrackTime = perf_counter()
    # Printing the relevant message to user.
    print(f'\t>> STEP 3: Running the Climatic model (MClim) is in progress:')

    if 'ACOverlayInps' in JsonData:  # AC over JPCP
        # Read the required variables.
        airTempFile, airTempFolder, latitude, longitude, daylightSaving, UTC, openMonth, designLife, unitWeight, \
        thermalConduct, heatCapacity, thickCum, ZCoord_Pvt, shortWaveAbsorb, climateType, climateModel = get_clim_inputs_AC_PCC(JsonData)

    else:  # JPCP
        # Read the required variables.
        airTempFile, airTempFolder, latitude, longitude, daylightSaving, UTC, openMonth, designLife, unitWeight, \
        thermalConduct, heatCapacity, thickCum, ZCoord_Pvt, shortWaveAbsorb, climateType, climateModel = get_clim_inputs(JsonData)
    
    # Run the MClim model.
    _, MAAT_F, Yr, Mo, Tpave_mek_F0, _, Prec_in, FreezeIndex, _, _, Dy, Hr, Tair_F, PHum, AnnualFreezeThaw = \
        FiniteDiffDempsey(airTempFile, airTempFolder, latitude, longitude, daylightSaving, UTC, openMonth,
                          designLife, unitWeight, thermalConduct, heatCapacity, thickCum, ZCoord_Pvt,
                          shortWaveAbsorb, climateType, climateModel)
    
    # Adding the build in curling to the pavement temperatures.
    Ttop = JsonData['PCCStructInps']['CurlWarpTempDiff'][0] / 2
    Tbot = -JsonData['PCCStructInps']['CurlWarpTempDiff'][0] / 2
    T_builtInCurl = np.repeat(np.linspace(Ttop, Tbot, num=11).reshape(1, -1), Tpave_mek_F0.shape[0], axis=0)
    Tpave_mek_F = Tpave_mek_F0.copy()
    Tpave_mek_F[:, -12:-1] = Tpave_mek_F0[:, -12:-1] + T_builtInCurl  # Apply built in curling to the PCC only.

    # Calculating the average annual number of wet days.
    monthDays = JsonData['monthDays']  # Number of days in each month
    monthNum = JsonData['mdfMonth']  # Number (index) of months
    AvgWetDaysAnnual = CalculateWetDays(Yr, Mo, Dy, monthNum, monthDays, designLife, Prec_in)

    # Calculating the base freezing index.
    BaseFreezingIndx = np.sum(Tpave_mek_F0[:, -1] <= 32) / Tpave_mek_F0.shape[0] * 100

    

    # Update the JsonData dictionary with the results.
    mclimdataout = {'PvtTemp_mek_F': Tpave_mek_F,
                    'AirTemp_F': Tair_F,
                    'Precipitation': Prec_in,
                    'PercentHumidity': PHum,
                    'MeanAnnualAirTemp_F': MAAT_F,
                    'FreezingIndx': FreezeIndex,
                    'BaseFreezingIndex': BaseFreezingIndx,
                    'FreezeThawCycles': AnnualFreezeThaw,
                    'YearList': Yr,
                    'MonthList': Mo,
                    'DayList': Dy,
                    'HourList': Hr,
                    'ZCoord': ZCoord_Pvt,
                    'AvgAnnualWetDays': AvgWetDaysAnnual}

    # mclimdataout = Numpy2List4JsonSave(mclimdataout)
    # outjson = os.path.join(JsonData['output_folder'],  JsonData['input_filename'] + "-MClim-output.json")
    # with open(outjson, 'w') as f:
    #     json.dump(mclimdataout, f)

    # outpkl = os.path.join(JsonData['output_folder'], JsonData['input_filename'] + "-MClim-output.pkl")
    # with open(outpkl, 'wb') as f:
    #     pickle.dump(mclimdataout, f)

    JsonData['MClim'] = mclimdataout

    Yr = JsonData['MClim']['YearList']
    Mo = JsonData['MClim']['MonthList']
    Dy = JsonData['MClim']['DayList']
    Hr = JsonData['MClim']['HourList']

    print("HERE2")
    # # Printing relevant message to user.
    txttoadd = f'\t\tAverage Annual wet days is {AvgWetDaysAnnual:.2f}.\n' \
                f'\t\tNumber of average annual Freezing/Thaw cycles is {AnnualFreezeThaw:.2f}.\n' \
                f'\t\tRunning the Climatic model (MClim) is DONE [{(perf_counter() - TrackTime):.3f} sec]'
    
    print(txttoadd)

    StatusTxtFile = JsonData['input_filename'] + '_status.txt'
    appendtxt(JsonData['output_folder'], StatusTxtFile, txttoadd)

   

    # Return the updated dictionary.
    return JsonData


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def CalculateWetDays(Years, Months, Days, monthNum, monthDays, DesignLife, Precipitation):
    """
    This function calculates the average annual wet Days over the pavement life time. It should be noted that a wet day
        is define as a days with more than 0.1" precipitation.
    :param Years: An array of years associated with each hour of pavement life.
    :param Months: An array of years associated with each hour of pavement life (starts from 1 to 12).
    :param Days: An array of years associated with each hour of pavement life (starts from 1 to 31).
    :param monthNum: An array of month numbers (in the range of 0 to 11).
    :param monthDays: An array of number of days in each month.
    :param DesignLife: The pavement design life (years).
    :param Precipitation: An array of hourly precipitation over the pavement life (in).
    :return: The average annual wet days in pavement design life.
    """
    YearsUnique = np.unique(Years)
    MonthNumber = DesignLife * 12  # Number of months in design life.
    PrecipitationMonthly = np.zeros((MonthNumber, 31))  # daily precipitation distribution over month.
    WetDaysMonthly = np.zeros(MonthNumber)  # Number of wet days in each month during life.
    WetDaysAnnual = np.zeros(len(YearsUnique) - 1)  # Number of wet days in each year during life.
    MonthCounter = -1  # A dummy variable to count the month number.
    YearCounter = 0  # A dummy variable to count the year number.
    StartOfYearIndx = 0  # A dummy variable to helps getting the annual wet days.
    for i in range(MonthNumber):
        # Adjust the month counter.
        MonthCounter += 1
        if MonthCounter == 12:
            MonthCounter = 0

        # Adjust the year counter at each January and calculate annual wet days of that year (NOTE: # If the first month
        #   is January, the year shouldn't be updated).
        if monthNum[MonthCounter] == 0 and i != 0:
            WetDaysAnnual[YearCounter] = np.sum(WetDaysMonthly[StartOfYearIndx:i])
            StartOfYearIndx = i
            YearCounter += 1

        # Calculate the monthly precipitation matrix (rows: hours, columns: days in month).
        PrecipitationHourly = np.zeros((24, 31))
        for j in range(monthDays[MonthCounter]):
            Indx = np.where((Years == YearsUnique[YearCounter]) &
                            (Months == (monthNum[MonthCounter] + 1)) &
                            (Days == (j + 1)))[0]
            PrecipitationHourly[:, j] = Precipitation[Indx]

        # Sum the hourly precipitation in each day and put it into the monthly distribution of precipitation matrix.
        PrecipitationMonthly[i, :] = np.sum(PrecipitationHourly, axis=0)

        # Calculate the number of wet days during the ith month.
        WetDaysMonthly[i] = len(np.where(PrecipitationMonthly[i, :] > 0.1)[0])

    # Adjust the WetDaysAnnual matrix, as the first and last elements are not full year and are complimentary.
    if monthNum[0] != 0:
        WetDaysAnnual[0] += np.sum(WetDaysMonthly[StartOfYearIndx:])
    else:
        WetDaysAnnual[YearCounter] = np.sum(WetDaysMonthly[StartOfYearIndx:])

    # Calculating the average annual precipitation.
    WetDays = WetDaysAnnual.mean()

    # Return the result
    return WetDays
