# ================================================
# HPMS Data Processing Script
# ------------------------------------------------
# Description:
# ---
# This script provides classes and methods to 
# handle reading input HPMS data generated
# by UPDAPS.
# ------------------------------------------------
# Author/Contacts:
# ---
# Eric Shoemaker
# Email: shoemaker30@marshall.edu
# Github: shoemaker30
# ---
# Dr. James Bryce
# Email: james.bryce@mail.wvu.edu
# ---
# ================================================

# --------
# Imports
# -------

# Python Libraries
import os
import pandas as pd

# Custom
import configuration as cf
from climate_file_handling import read_matlab_file

# -------------------
# Classes & Functions
# -------------------

# This function scales a single given value into the range 0,1 using the MinMaxScaler formula. 
# The 'range' parameter is a tuple (min, max) giving the possible value range the given feature could take. 
def min_max_scaler(range, value):
    if range[0] == range[1]:
        return value
    return (value - range[0]) / (range[1] - range[0])
def unscale_min_max_scaler(range, value):
    if range[0] == range[1]:
        return value
    return value * (range[1] - range[0]) + range[0]

# This class handles the storing of data from pavement samples.
class Pavement:

    # Read the input and output dictionaries
    def __init__(self, input_dict, output_dict=None):

        # Classifications of the Pavement
        self.f_system = input_dict['f_system']
        self.state_code = input_dict['state_code']
        self.HPMS_surface_type = input_dict['HPMS_surface_type']
        self.HPMS_base_type = input_dict['HPMS_base_type']
        self.vehicleSpeed = input_dict['vehicleSpeed']
        self.projectname = input_dict['projectname']

        # Properties about the physical structure of the pavement, link to climate data file, ALS, and class distributions of data
        self.input_dict = input_dict    

        # relevant input data from the pavement sample, normalized
        self.phys_prop = self.scale_struct_properties(self.get_struct_properties())
        self.traffic_load = self.scale_traffic_loading(self.get_processed_traffic_data())
        self.climate = self.scale_climate_data(self.get_processed_climate_data())

        # Get relevant output data from the HPMS.
        # Monthly progression of several pavement distresses as calculated by UPDAPS
        # Copy only the pavement distress data
        if output_dict is None:
            self.output_dict = None
        else:
            self.output_dict = {}
            for distress_type in cf.PAVEMENT_DISTRESSES:
                try:
                    self.output_dict[distress_type] = output_dict[distress_type]
                except KeyError:
                    self.output_dict[distress_type] = None

    # performance field measurements at one point in time
    def get_measured_performance(self):
        return self.input_dict['measuredPerformance']

    # return the surface thickness of the current pavement
    def get_AC_surface_thickness(self):
        if self.HPMS_surface_type == 2:
            return self.input_dict['layerTH'][0]
        elif self.HPMS_surface_type == 6:
            return self.input_dict['layerTH'][0] + self.input_dict['layerTH'][1]
        elif self.HPMS_surface_type == 7:
            return self.input_dict['ACOverlayInps']['TH_in']

    # return the climate data obtained from the MATLAB files
    def get_MATLAB_climate_data(self):
        try:
            return read_matlab_file(cf.climate_data_dir + '/' + self.input_dict['fnameAirTemp'], 1980, 2000, 8, averages=True)
        except(FileNotFoundError):
            pass

    ### ================================================================================================================================================
    ### ================================================================================================================================================
    ### METHODS USED FOR CREATING INPUT/OUTPUT VECTORS FOR THE ANN MODELS

    # get the progression of a specific distress type (IRI_ftmile, Rut_total_in, etc.)
    def get_distress(self, distress_type):
        return self.output_dict[distress_type][:240]
    
    # scale a distress array
    def scale_distress(self, distress, distress_type):
        for i in range(240):
            distress[i] = min_max_scaler(cf.FEATURE_NAMES[distress_type], distress[i])
        return distress

    # This function extracts the pavement structural data from the input HPMS dictionary
    def get_struct_properties(self):
        if self.HPMS_surface_type == 2:
            return {
                'HPMS_base_type'        : self.input_dict['HPMS_base_type'],
                'f_system'              : self.input_dict['f_system'],
                'vehicleSpeed'          : self.input_dict['vehicleSpeed'],
                'gwt'                   : self.input_dict['gwt'],
                'p4Subg'                : self.input_dict['p4Subg'],
                'p200Subg'              : self.input_dict['p200Subg'],
                'expansion_factor'      : self.input_dict['expansion_factor'],
                'layerGD[1]'            : self.input_dict['layerGD'][1],
                'layerPR[1]'            : self.input_dict['layerPR'][1],
                'layerTH[0]'            : self.input_dict['layerTH'][0],
                'layerTH[1]'            : self.input_dict['layerTH'][1],
                'layerMR[1]'            : self.input_dict['layerMR'][1],
                'layerMR[2]'            : self.input_dict['layerMR'][2]
            }
        elif self.HPMS_surface_type == 5:
            return {
                'HPMS_base_type'        : self.input_dict['HPMS_base_type'],
                'f_system'              : self.input_dict['f_system'],
                'vehicleSpeed'          : self.input_dict['vehicleSpeed'],
                'gwt'                   : self.input_dict['gwt'],
                'PCCStructInps_p4Subg'  : self.input_dict['PCCStructInps']['p4Subg'],
                'PCCStructInps_p200Subg': self.input_dict['PCCStructInps']['p200Subg'],
                'expansion_factor'      : self.input_dict['expansion_factor'],
                'PCCStructInps_gd[1]'   : self.input_dict['PCCStructInps']['gd'][1],
                'PCCStructInps_TH_in[0]': self.input_dict['PCCStructInps']['TH_in'][0],
                'PCCStructInps_TH_in[1]': self.input_dict['PCCStructInps']['TH_in'][1],
                'PCCStructInps_ModulusOfRupture'    : self.input_dict['PCCStructInps']['ModulusOfRupture'][0],
                'PCCStructInps_ElasticModulus_MEK'  : self.input_dict['PCCStructInps']['ElasticModulusMEK'][0],
                'PCCStructInps_MR[1]'   : self.input_dict['PCCStructInps']['MR'][1],
                'PCCStructInps_MR[2]'   : self.input_dict['PCCStructInps']['MR'][2]
            }
        elif self.HPMS_surface_type == 6:
            return {
                'HPMS_base_type'        : self.input_dict['HPMS_base_type'],
                'f_system'              : self.input_dict['f_system'],
                'vehicleSpeed'          : self.input_dict['vehicleSpeed'],
                'gwt'                   : self.input_dict['gwt'],
                'p4Subg'                : self.input_dict['p4Subg'],
                'p200Subg'              : self.input_dict['p200Subg'],
                'expansion_factor'      : self.input_dict['expansion_factor'],
                'layerGD[2]'            : self.input_dict['layerGD'][2],
                'layerPR[2]'            : self.input_dict['layerPR'][2],
                'layerTH[0]'            : self.input_dict['layerTH'][0],
                'layerTH[1]'            : self.input_dict['layerTH'][1],
                'layerTH[2]'            : self.input_dict['layerTH'][2],
                'layerMR[2]'            : self.input_dict['layerMR'][2],
                'layerMR[3]'            : self.input_dict['layerMR'][3]
            }
        elif self.HPMS_surface_type == 7:
            return {
                'HPMS_base_type'        : self.input_dict['HPMS_base_type'],
                'f_system'              : self.input_dict['f_system'],
                'vehicleSpeed'          : self.input_dict['vehicleSpeed'],
                'gwt'                   : self.input_dict['gwt'],
                'PCCStructInps_p4Subg'  : self.input_dict['PCCStructInps']['p4Subg'],
                'PCCStructInps_p200Subg': self.input_dict['PCCStructInps']['p200Subg'],
                'expansion_factor'      : self.input_dict['expansion_factor'],
                'PCCStructInps_gd[2]'   : self.input_dict['PCCStructInps']['gd'][2],
                'PCCStructInps_TH_in[1]': self.input_dict['PCCStructInps']['TH_in'][1],
                'PCCStructInps_TH_in[2]': self.input_dict['PCCStructInps']['TH_in'][2],
                'PCCStructInps_ModulusOfRupture'    : self.input_dict['PCCStructInps']['ModulusOfRupture'][0],
                'PCCStructInps_ElasticModulus_MEK'  : self.input_dict['PCCStructInps']['ElasticModulusMEK'][0],
                'PCCStructInps_MR[2]'   : self.input_dict['PCCStructInps']['MR'][2],
                'PCCStructInps_MR[3]'   : self.input_dict['PCCStructInps']['MR'][3],
                'PCCStructInps_p4Subg'  : self.input_dict['PCCStructInps']['p4Subg'],
                'PCCStructInps_p200Subg'  : self.input_dict['PCCStructInps']['p200Subg'],
                'PCCStructInps_p002mmSubg'  : self.input_dict['PCCStructInps']['p002mmSubg'],
                'PCCStructInps_SubgradePI'  : self.input_dict['PCCStructInps']['SubgradePI'],
                'ACOverlayInps_TH_in'   : self.input_dict['ACOverlayInps']['TH_in']
            }

    # This function scales an array of structural properties returned by get_struct_properties()
    def scale_struct_properties(self, struct_props):
        if struct_props is not None:
            for item in struct_props:
                struct_props[item] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)][item], struct_props[item])
            return struct_props
    
    # This function extracts the amount of load applications of each axel type by weight class each month from the 
    # data output by the UPDAPS traffic processing module for this pavement.
    # Data is scaled into the range (0, 1) based on the ranges defined in configuration.py
    def scale_traffic_loading(self, traffic_loading):
        if traffic_loading is not None:
            for m in range(240):
                for w in range(49):
                    traffic_loading['NAij_SitwMoQnt'][m][w] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['NAij_SitwMoQnt'][w], traffic_loading['NAij_SitwMoQnt'][m][w])
                    traffic_loading['NAij_TatwMoQnt'][m][w] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['NAij_TatwMoQnt'][w], traffic_loading['NAij_TatwMoQnt'][m][w])
                for w in range(48):
                    traffic_loading['NAij_TrtwMoQnt'][m][w] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['NAij_TrtwMoQnt'][w], traffic_loading['NAij_TrtwMoQnt'][m][w])
                    traffic_loading['NAij_QdtwMoQnt'][m][w] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['NAij_QdtwMoQnt'][w], traffic_loading['NAij_QdtwMoQnt'][m][w])
            return traffic_loading

    # This function scales climate data returned from the UDPAPS climate processing module into the range (0,1)
    def scale_climate_data(self, climate_data):
        if climate_data is not None:
            if self.HPMS_surface_type == 2 or self.HPMS_surface_type == 6:
                for m in range(240):
                    climate_data['Tsurf_mek_F'][m] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['Tsurf_mek_F'], climate_data['Tsurf_mek_F'][m])
                    climate_data['Precipitation'][m] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['Precipitation'], climate_data['Precipitation'][m])
                climate_data['Prec_in_an_avg'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['Prec_in_an_avg'], climate_data['Prec_in_an_avg'])
                climate_data['FreezingIndx'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['FreezingIndx'], climate_data['FreezingIndx'])
                climate_data['MeanAnnualAirTemp_F'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['MeanAnnualAirTemp_F'], climate_data['MeanAnnualAirTemp_F'])
            elif self.HPMS_surface_type == 7:
                for m in range(240):
                    climate_data['AirTemp_F'][m] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['AirTemp_F'], climate_data['AirTemp_F'][m])
                    climate_data['Precipitation'][m] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['Precipitation'], climate_data['Precipitation'][m])
                    climate_data['PercentHumidity'][m] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['PercentHumidity'], climate_data['PercentHumidity'][m])
                climate_data['FreezeThawCycles'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['FreezeThawCycles'], climate_data['FreezeThawCycles'])
                climate_data['FreezingIndx'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['FreezingIndx'], climate_data['FreezingIndx'])
                climate_data['BaseFreezingIndex'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['BaseFreezingIndex'], climate_data['BaseFreezingIndex'])
                climate_data['MeanAnnualAirTemp_F'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['MeanAnnualAirTemp_F'], climate_data['MeanAnnualAirTemp_F'])
                climate_data['AvgAnnualWetDays'] = min_max_scaler(cf.FEATURE_NAMES[str(self.HPMS_surface_type)]['AvgAnnualWetDays'], climate_data['AvgAnnualWetDays'])
            return climate_data


    # This function returns the results of the traffic processing module for this pavement
    def get_processed_traffic_data(self):
        
        # files run through the taffic processing module follow this naming convention:
        # <projectname>-TrafficResult.pkl
        traffic_file_path = os.path.join(cf.axel_load_files, (self.projectname + '-TrafficResult.pkl'))

        # Check if the file exists
        if os.path.exists(traffic_file_path):

            # extract the necessary data from the file
            file_data = pd.read_pickle(traffic_file_path)
            return {
                'NAij_SitwMoQnt' : file_data['NAij_SitwMoQnt'],
                'NAij_TatwMoQnt' : file_data['NAij_TatwMoQnt'],
                'NAij_TrtwMoQnt' : file_data['NAij_TrtwMoQnt'],
                'NAij_QdtwMoQnt' : file_data['NAij_QdtwMoQnt']
            }
        
        else:
            print("No traffic file for sample",self.projectname)
            # if it doesn't exist, run the traffic module first to create the file 
            #traffic_process_main(self.input)

    # This function returns the results of the climate processing module for this pavement
    # set filtered to true if wanting surface temperature to be reduced to monthly averages for ANN inputs
    def get_processed_climate_data(self):
        
        # files run through the climate processing module follow this naming convention:
        # <projectname>_MCLIM-output-summary.pkl
        mclim_file_path = os.path.join(cf.mclim_files, (self.projectname + '_MCLIM-output-summary.pkl'))

        # Check if the file exists
        # return the contents of the file
        if os.path.exists(mclim_file_path):

            climate_data = pd.read_pickle(mclim_file_path)

            if self.HPMS_surface_type == 2 or self.HPMS_surface_type == 6:
                summary = {
                    'Prec_in_an_avg' : climate_data['Prec_in_an_avg'],
                    'FreezingIndx' : climate_data['FreezingIndx'],
                    'MeanAnnualAirTemp_F' : climate_data['MeanAnnualAirTemp_F'],
                    'Tsurf_mek_F' : [],
                    'Precipitation' : []
                }
                # Additionally, save the values needed for manual IRI caluation (in PredictionModel class) as fields for access later
                self.FreezingIndx = climate_data['FreezingIndx']
                self.Prec_in_an_avg = climate_data['Prec_in_an_avg']

                # Get average surface temperature and precipitation per month and append to output summary
                current_month = climate_data['MonthList'][0]    # keep track of which month is being averaged
                month_total_temp = 0                            # keep track of the sum of hourly temperatures in the month
                month_total_precip = 0
                month_total_hours = 0                           # keep track of the amount of temperatures in the month
                month_start = 0                                 # keep track of the list index where the current month starts
                for i in range(len(climate_data['MonthList'])):
                    if climate_data['MonthList'][i] != current_month or i == len(climate_data['MonthList']) - 1:
                        summary['Tsurf_mek_F'].append(month_total_temp / (i - month_start))    # append average month temperature to output
                        summary['Precipitation'].append(month_total_precip / (i - month_start))    # append average month precip to output
                        current_month = climate_data['MonthList'][i]
                        month_start = i
                        month_total_temp = 0 
                        month_total_precip = 0
                        month_total_hours = 0
                    month_total_temp += climate_data['Tsurf_mek_F'][i]
                    month_total_precip += climate_data['Precipitation'][i]
                return summary
            
            elif self.HPMS_surface_type == 7:
                summary = {
                    'FreezingIndx' : climate_data['FreezingIndx'],
                    'BaseFreezingIndex' : climate_data['BaseFreezingIndex'],
                    'MeanAnnualAirTemp_F' : climate_data['MeanAnnualAirTemp_F'],
                    'AvgAnnualWetDays' : climate_data['AvgAnnualWetDays'],
                    'FreezeThawCycles' : climate_data['FreezeThawCycles'],
                    'AirTemp_F' : [],
                    'Precipitation' : [],
                    'PercentHumidity' : []
                }

                # Additionally, save the values needed for manual IRI caluation (in PredictionModel class) as fields for access later
                self.FreezingIndx = climate_data['FreezingIndx']
                self.Precipitation = climate_data['Precipitation']

                # Get average surface temperature and precipitation per month and append to output summary
                current_month = climate_data['MonthList'][0]    # keep track of which month is being averaged
                month_total_temp = 0                            # keep track of the sum of hourly temperatures in the month
                month_total_precip = 0
                month_total_humid = 0
                month_total_hours = 0                           # keep track of the amount of temperatures in the month
                month_start = 0                                 # keep track of the list index where the current month starts
                for i in range(len(climate_data['MonthList'])):
                    if climate_data['MonthList'][i] != current_month or i == len(climate_data['MonthList']) - 1:
                        summary['AirTemp_F'].append(month_total_temp / (i - month_start))    # append average month temperature to output
                        summary['Precipitation'].append(month_total_precip / (i - month_start))    # append average month precip to output
                        summary['PercentHumidity'].append(month_total_humid / (i - month_start))
                        current_month = climate_data['MonthList'][i]
                        month_start = i
                        month_total_temp = 0 
                        month_total_precip = 0
                        month_total_humid = 0
                        month_total_hours = 0
                    month_total_temp += climate_data['AirTemp_F'][i]
                    month_total_precip += climate_data['Precipitation'][i]
                    month_total_humid += climate_data['PercentHumidity'][i]
                return summary
            
        else:
            print("No climate file found for sample", self.projectname)
            '''
            # files run through the climate processing module follow this naming convention:
            # <projectname>_MCLIM-output-summary.pkl
            mclim_file_path = os.path.join(cf.mclim_files, (self.projectname + '_MCLIM-output-summary.pkl'))

            # Check if the file exists
            if os.path.exists(mclim_file_path):
                    
                # return the contents of the file
                return pd.read_pickle(mclim_file_path)
            '''

    # This function returns 240 in/out vectors (each representing one month) to be used with the ANN models.
    # ** This function only works is an output dictionary was given with the sample.
    #       If making predictions on new data, we can only get one month at a time since
    #       we need predicted outputs from ANN models. ** 
    def get_monthly_vectors(self, distress_type):

        x_ = None
        y_ = None

        distress = self.scale_distress(self.get_distress(distress_type), distress_type) 

        if self.phys_prop is not None and self.traffic_load is not None:
            if self.HPMS_surface_type == 5 or self.climate is not None:

                x_ = []
                y_ = []

                # read the input data into arrays for each month
                for i in range(240):

                    # get the age of the pavement (in months)
                    age = i
                    age = min_max_scaler(cf.FEATURE_NAMES['Age'], age)
        
                    # build the vector for the current month 'i'
                    x = [age]                                       # pavement age (in months)
                    x.extend(self.phys_prop.values())                    # physical properties of the pavement
                    x.extend(self.traffic_load['NAij_SitwMoQnt'][i])     # single axle applications
                    x.extend(self.traffic_load['NAij_TatwMoQnt'][i])     # tandem axle applications
                    x.extend(self.traffic_load['NAij_TrtwMoQnt'][i])     # tridem axle applications
                    x.extend(self.traffic_load['NAij_QdtwMoQnt'][i])     # quad axle applications

                    if self.HPMS_surface_type == 2 or self.HPMS_surface_type == 6:

                        x.append(self.climate['MeanAnnualAirTemp_F'])        # mean annual air temperature
                        x.append(self.climate['Prec_in_an_avg'])             # average precipitation overall
                        x.append(self.climate['FreezingIndx'])               # freezing index
                        x.append(self.climate['Tsurf_mek_F'][i])             # surface temperature for current month
                        x.append(self.climate['Precipitation'][i])           # precipitation for current month
                    
                    elif self.HPMS_surface_type == 7:
                        x.append(self.climate['MeanAnnualAirTemp_F'])        # mean annual air temperature
                        x.append(self.climate['AvgAnnualWetDays'])           # average annual wet days
                        x.append(self.climate['FreezingIndx'])               # freezing index
                        x.append(self.climate['BaseFreezingIndex'])          # freezing index of base
                        x.append(self.climate['FreezingIndx'])               # freezing index
                        x.append(self.climate['FreezeThawCycles'])           # annual freeze/thaw cycles
                        x.append(self.climate['AirTemp_F'][i])               # air temperature for current month
                        x.append(self.climate['Precipitation'][i])           # precipitation for current month
                        x.append(self.climate['PercentHumidity'][i])         # humidity for current month

                    # append the distress from last month 'i-1'
                    # the first distress starts at 0 unless it is IRI. 
                    # Starting IRI is specified in the input dictionary
                    if i == 0:
                        if distress_type == 'IRI_ftmile':
                            iriO = 63  
                            x.append(min_max_scaler(cf.FEATURE_NAMES['IRI_ftmile'], iriO))
                        else:
                            x.append(0)
                    else:
                        x.append(distress[i-1])

                    # create an output vector
                    y = [distress[i]]

                    # append the input/output vectors for this month to the list of vectors for all months
                    x_.append(x)   
                    y_.append(y)

        return x_, y_
    
    # This function returns an in/out vector (representing one month [0-239]) to be used with the ANN models.
    # The distress value from the previous month should be passed as a parameter.
    def get_vector_at_month(self, distress_type, month, prev_distress_value):

        if self.phys_prop is not None and self.traffic_load is not None:
            if self.HPMS_surface_type == 5 or self.climate is not None:

                # get the age of the pavement (in months)
                age = month
                age = min_max_scaler(cf.FEATURE_NAMES['Age'], age)

                # build the vector for the current month
                x = [age]                                           # pavement age (in months)
                x.extend(self.phys_prop.values())                        # physical properties of the pavement
                x.extend(self.traffic_load['NAij_SitwMoQnt'][month])     # single axle applications
                x.extend(self.traffic_load['NAij_TatwMoQnt'][month])     # tandem axle applications
                x.extend(self.traffic_load['NAij_TrtwMoQnt'][month])     # tridem axle applications
                x.extend(self.traffic_load['NAij_QdtwMoQnt'][month])     # quad axle applications

                if self.HPMS_surface_type == 2 or self.HPMS_surface_type == 6:

                    x.append(self.climate['MeanAnnualAirTemp_F'])        # mean annual air temperature
                    x.append(self.climate['Prec_in_an_avg'])             # average precipitation overall
                    x.append(self.climate['FreezingIndx'])               # freezing index
                    x.append(self.climate['Tsurf_mek_F'][month])         # surface temperature for current month
                    x.append(self.climate['Precipitation'][month])       # precipitation for current month
                
                elif self.HPMS_surface_type == 7:
                    x.append(self.climate['MeanAnnualAirTemp_F'])        # mean annual air temperature
                    x.append(self.climate['AvgAnnualWetDays'])           # average annual wet days
                    x.append(self.climate['FreezingIndx'])               # freezing index
                    x.append(self.climate['BaseFreezingIndex'])          # freezing index of base
                    x.append(self.climate['FreezingIndx'])               # freezing index
                    x.append(self.climate['FreezeThawCycles'])           # annual freeze/thaw cycles
                    x.append(self.climate['AirTemp_F'][month])           # air temperature for current month
                    x.append(self.climate['Precipitation'][month])       # precipitation for current month
                    x.append(self.climate['PercentHumidity'][month])     # humidity for current month

                # append the distress value from the previous month
                x.append(min_max_scaler(cf.FEATURE_NAMES[distress_type], prev_distress_value))
     
                return [x]
        
    ### ================================================================================================================================================