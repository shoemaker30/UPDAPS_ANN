# ================================================
# Configuration Script
# ------------------------------------------------
# Description:
# ---
# This script provides variables/constants used by
# the entire application. These include file
# paths to data files, tensorflow metadata, etc.,
# feature names and ranges of observed feature
# values, and the supported distress types.
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
### Supported Pavements --------------------------------------------------
SURFACE_TYPES = [
    2,  # asphalt
    5,  # CRCP
    6,  # asphalt overlay over existing asphalt
    7   # asphalt overlay over pcc
]

### Supported Types of Pavement Distress ---------------------------------
PAVEMENT_DISTRESSES = [
    'Rut_total_in',
    'TotalFatigueCrack_percent',
    'ReflectiveCrack_percent',
    'Punchout_occur_per_mile',
    'IRI_ftmile' 
]

### File paths  ----------------------------------------------------------
mclim_files = r'processed_climate'          # climate processing module outputs
axel_load_files = r'processed_traffic'      # traffic processing module outputs
climate_data_dir = r'raw_climate'           # MATLAB files from msu server
tensorflow_models = r'tensorflow_models'    # ann models metadata

### ANN Model Names ------------------------------------------------------
ANN_MODEL_NAMES = [
    'S2_ANN_IRI_ftmile',
    'S2_ANN_Rut_total_in',
    'S2_ANN_TotalFatigueCrack_percent',
    'S5_ANN_IRI_ftmile',
    'S5_ANN_Punchout_occur_per_mile',
    'S6_ANN_IRI_ftmile',
    'S6_ANN_Rut_total_in',
    'S6_ANN_TotalFatigueCrack_percent',
    'S7_ANN_IRI_ftmile',
    'S7_ANN_Rut_total_in',
    'S7_ANN_TotalFatigueCrack_percent',
    'S7_ANN_ReflectiveCrack_percent'
]

### ANN Architectures ----------------------------------------------------
### This is applied for new ANNs. Existing ANN models are not affected
### by changing the values here.
ANN_MODELS = {
    # --- Surface Type 2 ----------------------
    '2':{
        'input' : {
            'neurons': 214
        },
        'hidden' : {
            'neurons' : 143,
            'activation' : 'sigmoid'
        },
        'output' : {
            'neurons' : 1
        }
    },
    # --- Surface Type 5 ----------------------
    '5':{
        'input' : {
            'neurons': 210
        },
        'hidden' : {
            'neurons' : 107,
            'activation' : 'sigmoid'
        },
        'output' : {
            'neurons' : 1
        }
    },
    # --- Surface Type 6 ----------------------
    '6':{
        'input' : {
            'neurons': 215
        },
        'hidden' : {
            'neurons' : 143,
            'activation' : 'sigmoid'
        },
        'output' : {
            'neurons' : 1
        }
    },
    '7':{
        'input' : {
            'neurons': 222
        },
        'hidden' : {
            'neurons' : 149,
            'activation' : 'sigmoid'
        },
        'output' : {
            'neurons' : 1
        }
    },
}


### Feature Names (along with Min/Max Ranges for each feature) -------------
### MinMax Scaler is applied to these features using the provided min,max ranges
FEATURE_NAMES = {
    # --- Pavement Age ------------------------
    'Age' : (0, 239),

    # --- Disress Types -----------------------
    'Rut_total_in' : (0, 2),
    'TotalFatigueCrack_percent' : (0, 100),
    'ReflectiveCrack_percent' : (0, 100),
    'Punchout_occur_per_mile' : (0, 55),
    'IRI_ftmile' : (63, 170),

    # --- Surface Type 2 ----------------------
    '2':{
        'HPMS_base_type'        : (1, 8),
        'f_system'              : (1, 6),
        'vehicleSpeed'          : (15, 80),
        'gwt'                   : (10, 40),
        'p4Subg'                : (30, 100),
        'p200Subg'              : (4.2, 86.8),
        'expansion_factor'      : (1, 3022.998),
        'layerGD[1]'            : (107.9, 145),
        'layerPR[1]'            : (0.3, 0.35),
        'layerTH[0]'            : (2, 22),
        'layerTH[1]'            : (0.2, 36),
        'layerMR[1]'            : (22929, 500000),
        'layerMR[2]'            : (14100, 22929),
        'FreezingIndx'          : (0, 3776.6),
        'MeanAnnualAirTemp_F'   : (19.9, 77),
        'Prec_in_an_avg'        : (97, 3466),
        'Tsurf_mek_F'           : (-21.7, 124.4),
        'Precipitation'         : (0, 0.07),
        'NAij_SitwMoQnt'                    : [
            (0.0,130.2),	# weight group 1000
            (0.0,8574.6),	# weight group 2000
            (0.0,10701.2),	# weight group 3000
            (0.0,18172.2),	# weight group 4000
            (0.0,16671.8),	# weight group 5000
            (0.0,16609.8),	# weight group 6000
            (0.0,14948.2),	# weight group 7000
            (0.0,12821.6),	# weight group 8000
            (0.0,23374.0),	# weight group 9000
            (0.0,22710.6),	# weight group 10000
            (0.0,24328.8),	# weight group 11000
            (0.0,12871.2),	# weight group 12000
            (0.0,9585.2),	# weight group 13000
            (0.0,5505.6),	# weight group 14000
            (0.0,5288.6),	# weight group 15000
            (0.0,3441.0),	# weight group 16000
            (0.0,3434.8),	# weight group 17000
            (0.0,1990.2),	# weight group 18000
            (0.0,1531.4),	# weight group 19000
            (0.0,979.6),	# weight group 20000
            (0.0,520.8),	# weight group 21000
            (0.0,508.4),	# weight group 22000
            (0.0,310.0),	# weight group 23000
            (0.0,310.0),	# weight group 24000
            (0.0,186.0),	# weight group 25000
            (0.0,192.2),	# weight group 26000
            (0.0,111.6),	# weight group 27000
            (0.0,111.6),	# weight group 28000
            (0.0,74.4),	# weight group 29000
            (0.0,68.2),	# weight group 30000
            (0.0,37.2),	# weight group 31000
            (0.0,18.6),	# weight group 32000
            (0.0,18.6),	# weight group 33000
            (0.0,12.4),	# weight group 34000
            (0.0,12.4),	# weight group 35000
            (0.0,0.0),	# weight group 36000
            (0.0,0.0),	# weight group 37000
            (0.0,0.0),	# weight group 38000
            (0.0,0.0),	# weight group 39000
            (0.0,0.0),	# weight group 40000
            (0.0,0.0),	# weight group 41000
            (0.0,0.0),	# weight group 42000
            (0.0,0.0),	# weight group 43000
            (0.0,0.0),	# weight group 44000
            (0.0,0.0),	# weight group 45000
            (0.0,0.0),	# weight group 46000
            (0.0,0.0),	# weight group 47000
            (0.0,0.0),	# weight group 48000
            (0.0,0.0),	# weight group 49000
        ],
        'NAij_TatwMoQnt'                    : [
            (0.0,0.0),	# weight group 2000
            (0.0,998.2),	# weight group 4000
            (0.0,5840.4),	# weight group 6000
            (0.0,10118.4),	# weight group 8000
            (0.0,13230.8),	# weight group 10000
            (0.0,15227.2),	# weight group 12000
            (0.0,12815.4),	# weight group 14000
            (0.0,11966.0),	# weight group 16000
            (0.0,11693.2),	# weight group 18000
            (0.0,11674.6),	# weight group 20000
            (0.0,11550.6),	# weight group 22000
            (0.0,12288.4),	# weight group 24000
            (0.0,12914.6),	# weight group 26000
            (0.0,13385.8),	# weight group 28000
            (0.0,13020.0),	# weight group 30000
            (0.0,11091.8),	# weight group 32000
            (0.0,7688.0),	# weight group 34000
            (0.0,4426.8),	# weight group 36000
            (0.0,3224.0),	# weight group 38000
            (0.0,1922.0),	# weight group 40000
            (0.0,1345.4),	# weight group 42000
            (0.0,1016.8),	# weight group 44000
            (0.0,744.0),	# weight group 46000
            (0.0,545.6),	# weight group 48000
            (0.0,347.2),	# weight group 50000
            (0.0,223.2),	# weight group 52000
            (0.0,130.2),	# weight group 54000
            (0.0,62.0),	# weight group 56000
            (0.0,37.2),	# weight group 58000
            (0.0,24.8),	# weight group 60000
            (0.0,18.6),	# weight group 62000
            (0.0,12.4),	# weight group 64000
            (0.0,6.2),	# weight group 66000
            (0.0,0.0),	# weight group 68000
            (0.0,0.0),	# weight group 70000
            (0.0,0.0),	# weight group 72000
            (0.0,0.0),	# weight group 74000
            (0.0,0.0),	# weight group 76000
            (0.0,0.0),	# weight group 78000
            (0.0,0.0),	# weight group 80000
            (0.0,0.0),	# weight group 82000
            (0.0,0.0),	# weight group 84000
            (0.0,0.0),	# weight group 86000
            (0.0,0.0),	# weight group 88000
            (0.0,0.0),	# weight group 90000
            (0.0,0.0),	# weight group 92000
            (0.0,0.0),	# weight group 94000
            (0.0,0.0),	# weight group 96000
            (0.0,0.0),	# weight group 98000

        ],
        'NAij_TrtwMoQnt'                        :[
            (0.0,0.0),	# weight group 6000
            (0.0,353.4),	# weight group 9000
            (0.0,762.6),	# weight group 12000
            (0.0,421.6),	# weight group 15000
            (0.0,254.2),	# weight group 18000
            (0.0,210.8),	# weight group 21000
            (0.0,192.2),	# weight group 24000
            (0.0,167.4),	# weight group 27000
            (0.0,291.4),	# weight group 30000
            (0.0,254.2),	# weight group 33000
            (0.0,210.8),	# weight group 36000
            (0.0,161.2),	# weight group 39000
            (0.0,99.2),	    # weight group 42000
            (0.0,55.8),	    # weight group 45000
            (0.0,117.8),	# weight group 48000
            (0.0,99.2),	    # weight group 51000
            (0.0,80.6),	    # weight group 54000
            (0.0,68.2),	    # weight group 57000
            (0.0,74.4),	    # weight group 60000
            (0.0,86.8),	    # weight group 63000
            (0.0,99.2),	    # weight group 66000
            (0.0,130.2),	# weight group 69000
            (0.0,130.2),	# weight group 72000
            (0.0,142.6),	# weight group 75000
            (0.0,161.2),	# weight group 78000
            (0.0,167.4),	# weight group 81000
            (0.0,148.8),	# weight group 84000
            (0.0,142.6),	# weight group 87000
            (0.0,130.2),	# weight group 90000
            (0.0,111.6),	# weight group 93000
            (0.0,93.0),	# weight group 96000
            (0.0,74.4),	# weight group 99000
            (0.0,55.8),	# weight group 102000
            (0.0,43.4),	# weight group 105000
            (0.0,31.0),	# weight group 108000
            (0.0,24.8),	# weight group 111000
            (0.0,18.6),	# weight group 114000
            (0.0,12.4),	# weight group 117000
            (0.0,12.4),	# weight group 120000
            (0.0,6.2),	# weight group 123000
            (0.0,6.2),	# weight group 126000
            (0.0,6.2),	# weight group 129000
            (0.0,0.0),	# weight group 132000
            (0.0,0.0),	# weight group 135000
            (0.0,0.0),	# weight group 138000
            (0.0,0.0),	# weight group 141000
            (0.0,0.0),	# weight group 144000
            (0.0,0.0),	# weight group 147000
        ],
        'NAij_QdtwMoQnt'                        :[
            (0.0,0.0),	# weight group 6000
            (0.0,0.0),	# weight group 9000
            (0.0,0.0),	# weight group 12000
            (0.0,0.0),	# weight group 15000
            (0.0,0.0),	# weight group 18000
            (0.0,0.0),	# weight group 21000
            (0.0,0.0),	# weight group 24000
            (0.0,0.0),	# weight group 27000
            (0.0,0.0),	# weight group 30000
            (0.0,0.0),	# weight group 33000
            (0.0,0.0),	# weight group 36000
            (0.0,0.0),	# weight group 39000
            (0.0,0.0),	# weight group 42000
            (0.0,0.0),	# weight group 45000
            (0.0,0.0),	# weight group 48000
            (0.0,0.0),	# weight group 51000
            (0.0,0.0),	# weight group 54000
            (0.0,0.0),	# weight group 57000
            (0.0,0.0),	# weight group 60000
            (0.0,0.0),	# weight group 63000
            (0.0,0.0),	# weight group 66000
            (0.0,0.0),	# weight group 69000
            (0.0,0.0),	# weight group 72000
            (0.0,0.0),	# weight group 75000
            (0.0,0.0),	# weight group 78000
            (0.0,0.0),	# weight group 81000
            (0.0,0.0),	# weight group 84000
            (0.0,0.0),	# weight group 87000
            (0.0,0.0),	# weight group 90000
            (0.0,0.0),	# weight group 93000
            (0.0,0.0),	# weight group 96000
            (0.0,0.0),	# weight group 99000
            (0.0,0.0),	# weight group 102000
            (0.0,0.0),	# weight group 105000
            (0.0,0.0),	# weight group 108000
            (0.0,0.0),	# weight group 111000
            (0.0,0.0),	# weight group 114000
            (0.0,0.0),	# weight group 117000
            (0.0,0.0),	# weight group 120000
            (0.0,0.0),	# weight group 123000
            (0.0,0.0),	# weight group 126000
            (0.0,0.0),	# weight group 129000
            (0.0,0.0),	# weight group 132000
            (0.0,0.0),	# weight group 135000
            (0.0,0.0),	# weight group 138000
            (0.0,0.0),	# weight group 141000
            (0.0,0.0),	# weight group 144000
            (0.0,0.0),	# weight group 147000
        ]
    },
    # --- Surface Type 5 ----------------------
    '5': {
        'HPMS_base_type'                    : (1, 8),
        'f_system'                          : (1, 6),
        'vehicleSpeed'                      : (15, 80),
        'gwt'                               : (10, 40),
        'PCCStructInps_p4Subg'              : (50.5, 99.6),
        'PCCStructInps_p200Subg'            : (9.6, 86.8),
        'expansion_factor'                  : (1, 130.15),
        'PCCStructInps_gd[1]'               : (130, 145),
        'PCCStructInps_TH_in[0]'            : (6, 14.3),
        'PCCStructInps_TH_in[1]'            : (1.7, 24),
        'PCCStructInps_ModulusOfRupture'    : (488.9, 666),
        'PCCStructInps_ElasticModulus_MEK'  : (3068648, 4954918),
        'PCCStructInps_MR[1]'               : (24000, 1000000),
        'PCCStructInps_MR[2]'               : (12000, 16643),
        'NAij_SitwMoQnt'                    : [
            (0.0,55.8),	    # weight group 1000
            (0.0,2976.0),	# weight group 2000
            (0.0,5133.6),	# weight group 3000
            (0.0,11135.2),	# weight group 4000
            (0.0,10075.0),	# weight group 5000
            (0.0,10856.2),	# weight group 6000
            (0.0,12034.2),	# weight group 7000
            (0.0,13354.8),	# weight group 8000
            (0.0,25426.2),	# weight group 9000
            (0.0,25661.8),	# weight group 10000
            (0.0,28520.0),	# weight group 11000
            (0.0,14768.4),	# weight group 12000
            (0.0,10899.6),	# weight group 13000
            (0.0,6076.0),	# weight group 14000
            (0.0,5877.6),	# weight group 15000
            (0.0,3999.0),	# weight group 16000
            (0.0,4253.2),	# weight group 17000
            (0.0,2697.0),	# weight group 18000
            (0.0,2145.2),	# weight group 19000
            (0.0,1308.2),	# weight group 20000
            (0.0,601.4),	# weight group 21000
            (0.0,471.2),	# weight group 22000
            (0.0,248.0),	# weight group 23000
            (0.0,229.4),	# weight group 24000
            (0.0,111.6),	# weight group 25000
            (0.0,105.4),	# weight group 26000
            (0.0,68.2),	# weight group 27000
            (0.0,62.0),	# weight group 28000
            (0.0,37.2),	# weight group 29000
            (0.0,37.2),	# weight group 30000
            (0.0,12.4),	# weight group 31000
            (0.0,6.2),	# weight group 32000
            (0.0,6.2),	# weight group 33000
            (0.0,0.0),	# weight group 34000
            (0.0,0.0),	# weight group 35000
            (0.0,0.0),	# weight group 36000
            (0.0,0.0),	# weight group 37000
            (0.0,0.0),	# weight group 38000
            (0.0,0.0),	# weight group 39000
            (0.0,0.0),	# weight group 40000
            (0.0,0.0),	# weight group 41000
            (0.0,0.0),	# weight group 42000
            (0.0,0.0),	# weight group 43000
            (0.0,0.0),	# weight group 44000
            (0.0,0.0),	# weight group 45000
            (0.0,0.0),	# weight group 46000
            (0.0,0.0),	# weight group 47000
            (0.0,0.0),	# weight group 48000
            (0.0,0.0),	# weight group 49000
        ],
        'NAij_TatwMoQnt'                    : [
            (0.0,0.0),	# weight group 2000
            (0.0,1078.8),	# weight group 4000
            (0.0,6107.0),	# weight group 6000
            (0.0,11222.0),	# weight group 8000
            (0.0,15531.0),	# weight group 10000
            (0.0,18339.6),	# weight group 12000
            (0.0,15171.4),	# weight group 14000
            (0.0,13509.8),	# weight group 16000
            (0.0,12462.0),	# weight group 18000
            (0.0,11860.6),	# weight group 20000
            (0.0,11284.0),	# weight group 22000
            (0.0,10744.6),	# weight group 24000
            (0.0,10416.0),	# weight group 26000
            (0.0,10478.0),	# weight group 28000
            (0.0,10664.0),	# weight group 30000
            (0.0,10422.2),	# weight group 32000
            (0.0,9014.8),	# weight group 34000
            (0.0,6441.8),	# weight group 36000
            (0.0,4284.2),	# weight group 38000
            (0.0,2157.6),	# weight group 40000
            (0.0,1233.8),	# weight group 42000
            (0.0,731.6),	# weight group 44000
            (0.0,452.6),	# weight group 46000
            (0.0,303.8),	# weight group 48000
            (0.0,192.2),	# weight group 50000
            (0.0,117.8),	# weight group 52000
            (0.0,74.4),	# weight group 54000
            (0.0,37.2),	# weight group 56000
            (0.0,24.8),	# weight group 58000
            (0.0,12.4),	# weight group 60000
            (0.0,12.4),	# weight group 62000
            (0.0,0.0),	# weight group 64000
            (0.0,0.0),	# weight group 66000
            (0.0,0.0),	# weight group 68000
            (0.0,0.0),	# weight group 70000
            (0.0,0.0),	# weight group 72000
            (0.0,0.0),	# weight group 74000
            (0.0,0.0),	# weight group 76000
            (0.0,0.0),	# weight group 78000
            (0.0,0.0),	# weight group 80000
            (0.0,0.0),	# weight group 82000
            (0.0,0.0),	# weight group 84000
            (0.0,0.0),	# weight group 86000
            (0.0,0.0),	# weight group 88000
            (0.0,0.0),	# weight group 90000
            (0.0,0.0),	# weight group 92000
            (0.0,0.0),	# weight group 94000
            (0.0,0.0),	# weight group 96000
            (0.0,0.0),	# weight group 98000
        ],
        'NAij_TrtwMoQnt'                        :[
            (0.0,0.0),	# weight group 6000
            (0.0,434.0),	# weight group 9000
            (0.0,861.8),	# weight group 12000
            (0.0,378.2),	# weight group 15000
            (0.0,204.6),	# weight group 18000
            (0.0,173.6),	# weight group 21000
            (0.0,155.0),	# weight group 24000
            (0.0,155.0),	# weight group 27000
            (0.0,291.4),	# weight group 30000
            (0.0,272.8),	# weight group 33000
            (0.0,210.8),	# weight group 36000
            (0.0,179.8),	# weight group 39000
            (0.0,86.8),	# weight group 42000
            (0.0,55.8),	# weight group 45000
            (0.0,99.2),	# weight group 48000
            (0.0,86.8),	# weight group 51000
            (0.0,55.8),	# weight group 54000
            (0.0,74.4),	# weight group 57000
            (0.0,86.8),	# weight group 60000
            (0.0,105.4),	# weight group 63000
            (0.0,117.8),	# weight group 66000
            (0.0,142.6),	# weight group 69000
            (0.0,155.0),	# weight group 72000
            (0.0,173.6),	# weight group 75000
            (0.0,186.0),	# weight group 78000
            (0.0,186.0),	# weight group 81000
            (0.0,173.6),	# weight group 84000
            (0.0,167.4),	# weight group 87000
            (0.0,148.8),	# weight group 90000
            (0.0,130.2),	# weight group 93000
            (0.0,105.4),	# weight group 96000
            (0.0,86.8),	# weight group 99000
            (0.0,68.2),	# weight group 102000
            (0.0,49.6),	# weight group 105000
            (0.0,37.2),	# weight group 108000
            (0.0,31.0),	# weight group 111000
            (0.0,18.6),	# weight group 114000
            (0.0,18.6),	# weight group 117000
            (0.0,12.4),	# weight group 120000
            (0.0,6.2),	# weight group 123000
            (0.0,6.2),	# weight group 126000
            (0.0,6.2),	# weight group 129000
            (0.0,0.0),	# weight group 132000
            (0.0,0.0),	# weight group 135000
            (0.0,0.0),	# weight group 138000
            (0.0,0.0),	# weight group 141000
            (0.0,0.0),	# weight group 144000
            (0.0,0.0),	# weight group 147000
        ],
        'NAij_QdtwMoQnt'                        :[
            (0.0,0.0),	# weight group 6000
            (0.0,0.0),	# weight group 9000
            (0.0,0.0),	# weight group 12000
            (0.0,0.0),	# weight group 15000
            (0.0,0.0),	# weight group 18000
            (0.0,0.0),	# weight group 21000
            (0.0,0.0),	# weight group 24000
            (0.0,0.0),	# weight group 27000
            (0.0,0.0),	# weight group 30000
            (0.0,0.0),	# weight group 33000
            (0.0,0.0),	# weight group 36000
            (0.0,0.0),	# weight group 39000
            (0.0,0.0),	# weight group 42000
            (0.0,0.0),	# weight group 45000
            (0.0,0.0),	# weight group 48000
            (0.0,0.0),	# weight group 51000
            (0.0,0.0),	# weight group 54000
            (0.0,0.0),	# weight group 57000
            (0.0,0.0),	# weight group 60000
            (0.0,0.0),	# weight group 63000
            (0.0,0.0),	# weight group 66000
            (0.0,0.0),	# weight group 69000
            (0.0,0.0),	# weight group 72000
            (0.0,0.0),	# weight group 75000
            (0.0,0.0),	# weight group 78000
            (0.0,0.0),	# weight group 81000
            (0.0,0.0),	# weight group 84000
            (0.0,0.0),	# weight group 87000
            (0.0,0.0),	# weight group 90000
            (0.0,0.0),	# weight group 93000
            (0.0,0.0),	# weight group 96000
            (0.0,0.0),	# weight group 99000
            (0.0,0.0),	# weight group 102000
            (0.0,0.0),	# weight group 105000
            (0.0,0.0),	# weight group 108000
            (0.0,0.0),	# weight group 111000
            (0.0,0.0),	# weight group 114000
            (0.0,0.0),	# weight group 117000
            (0.0,0.0),	# weight group 120000
            (0.0,0.0),	# weight group 123000
            (0.0,0.0),	# weight group 126000
            (0.0,0.0),	# weight group 129000
            (0.0,0.0),	# weight group 132000
            (0.0,0.0),	# weight group 135000
            (0.0,0.0),	# weight group 138000
            (0.0,0.0),	# weight group 141000
            (0.0,0.0),	# weight group 144000
            (0.0,0.0),	# weight group 147000
        ]
    },
    # --- Surface Type 6 ----------------------
    '6': {
        'HPMS_base_type'        : (1, 8),
        'f_system'              : (1, 6),
        'vehicleSpeed'          : (15, 80),
        'gwt'                   : (10, 40),
        'p4Subg'                : (30, 100),
        'p200Subg'              : (4.2, 86.8),
        'expansion_factor'      : (1, 3022.998),
        'layerGD[2]'            : (97.7, 145),
        'layerPR[2]'            : (0.3, 0.35),
        'layerTH[0]'            : (0.2, 12),
        'layerTH[1]'            : (0.1, 21),
        'layerTH[2]'            : (6, 6),
        'layerMR[2]'            : (22929, 500000),
        'layerMR[3]'            : (14100, 22929),
        'FreezingIndx'          : (0, 1730.5),         
        'MeanAnnualAirTemp_F'   : (31.8, 77.1),
        'Prec_in_an_avg'        : (99.3, 3466),
        'Tsurf_mek_F'           : (-7.7, 144.8),
        'Precipitation'         : (0, 0.05),
        'NAij_SitwMoQnt'                            :[
            (0.0,68.2),	# weight group 1000
            (0.0,4600.4),	# weight group 2000
            (0.0,5778.4),	# weight group 3000
            (0.0,11290.2),	# weight group 4000
            (0.0,10558.6),	# weight group 5000
            (0.0,11668.4),	# weight group 6000
            (0.0,11866.8),	# weight group 7000
            (0.0,11699.4),	# weight group 8000
            (0.0,20466.2),	# weight group 9000
            (0.0,19920.6),	# weight group 10000
            (0.0,21644.2),	# weight group 11000
            (0.0,11042.2),	# weight group 12000
            (0.0,8072.4),	# weight group 13000
            (0.0,4439.2),	# weight group 14000
            (0.0,4023.8),	# weight group 15000
            (0.0,2523.4),	# weight group 16000
            (0.0,2523.4),	# weight group 17000
            (0.0,1599.6),	# weight group 18000
            (0.0,1295.8),	# weight group 19000
            (0.0,917.6),	# weight group 20000
            (0.0,502.2),	# weight group 21000
            (0.0,477.4),	# weight group 22000
            (0.0,266.6),	# weight group 23000
            (0.0,266.6),	# weight group 24000
            (0.0,161.2),	# weight group 25000
            (0.0,148.8),	# weight group 26000
            (0.0,93.0),	# weight group 27000
            (0.0,99.2),	# weight group 28000
            (0.0,55.8),	# weight group 29000
            (0.0,49.6),	# weight group 30000
            (0.0,37.2),	# weight group 31000
            (0.0,18.6),	# weight group 32000
            (0.0,18.6),	# weight group 33000
            (0.0,12.4),	# weight group 34000
            (0.0,12.4),	# weight group 35000
            (0.0,0.0),	# weight group 36000
            (0.0,0.0),	# weight group 37000
            (0.0,0.0),	# weight group 38000
            (0.0,0.0),	# weight group 39000
            (0.0,0.0),	# weight group 40000
            (0.0,0.0),	# weight group 41000
            (0.0,0.0),	# weight group 42000
            (0.0,0.0),	# weight group 43000
            (0.0,0.0),	# weight group 44000
            (0.0,0.0),	# weight group 45000
            (0.0,0.0),	# weight group 46000
            (0.0,0.0),	# weight group 47000
            (0.0,0.0),	# weight group 48000
            (0.0,0.0),	# weight group 49000

        ],
        'NAij_TatwMoQnt'                            : [
            (0.0,0.0),	    # weight group 2000
            (0.0,787.4),	# weight group 4000
            (0.0,4476.4),	# weight group 6000
            (0.0,8469.2),	# weight group 8000
            (0.0,11687.0),	# weight group 10000
            (0.0,13950.0),	# weight group 12000
            (0.0,11860.6),	# weight group 14000
            (0.0,10757.0),	# weight group 16000
            (0.0,9870.4),	# weight group 18000
            (0.0,9145.0),	# weight group 20000
            (0.0,8531.2),	# weight group 22000
            (0.0,8109.6),	# weight group 24000
            (0.0,7867.8),	# weight group 26000
            (0.0,7849.2),	# weight group 28000
            (0.0,7675.6),	# weight group 30000
            (0.0,7055.6),	# weight group 32000
            (0.0,5877.6),	# weight group 34000
            (0.0,4309.0),	# weight group 36000
            (0.0,3186.8),	# weight group 38000
            (0.0,1897.2),	# weight group 40000
            (0.0,1271.0),	# weight group 42000
            (0.0,830.8),	# weight group 44000
            (0.0,520.8),	# weight group 46000
            (0.0,347.2),	# weight group 48000
            (0.0,204.6),	# weight group 50000
            (0.0,130.2),	# weight group 52000
            (0.0,62.0),	    # weight group 54000
            (0.0,37.2),	    # weight group 56000
            (0.0,18.6),	    # weight group 58000
            (0.0,12.4),	    # weight group 60000
            (0.0,12.4),	    # weight group 62000
            (0.0,6.2),	    # weight group 64000
            (0.0,0.0),	    # weight group 66000
            (0.0,0.0),	    # weight group 68000
            (0.0,0.0),	    # weight group 70000
            (0.0,0.0),	    # weight group 72000
            (0.0,0.0),	    # weight group 74000
            (0.0,0.0),	    # weight group 76000
            (0.0,0.0),	    # weight group 78000
            (0.0,0.0),	    # weight group 80000
            (0.0,0.0),	    # weight group 82000
            (0.0,0.0),	    # weight group 84000
            (0.0,0.0),	    # weight group 86000
            (0.0,0.0),	    # weight group 88000
            (0.0,0.0),	    # weight group 90000
            (0.0,0.0),	    # weight group 92000
            (0.0,0.0),	    # weight group 94000
            (0.0,0.0),	    # weight group 96000
            (0.0,0.0),	    # weight group 98000

        ],
        'NAij_TrtwMoQnt'                            : [
            (0.0,0.0),	# weight group 6000
            (0.0,316.2),	# weight group 9000
            (0.0,663.4),	# weight group 12000
            (0.0,353.4),	# weight group 15000
            (0.0,217.0),	# weight group 18000
            (0.0,179.8),	# weight group 21000
            (0.0,148.8),	# weight group 24000
            (0.0,130.2),	# weight group 27000
            (0.0,192.2),	# weight group 30000
            (0.0,142.6),	# weight group 33000
            (0.0,111.6),	# weight group 36000
            (0.0,86.8),	# weight group 39000
            (0.0,49.6),	# weight group 42000
            (0.0,37.2),	# weight group 45000
            (0.0,86.8),	# weight group 48000
            (0.0,80.6),	# weight group 51000
            (0.0,62.0),	# weight group 54000
            (0.0,68.2),	# weight group 57000
            (0.0,68.2),	# weight group 60000
            (0.0,86.8),	# weight group 63000
            (0.0,93.0),	# weight group 66000
            (0.0,124.0),	# weight group 69000
            (0.0,124.0),	# weight group 72000
            (0.0,136.4),	# weight group 75000
            (0.0,161.2),	# weight group 78000
            (0.0,155.0),	# weight group 81000
            (0.0,142.6),	# weight group 84000
            (0.0,130.2),	# weight group 87000
            (0.0,117.8),	# weight group 90000
            (0.0,99.2),	# weight group 93000
            (0.0,80.6),	# weight group 96000
            (0.0,62.0),	# weight group 99000
            (0.0,49.6),	# weight group 102000
            (0.0,37.2),	# weight group 105000
            (0.0,31.0),	# weight group 108000
            (0.0,24.8),	# weight group 111000
            (0.0,18.6),	# weight group 114000
            (0.0,12.4),	# weight group 117000
            (0.0,6.2),	# weight group 120000
            (0.0,6.2),	# weight group 123000
            (0.0,6.2),	# weight group 126000
            (0.0,6.2),	# weight group 129000
            (0.0,0.0),	# weight group 132000
            (0.0,0.0),	# weight group 135000
            (0.0,0.0),	# weight group 138000
            (0.0,0.0),	# weight group 141000
            (0.0,0.0),	# weight group 144000
            (0.0,0.0),	# weight group 147000

        ],
        'NAij_QdtwMoQnt'                            :[
            (0.0,0.0),	# weight group 6000
            (0.0,0.0),	# weight group 9000
            (0.0,0.0),	# weight group 12000
            (0.0,0.0),	# weight group 15000
            (0.0,0.0),	# weight group 18000
            (0.0,0.0),	# weight group 21000
            (0.0,0.0),	# weight group 24000
            (0.0,0.0),	# weight group 27000
            (0.0,0.0),	# weight group 30000
            (0.0,0.0),	# weight group 33000
            (0.0,0.0),	# weight group 36000
            (0.0,0.0),	# weight group 39000
            (0.0,0.0),	# weight group 42000
            (0.0,0.0),	# weight group 45000
            (0.0,0.0),	# weight group 48000
            (0.0,0.0),	# weight group 51000
            (0.0,0.0),	# weight group 54000
            (0.0,0.0),	# weight group 57000
            (0.0,0.0),	# weight group 60000
            (0.0,0.0),	# weight group 63000
            (0.0,0.0),	# weight group 66000
            (0.0,0.0),	# weight group 69000
            (0.0,0.0),	# weight group 72000
            (0.0,0.0),	# weight group 75000
            (0.0,0.0),	# weight group 78000
            (0.0,0.0),	# weight group 81000
            (0.0,0.0),	# weight group 84000
            (0.0,0.0),	# weight group 87000
            (0.0,0.0),	# weight group 90000
            (0.0,0.0),	# weight group 93000
            (0.0,0.0),	# weight group 96000
            (0.0,0.0),	# weight group 99000
            (0.0,0.0),	# weight group 102000
            (0.0,0.0),	# weight group 105000
            (0.0,0.0),	# weight group 108000
            (0.0,0.0),	# weight group 111000
            (0.0,0.0),	# weight group 114000
            (0.0,0.0),	# weight group 117000
            (0.0,0.0),	# weight group 120000
            (0.0,0.0),	# weight group 123000
            (0.0,0.0),	# weight group 126000
            (0.0,0.0),	# weight group 129000
            (0.0,0.0),	# weight group 132000
            (0.0,0.0),	# weight group 135000
            (0.0,0.0),	# weight group 138000
            (0.0,0.0),	# weight group 141000
            (0.0,0.0),	# weight group 144000
            (0.0,0.0),	# weight group 147000
        ],
    },
    # --- Surface Type 7 ----------------------
    '7': {
        'HPMS_base_type'        : (1, 8),
        'f_system'              : (1, 6),
        'vehicleSpeed'          : (20, 80),
        'gwt'                   : (10, 40),
        'PCCStructInps_p4Subg'  : (30, 100),
        'PCCStructInps_p200Subg': (4.2, 86.8),
        'PCCStructInps_p002mmSubg' : (0.3, 36.1),
        'expansion_factor'      : (1, 3992.5),
        'PCCStructInps_gd[2]'   : (127.2, 145),
        'PCCStructInps_TH_in[1]': (6, 16),
        'PCCStructInps_TH_in[2]': (0.7, 26),
        'PCCStructInps_ModulusOfRupture'    : (478, 666),
        'PCCStructInps_ElasticModulus_MEK'  : (2500000, 4954918),
        'PCCStructInps_MR[2]'   : (24000, 1000000),
        'PCCStructInps_MR[3]'   : (12000, 16643),
        'PCCStructInps_SubgradePI'  : (0, 32.8),
        'ACOverlayInps_TH_in'   : (1, 12),
        'FreezingIndx'          : (1230, 8122),
        'BaseFreezingIndex'     : (0.44, 56.68),
        'MeanAnnualAirTemp_F'   : (3.37, 22.35),
        'FreezeThawCycles'      : (0, 354),
        'AirTemp_F'             : (-26.1, 35.2),
        'Precipitation'         : (0, 1.3),
        'PercentHumidity'       : (11.8, 96.4),
        'AvgAnnualWetDays'      : (38, 282),
        'NAij_SitwMoQnt'                            :[
                (0.0,204.6),	# weight group 1000
                (0.0,5611.0),	# weight group 2000
                (0.0,10422.2),	# weight group 3000
                (0.0,27751.2),	# weight group 4000
                (0.0,25500.6),	# weight group 5000
                (0.0,25252.6),	# weight group 6000
                (0.0,22512.2),	# weight group 7000
                (0.0,18190.8),	# weight group 8000
                (0.0,33399.4),	# weight group 9000
                (0.0,32717.4),	# weight group 10000
                (0.0,35612.8),	# weight group 11000
                (0.0,18042.0),	# weight group 12000
                (0.0,13106.8),	# weight group 13000
                (0.0,6987.4),	# weight group 14000
                (0.0,6361.2),	# weight group 15000
                (0.0,3968.0),	# weight group 16000
                (0.0,3937.0),	# weight group 17000
                (0.0,2473.8),	# weight group 18000
                (0.0,2008.8),	# weight group 19000
                (0.0,1419.8),	# weight group 20000
                (0.0,768.8),	# weight group 21000
                (0.0,694.4),	# weight group 22000
                (0.0,396.8),	# weight group 23000
                (0.0,384.4),	# weight group 24000
                (0.0,229.4),	# weight group 25000
                (0.0,229.4),	# weight group 26000
                (0.0,130.2),	# weight group 27000
                (0.0,124.0),	# weight group 28000
                (0.0,68.2),	# weight group 29000
                (0.0,68.2),	# weight group 30000
                (0.0,43.4),	# weight group 31000
                (0.0,24.8),	# weight group 32000
                (0.0,18.6),	# weight group 33000
                (0.0,12.4),	# weight group 34000
                (0.0,12.4),	# weight group 35000
                (0.0,6.2),	# weight group 36000
                (0.0,6.2),	# weight group 37000
                (0.0,0.0),	# weight group 38000
                (0.0,0.0),	# weight group 39000
                (0.0,0.0),	# weight group 40000
                (0.0,0.0),	# weight group 41000
                (0.0,0.0),	# weight group 42000
                (0.0,0.0),	# weight group 43000
                (0.0,0.0),	# weight group 44000
                (0.0,0.0),	# weight group 45000
                (0.0,0.0),	# weight group 46000
                (0.0,0.0),	# weight group 47000
                (0.0,0.0),	# weight group 48000
                (0.0,0.0),	# weight group 49000
        ],
        'NAij_TatwMoQnt'                            : [
                (0.0,6.2),	    # weight group 2000
                (0.0,1333.0),	# weight group 4000
                (0.0,7495.8),	# weight group 6000
                (0.0,13968.6),	# weight group 8000
                (0.0,19412.2),	# weight group 10000
                (0.0,23274.8),	# weight group 12000
                (0.0,19747.0),	# weight group 14000
                (0.0,17893.2),	# weight group 16000
                (0.0,16417.6),	# weight group 18000
                (0.0,15221.0),	# weight group 20000
                (0.0,14198.0),	# weight group 22000
                (0.0,13497.4),	# weight group 24000
                (0.0,13125.4),	# weight group 26000
                (0.0,14607.2),	# weight group 28000
                (0.0,17031.4),	# weight group 30000
                (0.0,17428.2),	# weight group 32000
                (0.0,14030.6),	# weight group 34000
                (0.0,8270.8),	# weight group 36000
                (0.0,5338.2),	# weight group 38000
                (0.0,3162.0),	# weight group 40000
                (0.0,2126.6),	# weight group 42000
                (0.0,1382.6),	# weight group 44000
                (0.0,880.4),	# weight group 46000
                (0.0,558.0),	# weight group 48000
                (0.0,328.6),	# weight group 50000
                (0.0,192.2),	# weight group 52000
                (0.0,111.6),	# weight group 54000
                (0.0,62.0),	# weight group 56000
                (0.0,37.2),	# weight group 58000
                (0.0,12.4),	# weight group 60000
                (0.0,12.4),	# weight group 62000
                (0.0,6.2),	# weight group 64000
                (0.0,6.2),	# weight group 66000
                (0.0,0.0),	# weight group 68000
                (0.0,0.0),	# weight group 70000
                (0.0,0.0),	# weight group 72000
                (0.0,0.0),	# weight group 74000
                (0.0,0.0),	# weight group 76000
                (0.0,0.0),	# weight group 78000
                (0.0,0.0),	# weight group 80000
                (0.0,0.0),	# weight group 82000
                (0.0,0.0),	# weight group 84000
                (0.0,0.0),	# weight group 86000
                (0.0,0.0),	# weight group 88000
                (0.0,0.0),	# weight group 90000
                (0.0,0.0),	# weight group 92000
                (0.0,0.0),	# weight group 94000
                (0.0,0.0),	# weight group 96000
                (0.0,0.0),	# weight group 98000
        ],
        'NAij_TrtwMoQnt'                            : [
                (0.0,0.0),	    # weight group 6000
                (0.0,539.4),	# weight group 9000
                (0.0,1091.2),	# weight group 12000
                (0.0,570.4),	# weight group 15000
                (0.0,334.8),	# weight group 18000
                (0.0,266.6),	# weight group 21000
                (0.0,235.6),	# weight group 24000
                (0.0,248.0),	# weight group 27000
                (0.0,427.8),	# weight group 30000
                (0.0,396.8),	# weight group 33000
                (0.0,310.0),	# weight group 36000
                (0.0,229.4),	# weight group 39000
                (0.0,155.0),	# weight group 42000
                (0.0,93.0),	    # weight group 45000
                (0.0,179.8),	# weight group 48000
                (0.0,105.4),	# weight group 51000
                (0.0,93.0),	    # weight group 54000
                (0.0,99.2),	    # weight group 57000
                (0.0,111.6),	# weight group 60000
                (0.0,136.4),	# weight group 63000
                (0.0,148.8),	# weight group 66000
                (0.0,204.6),	# weight group 69000
                (0.0,210.8),	# weight group 72000
                (0.0,229.4),	# weight group 75000
                (0.0,254.2),	# weight group 78000
                (0.0,248.0),	# weight group 81000
                (0.0,229.4),	# weight group 84000
                (0.0,217.0),	# weight group 87000
                (0.0,192.2),	# weight group 90000
                (0.0,161.2),	# weight group 93000
                (0.0,136.4),	# weight group 96000
                (0.0,111.6),	# weight group 99000
                (0.0,80.6),	# weight group 102000
                (0.0,62.0),	# weight group 105000
                (0.0,49.6),	# weight group 108000
                (0.0,37.2),	# weight group 111000
                (0.0,24.8),	# weight group 114000
                (0.0,18.6),	# weight group 117000
                (0.0,12.4),	# weight group 120000
                (0.0,12.4),	# weight group 123000
                (0.0,6.2),	# weight group 126000
                (0.0,6.2),	# weight group 129000
                (0.0,6.2),	# weight group 132000
                (0.0,0.0),	# weight group 135000
                (0.0,0.0),	# weight group 138000
                (0.0,0.0),	# weight group 141000
                (0.0,0.0),	# weight group 144000
                (0.0,0.0),	# weight group 147000
        ],
        'NAij_QdtwMoQnt'                            :[
                (0.0,0.0),	# weight group 6000
                (0.0,0.0),	# weight group 9000
                (0.0,0.0),	# weight group 12000
                (0.0,0.0),	# weight group 15000
                (0.0,0.0),	# weight group 18000
                (0.0,0.0),	# weight group 21000
                (0.0,0.0),	# weight group 24000
                (0.0,0.0),	# weight group 27000
                (0.0,0.0),	# weight group 30000
                (0.0,0.0),	# weight group 33000
                (0.0,0.0),	# weight group 36000
                (0.0,0.0),	# weight group 39000
                (0.0,0.0),	# weight group 42000
                (0.0,0.0),	# weight group 45000
                (0.0,0.0),	# weight group 48000
                (0.0,0.0),	# weight group 51000
                (0.0,0.0),	# weight group 54000
                (0.0,0.0),	# weight group 57000
                (0.0,0.0),	# weight group 60000
                (0.0,0.0),	# weight group 63000
                (0.0,0.0),	# weight group 66000
                (0.0,0.0),	# weight group 69000
                (0.0,0.0),	# weight group 72000
                (0.0,0.0),	# weight group 75000
                (0.0,0.0),	# weight group 78000
                (0.0,0.0),	# weight group 81000
                (0.0,0.0),	# weight group 84000
                (0.0,0.0),	# weight group 87000
                (0.0,0.0),	# weight group 90000
                (0.0,0.0),	# weight group 93000
                (0.0,0.0),	# weight group 96000
                (0.0,0.0),	# weight group 99000
                (0.0,0.0),	# weight group 102000
                (0.0,0.0),	# weight group 105000
                (0.0,0.0),	# weight group 108000
                (0.0,0.0),	# weight group 111000
                (0.0,0.0),	# weight group 114000
                (0.0,0.0),	# weight group 117000
                (0.0,0.0),	# weight group 120000
                (0.0,0.0),	# weight group 123000
                (0.0,0.0),	# weight group 126000
                (0.0,0.0),	# weight group 129000
                (0.0,0.0),	# weight group 132000
                (0.0,0.0),	# weight group 135000
                (0.0,0.0),	# weight group 138000
                (0.0,0.0),	# weight group 141000
                (0.0,0.0),	# weight group 144000
                (0.0,0.0),	# weight group 147000
        ]
    }
}