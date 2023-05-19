# Overview of Project Directory

### scripts/

- **hpms_data_handling** </br> 
This script implements the Pavement class which provides methods for reading HPMS input data and
transforming the data for usage with ANN models.

- **prediction_model.py** </br>
This script contains the implementation of the ANN models used to predict distress progression. It contains functions for training, testing, and evaluating the models. 

- **configuration.py** </br>
This script holds configuration variables used by the other scripts. These valriables include:
    - File paths to training data, all output files written, etc.
    - Surface types which are currently supported.
    - Pavement distresses which are currently supported. 

- **_MClim_Standalone_S2_S6/** </br>
This folder contains the Climate Processing Module from UPDAPS. This module is used to calculate
Freezing Index, Surface Temperature, Precipitation, etc. for a pavement sample of surface type 
2 or 6. 

- **_MClim_Standalone_S7/** </br>
This folder contains the Climate Processing Module from UPDAPS. This module is used to calculate
Freezing Index, Surface Temperature, Precipitation, etc. for a pavement sample of surface type 7. 

- **UPDAPS_Traffic_Processing_Module/** </br>
This folder contains the Traffic Processing Module from UPDAPS. This module is used to calculate
the amount of applications by each axle type by weight class per month. This module can be used
on pavements of all surface types. 

### tensorflow_models/
This folder contains metadata for the ANN models. As some of these files are in binary or ProtocolBuffer, 
so *do not edit them directly*! These files are accessed within funtions of *prediction_model.py*

### surf_<surf_type>/
These folders contain a small sample of training and testing data for pavement of one surface type.


#### UPDAPS Prediction Model
# Prediction and Training Operations

**See */documentation/usage_example.ipynb* for an example run of these procedures.**

This markdown explains the prediction and training processes used for the UPDAPS prediction models. Both of these processes are handled by the *PredictionModel* class located in *prediction_model.py*. The flow of both processes can be described in 3 parts: 
1. Preprocessing
2. Ann Predictions
3. Manual Calculations

### Preprocessing
The preprocessing phase transforms an HPMS input file into an input vector suitable to be input into an ANN. An HPMS file for this project must be of type *.JSON* and must follow the naming convention.

Convention: `<StateAbbr><ProjectID>_S<SurfaceType>B<BaseType>_st<StateCode><FunctionalSystem>-Python-output.json`

Example file name: `AL1685_S5B2_st1fs1-Python-output.json`

First, a *Pavement* object is instantiated. The constructor receives the file path to an HPMS file, and some classification data (surface type, funtional class, etc.) is extracted and saved as fields in the *Pavement* object. 

Using functions in the *Pavement* class, all needed data is extracted and scaled into the range 0,1 by calling the `get_monthly_vectors(distress=<distresses> scaled=<True>)`. Two 2D lists are returned. 

#### How the Pavement Class is used 

1. Several methods are called to get all needed input and output data for the current Pavement.

    * `get_struct_properties()`
        * returns a dictionary of information about the physical pavement (surface thickness, vehicle speed, etc.) read directly from HPMS

    * `get_processed_traffic_data()`
        * returns dictionary obtained using the UPDAPS traffic processing module; these values are:
            * **NAij_SitwMoQnt** *(np.ndarray)* : amount of single axles that passed over the pavement each month. Each month includes the amount of axles in each of 49 weight groups (shape=(240,49))
            * **NAij_TatwMoQnt** *(np.ndarray)* : amount of tandem axles that passed over the pavement each month. Each month includes the amount of axles in each of 49 weight groups (shape=(240,49))
            * **NAij_TrtwMoQnt** *(np.ndarray)* : amount of tridem axles that passed over the pavement each month. Each month includes the amount of axles in each of 48 weight groups (shape=(240,48))
            * **NAij_QdtwMoQnt** *(np.ndarray)* : amount of quad axles that passed over the pavement each month. Each month includes the amount of axles in each of 48 weight groups (shape=(240,48))

    * `get_processed_climate_data()`
        * returns dictionary obtained using the UPDAPS climate processing module; these values are:
            * **MeanAnnualAirTemp_F** *(float)* : average annual air temperature over the 20 year period
            * **Precipitation** *(np.ndarray)* : precipitation averaged for each month (shape=(240,))
            * **FreezingIndx** *(float)* : freezing index

        Depending on the HPMS surface type of the pavement being read, some additional variables are also read. 
        * If surface type 2 or 6:
            * **Tsurf_mek_F** *(np.ndarray)* : pavement surface temperature averaged for each month (shape=(240,))
            * **Prec_in_an_avg** *(float)* : average precipitation over the 20 year period

        * If surface type 7:
            * **BaseFreezingIndex** *(float)* : freezing index of the pavement base
            * **AvgAnnualWetDays** *(int)* : average annual number of wet-days over the 20 year period
            * **PercentHumidity** *(np.ndarray)* : percent humidity averaged for each month (shape=(240,))

    * `get_distress(<distress_type>)`
        * returns one of the following depending on the parameter value:
            * **Rut_total_in** *(np.ndarray)* : cumulative amount of rutting in inches marked at each month (shape=(240,))
            * **TotalFatigueCrack_percent**  *(np.ndarray)* : cumulative amount of fatigue cracking as a percentage marked at each month (shape=(240,))
            * **TotalReflectiveCrack_percent**  *(np.ndarray)* : (for surface type 7 pavements) cumulative amount of reflective cracking as a percentage marked at each month (shape=(240,))
            * **IRI_ftmile** *(np.ndarray)* : cumulative roughness marked at each month (shape=(240,))
            * **Punchout_in_per_mile** *(np.ndarray)* : cumulative pucnhout marked at each month (shape=(240,))

*Note: TotalReflectiveCrack_percent will be None for some pavements (eg. AC pavements, etc.). Punchout_in_per_mile will also be None for some pavements (eg. pavements with overlays, etc.)*

2. Then all the data read so far is scaled into the range 0,1. This is accomplished by calling several methods and passing the data read from the functions above.

    * `scale_struct_properties()` to scale the return dict from `get_struct_properties()`
    * `scale_traffic_loading()` to scale the return dict from `get_processed_traffic_data()`
    * `scale_climate_data()` to scale the return dict from `get_processed_climate_data()`
    * `scale_distress()` to scale the return array from `get_distress()`

3. Either `get_monthly_vectors()` or `get_vector_at_month()` methods will return inputs to be passed to an ANN model.
    * `get_monthly_vectors()` uses the output dictionaries in the HPMS; therefore, this should be used for training and evaluating models.
    * `get_vector_at_month()` does not use the output dictionaries, so it creates a vector for the first month. A prediction should be made
        on the vector so that the prediction can be used within the vector for the next month. Therefore, this method is used when making
        new predictions. 

### ANN Predictions

This process is handled by the PredictionModel class. When instantiating this object, the desired output and surface type of in the input pavements is specified, `PredictionModel(<surface_type>)`. The constructor will automatically instantiate the needed ANNs in order to make the prediction (For example, an ANN to predict fatigue cracking and rutting will be needed for the prediction model to calculate IRI).

*It is important to avoid instantiating PredictionModels in a loop, since instantiating ANNs can take up to 1 second.*

Create a prediction model object in your code (specify the surface type)
```
pm = PredictionModel(surf_type=2)
```

Call forecast_pavement_condition() and pass an HPMS input dictionary (Loop through many file names to predict for many pavements.)
```
with open('surf_2/test/GA607_S2B1_st13fs5-Python-output.json', 'r') as json_file:
    json_data = json.load(json_file)
prediction = pm.forecast_pavement_condition(json_data['input'])
print(prediction)
```

### ANN Training

This process is handled by the Ann class. An Ann is instantiated by specifying the model name `Ann(<model_name>)`. There are a total of 9 different ANNs which can be used, and their metadata is saved in */tensorflow_models*:

* **S2_TotalFatigueCrack_percent** : predicts fatigue cracking for all HPMS surface type 2 (AC) pavements
* **S6_TotalFatigueCrack_percent** : predicts fatigue cracking for all HPMS surface type 6 (ACEAC) pavements
* **S7_TotalFatigueCrack_percent** : predicts fatigue cracking for all HPMS surface type 7 (ACPCC) pavements
* **S7_TotalReflectiveCrack_percent** : predicts reflective cracking for all HPMS surface type 7 (ACPCC) pavements
* **S2_Rut_total_in** : predicts rutting for all HPMS surface type 2 (AC) pavements
* **S6_Rut_total_in** : predicts rutting for all HPMS surface type 6 (ACEAC) pavements
* **S7_Rut_total_in** : predicts rutting for all HPMS surface type 7 (ACPCC) pavements
* **S5_IRIftmile** : predicts IRI for all HPMS surface type 5 (CRPC) pavements
* **S5_Punchout_in_per_mile** : predicts punchout for all HPMS surface type 5 (CRPC) pavements

After instantiating an Ann, `fit(<x_>, <y_>, <epochs>)` can be called. This method will train the Ann on the given training set over a specified amount of epochs (default 500). After training is complete, the new weights for the Ann will be saved, and a log of the MSE loss and accuracy per epoch will be created and saved into */tensorflow_models/<model_name>/<log_name>.csv.*


Create a prediction model object in your code
```
pm = PredictionModel(surf_type=2)
```
Call train_inner_ANN_models() and pass a directory of training data.
Also specify the amount of iterations to train on. DEFAULT=500 if not specified.
```
pm.train_inner_ANN_models('surf_2/train', itr=10)
```
