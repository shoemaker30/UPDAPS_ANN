# This Jupiter Notebook demonstrates the usage of the UPDAPS ANN library.
# The files for the library should be in the working directory. Otherwise,
# be sure to check the imports and file paths in configuration.py.
# ==========================================================================
# Author: Eric Shoemaker 
# Email: shoemaker30@marshall.edu
# --
# Contact me for any bug reports!
# ==========================================================================

# NOTE: The PredictionModel constructor instantiates Tensorflow ANN objects which can take time, 
#       so try to instantiate this object outside of loops. 
#
#
import json
from prediction_model import PredictionModel
import matplotlib.pyplot as plt
# ==========================================================================
# ==========================================================================
# SCENARIO 1 -- Predicting the IRI progression of a pavement sample.

# import the PredictionModel class
# from prediction_model import PredictionModel

# Create a prediction model object in your code (specify the surface type)
pm = PredictionModel(surf_type=2)

# Call forecast_pavement_condition() and pass an HPMS input dictionary
# (Loop through many file names to predict for many pavements.)
with open('surf_2/test/GA607_S2B1_st13fs5-Python-output.json', 'r') as json_file:
    json_data = json.load(json_file)
prediction = pm.forecast_pavement_condition(json_data['input'])
print(prediction)

# The prediction can be used with any visulization software (Excel, matplotlib, etc.)
# Here, I use a simple example with matplotlib
plt.plot(prediction['IRI_ftmile'])      # plot the prediction on a graph
plt.xlabel('month')                     # label the x axis
plt.ylabel('IRI')                       # label the y axis
plt.show()                              # display the graph

# ==========================================================================
# ==========================================================================
# SCENARIO 2 -- Training the ANN models of a Prediction Model

# import the prediction Model 
# from prediction_model import PredictionModel

# Create a prediction model object in your code
# Note: the constructor instantiates Tensorflow ANN objects which can take time, 
# so try to instantiate this object outside of loops.
pm = PredictionModel(surf_type=2)

# Call train_inner_ANN_models() and pass a directory of training data.
# Also specify the amount of iterations to train on. DEFAULT=500 if not specified.
pm.train_inner_ANN_models('surf_2/train', itr=10)

