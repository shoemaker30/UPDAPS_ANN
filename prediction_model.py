# ================================================
# Prediction Model Script
# ------------------------------------------------
# Description:
# ---
# This script provides classes and methods to 
# handle making predictions on an HPMS input
# dictionary by utilizing Tensorflow ANNs.
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
import tensorflow as tf
import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from math import log

# Custom
import configuration as cf
from console_loading_bar import ConsoleLoadingBar
from hpms_data_handling import unscale_min_max_scaler
from hpms_data_handling import Pavement

# -------------------
# Classes & Functions
# -------------------

# This class implements a model which utilized ANNs to make predictions on the distress
# prediction of pavements over a 20-year span of time. 
class PredictionModel:
    
    # Constructor. 
    def __init__(self, surf_type):
        
        self.surf_type = surf_type

        # Load the appropriate ANN models
        if self.surf_type == 2:
            self.ann = {
                'Rut_total_in': Ann('S2_ANN_Rut_total_in'),
                'TotalFatigueCrack_percent': Ann('S2_ANN_TotalFatigueCrack_percent')
            }
        elif self.surf_type == 5:
            self.ann = {
                'Punchout_occur_per_mile': Ann('S2_ANN_Punchout_occur_per_mile'),
                'IRI_ftmile': Ann('S5_ANN_IRI_ftmile')
            }
        elif self.surf_type == 6:
            self.ann = {
                'Rut_total_in': Ann('S6_ANN_Rut_total_in'),
                'TotalFatigueCrack_percent': Ann('S6_ANN_TotalFatigueCrack_percent')
            }
        elif self.surf_type == 7:
            self.ann = {
                'Rut_total_in': Ann('S7_ANN_Rut_total_in'),
                'TotalFatigueCrack_percent': Ann('S7_ANN_TotalFatigueCrack_percent'),
                'ReflectiveCrack_percent': Ann('S7_ANN_ReflectiveCrack_percent')
            }

    # This function predicts the distress progression of a pavement.
    # The returned value is a dictionary of numpy arrays.
    # Each array represents one distress type: [240 months (20 years) of distress progression]
    def forecast_pavement_condition(self, input_dict):

        # dictionary to hold predictions 
        prediction = {
            'Rut_total_in': [],
            'TotalFatigueCrack_percent': [],
            'ReflectiveCrack_percent': [],
            'IRI_ftmile': [],
            'Punchout_occur_per_mile': []
        }

        # read the input dictionary
        pav = Pavement(input_dict)  

        # ensure that the provided input dictionary is for a pavement of the 
        # correct surface type
        if pav.HPMS_surface_type != self.surf_type:
            print('Prediction model for surface type', str(self.surf_type), 'cannot make predictions on a pavement of surface type', str(pav.HPMS_surface_type))
            return prediction
        
        # make the predictions 
        for distress in self.ann: # (self.ann holds the names of the distress types which will be predicted on)
            print("Predicting", distress, '...')

            # initial input vector for month 0
            x = pav.get_vector_at_month(
                distress_type=distress,
                month=0,        
                prev_distress_value=0,     # distresses start at 0 (except for IRI)
            )
            # if x is None, then there was missing data when calling get_vector_at_month()
            # get_vector_at_month() returns None when climate, traffic, or structural data is missing.
            if x is None:
                print('Missing data from input pavement', pav.projectname)
                return prediction

            # make a prediction on the first month
            curr_prediction = self.ann[distress].predict(x)[0][0]
            prediction[distress].append(curr_prediction)
            
            # for the rest of the months
            for i in range(1, 240):
                curr_prediction = self.ann[distress].predict(pav.get_vector_at_month(
                    distress_type=distress,
                    month=i,
                    prev_distress_value=unscale_min_max_scaler(cf.FEATURE_NAMES[distress], curr_prediction)
                ))[0][0]
                prediction[distress].append(curr_prediction)
            
            # unscale the predictions out of range (0,1)
            for i in range(len(prediction)):
                prediction[distress][i] = unscale_min_max_scaler(cf.FEATURE_NAMES[distress], prediction[distress][i])

            print()

        # Now calculate the IRI from the predicted values
        if self.surf_type != 5: # (surface type 5 predicted IRI from an ANN already)

            if self.surf_type == 2 or self.surf_type == 6:
                for i in range(240):
                    prediction['IRI_ftmile'].append(
                        self.f_calc_IRI(
                            IRIo= 63,
                            FCtot= prediction['TotalFatigueCrack_percent'][i],
                            rutTOT= prediction['Rut_total_in'][i],
                            TC= 0, # Transverse cracking is 0 at this time
                            iMntCum=i,
                            FreezingIndx=   pav.FreezingIndx,
                            Prec_in_an_avg= pav.Prec_in_an_avg,
                            P200=           pav.input_dict['p200Subg'],
                            P4=             pav.input_dict['p4Subg'],
                            C_IRI=          pav.input_dict['cIRI']
                        )
                    )

            elif self.surf_type == 7:
                DesignLife = pav.input_dict['tDesign']
                PvtLife = (np.arange(12 * DesignLife + 1)) / 12  # Pavement design life as array (year).
                PvtLife = PvtLife[1:]
                for i in range(240):
                    prediction['IRI_ftmile'].append(
                        self.CalculateIRI_ACPCC(
                            PvtLife=PvtLife[i],
                            FreezingIndx=pav.FreezingIndx,
                            Precipitation=pav.Precipitation,
                            p200Subg=pav.input_dict['PCCStructInps']['p200Subg'],
                            p002mmSubg=pav.input_dict['PCCStructInps']['p002mmSubg'],
                            SubgradePI=pav.input_dict['PCCStructInps']['SubgradePI'],
                            FCtot=prediction['TotalFatigueCrack_percent'][i],
                            rutTot=prediction['Rut_total_in'][i],
                            tc=0,
                            iri_CalibCoeff = pav.input_dict['IRI_CalibCoeff'],  # [40.8, 0.575, 0.0014, 0.00825]
                            iri_initial = pav.input_dict['IRI_initial'] 
                        )
                    )

        return prediction

    # This function trains the ANN models used to make up this prediction model.
    # Check the console for progress of training.
    def train_inner_ANN_models(self, dir, itr):

        # for each inner ann...
        for distress in self.ann: # (self.ann holds the names of the distress types which will be predicted on)

            print("=================================")
            print("Training Process for",distress,'\n')

            features = []   # training feature values
            targets = []    # training target values

            # Print status to the user
            print('Reading files from', dir, '...')
            n = len(list(os.scandir(dir)))
            cld = ConsoleLoadingBar(n)

            # read the files in the directory into a training set
            for file in os.scandir(dir):
                if file.name.endswith(".json"):
                    with open(file.path, 'r') as json_file:
                        json_data = json.load(json_file)
                        pav = Pavement(json_data['input'], json_data['output'])
                        x, y = pav.get_monthly_vectors(distress)
                        if x is not None and y is not None:
                            features.extend(x)
                            targets.extend(y)
                cld.increment()

            features = np.array(features)
            targets = np.array(targets)

            print('Done.')

            # open an ann model to train
            print('Starting model training....')
            # train the ann on the training set
            self.ann[distress].fit(
                x_ = features,
                y_ = targets,
                itr=itr
            )

            # print status to user
            print('Done.')

    # calculate iri for surface type 2 and 6
    def f_calc_IRI(self, IRIo, FCtot, rutTOT, TC, iMntCum, FreezingIndx, Prec_in_an_avg, P200, P4, C_IRI):

        age = (iMntCum + 1) / 12  # months to years
        frost = log((Prec_in_an_avg + 1) * (FreezingIndx + 1) * P4)
        swell = log((Prec_in_an_avg + 1) * (FreezingIndx + 1) * P200)
        site_factor = (frost + swell) * age ** 1.5

        # C_IRI = [40, 0.4, 0.008, 0.015] # default coefficients
        return IRIo + C_IRI[0] * rutTOT + C_IRI[1] * FCtot + C_IRI[2] * TC + C_IRI[3] * site_factor
    
    # calculate IRI for surface type 7
    def CalculateIRI_ACPCC(PvtLife, FreezingIndx, Precipitation, p200Subg, p002mmSubg, SubgradePI, FCtot, rutTot, tc, iri_CalibCoeff, iri_initial):

        precip = np.sum((Precipitation)) / 20

        # Calculating the IRI.
        SiteFactor = PvtLife ** 1.5 * (log((precip + 1) * (FreezingIndx + 1) * p002mmSubg) +
                                    log((precip + 1) * (SubgradePI + 1) * p200Subg))
        return iri_initial + \
            iri_CalibCoeff[0] * rutTot + \
            iri_CalibCoeff[1] * FCtot + \
            iri_CalibCoeff[2] * tc + \
            iri_CalibCoeff[3] * SiteFactor
        
# This class provides a wrapper for TensorFlow Sequential models
class Ann():

    # Constructor.
    # Loads an existing ANN or creates a new one if the provided model name has not already been used. 
    def __init__(self, model_name, surface_type=None, distress=None):

        self.surface_type = surface_type
        self.distress = distress

        # read the model name 
        self.model_name = model_name
        self.working_dir = os.path.join(cf.tensorflow_models, model_name)
        self.ann_model = None
        
        # if this model does not exist, create a new ANN model
        if not os.path.exists(self.working_dir):

            # create new directory to save metadata
            os.mkdir(self.working_dir)      

            # create a new ann model
            self.ann_model = tf.keras.models.Sequential()   
            self.ann_model.add(tf.keras.layers.Input(
                cf.ANN_MODELS[str(self.surface_type)]['input']['neurons']
            ))
            self.ann_model.add(tf.keras.layers.Dense(
                cf.ANN_MODELS[str(self.surface_type)]['hidden']['neurons'], 
                activation=cf.ANN_MODELS[str(self.surface_type)]['hidden']['activation']
            ))
            self.ann_model.add(tf.keras.layers.Dense(
                cf.ANN_MODELS[str(self.surface_type)]['output']['neurons']
            ))

            # compile the model
            self.ann_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

        # if the model already exists, then load the existing model
        else:
            self.load()

    # This function trains the current model on a given data set.
    # x_ : 2D numpy array where each row is an input vector (from Pavement class get_pmf_vector())
    # y_ : 2D numpy array where each row is the cooresponding output vector (from Pavement class get_pmf_vector())
    def fit(self, x_, y_, itr=500, graph_output=False):

        # fit the model on the provided data
        train_record = self.ann_model.fit(x_, y_, epochs=itr)

        ## save a record of the training
        # create the filename with a timestamp
        dateTimeObj = datetime.now()
        train_record_filename = ('training_'+
        str(dateTimeObj.year)+'-'+str(dateTimeObj.month)+'-'+str(dateTimeObj.day)+'_'+
        str(dateTimeObj.hour)+'-'+str(dateTimeObj.minute)+'-'+str(dateTimeObj.second)+
        '.csv')
        # save the record with in the model's metadata
        pd.DataFrame({
            'loss':train_record.history['loss'],
            'accuracy':train_record.history['accuracy']
        }).to_csv(os.path.join(self.working_dir,train_record_filename))

        # save the updated model weights
        self.save()

        # display a graph of the training results
        if graph_output:
            plt.xlabel('epoch')
            plt.plot(train_record.history['loss'], label="loss", linestyle='--')
            plt.plot(train_record.history['accuracy'], label="accuracy")
            plt.legend()
            plt.show()

    def score(self, x_, y_, show_num=500):
        print("Making predictions on data set...")
        predictions = self.predict(x_)
        print("Done.")
        for i in range(show_num):
            plt.plot(i, predictions[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
            plt.plot(i, y_[i], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="green")
        plt.show()

    # This function receives a vector representing a pavement sample and returns a
    # prediction of pavement distress
    def predict(self, x_):
        return self.ann_model.predict(x_)

    # This function overwrites the current model. 
    def save(self):
        return self.ann_model.save(self.working_dir)

    # Load a model
    def load(self):
        self.ann_model = tf.keras.models.load_model(self.working_dir)
