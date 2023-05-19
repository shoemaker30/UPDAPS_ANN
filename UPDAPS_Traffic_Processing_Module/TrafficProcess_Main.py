# Title: Code for running the UPDAPS traffic processing module and saving the corresponding results as a pickle file.
#
# Author: Farhad Abdollahi (abdolla4@msu.edu) under supervision of Prof. Kutay (kutay@msu.edu)
# Date: 06/20/2022
#
# (7/22/2022) Edited to take inps as a dictionary rather than a JSON file input. 
# Also saves outputs to a specific directory instead of the same. -Eric
# ======================================================================================================================

# Importing the required libraries.
import pickle
import numpy as np
from UPDAPS_Traffic_Processing_Module.f_rawTraf_to_AXL2 import processTrafficData
from configuration import axel_load_files

def traffic_process_main(inps):
    """
    This is the main function for running the traffic processing module.
    :param inppath: The path of the input JSON file.
    :return: Nothing.
    """
    # Load the JSON file.
    #inps = json.load(open(inppath, 'r'))

    # Preparing variables for running the traffic processing module.
    tDesign = int(inps['tDesign'])
    try:
        grType = inps['grType']
    except:
        grType = inps['crType']
    try:
        monthStart = inps['monthStart']
    except:
        monthStart = inps['monthText']

    # Depending on where the als data is coming from, do some processing
    # if 'alsInterp' key exists in inps, it means it is coming from NAPCOMPlus code
    # Else, it is coming from the earlier UPDAPS I project.
    if 'als' in inps:
        # alsData = inps['alsInterp'].copy()
        # alsData['DataType'] = 1 # NAPCOMPlus data
        # truckClassNames = alsData['rownames']
        # siLoadsLb = array(alsData['siLoadsKips']) * 1000
        # taLoadsLb = array(alsData['taLoadsKips']) * 1000
        # trLoadsLb = array(alsData['trLoadsKips']) * 1000
        # qdLoadsLb = array(alsData['qdLoadsKips']) * 1000

        als = inps['als'].copy()
        alsData = {}
        alsData['DataType'] = 1 # NAPCOMPlus data
        truckClassNames = als['truckClassNames']
        alsData['alsSingle'] = np.array(als['alsSi'])
        alsData['alsTandem'] = np.array(als['alsTa'])
        alsData['alsTridem'] = np.array(als['alsTr'])
        alsData['alsQuad'] = np.array(als['alsQd'])
        siLoadsLb = np.array(als['siLoadsKips']) * 1000
        taLoadsLb = np.array(als['taLoadsKips']) * 1000
        trLoadsLb = np.array(als['trLoadsKips']) * 1000
        qdLoadsLb = np.array(als['qdLoadsKips']) * 1000
        alsData['siLoadsLb'] = siLoadsLb
        alsData['taLoadsLb'] = taLoadsLb
        alsData['trLoadsLb'] = trLoadsLb
        alsData['qdLoadsLb'] = qdLoadsLb

    else:
        alsData = {}
        alsData['alsSingle'] = inps['alsSingle']
        alsData['alsTandem'] = inps['alsTandem']
        alsData['alsTridem'] = inps['alsTridem']
        alsData['alsQuad'] = inps['alsQuad']
        alsData['siLoadsLb'] = inps['siLoadsLb']
        alsData['taLoadsLb'] = inps['taLoadsLb']
        alsData['trLoadsLb'] = inps['trLoadsLb']
        alsData['qdLoadsLb'] = inps['qdLoadsLb']
        alsData['DataType'] = 2 # Earlier method, from UPDAPS-I project
        truckClassNames = inps['truckClass']
        siLoadsLb = np.array(inps['siLoadsLb'])
        taLoadsLb = np.array(inps['taLoadsLb'])
        trLoadsLb = np.array(inps['trLoadsLb'])
        qdLoadsLb = np.array(inps['qdLoadsLb'])

    # _________________________________________________________________________________________
    # Process traffic inps to compute number of applications of single, tandem, tridem and quad axles.
    traf_N = processTrafficData(inps['aadtt'], inps['aptQd'], inps['aptSi'],
                                inps['aptTa'], inps['aptTr'], grType,
                                inps['ddf'], inps['ldf'], inps['mdf'], truckClassNames,
                                inps['classPercent'], inps['growthPercent'], monthStart, tDesign, alsData)

    # Writing the output as a pickle file.
    print("Writing File...")
    pickle.dump(traf_N, open(axel_load_files + '/'+
                                inps['projectname'] + '-TrafficResult.pkl', 'wb'))
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

'''
if __name__ == '__main__':
    # Check the number of inputs.
    if len(sys.argv) == 1:
        main(InputPath)
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('ERROR!!!')
        raise Exception(f'This code requires exactly one input, you entered <<<{len(sys.argv) - 1}>>> inputs.')
'''