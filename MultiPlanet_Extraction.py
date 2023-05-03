import numpy as np
import pandas as pd
import lifesim as ls
from Extraction import ML_Extraction
from auxiliary import path
import operator as op

#create bus ------------------------------------------------------------------------------------------------------------
ex_bus = ls.Bus()

#set basic scenario
ex_bus.data.options.set_scenario('baseline')

#import catalog
ex_bus.data.import_catalog(path+'05_output_files/standard_simulations/standard500_scen1_spectrum.hdf5')

#add the instrument, transmission, extraction and noise modules and connect them
instrument = ls.Instrument(name='inst')
ex_bus.add_module(instrument)

transm = ls.TransmissionMap(name='transm')
ex_bus.add_module(transm)

exozodi = ls.PhotonNoiseExozodi(name='exo')
ex_bus.add_module(exozodi)

localzodi = ls.PhotonNoiseLocalzodi(name='local')
ex_bus.add_module(localzodi)

star_leak = ls.PhotonNoiseStar(name='star')
ex_bus.add_module(star_leak)

extr = ML_Extraction(name='extr')
ex_bus.add_module(extr)


ex_bus.connect(('inst', 'transm'))
ex_bus.connect(('inst', 'exo'))
ex_bus.connect(('inst', 'local'))
ex_bus.connect(('inst', 'star'))
ex_bus.connect(('star', 'transm'))
ex_bus.connect(('extr', 'inst'))
ex_bus.connect(('extr', 'transm'))

#perform instrument.apply_options() in order to be able to create the extraction class
instrument.apply_options()


#define variables ------------------------------------------------------------------------------------------------------
mu = 0


#define parameters with which to slice your dataset --------------------------------------------------------------------
parameters = np.array([
        ['detected', op.eq, 1],
        ['radius_p', op.ge, 0.5],
        ['radius_p', op.le, 1.5],
        ['flux_p', op.ge, 0.32],
        ['flux_p', op.le, 1.776],
])

attributes = parameters.T[0]
operators = parameters.T[1]
numbers = parameters.T[2]


#create the mask containing only the datapoints as specified by the parameters -----------------------------------------
mask = ex_bus.data.catalog

for i in range(attributes.size):
    aux_mask = operators[i](mask[attributes[i]], numbers[i])
    mask = mask.loc[aux_mask]

ex_bus.data.catalog = mask


#Perform the extraction and save the file in /05_output_files/changeme.csv ---------------------------------------------
extr.main_parameter_extraction(n_run=1, mu=mu, filepath=path+'05_output_files/')