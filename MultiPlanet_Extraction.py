import numpy as np
import pandas as pd
import lifesim as ls
from Extraction import ML_Extraction
from auxiliary import path
import statistics as st
import scipy as sp
import sys as sys
from lifesim.util.radiation import black_body
from astropy import units as u
from functions import get_detection_threshold
import operator as op
from tqdm import tqdm

#create bus ------------------------------------------------------------------------------------------------------------
ex_bus = ls.Bus()

#set basic scenario
ex_bus.data.options.set_scenario('baseline')

#import catalog
ex_bus.data.import_catalog(path+'05_output_files/standard10_scen1_spectrum.hdf5')

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
n_MC = 10
mu = 0


#define parameters with which to slice your dataset
parameters = np.array([
        ['detected', op.eq, 1],
        ['radius_p', op.ge, 0.5],
        ['radius_p', op.le, 0.7],
        ['flux_p', op.ge, 0.32],
        ['flux_p', op.le, 1.776],
])

attributes = parameters.T[0]
operators = parameters.T[1]
numbers = parameters.T[2]


#create the mask containing only the datapoints as specified by the parameters
mask = ex_bus.data.catalog

for i in range(attributes.size):
    aux_mask = operators[i](mask[attributes[i]], numbers[i])
    mask = mask.loc[aux_mask]

ex_bus.data.catalog = mask


extr.main_parameter_extraction(n_MC=n_MC, mu=mu, )

'''
#extract the data from the defined subset of the catalog ---------------------------------------------------------------
#create lists to store extracted data
extracted_spectra = []
extracted_snrs = []
extracted_sigmas = []
extracted_Jmaxs = []
extracted_rss = []
extracted_phiss = []
extracted_Ts = []
extracted_Ts_sigma = []
extracted_Rs = []
extracted_Rs_sigma = []


#run the extraction algorithm and save the data
for i in tqdm(range(mask['radius_p'].size)):

    #define the extraction class and get the parameters
    extraction = ML_Extraction(bus=ex_bus,planet_number=i)
    spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma = extraction.MC_spectrum_extraction(n_MC=n_MC, plot=False)

    #store the data in the lists
    extracted_spectra.append(spectra)
    extracted_snrs.append(snrs)
    extracted_sigmas.append(sigmas)
    extracted_Jmaxs.append(Jmaxs)
    extracted_rss.append(rss)
    extracted_phiss.append(phiss)
    extracted_Ts.append(Ts)
    extracted_Ts_sigma.append(Ts_sigma)
    extracted_Rs.append(Rs)
    extracted_Rs_sigma.append(Rs_sigma)


#add the data to the bus catalog
ex_bus.data.catalog['extracted_spectra'] = extracted_spectra
ex_bus.data.catalog['extracted_snrs'] = extracted_snrs
ex_bus.data.catalog['extracted_sigmas'] = extracted_sigmas
ex_bus.data.catalog['extracted_Jmaxs'] = extracted_Jmaxs
ex_bus.data.catalog['extracted_rss'] = extracted_rss
ex_bus.data.catalog['extracted_phiss'] = extracted_phiss
ex_bus.data.catalog['extracted_Ts'] = extracted_Ts
ex_bus.data.catalog['extracted_Ts_sigma'] = extracted_Ts_sigma
ex_bus.data.catalog['extracted_Rs'] = extracted_Rs
ex_bus.data.catalog['extracted_Rs_sigma'] = extracted_Rs_sigma


#save the catalog
ex_bus.data.catalog.to_csv(path+'05_output_files/changeme.csv')
'''