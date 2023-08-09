import numpy as np
import lifesim as ls
from Extraction_FPF_test import ML_Extraction
from Extraction_auxiliary import *
import operator as op
import random as ran
import warnings

if __name__ == '__main__':

    # Define whether you are running the file locally or on the server (to define the correct paths)
    local = False

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # create bus --------------------------------------------------------------------------------------------------------
    ex_bus = ls.Bus()

    # set basic scenario
    ex_bus.data.options.set_scenario('baseline')

    # import catalog
    if (local == True):
        path = 'C:/Users/Philipp Binkert/OneDrive/ETH/Master_Thesis/'
        ex_bus.data.import_catalog(path + '05_output_files/standard_simulations/standard10_scen1_spectrum.hdf5')
    else:
        ex_bus.data.import_catalog('/home/ipa/quanz/student_theses/master_theses/2023/binkertp/MasterThesis/'
                                   'standard10_scen1_spectrum.hdf5')

    # set the resolution to the required level
    L_used = ex_bus.data.catalog['planet_flux_use'][0][0].size
    if (L_used == 31):
        ex_bus.data.options.set_manual(spec_res=20.)
    elif (L_used == 77):
        ex_bus.data.options.set_manual(spec_res=50.)
    elif (L_used == 154):
        ex_bus.data.options.set_manual(spec_res=100.)

    # add the instrument, transmission, extraction and noise modules and connect them
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

    # perform instrument.apply_options() in order to be able to create the extraction class
    instrument.apply_options()


    # define variables ------------------------------------------------------------------------------------------------------
    planet_number = 6
    n_processes = 40
    n_runs = 10000

    # Perform the extraction and save the file as changeme.csv ---------------------------------------------------------
    if (local == True):
        extr.FPF_test(planet_number=planet_number, n_runs=n_runs, n_processes=n_processes, filepath=path + '05_output_files/')

    else:
        extr.FPF_test(planet_number=planet_number, n_runs=n_runs, n_processes=n_processes,
                      filepath='/home/ipa/quanz/student_theses/master_theses/2023/binkertp'
                                                '/MasterThesis/')