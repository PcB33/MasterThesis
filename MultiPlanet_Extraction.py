import numpy as np
import lifesim as ls
from Extraction import ML_Extraction
from Extraction_auxiliary import *
import operator as op
import random as ran
import warnings

if __name__ == '__main__':

    # define whether you are running the file locally or on the server (to define the correct paths)
    local = False

    # suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # create bus --------------------------------------------------------------------------------------------------------
    ex_bus = ls.Bus()

    # set basic scenario
    ex_bus.data.options.set_scenario('baseline')

    # import catalog
    if (local == True):
        path = 'C:/Users/Philipp Binkert/OneDrive/ETH/Master_Thesis/'
        ex_bus.data.import_catalog(path+'05_output_files/standard_simulations/standard10_scen1_spectrum.hdf5')
    else:
        ex_bus.data.import_catalog('/home/ipa/quanz/student_theses/master_theses/2023/binkertp/MasterThesis/'
                                   'standard500_scen2_spectrum.hdf5')

    # set the resolution to the required level
    L_used = ex_bus.data.catalog['planet_flux_use'][0][0].size
    if (L_used == 31):
        ex_bus.data.options.set_manual(spec_res=20.)
    elif (L_used == 77):
        ex_bus.data.options.set_manual(spec_res=50.)
    elif (L_used == 154):
        ex_bus.data.options.set_manual(spec_res=100.)
    elif (L_used == 16):
        ex_bus.data.options.set_manual(spec_res=10.)

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


    # define variables --------------------------------------------------------------------------------------------------
    mu = 0
    whitening_limit = 0
    n_processes = 40
    n_universes = 100
    precision_limit = 1600
    max_FoS = False


    include_dips = False
    atmospheric_scenario = Earth_like


    # define parameters with which to slice your dataset ---------------------------------------------------------------
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


    # define the random universes to choose
    total_universes = ex_bus.data.catalog['nuniverse'].max() + 1
    random_universes = ran.sample(range(total_universes), n_universes)


    # create the mask containing only the datapoints as specified by the parameters -------------------------------------
    mask = ex_bus.data.catalog

    for i in range(attributes.size):
        aux_mask = operators[i](mask[attributes[i]], numbers[i])
        mask = mask.loc[aux_mask]


    # drop the rows not belonging to the randomly selected universes
    for i in range(total_universes):
        if i not in random_universes:
            mask = mask.drop(mask[mask['nuniverse'] == i].index)


    print('Number of planets selected:', mask['radius_p'].size)

    # save the filtered catalog to the bus
    ex_bus.data.catalog = mask

    # perform the extraction and save the file as changeme.csv ---------------------------------------------------------
    if (local == True):
        extr.main_parameter_extraction(n_run=1, mu=mu, whitening_limit=whitening_limit, n_processes=n_processes,
                                            precision_limit=precision_limit, max_FoS=max_FoS, include_dips=include_dips,
                                            atmospheric_scenario=atmospheric_scenario, filepath=path+'05_output_files/')
    else:
        extr.main_parameter_extraction(n_run=1, mu=mu, whitening_limit=whitening_limit,  n_processes=n_processes,
                                            precision_limit=precision_limit, max_FoS=max_FoS, include_dips=include_dips,
                                            atmospheric_scenario=atmospheric_scenario,
                                            filepath='/home/ipa/quanz/student_theses/master_theses/2023/binkertp'
                                                                                                '/MasterThesis/')