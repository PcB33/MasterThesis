import numpy as np
import lifesim as ls
from Extraction import ML_Extraction
import operator as op
import random as ran
import warnings

if __name__ == '__main__':

    #Define whether you are running the file locally or on the server (to define the correct paths)
    local = False

    #Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    #create bus --------------------------------------------------------------------------------------------------------
    ex_bus = ls.Bus()

    #set basic scenario
    ex_bus.data.options.set_scenario('baseline')

    #import catalog
    if (local == True):
        path = 'C:/Users/Philipp Binkert/OneDrive/ETH/Master_Thesis/'
        ex_bus.data.import_catalog(path+'05_output_files/standard_simulations/standard10_scen2_spectrum.hdf5')
    else:
        ex_bus.data.import_catalog('/home/ipa/quanz/student_theses/master_theses/2023/binkertp/MasterThesis/'
                                   'standard500_scen2_spectrum.hdf5')

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


    #define variables --------------------------------------------------------------------------------------------------
    mu = 0
    whitening_limit = 0
    n_processes = 32
    n_universes = 100
    precision_limit = 1600


    # define parameters with which to slice your dataset ----------------------------------------------------------------
    parameters = np.array([
            ['detected', op.eq, 1],
            ['radius_p', op.ge, 0.82],
            ['radius_p', op.le, 1.4],
            ['flux_p', op.ge, 0.356],
            ['flux_p', op.le, 1.107],
    ])

    attributes = parameters.T[0]
    operators = parameters.T[1]
    numbers = parameters.T[2]


    #define the random universes to choose
    total_universes = ex_bus.data.catalog['nuniverse'].max() + 1
    random_universes = ran.sample(range(total_universes), n_universes)


    #create the mask containing only the datapoints as specified by the parameters -------------------------------------
    mask = ex_bus.data.catalog

    for i in range(attributes.size):
        aux_mask = operators[i](mask[attributes[i]], numbers[i])
        mask = mask.loc[aux_mask]

    
    #drop the rows not belonging to the randomly selected universes
    for i in range(total_universes):
        if i not in random_universes:
            mask = mask.drop(mask[mask['nuniverse'] == i].index)



    print('Number of planets selected:', mask['radius_p'].size)

    #save the filtered catalog to the bus
    ex_bus.data.catalog = mask

    #Perform the extraction and save the file as changeme.csv ---------------------------------------------------------
    if (local == True):
        extr.main_parameter_extraction(n_run=1, mu=mu, whitening_limit=whitening_limit, n_processes=n_processes,
                                            precision_limit=precision_limit, filepath=path+'05_output_files/')
    else:
        extr.main_parameter_extraction(n_run=1, mu=mu, whitening_limit=whitening_limit,  n_processes=n_processes,
                                            precision_limit=precision_limit, filepath='/home/ipa/quanz/student_theses'
                                                                                      '/master_theses/2023/binkertp'
                                                                                      '/MasterThesis/')