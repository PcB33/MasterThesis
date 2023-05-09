import lifesim as ls
import pandas as pd
import numpy as np
from auxiliary import path, convert_to_minus_pi_to_pi, convert_to_zero_to_2pi
import os
import math
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from poliastro.constants.general import M_sun
from astropy.constants import G
from astropy import units as u
import scipy as sp

def get_detections(file, parameter_matrix):
    '''
    :param file: input file
    :param parameter_matrix: parameters to filter the data
    :return: mean # of detections, std of # of detections, median SNR of detections (=0 if no detections)
    '''

    #unpack matrix
    attributes = parameter_matrix.T[0]
    operators = parameter_matrix.T[1]
    numbers = parameter_matrix.T[2]

    #import input file to dataframe

    simulated_data = pd.read_csv(path + "05_output_files/" + file)

    #define variables
    N_universe = simulated_data['nuniverse'].max() + 1
    n_detections = np.empty(N_universe)
    mask = simulated_data

    #create masks to filter for each attribute
    for i in range(attributes.size):
        aux_mask = operators[i](simulated_data[attributes[i]],numbers[i])
        mask = mask.loc[aux_mask]

    #calculate the number of detections
    for i in range(N_universe):
        mask_i = mask.loc[mask['nuniverse'] == i]
        n_detections[i] = mask_i.shape[0]

    #calculate mean and std of detections
    mean_detections = np.mean(n_detections)
    std_detections = np.std(n_detections)

    #calculate median SNR; return 0 if there are no detections
    median_SNR = mask['snr_current'].median()

    if (math.isnan(median_SNR)==1):
        median_SNR = 0

    return mean_detections, std_detections, median_SNR


def move_planets(input_semimajor_p, input_ecc_p, input_inc_p, input_large_omega_p, input_small_omega_p, input_theta_p,
                 input_mass_s, input_distance_s, propagation_time):
    '''

    :param input_semimajor_p: series with all inputs
    :param input_ecc_p: series with all inputs
    :param input_inc_p: series with all inputs
    :param input_large_omega_p: series with all inputs
    :param input_small_omega_p: series with all inputs
    :param input_theta_p: series with all inputs
    :param input_mass_s: series with all inputs
    :param input_distance_s: series with all inputs
    :param propagation_time: time the planets should propagate in days

    :return: new_true_anomalies: true anomalies of all the planets after the propagation time has passed
             new_angseps: angular separations of all the planets after the propagation time has passed
    '''

    N_elements = input_semimajor_p.size
    new_true_anomalies = np.empty(N_elements)
    new_angseps = np.empty(N_elements)

    # loop through all of the elements in the input file
    for element in range(N_elements):
        # define planetary parameters
        semimajor_p = input_semimajor_p[element] * u.au
        ecc_p = input_ecc_p[element] * u.one
        inc_p = input_inc_p[element] * u.rad
        raan_p = input_large_omega_p[element] * u.rad
        argp_p = input_small_omega_p[element] * u.rad
        true_anomaly_p_raw = input_theta_p[element] * u.rad
        true_anomaly_p = convert_to_minus_pi_to_pi(true_anomaly_p_raw / u.rad) * u.rad

        # define Star
        Star = Body(None, k=input_mass_s[element] * M_sun.min() * G.min(), name='star', mass=input_mass_s[element])
        distance_star_earth = input_distance_s[element] * u.pc

        # define orbit
        orb = Orbit.from_classical(Star, semimajor_p, ecc_p, inc_p, raan_p, argp_p, true_anomaly_p)

        # propagate orbit
        new_state = orb.propagate(propagation_time * u.day)
        new_true_anomaly_p = convert_to_zero_to_2pi(new_state.nu / u.rad) * u.rad

        # calculate angular separation from true anomaly
        rp = semimajor_p * (1 - ecc_p ** 2) / (1 + ecc_p * np.cos(new_true_anomaly_p))
        rpproj = rp * np.sqrt(
            np.cos(argp_p + new_true_anomaly_p) ** 2 + np.cos(inc_p) ** 2 * np.sin(argp_p + new_true_anomaly_p) ** 2)
        new_angsep_element = (rpproj / u.au) / (distance_star_earth / u.pc) * u.arcsec

        # add new values to arrays
        new_true_anomalies[element] = new_true_anomaly_p / u.rad
        new_angseps[element] = new_angsep_element / u.arcsec

    return new_true_anomalies, new_angseps



def multivisit_0(input_file,n_visits,optimize_habitable,output_file):
    '''
    :param input_file: input file for the bus (PPOP)
    :param n_visits: number of visits to perform
    :param optimize_habitable: True if the optimizer should optimize for habitable planets, False for all planets
    :param output_file: name of the output file
    :return: Nothing; a file with the outputname is automatically saved
    '''

    #create bus
    bus=ls.Bus()

    #set basic scenario and make potential changes with .set_manual
    bus.data.options.set_scenario('baseline')
    bus.data.options.set_manual(t_search=2.5/n_visits*365*24*60*60)

    #import ppop data to bus
    bus.data.catalog_from_ppop(input_path=path+input_file)

    #remove potential stars from the bus; removing A stars and (M stars > 10parsec) is assumed in the standard scenarios 1&2
    bus.data.catalog_remove_distance(stype='A', mode='larger', dist=0.)
    bus.data.catalog_remove_distance(stype='M', mode='larger', dist=10.)

    #add the instrument, transmission and noise modules and connect them
    instrument = ls.Instrument(name='inst')
    bus.add_module(instrument)

    transm = ls.TransmissionMap(name='transm')
    bus.add_module(transm)

    exozodi = ls.PhotonNoiseExozodi(name='exo')
    bus.add_module(exozodi)

    localzodi = ls.PhotonNoiseLocalzodi(name='local')
    bus.add_module(localzodi)

    star_leak = ls.PhotonNoiseStar(name='star')
    bus.add_module(star_leak)

    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))
    bus.connect(('star', 'transm'))


    #add and connect the optimizer and ahgs modules
    opt = ls.Optimizer(name='opt')
    bus.add_module(opt)
    ahgs = ls.AhgsModule(name='ahgs')
    bus.add_module(ahgs)

    bus.connect(('transm', 'opt'))
    bus.connect(('inst', 'opt'))
    bus.connect(('opt', 'ahgs'))

    #optimize the search phase; set ...['habitable'] to True/False to optimize for # of planets in habitable zone
    bus.data.options.optimization['habitable'] = optimize_habitable

    #Loop that gets the SNR_1h, does the optimization and saves the file for every visit
    for i in range(n_visits):
        instrument.get_snr(save_mode=False)

        opt.ahgs()

        index = str(i+1)
        bus.data.catalog.to_csv(path+'05_output_files/Auxiliary/run_'+index+'.csv')

        if (i != (n_visits-1)):
            print('Propagating planet positions')
            bus.data.catalog['theta_p'], bus.data.catalog['angsep'] = move_planets(bus.data.catalog['semimajor_p'],
                                                                                   bus.data.catalog['ecc_p'],
                                                                                   bus.data.catalog['inc_p'],
                                                                                   bus.data.catalog['large_omega_p'],
                                                                                   bus.data.catalog['small_omega_p'],
                                                                                   bus.data.catalog['theta_p'],
                                                                                   bus.data.catalog['mass_s'],
                                                                                   bus.data.catalog['distance_s'],
                                                                                   2.5/n_visits*365)


    #create the final dataframe by importing the csv from the last visit
    final_df = pd.read_csv(path+'05_output_files/Auxiliary/run_'+str(n_visits)+'.csv')

    #add the snr_currents from each of the visits
    for i in range(n_visits-1):
        index = str(i+1)
        final_df['snr run '+index] = pd.read_csv(path+'05_output_files/Auxiliary/run_'+index+'.csv')['snr_current']


    #reorder columns
    cols = list(final_df.columns)
    cols.remove('snr_current')
    cols.append('snr_current')
    final_df = final_df.reindex(columns=cols)
    final_df = final_df.rename(columns={'snr_current': 'snr run '+str(n_visits)})


    #calculate total snr and whether a planet was detected (note that the total snr is named snr_current again for compatibility with the statistical analysis function)
    sum_sqrts = 0
    for i in range(n_visits):
        index = str(i+1)
        sum_sqrts += final_df['snr run '+index]**2

    final_df['snr_current'] = np.sqrt(sum_sqrts)
    final_df['final_detected'] = final_df.apply(lambda row: True if row['snr_current'] >= 7 else False, axis=1)


    #save final dataframe and delete the rest
    final_df.to_csv(path+'05_output_files/'+output_file)
    print('Output file saved')

    for i in range(n_visits):
        index = str(i+1)
        os.remove(path+'05_output_files/Auxiliary/run_'+index+'.csv')

    return


def get_detection_threshold(L, sigma):
    eta = np.linspace(0, 300, int(10 ** 5))

    fact = sp.special.factorial
    cdf = 1 / 2 ** L
    for l in range(0, L):
        cdf += fact(L) / (2 ** L * fact(l) * fact(L - l)) * sp.special.gammainc((L - l) / 2, eta / 2)

    eta_ind_sigma = np.searchsorted(cdf, sp.stats.norm.cdf(sigma))
    eta_threshold_sigma = eta[eta_ind_sigma]

    return eta_threshold_sigma


