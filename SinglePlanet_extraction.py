import numpy as np
import lifesim as ls
from Extraction import ML_Extraction
from Extraction_auxiliary import *
from auxiliary import path
import statistics as st
import scipy as sp
import sys as sys


#create bus ------------------------------------------------------------------------------------------------------------
ex_bus = ls.Bus()

#set basic scenario
ex_bus.data.options.set_scenario('baseline')

#import catalog
ex_bus.data.import_catalog(path+'05_output_files/standard_simulations/standard10_scen1_spectrum.hdf5')

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

#perform instrument.apply_options() in order to be able to create the extraction class
instrument.apply_options()

np.set_printoptions(threshold=sys.maxsize)

#define variables ------------------------------------------------------------------------------------------------------
planet_number = 5 #118 #5 #2785 #11 #24 #4 #2798 #17 #2 #0
n_runs = 1
mu=0
whitening_limit = 0
angsep_accuracy_def = 0.15
phi_accuracy_def = 10

include_dips = False
atmospheric_scenario = mix_40


#Call the main_parameter_extraction function ---------------------------------------------------------------------------
spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma, FPRs, FPR_maxs,\
    induced_dips, t_scores, SNR_ps_news, bayes_factors = \
    extr.main_parameter_extraction(n_run=n_runs, plot=True, mu=mu,
                                   whitening_limit=whitening_limit, single_planet_mode=True,
                                   planet_number=planet_number, include_dips=include_dips,
                                   atmospheric_scenario=atmospheric_scenario, filepath=path+'05_output_files/')


#Perform the data analysis ---------------------------------------------------------------------------------------------
#Get median and MAD of extracted positions (angular separation and azimuthal position)
r_median = st.median(rss)
r_MAD = sp.stats.median_abs_deviation(rss)

#modified phiss makes sure that e.g. azimuthal angle 1 and 359 are equivalent
phiss_modified = np.empty_like(phiss)
for i in range(phiss.size):
    phiss_modified[i] = np.minimum(np.abs(0.-phiss[i]),np.abs(360.-phiss[i]))

phi_median = st.median(phiss_modified)
phi_MAD = sp.stats.median_abs_deviation(phiss_modified)

print('')
print('true r (in angsep):',np.round(ex_bus.data.catalog['angsep'][planet_number],5))
print('retrieved r (in angsep):',np.round(r_median,5),'+/-',np.round(r_MAD,5))
print('estimated phi (in degrees):',int(phi_median),'+/-',int(phi_MAD))
print('')


#Get SNR by extraction and by photon statistics
mean_s = snrs.mean(axis=0)
std_s = np.std(snrs)
mean_FPR = FPRs.mean(axis=0)
std_FPR = np.std(FPRs)
mean_FPR_max = FPR_maxs.mean(axis=0)
std_FPR_max = np.std(FPR_maxs)

print('snr by photon statistics:',np.round(ex_bus.data.catalog['snr_current'][planet_number],3))
print('snr_extracted:',np.round(mean_s,5),'+/-',np.round(std_s,3))
print('FPR extracted:',np.round(mean_FPR,5),'+/-',np.round(std_FPR,3))
print('FPR max extracted:', np.round(mean_FPR_max,5),'+/-',np.round(std_FPR_max,3))
print('')


#Get detection threshold and median/MAD of the extracted Jmaxs
eta_threshold_5 = get_detection_threshold(L=extr.L,sigma=5)
eta_max_threshold_5 = get_detection_threshold_max(L=extr.L, sigma=5, radial_ang_pix=extr.image_size/2)

Jmax_median = st.median(Jmaxs)
Jmax_MAD = sp.stats.median_abs_deviation(Jmaxs)

print('Detection threshold eta (5 sigma) = ',np.round(eta_threshold_5,1))
print('Detection threshold eta_max (5 sigma) = ',np.round(eta_max_threshold_5,1))
print('Median maximum of cost function J:',np.round(Jmax_median,0),'+/-',np.round(Jmax_MAD,0))
print('')


#Count number of failed extractions. position_fails mean failed position extractions, total_fails means either
#   failed position extraction or J below threshold
position_fails = 0
total_fails = 0

for i in range(n_runs):
    if ((rss[i] > (ex_bus.data.catalog['angsep'][planet_number]*(1+angsep_accuracy_def))) or
            (rss[i] < (ex_bus.data.catalog['angsep'][planet_number]*(1-angsep_accuracy_def)))):
        position_fails += 1
        total_fails += 1
    elif ((np.abs(phiss[i]-extr.planet_azimuth)+phi_accuracy_def) % 360 > phi_accuracy_def+phi_accuracy_def):
        position_fails += 1
        total_fails += 1
    elif (Jmaxs[i] < eta_threshold_5):
        total_fails += 1

location_accuracy = (n_runs - position_fails)/n_runs
total_accuracy = (n_runs - total_fails)/n_runs

print('Failed location estimates: ',position_fails,' => ',location_accuracy*100,'% success rate')
print('Failed extractions: ',total_fails, ' => ',total_accuracy*100,'% success rate')
print('')


#Get estimated radius and temperature from the spectrum fit
T_median = st.median(Ts)
T_MAD = sp.stats.median_abs_deviation(Ts)
R_median = st.median(Rs)
R_MAD = sp.stats.median_abs_deviation(Rs)

print('true Tp (in Kelvin) = ',np.round(ex_bus.data.catalog['temp_p'][planet_number],2))
print('fitted Tp (in Kelvin) = ',np.round(T_median,2),'+/-',np.round(T_MAD,2))
print('true Rp (in R_earth) = ',np.round(ex_bus.data.catalog['radius_p'][planet_number],2))
print('fitted Rp (in R_earth) = ',np.round(R_median,2),'+/-',np.round(R_MAD,2))
print('')


#Count how many times planet is localized too close/far away from star
too_close = 0
too_far = 0

for j in range (len(rss)):
    if (rss[j] < (ex_bus.data.catalog['angsep'][planet_number]*(1-angsep_accuracy_def))):
        too_close += 1
    elif (rss[j] > (ex_bus.data.catalog['angsep'][planet_number]*(1+angsep_accuracy_def))):
        too_far += 1

print('Planet detected too close to star in ',too_close,' of ',len(rss),' cases')
print('Planet detected too far from star in ',too_far,' of ',len(rss),' cases')
print('')


# check dip for first planet
if (include_dips == False):
    print('No dips included')
else:
    strong_alpha = 0.09
    decisive_alpha = 0.01
    strong_jeffrey = 1.0
    decisive_jeffrey = 2.0

    strong_t_statistics = np.zeros_like(t_scores)
    decisive_t_statistics = np.zeros_like(t_scores)
    strong_bayesian_models = np.zeros_like(bayes_factors)
    decisive_bayesian_models = np.zeros_like(bayes_factors)

    for i in range(n_runs):
        for j in range (len(atmospheric_scenario.dip_molecules)):
            if (t_scores[i][j] <= strong_alpha):
                strong_t_statistics[i][j] = 1
            if (t_scores[i][j] <= decisive_alpha):
                decisive_t_statistics[i][j] = 1
            if (np.log10(bayes_factors[i][j]) >= strong_jeffrey):
                strong_bayesian_models[i][j] = 1
            if (np.log10(bayes_factors[i][j]) >= decisive_jeffrey):
                decisive_bayesian_models[i][j] = 1


    print('Dips induced:', induced_dips[0])
    print('New SNR photon statistics:', np.round(SNR_ps_news[0],3))
    print('t-scores:', t_scores[0])
    print('Dips extracted alpha=0.09:', strong_t_statistics[0])
    print('Dips extracted alpha=0.01:', decisive_t_statistics[0])

    print('Log of Bayes factors:', np.round(np.log10(bayes_factors[0]),3))
    print('Models with strong evidence:', strong_bayesian_models[0])
    print('Models with decisive evidence:', decisive_bayesian_models[0])