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

#create bus -----------------------------------------------------------------------------------------------------------
ex_bus = ls.Bus()

#set basic scenario
ex_bus.data.options.set_scenario('baseline')

#import catalog
ex_bus.data.import_catalog(path+'05_output_files/standard10_scen1_spectrum.hdf5')

#add the instrument, transmission and noise modules and connect them
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

ex_bus.connect(('inst', 'transm'))
ex_bus.connect(('inst', 'exo'))
ex_bus.connect(('inst', 'local'))
ex_bus.connect(('inst', 'star'))
ex_bus.connect(('star', 'transm'))

#perform instrument.apply_options() in order to be able to create the extraction class
instrument.apply_options()


#create test system (from old version) ---------------------------------------------------------------------------------

first_row=pd.DataFrame(ex_bus.data.catalog.iloc[0]).transpose()

first_row['distance_s'][0]=15.25
first_row['radius_s'][0]=1
first_row['temp_s'][0]=5778 * 1
first_row['l_sun'][0]=1
first_row['radius_p'][0]=1.25
first_row['semimajor_p'][0]=1.76
first_row['angsep'][0] = first_row['semimajor_p'][0]/first_row['distance_s'][0] #angular separation (arcsec)
first_row['z'][0]=3
first_row['temp_p'][0]=191.90
first_row['baseline'][0] = 15.817 #optimal
first_row['int_time'][0] = 400*55*60*60
first_row['hz_in'][0] = 0.75
first_row['hz_out'][0] = 1.76
first_row['hz_center'][0] = 1.254
first_row['habitable'][0] = True
first_row['s_in'][0] = 1.766
first_row['s_out'][0] = 0.324
first_row['detected'][0] = True
first_row['lat'][0] = np.pi/4

fgamma = black_body(mode='planet',
                    bins=ex_bus.data.inst['wl_bins'],
                    width=ex_bus.data.inst['wl_bin_widths'],
                    temp=first_row['temp_p'][0],
                    radius=first_row['radius_p'][0],
                    distance=first_row['distance_s'][0]) \
                    / ex_bus.data.inst['wl_bin_widths']

first_row['planet_flux_use'][0] = [fgamma]

#dummy
first_row['p_orb'][0] = 10
first_row['mass_p'][0] = 2
first_row['ecc_p'][0] = 0
first_row['inc_p'][0] = 0
first_row['large_omega_p'][0] = 1
first_row['small_omega_p'][0] = 1
first_row['theta_p'][0] = 1
first_row['albedo_geom_mir'][0] = 0.05
first_row['albedo_geom_mir'][0] = 0.05
first_row['sep_p'][0] = first_row['semimajor_p'][0]
first_row['maxangsep'][0] = first_row['sep_p'][0]
first_row['flux_p'][0] = 0 #?????
first_row['fp'][0] = 0 #??????
first_row['mass_s'][0] = 1
first_row['ra'][0] = 100
first_row['dec'][0] = 50
first_row['nuniverse'][0] = 0
first_row['nstar'][0] = 0
first_row['stype'][0] = 'G'
first_row['id'][0] = -1
first_row['name_s'][0] = 'None'
first_row['lon'][0] = 1
first_row['snr_1h'][0] = 0 #???
first_row['photon_rate_planet'][0] = 0 #?????
first_row['photon_rate_noise'][0] = 0 #????
first_row['snr_current'][0] = 26 #????
first_row['t_slew'][0] = 0

ex_bus.data.catalog=pd.concat([first_row, ex_bus.data.catalog],ignore_index=True)
nu = ex_bus.data.catalog
np.set_printoptions(threshold=sys.maxsize)


#define variables ------------------------------------------------------------------------------------------------------
planet_number = 0#2799 #17
n_MC = 1
angsep_accuracy_def = 0.15
phi_accuracy_def = 10


#define the extraction class -------------------------------------------------------------------------------------------
extraction = ML_Extraction(ex_bus, planet_number)


#Get the extracted values for one planet -------------------------------------------------------------------------------
#perform the extraction for one planet
spectra, snrs, sigmas, rss, phiss, Jmaxs_wl, Jmaxs, Ts, Ts_sigma, Rs, Rs_sigma = extraction.MC_spectrum_extraction(n_MC=n_MC, plot=False)


#Get median and MAD of extracted positions (angular separation and azimuthal position)
#r_pixel = rss
#r_angsep = 2*r_pixel*ex_bus.data.inst['hfov']/ex_bus.data.options.other['image_size']*180*3600/np.pi
#print(rss)
#r_median=st.median(r_angsep)
#r_MAD = sp.stats.median_abs_deviation(r_angsep)

r_median = st.median(rss)
r_MAD = sp.stats.median_abs_deviation(rss)

#phi_pixel = locs[:,1]
#phi_angle = phi_pixel*extraction.n_steps/360
#phi_median = st.median(phi_angle)
#phi_MAD = sp.stats.median_abs_deviation(phi_angle)

phi_median = st.median(phiss)
phi_MAD = sp.stats.median_abs_deviation(phiss)

print('')
print('true r (in angsep):',np.round(ex_bus.data.catalog['angsep'][planet_number],5))
print('retrieved r (in angsep):',np.round(r_median,5),'+/-',np.round(r_MAD,5))
print('estimated phi (in degrees):',int(phi_median),'+/-',int(phi_MAD))
print('')


#Get SNR by extraction and by photon statistics
mean_s = snrs.mean(axis=0)
std_s = np.std(snrs)

print('snr by photon count:',np.round(ex_bus.data.catalog['snr_current'][planet_number],5))
print('snr_extracted:',np.round(mean_s,5),'+/-',np.round(std_s,5))
print('')



#Get detection threshold
eta_threshold_5 = get_detection_threshold(L=len(ex_bus.data.inst['wl_bins']), sigma=5)
print('Detection threshold eta (5 sigma) = ',np.round(eta_threshold_5,0))


#Count number of failed extractions. position_fails mean failed position extractions, total_fails means either failed position extraction or J below threshold
position_fails = 0
total_fails = 0

for i in range(n_MC):
    if ((rss[i] > (ex_bus.data.catalog['angsep'][planet_number]*(1+angsep_accuracy_def))) or (rss[i] < (ex_bus.data.catalog['angsep'][planet_number]*(1-angsep_accuracy_def)))):
        position_fails += 1
        total_fails += 1
    elif ((np.abs(phiss[i]-extraction.planet_azimuth)+phi_accuracy_def) % 360 > phi_accuracy_def+phi_accuracy_def):
        position_fails += 1
        total_fails += 1
    #ToDo: How to get the "right" Jmax?
    elif (Jmaxs[i] < eta_threshold_5):
        total_fails += 1

location_accuracy = (n_MC - position_fails)/n_MC
total_accuracy = (n_MC - total_fails)/n_MC

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