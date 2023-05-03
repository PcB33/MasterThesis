import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auxiliary import path
from functions import get_detection_threshold
import corner as corner


#define variables ------------------------------------------------------------------------------------------------------
file = 'changeme.csv'
angsep_accuracy_def = 0.15
phi_accuracy_def = 10
true_phi = 0


#load the input file, calculate some required variables and define empty lists of quantities to be stored --------------
extracted_data = pd.read_csv(path+'/05_output_files/multiextraction/'+file)
n_planets = len(extracted_data.index)

L = len(eval(extracted_data['extracted_spectra'][0])[0])
eta_threshold_5 = get_detection_threshold(L, 5)


position_fails = 0
total_fails = 0

SNR_ps_used = []
SNR_ratios = []
Theta_ratios = []
T_ratios = []
R_ratios = []


#go through all of the values and add the desired quantities to the lists ----------------------------------------------
for i in range(n_planets):
    snr_ps = extracted_data['snr_current'][i]
    true_angsep = extracted_data['angsep'][i]
    true_phi = true_phi
    true_T = extracted_data['temp_p'][i]
    true_R = extracted_data['radius_p'][i]

    snr_i = eval(extracted_data['extracted_snrs'][i])[0]
    r_i = eval(extracted_data['extracted_rss'][i])[0]
    phi_i = eval(extracted_data['extracted_phiss'][i])[0]
    T_i = eval(extracted_data['extracted_Ts'][i])[0]
    R_i = eval(extracted_data['extracted_Rs'][i])[0]
    jmax_i = eval(extracted_data['extracted_Jmaxs'][i])[0]


    if (jmax_i < eta_threshold_5):
        total_fails += 1

    elif ((r_i > true_angsep*(1+angsep_accuracy_def)) or (r_i < true_angsep*(1-angsep_accuracy_def))):
        position_fails += 1
        total_fails += 1

        #ToDo think about this (if you include them, T_mean fucks up)
        '''
        SNR_ps_used.append(snr_ps)
        SNR_ratios.append(snr_i / snr_ps)
        Theta_ratios.append(r_i / true_angsep)
        T_ratios.append(T_i / true_T)
        R_ratios.append(R_i / true_R)
        '''

    elif ((np.abs(phi_i-true_phi)+phi_accuracy_def) % 360 > phi_accuracy_def+phi_accuracy_def):
        position_fails += 1
        total_fails += 1

        '''
        SNR_ps_used.append(snr_ps)
        SNR_ratios.append(snr_i / snr_ps)
        Theta_ratios.append(r_i / true_angsep)
        T_ratios.append(T_i / true_T)
        R_ratios.append(R_i / true_R)
        '''

    else:
        SNR_ps_used.append(snr_ps)
        SNR_ratios.append(snr_i / snr_ps)
        Theta_ratios.append(r_i / true_angsep)
        T_ratios.append(T_i / true_T)
        R_ratios.append(R_i / true_R)



#calculate the means and stds and print the results --------------------------------------------------------------------
SNR_ps_used = np.array(SNR_ps_used)
SNR_ratios = np.array(SNR_ratios)
Theta_ratios = np.array(Theta_ratios)
T_ratios = np.array(T_ratios)
R_ratios = np.array(R_ratios)


mean_SNR_ratio = SNR_ratios.mean()
std_SNR_ratio = np.std(SNR_ratios)

mean_Theta_ratio = Theta_ratios.mean()
std_Theta_ratio = np.std(Theta_ratios)

mean_T_ratio = T_ratios.mean()
std_T_ratio = np.std(T_ratios)

mean_R_ratio = R_ratios.mean()
std_R_ratio = np.std(R_ratios)

location_accuracy = (n_planets - position_fails)/n_planets
total_accuracy = (n_planets - total_fails)/n_planets


print('Failed extractions: ',total_fails, ' => ',np.round(total_accuracy*100,2),'% success rate')
print('Excluding the failed extractions in the following')
print('')
print('Failed location estimates: ',position_fails,' => ',np.round(location_accuracy*100,2),'% success rate')

print('SNR_est/SNR_ps = ',np.round(mean_SNR_ratio,2),'+/-',np.round(std_SNR_ratio,2))
print('Theta_est/Theta_true = ',np.round(mean_Theta_ratio,2),'+/-',np.round(std_Theta_ratio,2))
print('T_est/T_true = ',np.round(mean_T_ratio,2),'+/-',np.round(std_T_ratio,2))
print('R_est/R_true = ',np.round(mean_R_ratio,2),'+/-',np.round(std_R_ratio,2))
print('')


#create histogram plots like figure 12 LIFE II -------------------------------------------------------------------------
#lists with the variables to be shown as well as attributes
variables = [extracted_data['snr_current'], extracted_data['temp_p'], extracted_data['radius_p'], extracted_data['angsep']*1000]
number_bins = [200, 40, 40, 130]
x_labels = ['SNR$_\mathrm{pred}$', '$T_\mathrm{p}$ [K]', '$R_\mathrm{p}$ [$R_\oplus$]', '$\Theta_\mathrm{p}$ [mas]']
y_labels = ['# of planets per Universe', '# of planets per Universe', '# of planets per Universe', '# of planets per Universe']
x_lims = [[0,50], [140,310], [0.5,1.5], [0,100]]
y_lims = [[0,6], [0,2.5], [0,2.2], [0,5.2]]

n_universes = extracted_data['nuniverse'].max() + 1

#create figure and fill plots by running through for loop for all list entries
fig, axes = plt.subplots(len(variables), 1, figsize=(6.4,9.6),)

for i in range(len(variables)):
    ax = axes[i]
    ax.hist(variables[i], bins=number_bins[i],
         weights=np.ones_like(variables[i])/n_universes,
         color="darkblue", rwidth=0.9)
    ax.set_xlabel(x_labels[i], fontsize=10)
    ax.set_ylabel(y_labels[i], fontsize=7)
    ax.set_xlim(x_lims[i])
    ax.set_ylim(y_lims[i])
    ax.grid()

plt.tight_layout()
#plt.savefig(path+'/06_plots/distributionplot_changeme.png')
plt.show()


#create cornerplot like figure 13 LIFE II ------------------------------------------------------------------------------
#stack the plot input data
data_SNR_ps_used = SNR_ps_used
data_SNR_ratios = SNR_ratios
data_T_ratios = T_ratios
data_R_ratios = R_ratios
data_Theta_ratios = Theta_ratios

all_data = np.vstack([data_SNR_ps_used, data_SNR_ratios, data_T_ratios, data_R_ratios, data_Theta_ratios])

#define labels and quantiles
labels = ['SNR$_\mathrm{ps}$',
          'SNR$_\mathrm{est}$/SNR$_\mathrm{ps}$',
          '$T_\mathrm{est} / T_\mathrm{true}$',
          '$R_\mathrm{est} / R_\mathrm{true}$',
          '$\Theta_\mathrm{est} / \Theta_\mathrm{true}$']

quantiles = [.159, .841]

#define the main figure
fig = plt.figure(figsize=(12, 12))
fig = corner.corner(np.transpose(all_data), fig=fig, range=[(7, 30), (0.6, 1.4), (0.4, 1.6), (0.0, 2), (.88, 1.12)],
                    labels=labels, show_titles=True, hist_bin_factor=1.5, max_n_ticks=5,
                    plot_density=False, plot_contours=False, no_fill_contours=True, color='darkblue', quantiles=quantiles)


#set limits and ticks
for i in range(len(fig.axes)):
    ax = fig.axes[i]
    ax.grid(True)

    if i in [0, 5, 10, 15, 20]:
        ax.set_xticks([10, 15, 20, 25, 30])
        ax.set_xlim([7, 30])

    if i in [6, 11, 16, 21]:
        ax.set_xticks([0.8, 1., 1.2])
        ax.set_xlim([0.6, 1.4])
    if i in [5]:
        ax.set_yticks([0.8, 1., 1.2])
        ax.set_ylim([0.6, 1.4])

    if i in [12, 17, 22]:
        ax.set_xticks([0.5, 0.75, 1., 1.25, 1.5])
        ax.set_xlim([0.4, 1.6])
    if i in [10, 11]:
        ax.set_yticks([0.75, 1., 1.25, 1.5])
        ax.set_ylim([0.4, 1.6])

    if i in [18, 23]:
        ax.set_xticks([0.5, 1., 1.5])
        ax.set_xlim([0, 2.])
    if i in [15, 16, 17]:
        ax.set_yticks([0.5, 1., 1.5])
        ax.set_ylim([0, 2.])

    if i in [24]:
        ax.set_xticks([0.95, 1, 1.05])
        ax.set_xlim([0.88, 1.12])
    if i in [20, 21, 22, 23]:
        ax.set_yticks([0.95, 1, 1.05])
        ax.set_ylim([0.88, 1.12])


fig.axes[0].plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
#plt.savefig(path+'/06_plots/cornerplot_changeme.png')
plt.show()