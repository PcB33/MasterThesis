import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auxiliary import path
from Extraction_auxiliary import get_detection_threshold, get_detection_threshold_max
import corner as corner


#define variables ------------------------------------------------------------------------------------------------------
file = 'randomsample_all.csv'
angsep_accuracy_def = 1000#0.15
phi_accuracy_def = 1000#10
true_phi = 0
#Define according to what SNR method th filter for detected should be made (== 1, 2, 3)
defining_criteria = 2


#load the input file, calculate some required variables and define empty lists of quantities to be stored --------------
extracted_data = pd.read_csv(path+'/05_output_files/multiextraction/'+file)
n_planets = len(extracted_data.index)


L = len(eval(extracted_data['extracted_spectra'][0])[0])
radial_ang_px = 128
eta_threshold_5 = get_detection_threshold(L, 5)
eta_threshold_max_5 = get_detection_threshold_max(L, 5, radial_ang_px)


total_fails = 0
snr_limit_fails = 0
position_fails = 0
r_fails = 0
ang_fails = 0
FPR_calc_fails = 0
successes = 0

smallest_SNR_ps = 10000

#SNRs: SNR_ratios_1 -> Naive approach; SNR_ratios_2 -> Approach J true position; SNR_ratios_3 -> Approach J max position
SNR_ps_used = []
SNR_ratios_1 = []
SNR_ratios_2 = []
SNR_ratios_3 = []
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

    snr_1 = eval(extracted_data['extracted_snrs'][i])[0]
    snr_2 = eval(extracted_data['extracted_FPRs'][i])[0]
    snr_3 = eval(extracted_data['extracted_FPR_maxs'][i], {'inf': float('inf')})[0]
    r_extr = eval(extracted_data['extracted_rss'][i])[0]
    phi_extr = eval(extracted_data['extracted_phiss'][i])[0]
    T_extr = eval(extracted_data['extracted_Ts'][i])[0]
    R_extr = eval(extracted_data['extracted_Rs'][i])[0]
    jmax_extr = eval(extracted_data['extracted_Jmaxs'][i])[0]


    if (defining_criteria == 1 and snr_1 < 5):
        total_fails += 1
        snr_limit_fails += 1

    elif (defining_criteria == 2 and snr_2 < 0):
        total_fails += 1
        snr_limit_fails += 1

    elif (defining_criteria == 3 and snr_3 < 5):
        total_fails += 1
        snr_limit_fails += 1

    elif ((r_extr > true_angsep*(1+angsep_accuracy_def)) or (r_extr < true_angsep*(1-angsep_accuracy_def))):
        total_fails += 1
        position_fails += 1
        r_fails += 1

    elif ((np.abs(phi_extr-true_phi)+phi_accuracy_def) % 360 > phi_accuracy_def+phi_accuracy_def):
        total_fails += 1
        position_fails += 1
        ang_fails += 1

    else:
        successes += 1

        SNR_ps_used.append(snr_ps)
        SNR_ratios_1.append(snr_1 / snr_ps)
        Theta_ratios.append(r_extr / true_angsep)
        T_ratios.append(T_extr / true_T)
        R_ratios.append(R_extr / true_R)

        #if either of the two methods were unsuccessful in calculating the sigma, set them both to 1 for simplicity
        if (snr_2==10000 or snr_3==10000):
            FPR_calc_fails += 1

            SNR_ratios_2.append(1)
            SNR_ratios_3.append(1)

            if (snr_ps<=smallest_SNR_ps):
                smallest_SNR_ps = snr_ps

        else:
            SNR_ratios_2.append(snr_2 / snr_ps)
            SNR_ratios_3.append(snr_3 / snr_ps)


#calculate the means and stds and print the results --------------------------------------------------------------------
SNR_ps_used = np.array(SNR_ps_used)
SNR_ratios_1 = np.array(SNR_ratios_1)
SNR_ratios_2 = np.array(SNR_ratios_2)
SNR_ratios_3 = np.array(SNR_ratios_3)
Theta_ratios = np.array(Theta_ratios)
T_ratios = np.array(T_ratios)
R_ratios = np.array(R_ratios)

mean_SNR_ratio_1 = SNR_ratios_1.mean()
std_SNR_ratio_1 = np.std(SNR_ratios_1)

mean_SNR_ratio_2 = SNR_ratios_2.mean()
std_SNR_ratio_2 = np.std(SNR_ratios_2)

mean_SNR_ratio_3 = SNR_ratios_3.mean()
std_SNR_ratio_3 = np.std(SNR_ratios_3)

mean_Theta_ratio = Theta_ratios.mean()
std_Theta_ratio = np.std(Theta_ratios)

mean_T_ratio = T_ratios.mean()
std_T_ratio = np.std(T_ratios)

mean_R_ratio = R_ratios.mean()
std_R_ratio = np.std(R_ratios)

location_accuracy = (n_planets - position_fails)/n_planets
total_accuracy = (n_planets - total_fails)/n_planets

print('Total planets:',n_planets)
print('Successful extractions:',successes)
print('# failed snr detection limit (method as defined):',snr_limit_fails)
print('# failed angular separation:',r_fails)
print('# failed phi:',ang_fails)
print('# of times SNR ratio was set to one:', FPR_calc_fails,'(smallest SNR_ps:',np.round(smallest_SNR_ps,3),')')
print('')

print('Total failed extractions: ',total_fails, ' => ',np.round(total_accuracy*100,2),'% success rate')
print('Excluding the failed extractions in the following')
print('')
print('Failed location estimates: ',position_fails,' => ',np.round(location_accuracy*100,2),'% success rate')

print('SNR_1/SNR_ps = ',np.round(mean_SNR_ratio_1,2),'+/-',np.round(std_SNR_ratio_1,2))
print('SNR_2/SNR_ps = ',np.round(mean_SNR_ratio_2,2),'+/-',np.round(std_SNR_ratio_2,2))
print('SNR_3/SNR_ps = ',np.round(mean_SNR_ratio_3,2),'+/-',np.round(std_SNR_ratio_3,2))
print('Theta_est/Theta_true = ',np.round(mean_Theta_ratio,2),'+/-',np.round(std_Theta_ratio,2))
print('T_est/T_true = ',np.round(mean_T_ratio,2),'+/-',np.round(std_T_ratio,2))
print('R_est/R_true = ',np.round(mean_R_ratio,2),'+/-',np.round(std_R_ratio,2))
print('')


#plots to compare two of the SNRs --------------------------------------------------------------------------------------
#Compare SNR_ps to SNR_1
plt.scatter(x=SNR_ratios_1*SNR_ps_used,y=SNR_ps_used,color='black',marker='x',s=20,label='all planets')
plt.plot(np.linspace(0,200,100),np.linspace(0,200,100), color='blue',label='theoretical line')
plt.title('Ratio of photon statistics SNR to naive extracted SNR')
plt.xlabel('naive extracted SNR ')
plt.ylabel('photon statistics SNR')
plt.xlim((0,15))
plt.ylim((0,15))
plt.legend(loc='best')
plt.grid()
#Uncomment the following line to save
#plt.savefig(path+'/06_plots/SNR_ps_to_SNR_1_changeme.pdf')
plt.show()

#Compare SNR_ps to SNR_2
plt.scatter(x=SNR_ratios_2*SNR_ps_used,y=SNR_ps_used,color='black',marker='x',s=20,label='all planets')
plt.plot(np.linspace(0,200,100),np.linspace(0,200,100), color='blue',label='theoretical line')
plt.title('Ratio of photon statistics SNR to true position SNR')
plt.xlabel('true position SNR ')
plt.ylabel('photon statistics SNR')
plt.xlim((0,15))
plt.ylim((0,15))
plt.legend(loc='best')
plt.grid()
#Uncomment the following line to save
#plt.savefig(path+'/06_plots/SNR_ps_to_SNR_2_changeme.pdf')
plt.show()

#Compare SNR_ps to SNR_3
plt.scatter(x=SNR_ratios_3*SNR_ps_used,y=SNR_ps_used,color='black',marker='x',s=20,label='all planets')
plt.plot(np.linspace(0,200,100),np.linspace(0,200,100), color='blue',label='theoretical line')
plt.title('Ratio of photon statistics SNR to max position SNR')
plt.xlabel('max position SNR ')
plt.ylabel('photon statistics SNR')
plt.xlim((0,15))
plt.ylim((0,15))
plt.legend(loc='best')
plt.grid()
#Uncomment the following line to save
#plt.savefig(path+'/06_plots/SNR_ps_to_SNR_3_changeme.pdf')
plt.show()

#compare SNR_2 to SNR_3
plt.scatter(x=SNR_ratios_2*SNR_ps_used,y=SNR_ratios_3*SNR_ps_used,color='black',marker='x',s=20,label='detected planets')
plt.plot(np.linspace(0,200,100),np.linspace(0,200,100), color='blue',label='theoretical boundary')
plt.title('Ratio of true position FPR to maximum position FPR')
plt.xlabel('max FPR')
plt.ylabel('true FPR')
plt.xlim((0,15))
plt.ylim((0,15))
plt.legend(loc='best')
plt.grid()
#Uncomment the following line to save
#plt.savefig(path+'/06_plots/SNR_2_to_SNR_3_changeme.pdf')
plt.show()


#create histogram plots like figure 12 LIFE II -------------------------------------------------------------------------
#lists with the variables to be shown as well as attributes
variables = [extracted_data['snr_current'], extracted_data['temp_p'], extracted_data['radius_p'], extracted_data['angsep']*1000]
number_bins = [260, 40, 36, 180]
x_labels = ['SNR$_\mathrm{pred}$', '$T_\mathrm{p}$ [K]', '$R_\mathrm{p}$ [$R_\oplus$]', '$\Theta_\mathrm{p}$ [mas]']
y_labels = ['# of planets per Universe', '# of planets per Universe', '# of planets per Universe', '# of planets per Universe']
x_lims = [[0,50], [140,310], [0.5,1.5], [0,100]]
y_lims = [[0,6], [0,2.5], [0,2.2], [0,5.4]]

n_universes = extracted_data['nuniverse'].nunique()


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
#Uncomment the following line to save
#plt.savefig(path+'/06_plots/distributionplot_changeme.pdf')
plt.show()


#create cornerplot like figure 13 LIFE II ------------------------------------------------------------------------------
#stack the plot input data
data_SNR_ps_used = SNR_ps_used
data_SNR_ratios_1 = SNR_ratios_1
data_SNR_ratios_2 = SNR_ratios_2
data_SNR_ratios_3 = SNR_ratios_3
data_T_ratios = T_ratios
data_R_ratios = R_ratios
data_Theta_ratios = Theta_ratios

#Define the parameters you want to compare in the cornerplot here
all_data = np.vstack([data_SNR_ps_used, data_SNR_ratios_3, data_T_ratios, data_R_ratios, data_Theta_ratios])

#define labels and quantiles
labels = ['SNR$_\mathrm{ps}$',
          'SNR$_\mathrm{est}$/SNR$_\mathrm{ps}$',
          '$T_\mathrm{est} / T_\mathrm{true}$',
          '$R_\mathrm{est} / R_\mathrm{true}$',
          '$\Theta_\mathrm{est} / \Theta_\mathrm{true}$']

quantiles = [.159, .841]

#define the main figure
fig = plt.figure(figsize=(12, 12))
fig = corner.corner(np.transpose(all_data), fig=fig, range=[(7, 30), (0.5, 1.2), (0.4, 1.6), (0.0, 2), (.88, 1.12)],
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
        ax.set_xticks([0.65, 0.8, 0.95, 1.1])
        ax.set_xlim([0.5, 1.25])
    if i in [5]:
        ax.set_yticks([0.65, 0.8, 0.95, 1.1])
        ax.set_ylim([0.5, 1.25])

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
#Uncomment the following line to save
#plt.savefig(path+'/06_plots/cornerplot_changeme.pdf')
plt.show()