import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auxiliary import path
from Extraction_auxiliary import *
import corner as corner
from matplotlib.patches import Arrow, Circle, Ellipse

#define variables ------------------------------------------------------------------------------------------------------
file = 'run_dips/EEC_multiextraction_dips_R100_mix_40.csv'
angsep_accuracy_def = 0.15
phi_accuracy_def = 10
true_phi = 0
#Define according to what SNR method to filter for detected should be made (== 1, 2, 3); this SNR is also used in the corner plot
defining_criteria = 3


#load the input file, calculate some required variables and define empty lists of quantities to be stored --------------
extracted_data = pd.read_csv(path+'/05_output_files/multiextraction/'+file)
n_planets = len(extracted_data.index)

L = len(eval(extracted_data['extracted_spectra'][0])[0])
radial_ang_px = 128
eta_threshold_5 = get_detection_threshold(L, 5)
eta_threshold_max_5 = get_detection_threshold_max(L, 5, radial_ang_px)

molecules_list = [r'$CO_2$', r'$O_3$', r'$H_2O$']
#molecules_list = [r'$CO_2$']
if (eval(extracted_data['extracted_SNR_ps_new'][0])[0] == 0):
    n_molecules = 0
else:
    n_molecules = len(molecules_list)
strong_alpha = 0.09
decisive_alpha = 0.01
strong_jeffrey = 1.0
decisive_jeffrey = 2.0


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
T_true = []
T_ratios = []
R_true = []
R_ratios = []
induced_dips = []
SNR_dips_used = []
t_scores = []
bayes_factors = []


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
    induced_dip = eval(extracted_data['extracted_induced_dips'][i])[0]
    SNR_dip = eval(extracted_data['extracted_SNR_ps_new'][i])[0]
    t_score = eval(extracted_data['extracted_t_scores'][i])[0]
    bayes_factor = eval(extracted_data['extracted_bayes_factors'][i], {'inf': float('inf')}, {'nan': float('nan')})[0]
    if (any (num == np.inf for num in bayes_factor) or any (num == np.nan for num in bayes_factor)):
        for j in range(len(bayes_factor)):
            if (bayes_factor[j] == np.inf):
                bayes_factor[j] = 100
            elif (bayes_factor[j] == np.nan):
                bayes_factor[j] = -100

    if (defining_criteria == 1 and snr_1 < 5):
        total_fails += 1
        snr_limit_fails += 1

    elif (defining_criteria == 2 and snr_2 < 5):
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
        T_true.append(true_T)
        T_ratios.append(T_extr / true_T)
        R_true.append(true_R)
        R_ratios.append(R_extr / true_R)
        induced_dips.append(induced_dip)
        SNR_dips_used.append(SNR_dip)
        t_scores.append(t_score)
        bayes_factors.append(bayes_factor)

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
T_true = np.array(T_true)
T_ratios = np.array(T_ratios)
R_true = np.array(R_true)
R_ratios = np.array(R_ratios)
induced_dips = np.array(induced_dips)
SNR_dips_used = np.array(SNR_dips_used)
t_scores = np.array(t_scores)
bayes_factors = np.array(bayes_factors)

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
x = np.linspace(0,200,100)

#Compare SNR_ps to SNR_1
plt.scatter(x=SNR_ratios_1*SNR_ps_used,y=SNR_ps_used,color='black',marker='x',s=20,label='all planets')
plt.plot(x,x, color='blue',label='theoretical line')
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
best_fit = np.polyfit(x=SNR_ratios_2*SNR_ps_used,y=SNR_ps_used,deg=1)
plt.scatter(x=SNR_ratios_2*SNR_ps_used,y=SNR_ps_used,color='black',marker='x',s=20,label='all planets')
plt.plot(x,x, color='blue',label='theoretical line')
#plt.plot(x,best_fit[1]*x+best_fit[0],label='linear fit '+'(bias:'+str(np.round(best_fit[0],3))+')',color='orange',linestyle='--')
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
plt.plot(x,x, color='blue',label='theoretical line')
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
plt.plot(x,x, color='blue',label='theoretical boundary')
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
number_bins = [40, 40, 30, 180]
x_labels = ['SNR$_\mathrm{ps}$', '$T_\mathrm{p}$ [K]', '$R_\mathrm{p}$ [$R_\oplus$]', '$\Theta_\mathrm{p}$ [mas]']
y_labels = ['# of planets per Universe', '# of planets per Universe', '# of planets per Universe', '# of planets per Universe']
x_lims = [[0,175], [140,300], [0.7,1.5], [0,90]]
y_lims = [[0,2], [0,2], [0,2], [0,2]]

n_universes = extracted_data['nuniverse'].nunique()


#create figure and fill plots by running through for loop for all list entries
fig, axes = plt.subplots(len(variables), 1, figsize=(6.4,9.6),)
fig.suptitle('Rocky Cold Planets Distributions')

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

if (defining_criteria==1):
    data_SNR_ratios_used = data_SNR_ratios_1
elif (defining_criteria==2):
    data_SNR_ratios_used = data_SNR_ratios_2
elif (defining_criteria==3):
    data_SNR_ratios_used = data_SNR_ratios_3

#Define the parameters you want to compare in the cornerplot here
all_data = np.vstack([data_SNR_ps_used, data_SNR_ratios_used, data_T_ratios, data_R_ratios, data_Theta_ratios])

#define labels and quantiles
labels = ['SNR$_\mathrm{ps}$',
          'FPF$_\mathrm{est}$/SNR$_\mathrm{ps}$',
          '$T_\mathrm{est} / T_\mathrm{true}$',
          '$R_\mathrm{est} / R_\mathrm{true}$',
          '$\Theta_\mathrm{est} / \Theta_\mathrm{true}$']

quantiles = [.159, .841]

#define the main figure
fig = plt.figure(figsize=(12, 12))
fig = corner.corner(np.transpose(all_data), fig=fig, range=[(7, 30), (0.5, 1.2), (0.4, 1.6), (0.0, 2), (.88, 1.12)],
                    labels=labels, show_titles=True, hist_bin_factor=1.5, max_n_ticks=5,
                    plot_density=False, plot_contours=False, no_fill_contours=True, color='darkblue', quantiles=quantiles)

fig.text(0.95, 0.98, 'Exo-Earth Candidates', ha='right', va='top', fontsize=20, fontweight='bold')
fig.text(0.95, 0.93, 'Total planets: '+str(n_planets), ha='right', va='top', fontsize=16)
fig.text(0.95, 0.90, 'Failed detections: '+str(snr_limit_fails), ha='right', va='top', fontsize=16)
fig.text(0.95, 0.87, 'False location extractions: '+str(position_fails), ha='right', va='top', fontsize=16)
fig.text(0.95, 0.84, 'Overall Success Rate: '+str(np.round(total_accuracy*100,2))+'%', ha='right', va='top', fontsize=16)

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
        ax.set_yticks([0.5, 0.75, 1., 1.25, 1.5])
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

# add arrows & circles
#arrow_1 = Arrow(0.23, 0.08, -0.15, 0.03, width=0.1, transform=fig.axes[5].transAxes, color='black')
#arrow_2 = Arrow(0.85, 0.5, -0.15, 0.00, width=0.1, transform=fig.axes[16].transAxes, color='darkgreen')
#arrow_3 = Arrow(0.85, 0.5, -0.15, 0.00, width=0.1, transform=fig.axes[11].transAxes, color='darkgreen')

#fig.axes[5].add_patch(arrow_1)
#fig.axes[16].add_patch(arrow_2)
#fig.axes[11].add_patch(arrow_3)

#circle = Ellipse((0.05,0.15),width=0.3, height=0.2, angle=-50, transform=fig.axes[5].transAxes, fill=False, color='violet', alpha=0.8, linewidth=4)
#fig.axes[5].add_patch(circle)

#Uncomment the following line to save
#plt.savefig(path+'/06_plots/cornerplot_changeme.pdf')
plt.show()


# create plots with dips -----------------------------------------------------------------------------------------------
#Compare SNR_ps with and without the dip
plt.scatter(x=SNR_dips_used,y=SNR_ps_used,color='black',marker='x',s=20,label='all planets')
plt.plot(x,x, color='blue',label='theoretical line')
#plt.plot(x,best_fit[1]*x+best_fit[0],label='linear fit '+'(bias:'+str(np.round(best_fit[0],3))+')',color='orange',linestyle='--')
plt.title(r'Ratio of $\mathrm{SNR_{ps}}$ with and without induced dip')
plt.xlabel(r'$\mathrm{SNR_{ps}}$ with dip')
plt.ylabel(r'$\mathrm{SNR_{ps}}$ without dip')
plt.xlim((0,15))
plt.ylim((0,15))
plt.legend(loc='best')
plt.grid()
#Uncomment the following line to save
#plt.savefig(path+'/06_plots/SNR_ps_to_SNR_2_changeme.pdf')
plt.show()


n_induced_dips = np.zeros((n_molecules))

TP_t_statistics_strong = np.zeros((n_molecules,successes))
FP_t_statistics_strong = np.zeros((n_molecules,successes))
FN_t_statistics_strong = np.zeros((n_molecules,successes))
TN_t_statistics_strong = np.zeros((n_molecules,successes))

TP_t_statistics_decisive = np.zeros((n_molecules,successes))
FP_t_statistics_decisive = np.zeros((n_molecules,successes))
FN_t_statistics_decisive = np.zeros((n_molecules,successes))
TN_t_statistics_decisive = np.zeros((n_molecules,successes))

TP_bayesian_models_strong = np.zeros((n_molecules,successes))
FP_bayesian_models_strong = np.zeros((n_molecules,successes))
FN_bayesian_models_strong = np.zeros((n_molecules,successes))
TN_bayesian_models_strong = np.zeros((n_molecules,successes))

TP_bayesian_models_decisive = np.zeros((n_molecules,successes))
FP_bayesian_models_decisive = np.zeros((n_molecules,successes))
FN_bayesian_models_decisive = np.zeros((n_molecules,successes))
TN_bayesian_models_decisive = np.zeros((n_molecules,successes))


for i in range(n_molecules):
    n_induced_dips[i] = np.count_nonzero(induced_dips.T[i])

    if (n_induced_dips[i] == 0):
        print('No', molecules_list[i], 'dips induced')
        print('')

    else:
        TP_t_statistics_strong[i], FP_t_statistics_strong[i], FN_t_statistics_strong[i], \
            TN_t_statistics_strong[i], _, _ = get_rates(t_scores.T[i], induced_dips.T[i], strong_alpha, True)
        TP_bayesian_models_strong[i], FP_bayesian_models_strong[i], FN_bayesian_models_strong[i], \
            TN_bayesian_models_strong[i], _, _ = get_rates(np.log10(bayes_factors.T[i]), induced_dips.T[i], strong_jeffrey, False)
        TP_t_statistics_decisive[i], FP_t_statistics_decisive[i], FN_t_statistics_decisive[i], \
            TN_t_statistics_decisive[i], _, _ = get_rates(t_scores.T[i], induced_dips.T[i], decisive_alpha, True)
        TP_bayesian_models_decisive[i], FP_bayesian_models_decisive[i], FN_bayesian_models_decisive[i], \
            TN_bayesian_models_decisive[i], _, _ = get_rates(np.log10(bayes_factors.T[i]), induced_dips.T[i], decisive_jeffrey, False)


        print('Induced', molecules_list[i], 'dips:', int(n_induced_dips[i]), '(', np.round(n_induced_dips[i]/successes*100,2), '% of all planets)')
        print('')

        print('t-statistics (alpha = 0.01):')
        print('True positive detections:', int(np.sum(TP_t_statistics_decisive[i])), '(', np.round(np.sum(TP_t_statistics_decisive[i])/successes*100,2), '% of all planets)')
        print('False positive detections:', int(np.sum(FP_t_statistics_decisive[i])), '(', np.round(np.sum(FP_t_statistics_decisive[i]) / successes * 100, 2), '% of all planets)')
        print('False negative detections:', int(np.sum(FN_t_statistics_decisive[i])), '(', np.round(np.sum(FN_t_statistics_decisive[i]) / successes * 100, 2), '% of all planets)')
        print('True negative detections:', int(np.sum(TN_t_statistics_decisive[i])), '(', np.round(np.sum(TN_t_statistics_decisive[i]) / successes * 100, 2), '% of all planets)')

        print('')

        print('Bayesian model selection (Jeffrey-factor = 2):')
        print('True positive selections:', int(np.sum(TP_bayesian_models_decisive[i])), '(', np.round(np.sum(TP_bayesian_models_decisive[i]) / successes * 100, 2), '% of all planets)')
        print('False positive selections:', int(np.sum(FP_bayesian_models_decisive[i])), '(', np.round(np.sum(FP_bayesian_models_decisive[i]) / successes * 100, 2), '% of all planets)')
        print('False negative selections:', int(np.sum(FN_bayesian_models_decisive[i])), '(', np.round(np.sum(FN_bayesian_models_decisive[i]) / successes * 100, 2), '% of all planets)')
        print('True negative selections:', int(np.sum(TN_bayesian_models_decisive[i])), '(', np.round(np.sum(TN_bayesian_models_decisive[i]) / successes * 100, 2), '% of all planets)')

        print('---------')


        SNR_bins_1 = np.linspace(0,50,15, endpoint=False)
        SNR_bins_2 = np.linspace(50,100,5, endpoint=False)
        SNR_bins_3 = np.linspace(100,175,5)
        SNR_bins = np.concatenate((SNR_bins_1, SNR_bins_2, SNR_bins_3))

        SNR_t_statistics_ratios_strong = np.zeros_like(SNR_bins[:-1])
        SNR_t_statistics_ratios_decisive = np.zeros_like(SNR_bins[:-1])
        SNR_bayesian_models_ratios_strong = np.zeros_like(SNR_bins[:-1])
        SNR_bayesian_models_ratios_decisive = np.zeros_like(SNR_bins[:-1])

        T_bins = np.linspace(160,280,20)

        T_t_statistics_ratios_strong = np.zeros_like(T_bins[:-1])
        T_t_statistics_ratios_decisive = np.zeros_like(T_bins[:-1])
        T_bayesian_models_ratios_strong = np.zeros_like(T_bins[:-1])
        T_bayesian_models_ratios_decisive = np.zeros_like(T_bins[:-1])

        R_bins = np.linspace(0.8, 1.4, 20)

        R_t_statistics_ratios_strong = np.zeros_like(R_bins[:-1])
        R_t_statistics_ratios_decisive = np.zeros_like(R_bins[:-1])
        R_bayesian_models_ratios_strong = np.zeros_like(R_bins[:-1])
        R_bayesian_models_ratios_decisive = np.zeros_like(R_bins[:-1])


        for j in range(SNR_bins.size-1):
            SNR_t_statistics_in_bin_strong = []
            SNR_t_statistics_in_bin_decisive = []
            SNR_bayesian_models_in_bin_strong = []
            SNR_bayesian_models_in_bin_decisive = []

            for k in range(SNR_dips_used.size):
                if (SNR_dips_used[k] >= SNR_bins[j] and SNR_dips_used[k] < SNR_bins[j+1]):
                    SNR_t_statistics_in_bin_strong.append(TP_t_statistics_strong[i][k])
                    SNR_t_statistics_in_bin_decisive.append(TP_t_statistics_decisive[i][k])
                    SNR_bayesian_models_in_bin_strong.append(TP_bayesian_models_strong[i][k])
                    SNR_bayesian_models_in_bin_decisive.append(TP_bayesian_models_decisive[i][k])

            SNR_t_statistics_ratios_strong[j] = get_ratio_safe(SNR_t_statistics_in_bin_strong)
            SNR_t_statistics_ratios_decisive[j] = get_ratio_safe(SNR_t_statistics_in_bin_decisive)
            SNR_bayesian_models_ratios_strong[j] = get_ratio_safe(SNR_bayesian_models_in_bin_strong)
            SNR_bayesian_models_ratios_decisive[j] = get_ratio_safe(SNR_bayesian_models_in_bin_decisive)


        for j in range(T_bins.size - 1):
            T_t_statistics_in_bin_strong = []
            T_t_statistics_in_bin_decisive = []
            T_bayesian_models_in_bin_strong = []
            T_bayesian_models_in_bin_decisive = []

            for k in range(T_true.size):
                if (T_true[k] >= T_bins[j] and T_true[k] < T_bins[j + 1]):
                    T_t_statistics_in_bin_strong.append(TP_t_statistics_strong[i][k])
                    T_t_statistics_in_bin_decisive.append(TP_t_statistics_decisive[i][k])
                    T_bayesian_models_in_bin_strong.append(TP_bayesian_models_strong[i][k])
                    T_bayesian_models_in_bin_decisive.append(TP_bayesian_models_decisive[i][k])

            T_t_statistics_ratios_strong[j] = get_ratio_safe(T_t_statistics_in_bin_strong)
            T_t_statistics_ratios_decisive[j] = get_ratio_safe(T_t_statistics_in_bin_decisive)
            T_bayesian_models_ratios_strong[j] = get_ratio_safe(T_bayesian_models_in_bin_strong)
            T_bayesian_models_ratios_decisive[j] = get_ratio_safe(T_bayesian_models_in_bin_decisive)


        for j in range(R_bins.size - 1):
            R_t_statistics_in_bin_strong = []
            R_t_statistics_in_bin_decisive = []
            R_bayesian_models_in_bin_strong = []
            R_bayesian_models_in_bin_decisive = []

            for k in range(R_true.size):
                if (R_true[k] >= R_bins[j] and R_true[k] < R_bins[j + 1]):
                    R_t_statistics_in_bin_strong.append(TP_t_statistics_strong[i][k])
                    R_t_statistics_in_bin_decisive.append(TP_t_statistics_decisive[i][k])
                    R_bayesian_models_in_bin_strong.append(TP_bayesian_models_strong[i][k])
                    R_bayesian_models_in_bin_decisive.append(TP_bayesian_models_decisive[i][k])

            R_t_statistics_ratios_strong[j] = get_ratio_safe(R_t_statistics_in_bin_strong)
            R_t_statistics_ratios_decisive[j] = get_ratio_safe(R_t_statistics_in_bin_decisive)
            R_bayesian_models_ratios_strong[j] = get_ratio_safe(R_bayesian_models_in_bin_strong)
            R_bayesian_models_ratios_decisive[j] = get_ratio_safe(R_bayesian_models_in_bin_decisive)


        SNR_bins = SNR_bins[:-1]
        T_bins = T_bins[:-1]
        R_bins = R_bins[:-1]

        plt.scatter(x=SNR_bins,y=SNR_t_statistics_ratios_strong,color='grey',marker='x',s=20,label=r'$\alpha$=0.09')
        plt.plot(SNR_bins, SNR_t_statistics_ratios_strong, color='grey',alpha=0.5)
        plt.scatter(x=SNR_bins, y=SNR_t_statistics_ratios_decisive, color='black', marker='x', s=20, label=r'$\alpha$=0.01')
        plt.plot(SNR_bins, SNR_t_statistics_ratios_decisive, color='black', alpha=0.5)
        plt.title(molecules_list[i]+': detections using t-statistics')
        plt.xlabel(r'$\mathrm{SNR_{ps}}$ []')
        plt.ylabel(r'Fraction of planets []')
        plt.legend(loc='upper left')
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.show()

        plt.scatter(x=SNR_bins, y=SNR_bayesian_models_ratios_strong, color='grey', marker='x', s=20, label=r'$\mathrm{log_{10}}(K)$=1')
        plt.plot(SNR_bins, SNR_bayesian_models_ratios_strong, color='grey', alpha=0.5)
        plt.scatter(x=SNR_bins, y=SNR_bayesian_models_ratios_decisive, color='black', marker='x', s=20, label=r'$\mathrm{log_{10}}(K)$=2')
        plt.plot(SNR_bins, SNR_bayesian_models_ratios_decisive, color='black', alpha=0.5)
        plt.title(molecules_list[i] + ': detections using Bayesian model selection')
        plt.xlabel(r'$\mathrm{SNR_{ps}}$ []')
        plt.ylabel(r'Fraction of planets []')
        plt.legend(loc='upper left')
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.show()

        plt.scatter(x=T_bins, y=T_t_statistics_ratios_strong, color='grey', marker='x', s=20, label=r'$\alpha$=0.09')
        plt.plot(T_bins, T_t_statistics_ratios_strong, color='grey', alpha=0.5)
        plt.scatter(x=T_bins, y=T_t_statistics_ratios_decisive, color='black', marker='x', s=20, label=r'$\alpha$=0.01')
        plt.plot(T_bins, T_t_statistics_ratios_decisive, color='black', alpha=0.5)
        plt.title(molecules_list[i] + ': detections using t-statistics')
        plt.xlabel(r'T [K]')
        plt.ylabel(r'Fraction of planets []')
        plt.legend(loc='upper left')
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.show()

        plt.scatter(x=T_bins, y=T_bayesian_models_ratios_strong, color='grey', marker='x', s=20,
                    label=r'$\mathrm{log_{10}}(K)$=1')
        plt.plot(T_bins, T_bayesian_models_ratios_strong, color='grey', alpha=0.5)
        plt.scatter(x=T_bins, y=T_bayesian_models_ratios_decisive, color='black', marker='x', s=20,
                    label=r'$\mathrm{log_{10}}(K)$=2')
        plt.plot(T_bins, T_bayesian_models_ratios_decisive, color='black', alpha=0.5)
        plt.title(molecules_list[i] + ': detections using Bayesian model selection')
        plt.xlabel(r'T [K]')
        plt.ylabel(r'Fraction of planets []')
        plt.legend(loc='upper left')
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.show()

        plt.scatter(x=R_bins, y=R_t_statistics_ratios_strong, color='grey', marker='x', s=20, label=r'$\alpha$=0.09')
        plt.plot(R_bins, R_t_statistics_ratios_strong, color='grey', alpha=0.5)
        plt.scatter(x=R_bins, y=R_t_statistics_ratios_decisive, color='black', marker='x', s=20, label=r'$\alpha$=0.01')
        plt.plot(R_bins, R_t_statistics_ratios_decisive, color='black', alpha=0.5)
        plt.title(molecules_list[i] + ': detections using t-statistics')
        plt.xlabel(r'R [K]')
        plt.ylabel(r'Fraction of planets []')
        plt.legend(loc='upper left')
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.show()

        plt.scatter(x=R_bins, y=R_bayesian_models_ratios_strong, color='grey', marker='x', s=20,
                    label=r'$\mathrm{log_{10}}(K)$=1')
        plt.plot(R_bins, R_bayesian_models_ratios_strong, color='grey', alpha=0.5)
        plt.scatter(x=R_bins, y=R_bayesian_models_ratios_decisive, color='black', marker='x', s=20,
                    label=r'$\mathrm{log_{10}}(K)$=2')
        plt.plot(R_bins, R_bayesian_models_ratios_decisive, color='black', alpha=0.5)
        plt.title(molecules_list[i] + ': detections using Bayesian model selection')
        plt.xlabel(r'R [K]')
        plt.ylabel(r'Fraction of planets []')
        plt.legend(loc='upper left')
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.show()

        alpha_array = np.linspace(0, 1, 100)
        TPR_array_t_statistics = np.empty_like(alpha_array)
        FPR_array_t_statistics = np.empty_like(alpha_array)

        jeffrey_array = np.linspace(10, -7, 100) # theoretically from inf to -inf but this about covers it
        TPR_array_bayesian_models = np.empty_like(jeffrey_array)
        FPR_array_bayesian_models = np.empty_like(jeffrey_array)

        for l in range(alpha_array.size):
            _, _, _, _, TPR_array_t_statistics[l], FPR_array_t_statistics[l] = get_rates(t_scores.T[i],
                                                                                             induced_dips.T[i],
                                                                                             alpha_array[l], True)
        for l in range(jeffrey_array.size):
            _, _, _, _, TPR_array_bayesian_models[l], FPR_array_bayesian_models[l] = get_rates(np.log10(bayes_factors.T[i]), induced_dips.T[i],
                                                                           jeffrey_array[l], False)


        AUC_t_statistics = 0
        AUC_bayesian_models = 0

        # calculate the AUC using the trapez rule
        for j in range(1, FPR_array_t_statistics.size):
            AUC_t_statistics += (TPR_array_t_statistics[j] + TPR_array_t_statistics[j - 1]) * (FPR_array_t_statistics[j] - FPR_array_t_statistics[j-1]) / 2


        for j in range(1, FPR_array_bayesian_models.size):
            AUC_bayesian_models += (TPR_array_bayesian_models[j] + TPR_array_bayesian_models[j - 1]) * (FPR_array_bayesian_models[j] - FPR_array_bayesian_models[j - 1]) / 2


        print(molecules_list[i], 'AUC-value for t-statistics:', np.round(AUC_t_statistics,3))
        print(molecules_list[i], 'AUC-value for Bayesian model selection:', np.round(AUC_bayesian_models, 3))
        print('---------')
        print('')


        random_x = np.linspace(0,1,100)

        plt.plot(FPR_array_t_statistics, TPR_array_t_statistics, label='t-test classification', color='green')
        plt.plot(FPR_array_bayesian_models, TPR_array_bayesian_models, label='Bayesian model classification',
                 color='royalblue')
        plt.plot(random_x, random_x, label='random classification',color='dimgrey', linestyle='--')
        plt.title(molecules_list[i]+': ROC curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.grid()
        plt.show()