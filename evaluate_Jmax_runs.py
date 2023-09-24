import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from auxiliary import path
import scipy as sp
from Extraction_auxiliary import *
from matplotlib.patches import FancyArrow

# define input file and variables
file_Jmax_run = 'Jmax_run_planet2798_1000000.csv'
angsep = 0.08678
file_clusters_run = 'cluster_run_test_new2.csv'
L = 31
sigmas = 4

# validate one planet --------------------------------------------------------------------------------------------------
# calculate thresholds
naive_threshold = get_detection_threshold(L=L,sigma=sigmas)
max_threshold = get_detection_threshold_max(L=L, sigma=sigmas, angsep=angsep)

# define dataframes
Jmax_df = pd.read_csv(path+'/05_output_files/Jmax_runs/'+file_Jmax_run, header=None)
Jmax_array = Jmax_df.values.ravel()

# count the number of false positives for both thresholds
False_Positives_naive = 0
False_Positives_max = 0
total = Jmax_array.size

for i in range(Jmax_array.size):
    if (Jmax_array[i] >= naive_threshold):
        False_Positives_naive += 1
    if (Jmax_array[i] >= max_threshold):
        False_Positives_max += 1

FPF_naive = False_Positives_naive / total
FPF_max = False_Positives_max / total

FPF_norm = 1 - sp.stats.norm.cdf(sigmas)

print('Naive method (threshold = ',naive_threshold,'):, Total False Positives:',False_Positives_naive,'(of',total,
                                                                                                        'total runs)')
print('--> False Positive Fraction',FPF_naive)
print('--> FPF of',sigmas,'sigma:',FPF_norm)

print('Max method (threshold = ',max_threshold,'):, Total False Positives:',False_Positives_max,'(of',total,
                                                                                                        'total runs)')
print('--> False Positive Fraction',FPF_max)
print('--> FPF of',sigmas,'sigma:',FPF_norm)


# create histogram plot
bins = np.linspace(30,90,50)
x_array = np.linspace(30,90,200)
cdf_Jmax_array = cdf_Jmax(L, x_array, angsep)
pdf_Jmax_array = np.gradient(cdf_Jmax_array, x_array)

counts_to_norm, bin_edges = np.histogram(Jmax_array, bins=bins)
area_to_normalize = np.sum(counts_to_norm * (bin_edges[1] - bin_edges[0]))

plt.hist(x=Jmax_array, bins=bins, rwidth=0.9, color='darkblue', label='measured values')
plt.plot(x_array,area_to_normalize*pdf_Jmax_array,label='theoretical curve',color='maroon')
plt.title('Measured maximum J\u2032\u2032 values for pure noise')
plt.xlabel('max of cost function J\u2032\u2032 []')
plt.ylabel('Counts []')
#plt.ylim(0,10)
plt.axvline(x=max_threshold, color='black', linestyle='--', label='det. thres. '+str(sigmas)+r'$\sigma$')
plt.text(0.015,0.95,'n = '+str(total),transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', pad=0.5))
plt.legend(loc='best')
plt.grid()
plt.show()


# cluster evaluation ---------------------------------------------------------------------------------------------------
# define the dataframe and the required variables
clusters_df = pd.read_csv(path+'/05_output_files/Jmax_runs/'+file_clusters_run, header=None)

n_planets = int(clusters_df.iloc[0].size / 2)
n_runs_per_planet = clusters_df[0].size - 1

Jmaxs = clusters_df.iloc[1:,:n_planets]
clusters = clusters_df.iloc[1:,n_planets:]
angseps = clusters_df.iloc[0,:n_planets]

cluster_means = clusters.mean()
cluster_stds = clusters.std()

# create the plot of # of max/min vs angsep
plt.scatter(x=angseps,y=cluster_means,color='darkblue',marker='x',s=40)
plt.title(r'# of local minima & maxima')
plt.xlabel('Angular separation [mas]')
plt.ylabel(r'# of local minima & maxima []')
plt.xlim((0,0.30))
plt.ylim((0,6000))
plt.grid()
plt.show()


# function that returns cdf of J_max
def cdf_Jmax_clusters(L, J, clusters):
    cdf_Jmax = cdf_J(L, J) ** clusters

    return cdf_Jmax

# function that returns pdf of J_max
def test_pdf(x, total_clusters):
    test_cdf_Jmax = cdf_Jmax_clusters(L, x, total_clusters)
    test_pdf_Jmax = np.gradient(test_cdf_Jmax, x)

    return test_pdf_Jmax


test_bins = np.linspace(0,90,60)
x_array = np.linspace(0,90,200)
best_total_clusters = np.zeros((n_planets))
angsep_threshold = 0.05

# loop through all the planets
for i in range(n_planets):
    test_clusters = cluster_means.iloc[i]

    hist, bins = np.histogram(Jmaxs.iloc[:,i], bins=test_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    best_total_cluster, _ = sp.optimize.curve_fit(test_pdf, xdata=bin_centers, ydata=hist, p0=[10000])
    best_total_clusters[i] = best_total_cluster[0]

    # plot two random histograms
    if (i==6 or i == 7):

        cdf_Jmax_array = cdf_Jmax_clusters(L, x_array, best_total_cluster[0])
        pdf_Jmax_array = np.gradient(cdf_Jmax_array, x_array)

        counts_to_norm, bin_edges = np.histogram(Jmaxs.iloc[:,i], bins=bins)
        area_to_normalize = np.sum(counts_to_norm * (bin_edges[1] - bin_edges[0]))

        arrow1 = FancyArrow(0.41, 0.86, 0.1, 0.0, width=0.005, head_length=0.02, head_width=0.02, fc='green',
                                                                        ec='green', transform=plt.gca().transAxes)
        arrow2 = FancyArrow(0.51, 0.86, -0.1, 0.0, width=0.005, head_length=0.02, head_width=0.02, fc='green',
                                                                        ec='green', transform=plt.gca().transAxes)
        arrow3 = FancyArrow(0.43, 0.83, 0.1, 0.0, width=0.005, head_length=0.02, head_width=0.02, fc='green',
                                                                        ec='green', transform=plt.gca().transAxes)
        arrow4 = FancyArrow(0.53, 0.83, -0.1, 0.0, width=0.005, head_length=0.02, head_width=0.02, fc='green',
                                                                        ec='green', transform=plt.gca().transAxes)

        plt.hist(x=Jmaxs.iloc[:,i], bins=bins, rwidth=0.9, color='darkblue', label='measured values')
        plt.plot(x_array, area_to_normalize * pdf_Jmax_array, label='theoretical curve: \nbest fit n = '+
                                                                    str(round(best_total_cluster[0])), color='maroon')
        plt.title('Measured maximum J\u2032\u2032 values for pure noisy system')
        plt.xlabel('max of cost function J\u2032\u2032 []')
        plt.ylabel('Counts []')
        #plt.ylim(0,10)
        #plt.axvline(x=max_threshold, color='black', linestyle='--', label='det. thres. ' + str(sigmas) + r'$\sigma$')
        plt.text(0.025, 0.85, 'm = 1000', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', pad=0.5))
        plt.text(0.025, 0.94, 'ang. sep. = '+str(np.round(angseps[i],3))+' arcsec', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', pad=0.5))
        plt.legend(loc='best')
        if (i == 6):
            plt.gca().add_patch(arrow1)
            plt.gca().add_patch(arrow2)
        if (i == 7):
            plt.gca().add_patch(arrow3)
            plt.gca().add_patch(arrow4)
        plt.grid()
        plt.show()


# create scatter plots
angseps_below = []
angseps_above = []
best_total_clusters_below = []
best_total_clusters_above = []


for i in range(n_planets):
    if (angseps[i] < angsep_threshold):
        best_total_clusters_below.append(best_total_clusters[i])
        angseps_below.append(angseps[i])
    else:
        best_total_clusters_above.append(best_total_clusters[i])
        angseps_above.append(angseps[i])

angseps_below = np.array(angseps_below)
angseps_above = np.array(angseps_above)
best_total_clusters_below = np.array(best_total_clusters_below)
best_total_clusters_above = np.array(best_total_clusters_above)

fit_clusters_below = np.polyfit(angseps_below,best_total_clusters_below,deg=2)
fit_clusters_above = np.polyfit(angseps_above,best_total_clusters_above,deg=1)

x1 = np.linspace(0,angsep_threshold,100)
x2 = np.linspace(angsep_threshold,0.30,100)


plt.scatter(x=angseps_below,y=best_total_clusters_below,color='green',marker='x',s=40,label=r'best fit # of clusters')
plt.plot(x1,x1**2*fit_clusters_below[0]+x1*fit_clusters_below[1]+fit_clusters_below[2],color='black',
                                                                                            label='best fit line')
plt.title(r'Correlation of clusters and angular separation')
plt.xlabel('Angular separation [arcsec]')
plt.ylabel(r'# of clusters []')
plt.xlim((0,angsep_threshold))
plt.ylim((0,4000))
plt.legend(loc='upper left')
plt.grid()
plt.show()


plt.scatter(x=angseps_above,y=best_total_clusters_above,color='green',marker='x',s=40,label=r'best fit # of clusters')
plt.plot(x2,fit_clusters_above[0]*x2+fit_clusters_above[1],color='black',label='best fit line')
plt.title(r'Correlation of clusters and angular separation')
plt.xlabel('Angular separation [arcsec]')
plt.ylabel(r'# of clusters []')
plt.xlim((angsep_threshold, 0.30))
plt.ylim((4000,35000))
plt.legend(loc='best')
plt.grid()
plt.show()


plt.scatter(x=angseps,y=best_total_clusters,color='green',marker='x',s=40,label=r'best fit # of clusters')
plt.plot(x1,x1**2*fit_clusters_below[0]+x1*fit_clusters_below[1]+fit_clusters_below[2],color='black',
                                                                                            label='best fit line')
plt.plot(x2,fit_clusters_above[0]*x2+fit_clusters_above[1],color='black')
plt.title(r'Correlation of clusters and angular separation')
plt.xlabel('Angular separation [arcsec]')
plt.ylabel(r'# of clusters []')
plt.xlim((0,0.30))
plt.ylim((0,35000))
plt.legend(loc='best')
plt.grid()
plt.show()


print('Fit below threshold: y =',np.round(fit_clusters_below[0],3),'* x^2 +',np.round(fit_clusters_below[1],3),'* x +',
                                                                                    np.round(fit_clusters_below[2],3))
print('Fit above threshold: y =',np.round(fit_clusters_above[0],3),'* x +',np.round(fit_clusters_above[1],3))