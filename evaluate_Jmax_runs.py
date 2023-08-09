import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from auxiliary import path
import scipy as sp
from Extraction_auxiliary import *

file_Jmax_run = 'Jmax_run_planet5_10000.csv'
L = 31
image_size = 256
sigmas = 4

naive_threshold = get_detection_threshold(L=L,sigma=sigmas)
max_threshold = get_detection_threshold_max(L=L, sigma=sigmas, radial_ang_pix=image_size/2)


Jmax_df = pd.read_csv(path+'/05_output_files/Jmax_runs/'+file_Jmax_run, header=None)
Jmax_array = Jmax_df.values.ravel()

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

print('Naive method (threshold = ',naive_threshold,'):, Total False Positives:',False_Positives_naive,'(of',total,'total runs)')
print('--> False Positive Fraction',FPF_naive)
print('--> FPF of',sigmas,'sigma:',FPF_norm)

print('Max method (threshold = ',max_threshold,'):, Total False Positives:',False_Positives_max,'(of',total,'total runs)')
print('--> False Positive Fraction',FPF_max)
print('--> FPF of',sigmas,'sigma:',FPF_norm)


bins = np.linspace(30,90,50)
x_array = np.linspace(30,90,200)
cdf_Jmax_array = cdf_Jmax(L, x_array, image_size/2)
pdf_Jmax_array = np.gradient(cdf_Jmax_array, x_array)

counts_to_norm, bin_edges = np.histogram(Jmax_array, bins=bins)
area_to_normalize = np.sum(counts_to_norm * (bin_edges[1] - bin_edges[0]))

plt.hist(x=Jmax_array, bins=bins, rwidth=0.9, color='darkblue', label='measured values')
plt.plot(x_array,area_to_normalize*pdf_Jmax_array,label='theoretical curve',color='maroon')
plt.title('Measured maximum J\u2032\u2032 values for pure noise')
plt.xlabel('max of cost function J\u2032\u2032 []')
plt.ylabel('Counts []')
plt.axvline(x=max_threshold, color='black', linestyle='--', label='det. thres. '+str(sigmas)+r'$\sigma$')
plt.text(0.015,0.95,'n = '+str(total),transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', pad=0.5))
plt.legend(loc='best')
plt.grid()
plt.show()