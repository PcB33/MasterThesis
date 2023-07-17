import numpy as np
import matplotlib.pyplot as plt
from functions import get_detections
from auxiliary import path
import operator as op


# reproduce fig 3&12 LIFE I
def gridplot_life_I(input_file, radius, instellation):

    result_means = np.empty((radius.size - 1, instellation.size - 1))
    result_stds = np.empty((radius.size - 1, instellation.size - 1))
    result_SNRs = np.empty((radius.size - 1, instellation.size - 1))

    for i in range(radius.size - 1):
        for j in range(instellation.size - 1):
            parameters = np.array([
                ['detected', op.eq, 1],
                ['radius_p', op.ge, radius[i]],
                ['radius_p', op.le, radius[i + 1]],
                ['flux_p', op.ge, instellation[j]],
                ['flux_p', op.le, instellation[j + 1]],
            ])

            mean, std, SNR = get_detections(input_file, parameters)
            result_means[radius.size - 2 - i, j] = mean
            result_stds[radius.size - 2 - i, j] = std
            result_SNRs[radius.size - 2 - i, j] = SNR

    #plot the array as a grid
    fig, ax = plt.subplots()
    im = ax.imshow(result_means, cmap='gray', vmin=0.5, vmax=0.5)

    #add axis labels and title
    ax.set_xticks(np.arange(len(result_means[0])))
    ax.set_yticks(np.arange(len(result_means)))
    ax.set_xticklabels(np.log10(instellation[0:-1]))
    ax.set_yticklabels(np.flip(radius[0:-1]))
    ax.set_title("Number of detectable planets, scenario 2")

    #add the text in the grid cells
    for i in range(len(result_means)):
        for j in range(len(result_means[0])):
            text = ax.text(j, i, str(int(np.round(result_means[i, j],0)))+'\n'+'+/-'+str(int(np.round(result_stds[i,j],0))), ha='center', va='center', color='w', fontsize='small')

    #show (and save) the plot
    #fig.savefig(path+'06_plots/LIFE_I_Fig3b.png')
    plt.show()

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(result_SNRs, cmap='gray', vmin=0.5, vmax=0.5)

    #add axis labels and title
    ax2.set_xticks(np.arange(len(result_SNRs[0])))
    ax2.set_yticks(np.arange(len(result_SNRs)))
    ax2.set_xticklabels(np.log10(instellation[0:-1]))
    ax2.set_yticklabels(np.flip(radius[0:-1]))
    ax2.set_title("Median SNR of detected planets, scenario 2")

    #add the text in the grid cells
    for i in range(len(result_SNRs)):
        for j in range(len(result_SNRs[0])):
            text2 = ax2.text(j, i, str(int(np.round(result_SNRs[i, j],0))), ha='center', va='center', color='w', fontsize='small')

    #show (and save) the plot
    #fig2.savefig(path+'06_plots/LIFE_I_Fig12b.png')
    plt.show()

'''
# run gridplot_life_I
input_file = 'standard_simulations/standard10_scen2_spectrum.csv'
radius = np.array([0.5, 1.5, 3, 6])
instellation = np.array([10 ** -1.5, 10 ** -1, 10 ** -0.5, 10 ** 0, 10 ** 0.5, 10 ** 1, 10 ** 1.5, 10 ** 2, 10 ** 2.5, 10 ** 3])
gridplot_life_I(input_file,radius,instellation)
'''

#creates plots like figure 12 LIFE II
#currently unused; plot being created directly in Extraction_DataAnalysis.py
def histplot_planet_parameters(variable, title, x_label, y_label, x_lim, y_lim, number_bins, n_universes):

    #Define weights that normalize the histogram to the numbers per universe
    weights = np.ones_like(variable) / n_universes

    #create the plot and define its size
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(bottom=0.2)
    n, bins, _ = plt.hist(variable, bins=number_bins, weights=weights, color='darkblue', rwidth=0.9)

    #add descriptions
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.grid()

    plt.show()