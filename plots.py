import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from functions import get_detections
from auxiliary import path
import operator as op
from mpl_toolkits.axes_grid1 import AxesGrid
import os

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


#plot from the old lifesim (modules/plotting/plotter.py)
def plot_planet_SED_and_SNR(wl_bins, Fp, Fp_est, sigma, wl_min, wl_max, Fp_BB=None, snr_photon_stat=None, filename=None):

    fig, ax = plt.subplots()

    ax.plot(wl_bins * 1e6, Fp, color="mediumblue", linestyle="-", label="True spectrum")

    if snr_photon_stat is not None:
        ax.fill_between(wl_bins * 1e6, Fp * (1 - 1 / snr_photon_stat), Fp * (1 + 1 / snr_photon_stat),
                        color="mediumblue", alpha=0.1, label=r"Photon noise $\sigma$")

    if Fp_BB is not None:
        ax.plot(wl_bins * 1e6, Fp_BB, color="green", linestyle="-", label="Fit BB Spectrum")

    ax.plot(wl_bins * 1e6, Fp_est, color="red", marker=".", linestyle="")

    with np.errstate(divide='ignore'):
        ax.errorbar(wl_bins * 1e6, Fp_est, yerr=sigma, capsize=3, marker=".",
                    color="red", ecolor="red", linestyle="none",
                    elinewidth=1, alpha=0.4,
                    label="Estimated spectrum")

    ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=12)
    ax.set_ylabel(r"Planet flux $F_\lambda$ [ph $\mathrm{s}^{-1}$m$^{-2}\mu \mathrm{m}^{-1}$]", fontsize=12)
    ax.set_xlim(wl_min*10**6, wl_max*10**6)
    ax.set_ylim(0, 1.6 * np.max(Fp))
    ax.grid()
    ax.legend(fontsize=10)

    if filename is not None:
        plt.savefig("plots/analysis/" + filename + ".pdf", bbox_inches='tight')

    plt.show()


#from modules/plotting/plotanalysis.py
def plot_multi_map(maps, map_type, hfov_mas, colormap="inferno", vmin=None, vmax=None,
                   show_detections=False, filename_post=None, canvas=False):
    if len(np.shape(maps)) < 3:
        maps = [maps]

    n = len(maps)

    if n > 1:
        cbar_location = "bottom"
    else:
        cbar_location = "right"

    fig = plt.figure(figsize=(n * 4.8 + 1, 4.8), dpi=300)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, n),
                    axes_pad=0.2,
                    cbar_mode='each',
                    cbar_location=cbar_location,
                    cbar_pad=0.05,
                    cbar_size="5%"
                    )

    sf_mas = hfov_mas

    i = 0
    for ax in grid:
        im = ax.imshow(maps[i], cmap=colormap, origin="lower",
                       extent=[sf_mas, -sf_mas, -sf_mas, sf_mas], vmin=vmin, vmax=vmax)
        ax.set_xticks([-sf_mas / 2, 0, sf_mas / 2])
        ax.set_yticks([-sf_mas / 2, 0, sf_mas / 2])

        ax.tick_params(pad=1)
        plt.setp(ax.get_yticklabels(), rotation=90, va='center')

        ax.set_xlabel(r"$\Delta$RA [mas]")
        ax.set_ylabel(r"$\Delta$Dec [mas]")

        if show_detections:
            theta_max = np.unravel_index(np.argmax(maps[i], axis=None), maps[i].shape)
            t = tuple(ti / len(maps[i]) for ti in theta_max)

            ax.annotate(str(i + 1), xy=(t[1], t[0] - 0.02), xytext=(t[1], t[0] - 0.2), va="center", ha='center',
                        color="white",
                        arrowprops=dict(color='white', width=1, headwidth=10, shrink=0.1), xycoords=ax.transAxes,
                        fontsize=18)

        sfmt = mtick.ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits((-3, 3))
        sfmt.set_scientific(True)

        cbar = grid.cbar_axes[i].colorbar(im, format=sfmt)
        if n > 1:
            if type(map_type) == list:
                label = map_type[i]
            else:
                label = map_type

            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            cbar.ax.set_xlabel(label, fontsize=12)
        else:
            cbar.ax.set_ylabel(map_type, fontsize=12)

        if maps[i].max() == 1:
            cbar.set_ticks([maps[i].min(), 0, maps[i].max()])

        i += 1

    # plt.tight_layout()
    if filename_post is not None:
        #filename = os.path.join("plots", "analysis_maps", filename_post + ".pdf")
        plt.savefig(path+'05_output_files/Auxiliary/auxfig'+str(filename_post)+'.png', bbox_inches='tight')

    plt.show()

    if (canvas == True):
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return image