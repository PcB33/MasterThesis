import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import factorial as fact
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import AxesGrid


def get_detection_threshold(L, sigma):
    '''
    This function calculates the threshold above which one can be certain to 'sigma' sigmas that a detection is not
    a false positive. See LIFE II section 3.2 for a detailed description

    :param sigma: float; # of sigmas outside of which a false positive must lie []

    :return: float; threshold is terms of the cost function J []
    '''

    # create the input linspace
    eta = np.linspace(0, 300, int(10 ** 5))

    cdf = 1 / 2 ** L

    # calculate the cdf
    for l in range(0, L):
        cdf += fact(L) / (2 ** L * fact(l) * fact(L - l)) * sp.special.gammainc((L - l) / 2, eta / 2)

    # find the threshold value eta
    eta_ind_sigma = np.searchsorted(cdf, sp.stats.norm.cdf(sigma))
    eta_threshold_sigma = eta[eta_ind_sigma]

    return eta_threshold_sigma


# Transforms cartesian coordinates to polar coordinates
def cartesian2polar_for_map(outcoords, inputshape):

    y, x = outcoords
    x = x - (inputshape[0] - 0.5)
    y = y - (inputshape[0] - 0.5)

    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(-y, -x)
    phi_index = (phi + np.pi) * inputshape[1] / (2 * np.pi)

    return (r, phi_index)


# Produces a cartesian map from a given input image
def pol_to_cart_map(image, image_size):
    # create new column at end (360 deg) with same values as first column (0 deg) to get complete image
    image_new = np.empty((image.shape[0], image.shape[1] + 1))
    image_new[:, :-1] = image
    image_new[:, -1] = image[:, 0]

    cartesian_map = sp.ndimage.geometric_transform(image_new, cartesian2polar_for_map, order=1,
                                                   output_shape=(image_size, image_size), mode="constant",
                                                   extra_keywords={'inputshape': (image.shape)})

    return cartesian_map

# plot from the old lifesim (modules/plotting/plotter.py)
def plot_planet_SED_and_SNR(wl_bins, Fp, Fp_est, sigma, wl_min, wl_max, Fp_BB=None, snr_photon_stat=None,
                            filename=None):

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
    ax.set_xlim(wl_min * 10 ** 6, wl_max * 10 ** 6)
    ax.set_ylim(0, 1.6 * np.max(Fp))
    ax.grid()
    ax.legend(fontsize=10)

    if filename is not None:
        plt.savefig("plots/analysis/" + filename + ".pdf", bbox_inches='tight')

    plt.show()

# from modules/plotting/plotanalysis.py
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
        # filename = os.path.join("plots", "analysis_maps", filename_post + ".pdf")
        plt.savefig(str(filename_post) + '.png', bbox_inches='tight')

    plt.show()

    if (canvas == True):
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return image