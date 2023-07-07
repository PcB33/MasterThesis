import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import factorial as fact
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import AxesGrid
from lifesim.util.constants import c, h, k, radius_earth, m_per_pc
import mpmath as mp


def cdf_J(L, J):
    '''
    This function calculates the cumulative density function for the cost function J'' as described in LIFE II
    equation (31). It uses floats and is therefore only useable for values not too close to 1

    :param L: int; number of wavelength bins []
    :param J: float; value of the cost function []

    :return cdf: float; Cumulative probability density of sum of L bins of the chi-squared distribution []
    '''

    fact = sp.special.factorial

    # calculate the cdf as in equation (31) LIFE II
    cdf = 1 / 2 ** L
    for l in range(0, L):
        cdf += fact(L) / (2 ** L * fact(l) * fact(L - l)) * sp.special.gammainc((L - l) / 2, J / 2)

    return cdf


def cdf_Jmax(L, J, radial_ang_px):
    '''
    This function calculates the cumulative distribution function for the cost function J'' when factoring in that the
    maximum pixel is selected as described in Thesis. It calls upon the cdf_J and raises it to the power of the number
    of total pixels in the image. It uses floats and is therefore only useable for values not too close to 1

    :param L: int; number of wavelength bins []
    :param J: float; value of the cost function []
    :param radial_ang_px: int; number of pixels in the radial direction of the image (=image_size/2) []

    :return cdf_Jmax: float; Cumulative probability density of sum of L bins of the chi-squared distribution when
                                selecting the maximum pixel []
    '''

    cdf_Jmax = cdf_J(L, J)**(radial_ang_px**2)

    return cdf_Jmax


def pdf_J(L, J):
    '''
    This function calculates the probability density function for the cost function J'' as described in LIFE II
    equation (30). Note that the dirac delta function is not implemented

    :param L: int; number of wavelength bins []
    :param J: float; value of the cost function []

    :return pdf: float; Normalized probability density of sum of L bins of the chi-squared distribution []
    '''

    fact = sp.special.factorial

    # calculate the pdf as in equation (30) LIFE II
    pdf = 0
    for l in range(0, L):
        pdf += fact(L) / (2 ** L * fact(l) * fact(L - l)) * sp.stats.chi2.pdf(J, L - l)

    return pdf


def cdf_J_precision(L, J, precision):
    '''
    This function calculates the cumulative density function for the cost function J'' as described in LIFE II
    equation (31). It uses the mpmath package to achieve better performance for values close to 1

    :param L: int; number of wavelength bins []
    :param J: mpmath_object; value of the cost function []
    :param precision: int; degree of precision used by mpmath []

    :return cdf: mpmath_object; Cumulative probability density of sum of L bins of the chi-squared distribution []
    '''

    # set the precision
    mp.mp.dps = precision

    # calculate the cdf as in equation (31) LIFE II
    cdf = mp.power(mp.mp.mpf('0.5'), L)
    for l in range(0, L):
        cdf += mp.fac(L) / (mp.mp.power(mp.mp.mpf('2'), L) * mp.fac(l) * mp.fac(L - l)) *\
                    mp.gammainc((L - l) / 2, 0, J / 2, regularized=True)

    return cdf


def cdf_Jmax_precision(L, J, precision, radial_ang_px):
    '''
    This function calculates the cumulative distribution function for the cost function J'' when factoring in that the
    maximum pixel is selected as described in Thesis. It calls upon the cdf_J_precision and raises it to the power of
    the number of total pixels in the image. It uses the mpmath package to achieve better performance for values close
    to 1

    :param L: int; number of wavelength bins []
    :param J: mpmath_object; value of the cost function []
    :param precision: int; degree of precision used by mpmath []
    :param radial_ang_px: int; number of pixels in the radial direction of the image (=image_size/2) []

    :return cdf_Jmax: mpmath_object; Cumulative probability density of sum of L bins of the chi-squared distribution
                                        when selecting the maximum pixel []
    '''

    cdf_Jmax = cdf_J_precision(L, J, precision)**(radial_ang_px**2)

    return cdf_Jmax


def alt_sigma_calc(FPR, filepath):
    '''
    This function is used to calculate the number of standard deviations a given input value of a cdf function
    corresponds to. It does this by calling upon a lookup table, which is much faster than performing the direct
    conversion for large sigmas.

    :param FPR: mpmath_object; value of the cdf function that is to be converted []
    :param filepath: str; path to the lookup table

    :return: float; number of sigmas that are inside the confidence interval of the given cdf output value
    '''

    # load the lookup table; the two approaches are for the server/local data structure
    try:
        lookup_table = pd.read_csv(filepath+'prob2sigma_conversiontable_minimal.csv',dtype={'prob': str})
    except FileNotFoundError:
        lookup_table = pd.read_csv(filepath+'Auxiliary/'+'prob2sigma_conversiontable_minimal.csv', dtype={'prob': str})

    # create a series with all the probability values as mpmath objects
    mp_series = lookup_table['prob'].map(lambda x: mp.mpf(str(x)))

    # find the probability value closest to the input value
    abs_diff = np.abs(mp_series - FPR)
    min_index = np.argmin(abs_diff)

    # take the sigma corresponding to the closest value
    sigma = lookup_table['sigma'][min_index]

    return sigma


def get_detection_threshold(L, sigma):
    '''
    This function calculates the threshold above which one can be certain to 'sigma' sigmas that a detection is not
    a false positive. See LIFE II section 3.2 for a detailed description

    :param L: int; Number of wavelength bins as given by the wavelength range and the resolution parameter R []
    :param sigma: float; # of sigmas outside of which a false positive must lie []

    :return eta_threshold_sigma: float; threshold is terms of the cost function J []
    '''

    # create the input linspace
    eta = np.linspace(0, 300, int(10 ** 5))

    # calculate the cdf values for each element in the eta-linspace
    cdf = cdf_J(L, eta)

    # find the threshold value eta
    eta_ind_sigma = np.searchsorted(cdf, sp.stats.norm.cdf(sigma))
    eta_threshold_sigma = eta[eta_ind_sigma]

    return eta_threshold_sigma


def get_detection_threshold_max(L, sigma, radial_ang_pix):
    '''
    This function calculates the threshold above which one can be certain to 'sigma' sigmas that a detection is not
    a false positive. See LIFE II section 3.2 for a detailed description

    :param L: int; Number of wavelength bins as given by the wavelength range and the resolution parameter R []
    :param sigma: float; # of sigmas outside of which a false positive must lie []

    :return eta_threshold_sigma: float; threshold is terms of the cost function J []
    '''

    # create the input linspace
    eta = np.linspace(0, 200, int(10 ** 3))

    # calculate the cdf values for each element in the eta-linspace
    cdf = np.empty_like(eta)
    for i in range(eta.size):
        cdf[i] = cdf_Jmax(L, eta[i], radial_ang_pix)

    # find the threshold value eta
    eta_ind_sigma = np.searchsorted(cdf, sp.stats.norm.cdf(sigma))
    eta_threshold_sigma = eta[eta_ind_sigma]

    return eta_threshold_sigma


def BB_for_fit(wl_and_distS, Tp, Rp):
    '''
    This function calculates the flux received at Earth from an object with radius Rp radiating at temperature T,
    at distance dist_s and at wavelengths wl

    :param wl: np.ndarray of size L; wavelength bins in [m]
    :param Tp: float; planet temperature in [K]
    :param Rp: float; planet radius in [R_earth]

    :return fgamma: np.ndarray of size L; contains the total blackbody fluxes in each of the L wavelength bins
                    in [photons]
    '''

    # unpack the input array (required in this format for the scipy.optimizer function)
    wl = wl_and_distS[0]
    dist_s = wl_and_distS[1]

    fact1 = 2 * c / (wl ** 4)
    fact2 = (h * c) / (k * Tp * wl)

    # calculate the standard planck function
    fgamma = np.array(fact1 / (np.exp(fact2) - 1.0)) * 10 ** -6 * np.pi * (
            (Rp * radius_earth) / (dist_s * m_per_pc)) ** 2

    return fgamma


def get_Dsqr_mat(L):
    '''
    This function calcualtes the D^2 matrix as described in LIFE II Appendix B used for the calculation of the
    estimated planet flux. See this paper for the explanations of the exact calculations

    :param L : int; number of wavelegth bins []

    :return Dsqr_mat: matrix of dimensions (L,radial_ang_px,n_steps); required for calculation of the estimated
            planet flux []
    '''

    dif_diag = -2 * np.diag(np.ones(L))
    dif_pre = 1 * np.diag(np.ones(L - 1), 1)
    dif_post = 1 * np.diag(np.ones(L - 1), -1)

    D_mat = dif_diag + dif_pre + dif_post
    D_mat[0, :2] = np.array([-1, 1])
    D_mat[-1, -2:] = np.array([1, -1])

    Dsqr_mat = np.matmul(D_mat, D_mat)

    return Dsqr_mat


def cartesian2polar_for_map(outcoords, inputshape):
    '''
    Transforms cartesian coordinates into polar coordinates; auxiliary function for pol_to_cart_map (adapted from
    the old LIFEsim version)

    :param outcoords: format of the output coordinates
    :param inputshape: shape of the input coordinates

    :return (r,phi): tuple; indices r and phi of the output polar map
    '''

    y, x = outcoords
    x = x - (inputshape[0] - 0.5)
    y = y - (inputshape[0] - 0.5)

    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(-y, -x)
    phi_index = (phi + np.pi) * inputshape[1] / (2 * np.pi)

    return (r, phi_index)


def pol_to_cart_map(image, image_size):
    '''
    Produces a cartesian map from a given input image (adapted from the old LIFEsim version)

    :param image: input image
    :param image_size: int; size of the input image

    :return cartesian map: transformed image in cartesian coordinates
    '''

    # create new column at end (360 deg) with same values as first column (0 deg) to get complete image
    image_new = np.empty((image.shape[0], image.shape[1] + 1))
    image_new[:, :-1] = image
    image_new[:, -1] = image[:, 0]

    cartesian_map = sp.ndimage.geometric_transform(image_new, cartesian2polar_for_map, order=1,
                                                   output_shape=(image_size, image_size), mode="constant",
                                                   extra_keywords={'inputshape': (image.shape)})

    return cartesian_map


def plot_planet_SED_and_SNR(wl_bins, Fp, Fp_est, sigma, wl_min, wl_max, Fp_BB=None, snr_photon_stat=None,
                            filename=None):
    '''
    Plots the true and extracted blackbody curves (from the old lifesim version: modules/plotting/plotter.py)

    :param wl_bins: np.ndarray of size L; consists of all the wavelength bins (middle wavelength of each bin) in [m]
    :param Fp: np.ndarray of size L; true blackbody curve of the planet [photons]
    :param Fp_est: np.ndarray of size L; extracted fluxes for each wl bin [photons]
    :param sigma: np.ndarray of size L; uncertainties of the extracted fluxes for each wl bin [photons]
    :param wl_min: float; minimum wavelength captured by the instrument [m]
    :param wl_max: float; maximum wavelength captured by the instrument [m]
    :param Fp_BB: np.ndarray of size L; fitted blackbody curve of the planet [photons]
    :param snr_photon_stat: np.ndarray of size L; snr per wavelength bin as calculated using photon statistics
                                (no longer calculated in the new lifesim version, always set to 'None')
    :param filename: str; name of the file if the plot should be saved. If 'None', no plot is saved
    '''

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
    ax.set_title('Blackbody curve fit')
    ax.legend(fontsize=10)

    if filename is not None:
        plt.savefig("C:\\Users\\Philipp Binkert\\OneDrive\\ETH\\Master_Thesis\\06_plots\\" + filename + ".pdf",
                        bbox_inches='tight')

    plt.show()

    return


def plot_multi_map(maps, map_type, hfov_mas, colormap="inferno", vmin=None, vmax=None,
                   show_detections=False, filename_post=None):
    '''
    Plots the cost function J'' for each of the pixels in the image (from the old
    lifesim version: modules/plotting/plotanalysis.py)

    :param maps: np.ndarray; input map data
    :param map_type: str or list of str; labels for the maps
    :param hfov_mas: float; half field of view used to create the map
    :param colormap: str; style of the colormap
    :param vmin: float; minimum value of the colorbar to be shown
    :param vmax: float; maximum value of the colorbar to be shown
    :param show_detections: boolean; if True, mark the position of the detected planet in the plot
    :param filename_post: str; name of the file if the plot should be saved. If 'None', no plot is saved
    '''

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

        ax.set_title('Heatmap of the Cost function J\u2032\u2032')

        ax.set_xticks([-np.round(sf_mas / 2,0), 0, np.round(sf_mas / 2,0)])
        ax.set_yticks([-np.round(sf_mas / 2, 0), 0, np.round(sf_mas / 2, 0)])

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
        sfmt.set_scientific(False)


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

    if filename_post is not None:
        plt.savefig(filename_post + '.pdf', bbox_inches='tight')

    plt.show()

    return