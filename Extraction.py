import numpy as np
import pandas as pd
from Extraction_auxiliary import pol_to_cart_map, plot_planet_SED_and_SNR, plot_multi_map
from astropy import units as u
import matplotlib.pyplot as plt
import scipy as sp
from lifesim.util.radiation import black_body
from lifesim.core.modules import ExtractionModule
from lifesim.util.constants import c, h, k, radius_earth, m_per_pc
from tqdm import tqdm
import multiprocessing as multiprocessing
import mpmath as mp


class ML_Extraction(ExtractionModule):
    '''
    The ML extraction class is a signal extraction module based on the maximum likelihood method described by Dannert et
    al. (2022) in LIFE II:  Signal simulation, signal extraction and fundamental exoplanet parameters from single epoch
    observations. The code is largely based on the code for an older Lifesim version by Maurice Ottiger, as described
    in his Master Thesis "Exoplanet Detection Yield Simulations and Signal Analysis Studies for the LIFE Mission" (2020)
    '''

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Name of the module.
        """
        super().__init__(name=name)


    def get_B_C(self, signals, T):
        '''
        This functions calculates two matrices that aid in the calculation of the cost function J and the estimated
        planet flux based on the received signal and the transmission functions. See Appendix B of LIFE II for an
        explanation of the exact calculations

        :param signals: np.ndarray of shape (L,n_steps) containing the received signals in [photons]
        :param T: np.ndarray of shape (L,radial_ang_px,n_steps) containing the tm_chop transmission factor at each pixel
                on the image []

        Attributes
        --------------
        - B: np.ndarray of shape (L,radial_ang_px,n_steps); matrix used for the calculation of the estimated flux
                as described in Appendix B of LIFE II []
        - C: np.ndarray of shape (L,radial_ang_px,n_steps); matrix used for the calculation of the estimated flux
                and the cost function as described in Appendix B of LIFE II []
        - B_noise: np.ndarray of shape (L,radial_ang_px,n_steps); like B, but only for the noise (no planet signal).
                Used for the calculation of the signal to noise ratio in cost_func_MAP
        - C_noise: np.ndarray of shape (L,radial_ang_px,n_steps); like C, but only for the noise (no planet signal).
                Used for the calculation of the signal to noise ratio in cost_func_MAP
        '''

        # Get dimensions of transmission map
        (n_l, n_r, n_p) = T.shape

        # Get variance of received signals
        var = np.var(signals, axis=1, ddof=1)

        # Calculate B; in this step, the time series is included by the np.repeat function
        B_vec = (T ** 2).sum(axis=-1) / var[:, np.newaxis]
        self.B = np.repeat(B_vec[:, :, np.newaxis], n_p, axis=2)

        if (self.ideal == True):
            #If no noise is included, B_noise is obsolete
            self.B_noise = np.zeros_like(self.B)

        else:
            #Get variance of received noise (without signal) and calculate B analogously to before for this variance
            var_noise = np.var(self.noise, axis=1, ddof=1)
            B_vec_noise = (T ** 2).sum(axis=-1) / var_noise[:, np.newaxis]
            self.B_noise = np.repeat(B_vec_noise[:, :, np.newaxis], n_p, axis=2)


        # Take T twice back to back
        T_exp = np.tile(T, 2)


        # Calculate C with and without noise; also here, the time series is included
        C = np.empty((n_l, n_r, n_p // 2))
        C_noise = np.empty((n_l, n_r, n_p // 2))
        for i in range(n_p // 2):
            T_i = T_exp[:, :, n_p - i: 2 * n_p - i]
            C[:, :, i] = np.einsum("ij,ikj->ik", signals, T_i)
            C_noise[:, :, i] = np.einsum("ij,ikj->ik", self.noise, T_i)

        # Use antisymmetry of C(_noise) to speed up calculation
        C = np.concatenate((C, -C), axis=2)
        self.C = (C.T / var).T

        C_noise = np.concatenate((C_noise, -C_noise), axis=2)
        self.C_noise_only = (C_noise.T / var_noise).T
        self.C_noise_var = (C.T / var_noise).T

        return


    def cost_func(self, signals, plot=False):
        '''
        Ref:
            - Mugnier et al (2005): "Data Processing in Nulling Interf... "
            - Thiebaut & Mugnier (2005): "Maximum a posteriori planet detection and characterization with a nulling
                interferometer"

        This function first calculates the transmission function by defining a polar coordinate system and then calling
        the corresponding function from the transmission module.
        In a second step it calls get_B_C to get the matrices required for further calculation

        :param signals: np.ndarray of shape (L,n_steps) containing the received signals in [photons]
        :param plot: boolean; determines whether to show plot of the transmission function
        '''

        # Define the azimuthal coordinates
        phi_lin = np.linspace(0, 2 * np.pi, self.n_steps, endpoint=False)  # 1D array with azimuthal coordinates
        phi_mat = np.tile(phi_lin, (self.radial_ang_px, 1))  # 2D map with azimuthal coordinates

        # Define the radial coordinates
        theta_lin = np.linspace(0, 1, self.radial_ang_px,
                                endpoint=False)  # 1D array with radial separation coord [radians]
        theta_lin += 1 * 0.5 / self.radial_ang_px  # shift the coordinates to the "center" of the bins
        theta_mat = np.tile(theta_lin, (self.n_steps, 1)).T  # 2D array with radial separation coordinates [radians]

        # Define matrices which include the different wl parameters; the radial coordinate must be scaled by the hfov
        phi_mat_wl = np.zeros((self.L, self.radial_ang_px, self.n_steps))
        theta_mat_wl = np.zeros((self.L, self.radial_ang_px, self.n_steps))

        # define the inputs for the transmission function using the wl-independent hfov_cost
        for i in range(self.L):
            phi_mat_wl[i, :, :] = phi_mat
            theta_mat_wl[i, :, :] = theta_mat * self.hfov_cost

        # Define the input parameters for the transmission function
        d_alpha = theta_mat_wl * np.cos(phi_mat_wl)
        d_beta = theta_mat_wl * np.sin(phi_mat_wl)

        #Calculate the transmission map
        _, _, _, _, tm_chop = self.run_socket(s_name='transmission',
                                              method='transmission_map',
                                              map_selection=['tm_chop'],
                                              direct_mode=True,
                                              d_alpha=d_alpha,
                                              d_beta=d_beta,
                                              hfov=self.hfov_cost,
                                              image_size=self.image_size
                                              )


        # Plot the resulting transmission map for the first wl bin
        if (plot == True):
            plt.contour(tm_chop[0, :, :])
            plt.show()

        # Normalize the transmission map to instrument performance (required for compatability with functions written for
        #   the old lifesim version
        self.tm_chop = tm_chop * self.single_data_row['int_time'] / self.n_steps * self.data.inst[
            'telescope_area'] * self.data.inst['eff_tot'] * self.wl_bin_widths[:, np.newaxis, np.newaxis] * 10 ** 6


        # Calculate the matrices B and C as well as B_noise and C_noise
        self.get_B_C(signals, self.tm_chop)

        return


    def get_Dsqr_mat(self, L):
        '''
        This function calcualtes the D^2 matrix as described in LIFE II Appendix B used for the calculation of the
        estimated planet flux. See this paper for the explanations of the exact calculations

        :param L : number of wavelegth bins []

        :return Dsqr_mat: matrix of dimensions (L,radial_ang_px,n_steps) required for calculation of the estimated
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


    def get_F_estimate(self, mu=0):
        '''
        This function calculates the estimated flux received by the planet based on the matrices B and C which are
        given as input. See LIFE II Appendix B for an explanation of the calculations. The calculation is also done for
        the pure signal without noise ("white"); this is required for the SNR calculation in cost_func_MAP

        :param B: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calculation of the estimated flux
                as described in Appendix B of LIFE II []
        :param C: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calculation of the estimated flux
                and the cost function as described in Appendix B of LIFE II []
        :param B_noise: np.ndarray of shape (L,radial_ang_px,n_steps); like B, but only for the noise (without planet signal) []
        :param C_noise: np.ndarray of shape (L,radial_ang_px,n_steps); like C, but only for the noise (without planet signal) []
        :param mu: float, regularization parameter []

        :return F: np.ndarray of shape (L,radial_ang_px); total estimated flux (planet plus noise) received for each
                    wl bin at every pixel in the image in [photons]
        :return F_pos: np.ndarray of shape (L,radial_ang_px); positive part of total estimated flux received for each
                    wl bin at every pixel in the image in [photons]
        :return F_white: np.ndarray of shape (L,radial_ang_px); estimated planet flux (no noise) received for each
                    wl bin at every pixel in the image in [photons]
        :return F_white_pos: np.ndarray of shape (L,radial_ang_px); positive part of estimated planet flux received
                    for each wl bin at every pixel in the image in [photons]
        '''

        # Get dimensions of the matrix
        (n_l, n_r, n_p) = self.C.shape

        # If there is no regularization, the calculation can be simplified
        if mu == 0:
            #Calculation of the total estimated signal (planet signal plus noise)
            F = self.C / self.B
            F_pos = np.where(F >= 0, F, 0)

            ##Calculation of the estimated planet signal (no noise)
            F_noise = self.C_noise_only / self.B_noise
            F_noise_pos = np.where(F_noise >=0, F_noise, 0)
            F_white = F - F_noise
            F_white_pos = np.where(F_white >=0, F_white, 0)

        # If regularization is included, F is calculated according to B.6, additionally requiring calc. of the D matrix
        else:
            try:
                # Calculate D^2
                Dsqr_mat = self.get_Dsqr_mat(n_l)

                B_diags = np.array([np.diag(B[:, i, 0]) for i in range(n_r)])  # B is phi-independent
                B_diags_noise = np.array([np.diag(self.B_noise[:, i, 0]) for i in range(n_r)])


                # Calculate the inverse of (B+mu*D^2)
                S = B_diags + mu * Dsqr_mat
                Sinv = np.linalg.inv(S)

                S_noise = B_diags_noise + mu * Dsqr_mat
                Sinv_noise = np.linalg.inv(S_noise)

                # Calculate F and F_pos without and with noise
                F = np.einsum("rlm,mrp ->lrp", Sinv, self.C)
                F_noise = np.einsum("rlm,mrp ->lrp", Sinv_noise, self.C_noise_only)

                F_pos = np.empty_like(F)
                F_noise_pos = np.empty_like(F_noise)

                for r in range(n_r):  # r-range
                    for p in range(n_p):  # phi-range
                        # nnls: F = argmin_F(abs(S*F-C))
                        F_pos[:, r, p] = sp.optimize.nnls(S[r], self.C[:, r, p], maxiter=200)[0]
                        F_noise_pos[:, r, p] = sp.optimize.nnls(S_noise[r], self.C_noise_only[:, r, p], maxiter=200)[0]

                F_white = F - F_noise
                F_white_pos = np.where(F_white >= 0, F_white, 0)

            # For some values of mu, (B+mu*D^2) can be singular. In this case, calculate F without regularization and
            # print a corresponding warning
            except:
                print('Warning: singular matrix obtained for this value of mu; F is calculated for mu=0')
                F = self.C / self.B
                F_pos = np.where(F >= 0, F, 0)

                F_noise = self.C_noise_only / self.B_noise
                F_white = F - F_noise
                F_white_pos = np.where(F_white >= 0, F_white, 0)

        return F, F_pos, F_white, F_white_pos, F_noise, F_noise_pos


    def pdf_J(self,J, delta):

        fact = sp.special.factorial

        pdf = 1 / 2 ** self.L * delta
        for l in range(0, self.L):
            pdf += fact(self.L) / (2 ** self.L * fact(l) * fact(self.L - l)) * sp.stats.chi2.pdf(J, self.L - l)

        return pdf


    def cdf_J(self,J, precision):

        mp.mp.dps = precision

        cdf = mp.power(mp.mp.mpf('0.5'), self.L)
        for l in range(0, self.L):
            cdf += mp.fac(self.L) / (mp.mp.power(mp.mp.mpf('2'), self.L) * mp.fac(l) * mp.fac(self.L - l)) * mp.gammainc((self.L - l) / 2, 0, J / 2, regularized=True)

        return cdf


    def cost_func_MAP(self, mu=0, plot_maps=False):
        '''
        This function calculates the estimated planet flux and the cost function based on the transmission map, the
        received signal and (if applicable) the regularization parameter. See LIFE II Appendix B for a derivation
        of the exact calculations

        :param mu: float, regularization parameter []
        :param plot_maps: boolean, determines whether to plot the cost map

        :return Jmax: float; maximum cost function value (of all pixels) []
        :return theta_max: tuple; (r,phi) coordinates of the maximum cost function value in [radial pixel,azimuthal pixel]
        :return Fp_est: np.ndarray of size L; contains the estimated planet flux received at every
                    wavelength at the theta_max pixel coordinates in [photons]
        :return Fp_est_pos: np.ndarray of size L; contains the positive part of the estimated planet flux received
                    at every wavelength at the theta_max pixel coordinates in [photons]
        :return sigma_est: np.ndarray of size L; contains the sigmas for the extracted spectra per wavelength bin
                    in [photons]
        :return SNR_est: float; contains the extracted snr over the entire wavelength range []
        '''

        #Calculate the estimated planet flux
        F, F_pos, F_white, F_white_pos, F_noise, F_noise_pos = self.get_F_estimate(mu=mu)

        # Calculate the cost function and its maximum as well as the position of the maximum value of the cost function
        self.J = (F_pos * self.C).sum(axis=0)
        self.J_noise = (F_noise_pos * self.C_noise_only).sum(axis=0)
        self.J_FPR = (F_pos * self.C_noise_var).sum(axis=0)
        Jmax = np.max(self.J)
        theta_max = np.unravel_index(np.argmax(self.J, axis=None), self.J.shape)
        (r, p) = theta_max


        #Calculate the estimated flux (with noise and without) at the position of Jmax (total and positive part)
        F_est = F[:, r, p]
        F_est_pos = F_pos[:, r, p]


        true_r = int(self.single_data_row['angsep'] / 2 / self.hfov_cost * self.image_size / 180 / 3600 * np.pi)
        true_phi = 0

        J_true_pos = self.J_FPR[true_r, true_phi]

        precision=100
        FPR_sigma = np.inf

        while np.isinf(FPR_sigma):

            FPR = self.cdf_J(J_true_pos,precision)
            FPR_sigma = mp.mp.sqrt(2) * mp.erfinv(2*FPR-1)
            FPR_sigma = float(FPR_sigma)
            precision *= 2

            #Break the loop if the precision goes to high to avoid too long runtimes. Set FPR_sigma to 100 in this case
            #   in order to remove the value during analysis
            if (precision>self.precision_limit):
                if (plot_maps==True):
                    print('warning: FPR_sigma could not be calculated')
                FPR_sigma = 10000


        if (self.ideal == True):
            # Sigma equal to zero leads to errors in the curve fitting; take very small value instead
            sigma_est = np.ones((self.L)) * 10 ** (-9)

        else:
            # Calculate sigma at the position of Jmax
            sigma_est = self.B_noise[:, r, p] ** (-1 / 2)


        if (plot_maps == True):
            # Plot the cost function map
            j_map = pol_to_cart_map(self.J, self.image_size)
            plot_multi_map(j_map, "Cost Value",
                           self.hfov_cost * 3600000 * 180 / np.pi, "inferno")


            #Plot a histogramm of the values of J
            flat_J = self.J.flatten()
            j_array = np.linspace(0, 2 * np.max(flat_J), 100)

            plt.hist(flat_J,j_array)
            plt.title('J including signal')
            plt.ylim((0,10))
            plt.axvline(x=65, color='red', linestyle='--', label='det. thres.')
            plt.legend(loc='best')
            plt.show()

            #Plot a histogramm of the values of J considering only the noisy part of the signal
            flat_J_noise = self.J_noise.flatten()
            j_array_noise = np.linspace(0, 2* np.max(flat_J_noise), int(10 ** 2))

            weights_noise = np.ones_like(flat_J_noise)/flat_J_noise.size
            counts, bins, _ = plt.hist(flat_J_noise, j_array_noise, weights=weights_noise)

            #Delta must be =0 for the function to be normalized (pdf_J integrated =1)
            delta=0

            pdf_Jprime = sp.stats.chi2.pdf(j_array_noise,self.L)
            pdf_J2prime = self.pdf_J(j_array_noise,delta)

            plt.plot(j_array_noise,pdf_Jprime,label='J\u2032')
            plt.plot(j_array_noise,pdf_J2prime,label='J\u2032\u2032')
            plt.title('J only noise')
            plt.axvline(x=65, color='red', linestyle='--', label='det. thres.')
            plt.legend(loc='best')
            plt.show()


        return Jmax, theta_max, F_est, F_est_pos, sigma_est, FPR_sigma


    def BB_for_fit(self, wl, Tp, Rp):
        '''
        This function calculates the flux received at Earth from an object with radius Rp radiating at temperature T,
        at distance dist_s and at wavelengths wl

        :param wl: np.ndarray of size L; wavelength bins in [m]
        :param Tp: float, planet temperature in [K]
        :param Rp: float, planet radius in [R_earth]

        :return fgamma: np.ndarray of size L; contains the total blackbody fluxes in each of the L wavelength bins
                        in [photons]
        '''

        dist_s = self.single_data_row['distance_s']

        fact1 = 2 * c / (wl ** 4)
        fact2 = (h * c) / (k * Tp * wl)

        # Standard planck function
        fgamma = np.array(fact1 / (np.exp(fact2) - 1.0)) * 10 ** -6 * np.pi * (
                (Rp * radius_earth) / (dist_s * m_per_pc)) ** 2

        return fgamma


    def get_T_R_estimate(self, spectra, sigmas, p0=(300, 1.), absolute_sigma=True, plot_flux=False,
                         plot_BB=False):
        '''
        This function obtains the planet temperature and radius by fitting the input spectra and sigmas to blackbody curves

        :param spectra: np.ndarray of size L; contains the extracted planet flux in each wl-bin in [photons]
        :param sigmas: np.ndarray of size L; contains the sigmas of the extracted planet flux in each wl-bin in [photons]
        :param p0: tuple; initial values for the optimization for T and R in [K,R_earth]
        :param absolute_sigma: boolean; If True, sigma is used in an absolute sense and the estimated covariance
                    reflects these absolute values. If False, only the relative magnitudes of the sigma values matter
        :param plot_flux: boolean; determines whether to plot at all. If True, automatically plots the spectra
        :param plot_BB: boolean; determines whether to plot the fitted blackbody curve. Only applicable if plot_flux=True

        :return popt: np.ndarray of size 2; optimal values for (T,R) such that the sum of the squared residuals to the
                        blackbody curve is minimized. In [T,R_earth]
        :return pcov: np.ndarray of shape (2,2); covariance matrix of the obtained popt, a measure of uncertainty.
                        Dimensions [T^2] and [R_earth^2] on the first and second diagonal element
        :return perr: np.ndarray of size 2; standard deviation of the errors on T and R. In [T,R_earth]
        '''

        # Perform the fit of T and R and calculate the uncertainties

        popt, pcov = sp.optimize.curve_fit(self.BB_for_fit, self.wl_bins, spectra,
                                           sigma=sigmas, p0=p0, absolute_sigma=absolute_sigma, maxfev=10000)

        perr = np.sqrt(np.diag(pcov))

        # Plot the different quantities as defined
        if (plot_flux == True):
            # Get the blackbody curve for the estimated parameters
            Fp_fit = self.BB_for_fit(self.wl_bins, popt[0], popt[1])

            #In the old version, the snr per wavelength according to photon statistics was saved; this is no longer the
            #   case in the new lifesim version, therefor this option is no longer available. Can be re-implemented if
            #   the data is saved again
            snr_photon_stat = None

            if (plot_BB == True):
                Fp_BB = Fp_fit

            else:
                Fp_BB = None

            # Get the ideal blackbody flux of the planet
            Fp = black_body(mode='planet',
                            bins=self.wl_bins,
                            width=self.wl_bin_widths,
                            temp=self.single_data_row['temp_p'],
                            radius=self.single_data_row['radius_p'],
                            distance=self.single_data_row['distance_s']) / self.wl_bin_widths * 10 ** -6

            plot_planet_SED_and_SNR(self.wl_bins, Fp, spectra, sigmas, self.min_wl,
                                    self.max_wl, Fp_BB=Fp_BB, snr_photon_stat=snr_photon_stat)

        return popt, pcov, perr



    def single_spectrum_extraction(self, n_run=1, plot=False):
        '''
        This is the core function of the ML_Extraction class that executes the signal extraction for a single planet.
        In each run, the extracted spectrum, snr, sigma, planet positions and cost function values are calculated based
        on the parameters in the bus catalog and random noise generation.

        In a first step, the signal (including noise) is generated by calling the instrument.get_signals function.
        Next, the cost_function is called to generate the transmission maps and the matrices required to calculate the
        estimated planet flux and the cost function J.
        The third step consists of calling cost_func_MAP to get the estimated planet flux as well as the cost functions
        and the positions corresponding to the maxima of the cost function at every wavelength.
        Since the strength of the cost function maximum varies strongly between wavelength bins, get_correct_position
        is then called to assess the correct position based on at which wavelength the extraction is reliable, i.e.
        where the cost function maximum is high.
        Finally, the extracted spectra, sigma and snr are calculated for every wavelength bin using the presumed planet
        position and returned for further processing. With these extracted spectra and sigmas, the planet temperature
        and radius can be calculated by fitting the values to a blackbody curve

        :param n_run: integer, number of runs to perform
        :param plot: boolean; determines whether to show plots throughout the runs (only advised for small n_run)

        :return extracted spectra: np.ndarray of shape (n_run,L); contains the extracted spectra per wavelength bin for
                    each of the runs in [photons]
        :return extracted snrs: np.ndarray of size n_run; contains the extracted snrs for each of the runs over the
                    entire wavelength range []
        :return extracted sigmas: np.ndarray of shape (n_run,L); contains the sigmas for the extracted spectra
                    per wavelength bin for each of the runs in [photons]
        :return extracted Jmaxs: np.ndarray of size n_run containing the maximum values of J over all wavelengths
                    for every run []
        :return rss: np.ndarray of size n_run; contains the extracted angular separations for each of the runs
                    in [arcsec]
        :return: phis: np.ndarray of size n_run; contains the extracted azimuthal positions on the image for each of
                    the runs in [degrees]
        :return Ts: np.ndarray of size n_run; extracted planet temperatures of every run in [K]
        :return Ts_sigma: np.ndarray of size n_run; uncertainties of the extracted planet temperatures of every run in [K]
        :return Rs: np.ndarray of size n_run; extracted planet radii of every run in [K]
        :return Rs_sigma: np.ndarray of size n_run; uncertainties of the extracted planet radii of every run in [K]
        '''

        extracted_spectra = []
        extracted_snrs = []
        extracted_sigmas = []
        extracted_Jmaxs = []
        rss = []
        phiss = []
        Ts = []
        Ts_sigma = []
        Rs = []
        Rs_sigma = []
        FPRs = []


        for n in range(n_run):
            if (plot == True):
                print('run:', n)

            # Get the signals (with and without noise) for the specified constellation
            #ToDo question: The noise/2...
            self.signals, self.ideal_signals = self.run_socket(s_name='instrument',
                                                               method='get_signal',
                                                               temp_s=self.single_data_row['temp_s'],
                                                               radius_s=self.single_data_row['radius_s'],
                                                               distance_s=self.single_data_row['distance_s'],
                                                               lat_s=self.single_data_row['lat'],
                                                               z=self.single_data_row['z'],
                                                               angsep=self.single_data_row['angsep'],
                                                               flux_planet_spectrum=[self.wl_bins * u.meter,
                                                                                     self.single_data_row[
                                                                                         'planet_flux_use'][
                                                                                         0] / 3600 /
                                                                                     self.data.inst[
                                                                                         'telescope_area'] /
                                                                                     self.data.inst[
                                                                                         'eff_tot'] / self.wl_bin_widths * u.photon / u.second / (
                                                                                             u.meter ** 3)],
                                                               integration_time=self.single_data_row['int_time'],
                                                               phi_n=self.n_steps,
                                                               extraction_mode=True)


            # Create the transmission maps and the auxiliary matrices B&C
            if (self.ideal == True):
                # define noise as an array with all ones (to be evaluated in an if-clause in self.get_B_C)
                self.noise = np.zeros_like(self.ideal_signals)

                self.cost_func(signals=self.ideal_signals, plot=plot)

            else:
                # define noise as total signal minus planet signal
                self.noise = self.signals - self.ideal_signals

                self.cost_func(signals=self.signals, plot=plot)

            # Get the extracted signals and cost function maxima
            Jmax, theta_max, Fp_est, Fp_est_pos, sigma_est, FPR_extr = self.cost_func_MAP(mu=self.mu, plot_maps=plot)


            # Calculate the position of the maximum of the cost function
            (r, phi) = theta_max

            # Calculate the best fit temperature and radius along with the corresponding uncertainties
            try:
                popt, pcov, perr = self.get_T_R_estimate(Fp_est, sigma_est, plot_flux=plot, plot_BB=plot)

            except RuntimeError:
                print('T and R not found')
                popt = np.array([self.single_data_row['temp_p'], self.single_data_row['radius_p']])
                perr = np.array([0.1,0.1])

            T_est = popt[0]
            R_est = popt[1]
            T_sigma = perr[0]
            R_sigma = perr[1]

            #ToDo question: remove this SNR calculation?
            #Calculate the SNR
            if (self.ideal == True):
                # SNR is not a well-defined quantity
                SNR_est = np.inf


            else:
                # Calculate the blackbody flux of the planet according to the estimated parameters
                est_flux = black_body(mode='planet',
                                      bins=self.wl_bins,
                                      width=self.wl_bin_widths,
                                      temp=self.single_data_row['temp_p'],
                                      radius=self.single_data_row['radius_p'],
                                      distance=self.single_data_row['distance_s']
                                      )


                #Calculate the flux retrieved from the planet with the assumed parameters
                _, self.est_ideal_signal = self.run_socket(s_name='instrument',
                                                           method='get_signal',
                                                           temp_s=self.single_data_row['temp_s'],
                                                           radius_s=self.single_data_row['radius_s'],
                                                           distance_s=self.single_data_row['distance_s'],
                                                           lat_s=self.single_data_row['lat'],
                                                           z=self.single_data_row['z'],
                                                           angsep=2 * r * self.hfov_cost / self.image_size * 180 * 3600 / np.pi,
                                                           flux_planet_spectrum=[self.wl_bins * u.meter,
                                                                                 est_flux / self.wl_bin_widths * u.photon / u.second / ( u.meter ** 3)],
                                                           integration_time=self.single_data_row['int_time'],
                                                           phi_n=self.n_steps,
                                                           extraction_mode=True)


                #Recalculate the extracted signal according to the extracted parameters
                self.noise = self.signals - self.est_ideal_signal
                self.get_B_C(self.signals, self.tm_chop)
                _, _, _, est_signal_pos, _, _ = self.get_F_estimate(mu=self.mu)

                #Take the signal at the derived position
                est_signal_pos = est_signal_pos[:,r,phi]

                # Calculate the total SNR over all wavelength bins
                SNR_est = np.sum((est_signal_pos / sigma_est) ** 2) ** (1 / 2)


            # Add the quantities from this run to the final list
            extracted_spectra.append(Fp_est)
            extracted_snrs.append(SNR_est)
            extracted_sigmas.append(sigma_est)
            extracted_Jmaxs.append(Jmax)

            rss.append(2 * r * self.hfov_cost / self.image_size * 180 * 3600 / np.pi)
            phiss.append(phi * self.n_steps / 360)

            Ts.append(T_est)
            Rs.append(R_est)
            Ts_sigma.append(T_sigma)
            Rs_sigma.append(R_sigma)
            FPRs.append(FPR_extr)

        # Convert the final lists to arrays
        extracted_spectra = np.array(extracted_spectra)
        extracted_snrs = np.array(extracted_snrs)
        extracted_sigmas = np.array(extracted_sigmas)
        extracted_Jmaxs = np.array(extracted_Jmaxs)

        rss = np.array(rss)
        phiss = np.array(phiss)

        Ts = np.array(Ts)
        Ts_sigma = np.array(Ts_sigma)
        Rs = np.array(Rs)
        Rs_sigma = np.array(Rs_sigma)
        FPRs = np.array(FPRs)


        # Plot the extracted spectra and sigmas along with the true blackbody function
        if (plot == True):
            sigma_est_mean = extracted_sigmas.mean(axis=0)
            Fp_est_mean = extracted_spectra.mean(axis=0)
            Ts_mean = Ts.mean(axis=0)
            Rs_mean = Rs.mean(axis=0)

            Fp_BB = self.BB_for_fit(self.wl_bins, Ts_mean, Rs_mean)

            # In the old version, the snr per wavelength according to photon statistics was saved; this is no longer the
            #   case in the new lifesim version, therefor this option is no longer available. Can be re-implemented if
            #   the data is saved again
            snr_photon_stat = None

            Fp = black_body(mode='planet',
                            bins=self.wl_bins,
                            width=self.wl_bin_widths,
                            temp=self.single_data_row['temp_p'],
                            radius=self.single_data_row['radius_p'],
                            distance=self.single_data_row['distance_s']) / self.wl_bin_widths * 10 ** -6

            plot_planet_SED_and_SNR(self.wl_bins, Fp, Fp_est_mean, sigma_est_mean,
                                    self.min_wl,
                                    self.max_wl, Fp_BB=Fp_BB,
                                    snr_photon_stat=snr_photon_stat)

        return extracted_spectra, extracted_snrs, extracted_sigmas, extracted_Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma, FPRs



    def main_parameter_extraction(self, n_run=1, mu=0, n_processes = 1, precision_limit=3200, plot=False, ideal=False, single_planet_mode=False, planet_number=0, save_mode=True, filepath=None):
        '''
        The main_parameter_extraction function is the function that should get called by other files. In defines all the
        required parameters and then runs single_spectrum_extraction for either one specified planet (single_planet_mode=True)
        or for all of the planets in the catalog. In single_planet_mode=True, the extracted quantities are returned.
        If =False, then a number of processes equal to n_processes are created to run the extraction of the planets in
        the catalog of the bus in parallel. The output is directly adjusted and saved (provided save_mode=True)
        in path/changeme.csv.

        :param n_run: integer; number of runs to perform for each planet
        :param mu: float; regularization parameter for the calculation of the cost function J as described in section 3.1
                    of LIFE II []
        :param n_processes: integer; number of processes to run in parallel for multi-planet extraction (ignored if
                    single_planet_mode=True)
        :param plot: boolean; determines whether to show plots throughout the runs (only advised for small n_run)
        :param ideal: boolean; if True, no noise is included in the extraction
        :param single_planet_mode: boolean; if True, signal extraction is performed for only one planet in the catalog
                    as defined by the following parameter
        :param planet_number: integer; index of the planet in the catalog of the bus of which the signal is to be
                    extracted. Only applicable if single_planet_mode=True
        :param save_mode: boolean; if True, the catalog with added columns for the extracted parameters will be saved
                    in path/changeme.csv. Only applicable if single_planet_mode=False
        :param filepath: str; path to where the the file will be saved (only applicable if save_mode=True and
                    single_planet_mode=False)

        :return spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma; Only applicable if
                    single_planet_mode=True, else nothing is returned. See single_spectrum_extraction for description

        Attributes
        -----------
        Most of the attributes are simply abbreviations of the respective objects inherited from the bus. These are:
        - 'n_planets': Number of planets in the catalog []
        - 'L': Number of wavelength bins as given by the wavelength range and the resolution parameter R []
        - 'min_wl': minimum wavelength captured by the instrument [m]
        - 'max_wl': maximum wavelength captured by the instrument [m]
        - 'wl_bins': np.ndarray of size L, consists of all the wavelength bins (middle wavelength of each bin) in [m]
        - 'wl_bin_widths': np.ndarray of size L, consists of all the widths of the wavelength bins in [m]
        - 'image_size': integer, precision that the image can be resolved to in one dimension in [number of pixels]
        - 'radial_ang_px': integer, precision that the image can be resolved to in the radial coordinate in [number of pixels]
        - 'hfov': np.ndarray of size L, consists of the half field of views of the instrument in each wavelength bin in [radians]
        - 'mu': float, regularization parameter for the calculation of the cost function J as described above []
        - 'n_run': integer; number of runs to perform for each planet []
        - 'ideal': boolean; if True, no noise is included in the extraction []
        - 'hfov_cost': float; Half field of view used for the calculation of the cost map. This must be chosen such that
                            that all of the planets are inside it (here 0.4 arcsec) [radian]

        Two attributes concerning the angular dimensioning of the image:
        - 'planet_azimuth': Angular position of the planet on the image. Contrary to the radial position, which is given by
                            the angular separation, this attribute is not known a priori from the simulation and thus set to
                            0 [degrees] for simplicity. Changing this or making it adaptable would require some modification
                            of the transmission functions to comply with the rest of the code
        - 'n_steps':        Resolution of the angular coordinate. Given as an integer, e.g. 360 corresponds to the angular
                            resolution being split into 360 parts, i.e. one part has a size of 1 degree []

        The following attribute is planet-specific and must be redefined for each new planet:
        - 'single_data_row': pd.series, catalog data from the bus corresponding to the planet described by planet_number
        '''

        self.n_planets = len(self.data.catalog.index)
        self.L = self.data.inst['wl_bins'].size
        self.min_wl = self.data.inst['wl_bin_edges'][0]
        self.max_wl = self.data.inst['wl_bin_edges'][-1]
        self.wl_bins = self.data.inst['wl_bins']
        self.wl_bin_widths = self.data.inst['wl_bin_widths']
        self.image_size = self.data.options.other['image_size']
        self.radial_ang_px = int(self.image_size / 2)
        self.hfov = self.data.inst['hfov']
        self.mu = mu
        self.n_run = n_run
        self.ideal = ideal
        self.precision_limit = precision_limit

        self.planet_azimuth = 0
        self.n_steps = 360


        #if in single_planet_mode, define the planet-specific parameters, run single_spectrum_extraction once and return
        # the extracted parameters
        if (single_planet_mode==True):
            self.single_data_row = self.data.catalog.iloc[planet_number]
            self.hfov_cost = self.single_data_row['angsep'] * 1.2 / 3600 / 180 * np.pi

            spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma, FPRs = self.single_spectrum_extraction(n_run=n_run, plot=plot)

            return spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma, FPRs


        #if single_planet_mode=False, proceed here
        else:

            #create lists to store extracted data
            extracted_spectra_tot = []
            extracted_snrs_tot = []
            extracted_sigmas_tot = []
            extracted_Jmaxs_tot = []
            extracted_rss_tot = []
            extracted_phiss_tot = []
            extracted_Ts_tot = []
            extracted_Ts_sigma_tot = []
            extracted_Rs_tot = []
            extracted_Rs_sigma_tot = []
            extracted_FPRs_tot = []


            # Divide the planets into equal ranges for parallel processing
            n_processes = n_processes
            planet_indices = []

            for i in range(n_processes):
                lower_index = int(np.floor(self.n_planets/n_processes*i))
                if (i == n_processes):
                    upper_index = int(self.n_planets)
                else:
                    upper_index = int(np.floor(self.n_planets/n_processes*(i+1)))
                index_range = np.arange(lower_index,upper_index,1)
                planet_indices.append(index_range)


            #Define processes, queues and events:
            #   The processes are the objects which will run the extraction in parallel
            #   The queues are where the processes store their output
            #   The process has an event attributed to it; it's function is to be 'set' when the process is completed so
            #       that the main code can continue as soon as all events are set, i.e. all processes are finished
            processes = []
            events = []
            res_queue = multiprocessing.Queue()
            num_queue = multiprocessing.Queue()

            for i in range(n_processes):
                e = multiprocessing.Event()
                p = multiprocessing.Process(target=self.execute_multiprocessing, args=[planet_indices[i], res_queue, num_queue, e, i])
                p.start()
                events.append(e)
                processes.append(p)


            #Wait for all processes to finish
            for event in events:
                event.wait()


            #Get the results from the queues and store them in lists. The numbers list is used to keep track of what
            #   order the queues finished in
            results = []
            numbers = []

            for i in range(n_processes):
                result = res_queue.get()
                number = num_queue.get()
                results.append(result)
                numbers.append(number)

            #Add the results to the main list in the correct order
            for i in range(n_processes):
                place_in_queue = int(numbers.index(i))

                extracted_spectra_tot.append(results[place_in_queue][0])
                extracted_snrs_tot.append(results[place_in_queue][1])
                extracted_sigmas_tot.append(results[place_in_queue][2])
                extracted_Jmaxs_tot.append(results[place_in_queue][3])
                extracted_rss_tot.append(results[place_in_queue][4])
                extracted_phiss_tot.append(results[place_in_queue][5])
                extracted_Ts_tot.append(results[place_in_queue][6])
                extracted_Ts_sigma_tot.append(results[place_in_queue][7])
                extracted_Rs_tot.append(results[place_in_queue][8])
                extracted_Rs_sigma_tot.append(results[place_in_queue][9])
                extracted_FPRs_tot.append(results[place_in_queue][10])



            #Add the data to the bus catalog
            self.data.catalog['extracted_spectra'] = sum(extracted_spectra_tot, [])
            self.data.catalog['extracted_snrs'] = sum(extracted_snrs_tot, [])
            self.data.catalog['extracted_sigmas'] = sum(extracted_sigmas_tot, [])
            self.data.catalog['extracted_Jmaxs'] = sum(extracted_Jmaxs_tot, [])
            self.data.catalog['extracted_rss'] = sum(extracted_rss_tot, [])
            self.data.catalog['extracted_phiss'] = sum(extracted_phiss_tot, [])
            self.data.catalog['extracted_Ts'] = sum(extracted_Ts_tot, [])
            self.data.catalog['extracted_Ts_sigma'] = sum(extracted_Ts_sigma_tot, [])
            self.data.catalog['extracted_Rs'] = sum(extracted_Rs_tot, [])
            self.data.catalog['extracted_Rs_sigma'] = sum(extracted_Rs_sigma_tot, [])
            self.data.catalog['extracted_FPRs'] = sum(extracted_FPRs_tot, [])

            #save the catalog
            if (save_mode==True):
                self.data.catalog.to_csv(filepath+'changeme.csv')


            print('main_parameter_extraction completed')

            return


    def execute_multiprocessing(self, process_range, res, num, event, n_process):
        '''
        This function is called by each of the parallel running processes and executes the signal extraction for the
        planets in its range.

        :param process_range: np.array; contains the indices of the planets in the catalog that the process should extract
        :param res: multiprocessing.Queue; this is where the results are stored
        :param num: multiprocessing.Queue; keeps track of the order in which the processes fill up the results queue
        :param event: multiprocessing.Event; this object is 'set' as soon as the process is complete
        :param n_process: int; indicates the process number [0, n_processes] (label-like)

        :return: No return; the output is stored in the res Queue
        '''

        #Create lists to store extracted data
        extracted_spectra = []
        extracted_snrs = []
        extracted_sigmas = []
        extracted_Jmaxs = []
        extracted_rss = []
        extracted_phiss = []
        extracted_Ts = []
        extracted_Ts_sigma = []
        extracted_Rs = []
        extracted_Rs_sigma = []
        extracted_FPRs = []

        print('Process #',n_process,' started')

        #Loop through all of the planets in the process range
        for j in tqdm(process_range):
            self.single_data_row = self.data.catalog.iloc[j]
            self.hfov_cost = self.single_data_row['angsep'] * 1.2 / 3600 / 180 * np.pi

            #Call the extraction function for a single planet
            spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma, FPRs = self.single_spectrum_extraction(
                n_run=self.n_run, plot=False)

            #Store the data in the lists
            extracted_spectra.append(spectra.tolist())
            extracted_snrs.append(snrs.tolist())
            extracted_sigmas.append(sigmas.tolist())
            extracted_Jmaxs.append(Jmaxs.tolist())
            extracted_rss.append(rss.tolist())
            extracted_phiss.append(phiss.tolist())
            extracted_Ts.append(Ts.tolist())
            extracted_Ts_sigma.append(Ts_sigma.tolist())
            extracted_Rs.append(Rs.tolist())
            extracted_Rs_sigma.append(Rs_sigma.tolist())
            extracted_FPRs.append(FPRs.tolist())

        #Add the process number and the results to the queue and set the event
        num.put(n_process)
        res.put([extracted_spectra, extracted_snrs, extracted_sigmas, extracted_Jmaxs, extracted_rss, extracted_phiss, extracted_Ts, extracted_Ts_sigma, extracted_Rs, extracted_Rs_sigma, extracted_FPRs])
        event.set()
        print('Process #', n_process, ' finished')

        return