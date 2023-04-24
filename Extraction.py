import numpy as np
import lifesim as ls
import pandas as pd
from auxiliary import m_per_pc, pol_to_cart_map, path
from lifesim.util.constants import c, h, k, radius_earth
from astropy import units as u
import matplotlib.pyplot as plt
import scipy as sp
from lifesim.util.radiation import black_body
from plots import plot_planet_SED_and_SNR, plot_multi_map
from functions import get_detection_threshold
from lifesim.core.modules import ExtractionModule

class ML_Extraction(ExtractionModule):
    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Name of the module.
        """
        super().__init__(name=name)


    '''
        The ML extraction class is a signal extraction module based on the maximum likelihood method described by Dannert et
        al. (2022) in LIFE II:  Signal simulation, signal extraction and fundamental exoplanet parameters from single epoch
        observations. The code is largely based on the code for an older Lifesim version by Maurice Ottiger, as described
        in his Master Thesis "Exoplanet Detection Yield Simulations and Signal Analysis Studies for the LIFE Mission" (2020)

        Parameters
        ------
        bus: lifesim bus object
            A bus with an already existing dataset (i.e. that already has simulation data)
        planet number: index (integer)
            Index of the planet in the catalog of the bus of which the signal is to be extracted
        mu: float
            Regularization parameter for the calculation of the cost function J as described in section 3.1 of LIFE II

        Attributes
        -----------
        Most of the attributes are simply abbreviations of the respective objects inherited from the bus. These are:
        - 'bus': This is just the bus
        - 'ins': Instrument module attached to the bus
        - 'exozodi': Exozodi module attached to the bus
        - 'localzodi': Localzodi module attached ot the bus
        - 'transm': Transmission module attached to the bus
        - 'L': Number of wavelength bins as given by the wavelength range and the resolution parameter R []
        - 'min_wl': minimum wavelength captured by the instrument [m]
        - 'max_wl': maximum wavelength captured by the instrument [m]
        - 'wl_bins': np.ndarray of size L, consists of all the wavelength bins (middle wavelength of each bin) in [m]
        - 'wl_bin_widths': np.ndarray of size L, consists of all the widths of the wavelength bins in [m]
        - 'image_size': integer, precision that the image can be resolved to in one dimension in [number of pixels]
        - 'radial_ang_px': integer, precision that the image can be resolved to in the radial coordinate in [number of pixels]
        - 'hfov': np.ndarray of size L, consists of the half field of views of the instrument in each wavelength bin in [radians]
        - 'mu': float, regularization parameter for the calculation of the cost function J as described above []
        - 'planet_number': index of the planet in the catalog of the bus of which the signal is to be extracted []
        - 'single_data_row': pd.series, catalog data from the bus corresponding to the planet described by planet_number

        Two attributes concerning the angular dimensioning of the image:
        - 'planet_azimuth': Angular position of the planet on the image. Contrary to the radial position, which is given by
                            the angular separation, this attribute is not known a priori from the simulation and thus set to
                            0 [degrees] for simplicity. Changing this or making it adaptable would require some modification
                            of the transmission functions to comply with the rest of the code
        - 'n_steps':        Resolution of the angular coordinate. Given as an integer, e.g. 360 corresponds to the angular
                            resolution being split into 360 parts, i.e. one part has a size of 1 degree []

        Some detection limits related to the cost function J with positivity constraints as described in LIFE II section 3.2:
        - 'fivesigmathreshold':     float, 5 sigma certainty []
        - 'foursigmathreshold':     float, 4 sigma certainty []
        - 'threesigmathreshold':    float, 3 sigma certainty []
        - 'twosigmathreshold':      float, 2 sigma certainty []
        '''
    '''
    def __init__(self,
                 bus,
                 planet_number,
                 mu=0):
        self.bus = bus
        self.inst = bus.modules['inst']
        self.exozodi = bus.modules['exo']
        self.localzodi = bus.modules['local']
        self.transm = bus.modules['transm']
        self.L = self.bus.data.inst['wl_bins'].size
        self.min_wl = self.bus.data.inst['wl_bin_edges'][0]
        self.max_wl = self.bus.data.inst['wl_bin_edges'][-1]
        self.wl_bins = self.bus.data.inst['wl_bins']
        self.wl_bin_widths = self.bus.data.inst['wl_bin_widths']
        # ToDo question: The image size was 256; I changed it to 512 to make it consistent with the old version in options.py line 109
        self.image_size = self.bus.data.options.other['image_size']
        self.radial_ang_px = int(self.image_size / 2)
        self.hfov = self.bus.data.inst['hfov']

        self.mu = mu
        self.planet_number = planet_number
        self.single_data_row = bus.data.catalog.iloc[planet_number]
        # ToDo: Think about if this is valid
        self.hfov_cost = self.single_data_row['angsep'] * 1.5 / 3600 / 180 * np.pi

        # this is the position of the planet on the polar map; can be chosen arbitrarily, although if !=0 the transition map must be rolled, see old lifesim
        self.planet_azimuth = 0
        self.n_steps = 360

        # Get the detection limit for J
        self.fivesigmathreshold = get_detection_threshold(L=self.L, sigma=5)
        '''

    def get_B_C(self, signals, T):
        '''
        This functions calculates two matrices that aid in the calculation of the cost function J and the estimated
        planet flux based on the received signal and the transmission functions. See Appendix B of LIFE II for an
        explanation of the exact calculations

        :param signals: np.ndarray of shape (L,n_steps) containing the received signals in [photons]
        :param T: np.ndarray of shape (L,radial_ang_px,n_steps) containing the tm_chop transmission factor at each pixel
                on the image []

        :return B: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                as described in Appendix B of LIFE II []
        :return C: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                and the cost function as described in Appendix B of LIFE II []
        '''

        # Get dimensions of transmission map
        (n_l, n_r, n_p) = T.shape

        # Get variance of received signals
        var = np.var(signals, axis=1, ddof=1)

        # Calculate B; in this step, the time series is included by the np.repeat function
        B_vec = (T ** 2).sum(axis=-1) / var[:, np.newaxis]
        B = np.repeat(B_vec[:, :, np.newaxis], n_p, axis=2)

        # Take T twice back to back
        T_exp = np.tile(T, 2)

        # Calculate C; also here, the time series is included
        C = np.empty((n_l, n_r, n_p // 2))
        for i in range(n_p // 2):
            T_i = T_exp[:, :, n_p - i: 2 * n_p - i]
            C[:, :, i] = np.einsum("ij,ikj->ik", signals, T_i)

        # Use antisymmetry of C to speed up calculation
        C = np.concatenate((C, -C), axis=2)
        C = (C.T / var).T

        return B, C

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
                                              hfov=self.hfov,
                                              image_size=self.image_size
                                              )
        '''
        #Calculate the transmission map
        _, _, _, _, tm_chop = self.transm.transmission_map(map_selection=['tm_chop'],
                                                           direct_mode=True,
                                                           d_alpha=d_alpha,
                                                           d_beta=d_beta,
                                                           hfov=self.hfov,
                                                           image_size=self.image_size)
                                                           '''

        # Plot the resulting transmission map for the first wl bin
        if (plot == True):
            plt.contour(tm_chop[0, :, :])
            plt.show()

        # Normalize the transmission map to instrument performance (required for compatability with functions written for
        # the old lifesim version
        self.tm_chop = tm_chop * self.single_data_row['int_time'] / self.n_steps * self.data.inst[
            'telescope_area'] * self.data.inst['eff_tot'] * self.wl_bin_widths[:, np.newaxis, np.newaxis] * 10 ** 6

        # Calculate the matrices B and C
        self.B, self.C = self.get_B_C(signals, self.tm_chop)

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

    def get_F_estimate(self, B, C, mu=0):
        '''
        This function calculates the estimated flux received by the planet based on the matrices B and C which are
        given as input. See LIFE II Appendix B for an explanation of the calculations

        :param B: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                as described in Appendix B of LIFE II []
        :param C: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                and the cost function as described in Appendix B of LIFE II []
        :param mu: float, regularization parameter []

        :return F: np.ndarray of shape (L,radial_ang_px); estimated flux received by the planet for each wl bin at every
                pixel in the image in [photons]
        :return F_pos: np.ndarray of shape (L,radial_ang_px); positive part of estimated flux received by the planet for
                each wl bin at every pixel in the image in [photons]
        '''

        # Get dimensions of the matrix
        (n_l, n_r, n_p) = C.shape

        # If there is no regularization, the calculation can be simplified
        if mu == 0:
            # ToDo question: Is this element-wise division?
            F = C / B
            F_pos = np.where(F >= 0, F, 0)

        # If regularization is included, F is calculated according to B.6, additionally requiring calc. of the D matrix
        else:
            try:
                # Calculate D^2
                # ToDo: Question: In the old version, D_sqr was divided by wl_bin_widths.mean**4, no idea why. It produces BS here
                Dsqr_mat = self.get_Dsqr_mat(n_l)  # / self.bus.data.inst['wl_bin_widths'].mean() ** 4

                B_diags = np.array([np.diag(B[:, i, 0]) for i in range(n_r)])  # B is phi-independent
                # print(B_diags.shape)
                # print(Dsqr_mat.shape)

                # Calculate the inverse of (B+mu*D^2)
                S = B_diags + mu * Dsqr_mat
                # print(S.shape)
                Sinv = np.linalg.inv(S)

                # Calculate F and F_pos
                F = np.einsum("rlm,mrp ->lrp", Sinv, C)

                F_pos = np.empty_like(F)

                for r in range(n_r):  # r-range
                    for p in range(n_p):  # phi-range
                        # nnls: F = argmin_F(abs(S*F-C))
                        F_pos[:, r, p] = sp.optimize.nnls(S[r], C[:, r, p], maxiter=200)[0]

            # For some values of mu, (B+mu*D^2) can be singular. In this case, calculate F without regularization and
            # print a corresponding warning
            except:
                print('Warning: singular matrix obtained for this value of mu; F is calculated for mu=0')
                F = C / B
                F_pos = np.where(F >= 0, F, 0)

        return F, F_pos

    def cost_func_MAP(self, mu=0, plot_maps=False):
        # ToDo this is from the old new cost_func_MAP function
        # ToDo comment this function
        '''
        This function calculates the estimated planet flux and the cost function based on the transmission map, the
        received signal and (if applicable) the regularization parameter. See LIFE II Appendix B for a derivation
        of the exact calcualtions

        :param mu: float, regularization parameter []
        :param plot_maps: boolean, determines whether to plot the cost map

        :return Js: np.ndarray of shape (L,radial_ang_px,n_steps); cost function calculated at each wavelength for every
                    pixel on the image []
        :return Jmaxs: np.ndarray of size L; maximum cost function value (of all pixels) at every wavelength
        :return theta_maxs: np.ndarray of shape (L,2); (r,phi) coordinates of the maximum cost function value at every
                    wavelength in [radial pixel,azimuthal pixel]
        :return rs: np.ndarray of size L; r coordinate at every wavelength of the maximum cost function value in [pixel]
        :return phis: np.ndarray of size L; phi coordinate at every wavelength of the maximum cost function value in [pixel]
        :return Fps: np.ndarray of shape (l,radial_ang_px,n_steps); contains the estimated planet flux received at each
                    wavelength at every pixel in the image in [photons]
        :return Fp_poss: np.ndarray of shape (l,radial_ang_px,n_steps); contains the positive part of the estimated
                    planet flux received at each wavelength at every pixel in the image in [photons]
        :return sigmas: np.ndarray of shape (l,radial_ang_px,n_steps); contains the uncertainty of the estimated planet
                    flux received at each wavelength at every pixel in the image in [photons]
        '''

        F, F_pos = self.get_F_estimate(self.B, self.C, mu=mu)

        Js = []
        for wl in range(self.L):
            J_wl = F_pos[wl] * self.C[wl]
            Js.append(J_wl)
        Js = np.array(Js)

        # ToDo Question: According to LIFE II (B.8), this should have a minus sign. That fucks it up though
        J = (F_pos * self.C).sum(axis=0)
        self.J = J

        theta_max = np.unravel_index(np.argmax(J, axis=None), J.shape)
        (r, p) = theta_max

        Fp_est = F[:, r, p]
        Fp_est_pos = F_pos[:, r, p]
        sigma_est = self.B[:, r, p] ** (-1 / 2)
        SNR_est = np.sum((Fp_est_pos / sigma_est) ** 2) ** (1 / 2)

        if (plot_maps == True):
            # ToDo: Remove scale factor BS
            j_map = pol_to_cart_map(J, self.image_size, scale_factor=1)
            plot_multi_map(j_map, "Cost Value",
                           self.hfov_cost * 3600000 * 180 / np.pi, "inferno")

        # print(sigma_est)
        return J, Js, theta_max, Fp_est, Fp_est_pos, sigma_est, SNR_est

    def MC_spectrum_extraction(self, n_MC=100, plot=False, ideal=False):
        '''
        This is the core function of the ML_Extraction class that executes the Monte Carlo simulation of the signal
        extraction. In each MC run, the extracted spectrum, snr, sigma, planet positions and cost function values are
        calculated based on the parameters in the bus catalog and random noise generation.

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

        :param n_MC: integer, number of MC runs to perform
        :param plot: boolean; determines whether to show plots throughout the runs (only advised for small n_MC)
        :param ideal: boolean; if True, no noise is included in the extraction

        :return extracted spectra: np.ndarray of shape (n_MC,L); contains the extracted spectra per wavelength bin for
                    each of the MC runs in [photons]
        :return extracted snrs: np.ndarray of size n_MC; contains the extracted snrs for each of the MC runs over the
                    entire wavelength range []
        :return extracted sigmas: np.ndarray of shape (n_MC,L); contains the sigmas for the extracted spectra
                    per wavelength bin for each of the MC runs in [photons]
        :return rss: np.ndarray of size n_MC; contains the extracted angular separations for each of the MC runs
                    in [arcsec]
        :return: phis: np.ndarray of size n_MC; contains the extracted azimuthal positions on the image for each of
                    the MC runs in [degrees]
        :return extracted_Jmaxs_wl: np.ndarray of shape (n_MC,L) containing the maximum values of J at every wavelength
                    for every MC run []
        :return extracted Jmaxs: np.ndarray of size n_MC containing the maximum values of J over all wavelengths
                    for every MC run []
        :return Ts: np.ndarray of size n_MC; extracted planet temperatures of every MC run in [K]
        :return Ts_sigma: np.ndarray of size n_MC; uncertainties of the extracted planet temperatures of every MC run in [K]
        :return Rs: np.ndarray of size n_MC; extracted planet radii of every MC run in [K]
        :return Rs_sigma: np.ndarray of size n_MC; uncertainties of the extracted planet radii of every MC run in [K]
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

        for n in range(n_MC):
            if (plot == True):
                print('MC run:', n)

            # Get the signals (with and without noise) for the specified constellation
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
                                                               extraction_mode=True
                                                               )

            '''
            # Get the signals (with and without noise) for the specified constellation
            self.signals, self.ideal_signals = self.inst.get_signal(temp_s=self.single_data_row['temp_s'],
                                                                    radius_s=self.single_data_row['radius_s'],
                                                                    distance_s=self.single_data_row['distance_s'],
                                                                    lat_s=self.single_data_row['lat'],
                                                                    z=self.single_data_row['z'],
                                                                    angsep=self.single_data_row['angsep'],
                                                                    flux_planet_spectrum=[self.wl_bins * u.meter,
                                                                                          self.single_data_row[
                                                                                              'planet_flux_use'][
                                                                                              0] / 3600 /
                                                                                          self.bus.data.inst[
                                                                                              'telescope_area'] /
                                                                                          self.bus.data.inst[
                                                                                              'eff_tot'] / self.wl_bin_widths * u.photon / u.second / (
                                                                                                      u.meter ** 3)],
                                                                    integration_time=self.single_data_row['int_time'],
                                                                    phi_n=self.n_steps,
                                                                    extraction_mode=True
                                                                    )
                                                                    '''

            # Create the transmission maps and the auxiliary matrices B&C
            if (ideal == True):
                self.cost_func(signals=self.ideal_signals, plot=plot)
            else:
                self.cost_func(signals=self.signals, plot=plot)

            # Get the extracted signals and cost functions
            J, Js, theta_max, Fp_est, Fp_est_pos, sigma_est, SNR_est = self.cost_func_MAP(mu=self.mu, plot_maps=plot)

            # Calculate the maximum of the cost function and its position
            Jmax = J.max()
            (r, phi) = theta_max

            # Calculate the best fit temperature and radius along with the corresponding uncertainties
            popt, pcov, perr = self.get_T_R_estimate(Fp_est, sigma_est, plot_flux=plot, plot_BB=plot,
                                                     plot_snr=plot)

            # Add the quantities from this MC run to the final list
            extracted_spectra.append(Fp_est)
            extracted_snrs.append(SNR_est)
            extracted_sigmas.append(sigma_est)
            extracted_Jmaxs.append(Jmax)

            rss.append(2 * r * self.hfov_cost / self.image_size * 180 * 3600 / np.pi)
            phiss.append(phi * self.n_steps / 360)

            Ts.append(popt[0])
            Rs.append(popt[1])
            Ts_sigma.append(perr[0])
            Rs_sigma.append(perr[1])

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


        # Plot the extracted spectra and sigmas along with the true blackbody function
        if (plot == True):
            sigma_est_mean = extracted_sigmas.mean(axis=0)
            Fp_est_mean = extracted_spectra.mean(axis=0)

            Fp_BB = None

            # ToDo Question: We don't save snr per wavelength in the new version.. --> Leave out option?
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

        return extracted_spectra, extracted_snrs, extracted_sigmas, extracted_Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma


    def main_parameter_extraction(self, n_MC=100, mu=0, plot=False, ideal=False, single_planet_mode=False, planet_number=0, save_mode=True):
        '''
        Description

        :param n_MC:
        :param mu:
        :return:
        '''

        self.n_planets = len(self.data.catalog.index)
        self.L = self.data.inst['wl_bins'].size
        self.min_wl = self.data.inst['wl_bin_edges'][0]
        self.max_wl = self.data.inst['wl_bin_edges'][-1]
        self.wl_bins = self.data.inst['wl_bins']
        self.wl_bin_widths = self.data.inst['wl_bin_widths']
        # ToDo question: The image size was 256; I changed it to 512 to make it consistent with the old version in options.py line 109
        self.image_size = self.data.options.other['image_size']
        self.radial_ang_px = int(self.image_size / 2)
        self.hfov = self.data.inst['hfov']

        self.mu = mu
        self.n_MC = n_MC

        # this is the position of the planet on the polar map; can be chosen arbitrarily, although if !=0 the transition map must be rolled, see old lifesim
        self.planet_azimuth = 0
        self.n_steps = 360


        if (single_planet_mode==True):
            self.planet_number = planet_number
            self.single_data_row = self.data.catalog.iloc[planet_number]
            # ToDo: Think about if this is valid
            self.hfov_cost = self.single_data_row['angsep'] * 1.5 / 3600 / 180 * np.pi

            spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma = self.MC_spectrum_extraction(n_MC=n_MC, plot=plot, ideal=ideal)

            return spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma


        else:

            # create lists to store extracted data
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

            for i in range(self.n_planets):
                self.planet_number = i
                self.single_data_row = self.data.catalog.iloc[i]
                self.hfov_cost = self.single_data_row['angsep'] * 1.5 / 3600 / 180 * np.pi

                spectra, snrs, sigmas, Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma = self.MC_spectrum_extraction(n_MC=self.n_MC, plot=plot, ideal=ideal)

                # store the data in the lists
                extracted_spectra.append(spectra)
                extracted_snrs.append(snrs)
                extracted_sigmas.append(sigmas)
                extracted_Jmaxs.append(Jmaxs)
                extracted_rss.append(rss)
                extracted_phiss.append(phiss)
                extracted_Ts.append(Ts)
                extracted_Ts_sigma.append(Ts_sigma)
                extracted_Rs.append(Rs)
                extracted_Rs_sigma.append(Rs_sigma)


            # add the data to the bus catalog
            self.data.catalog['extracted_spectra'] = extracted_spectra
            self.data.catalog['extracted_snrs'] = extracted_snrs
            self.data.catalog['extracted_sigmas'] = extracted_sigmas
            self.data.catalog['extracted_Jmaxs'] = extracted_Jmaxs
            self.data.catalog['extracted_rss'] = extracted_rss
            self.data.catalog['extracted_phiss'] = extracted_phiss
            self.data.catalog['extracted_Ts'] = extracted_Ts
            self.data.catalog['extracted_Ts_sigma'] = extracted_Ts_sigma
            self.data.catalog['extracted_Rs'] = extracted_Rs
            self.data.catalog['extracted_Rs_sigma'] = extracted_Rs_sigma

            #save the catalog
            if (save_mode==True):
                self.data.catalog.to_csv(path + '05_output_files/changeme.csv')

            print('We are all Patriots!!')

            return



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
                         plot_BB=False, plot_snr=False):
        '''
        This function obtains the planet temperature and radius by fitting the input spectra and sigmas to blackbody curves

        :param spectra: np.ndarray of size L; contains the extracted planet flux in each wl-bin in [photons]
        :param sigmas: np.ndarray of size L; contains the sigmas of the extracted planet flux in each wl-bin in [photons]
        :param p0: tuple; initial values for the optimization for T and R in [K,R_earth]
        :param absolute_sigma: boolean; If True, sigma is used in an absolute sense and the estimated covariance
                    reflects these absolute values. If False, only the relative magnitudes of the sigma values matter
        :param plot_flux: boolean; determines whether to plot at all. If True, automatically plots the spectra
        :param plot_BB: boolean; determines whether to plot the fitted blackbody curve. Only applicable if plot_flux=True
        :param plot_snr: boolean; determines whether to plot the SNR based on photon statistics for each wl-bin

        :return popt: np.ndarray of size 2; optimal values for (T,R) such that the sum of the squared residuals to the
                        blackbody curve is minimized. In [T,R_earth]
        :return pcov: np.ndarray of shape (2,2); covariance matrix of the obtained popt, a measure of uncertainty.
                        Dimensions [T^2] and [R_earth^2] on the first and second diagonal element
        :return perr: np.ndarray of size 2; standard deviation of the errors on T and R. In [T,R_earth]
        '''

        # Perform the fit of T and R and calculate the uncertainties
        popt, pcov = sp.optimize.curve_fit(self.BB_for_fit, self.wl_bins, spectra,
                                           sigma=sigmas, p0=p0, absolute_sigma=absolute_sigma)
        perr = np.sqrt(np.diag(pcov))

        # Plot the differnt quantities as defined
        if (plot_flux == True):
            '''
            print('--> true   Tp = %.4f K' % (self.single_data_row['temp_p']))
            print('--> fitted Tp = %.4f +- %.4f K' % (popt[0], perr[0]))
            print('--> true   Rp = %.4f Re' % (self.single_data_row['radius_p']))
            print('--> fitted Rp = %.4f +- %.4f Re' % (popt[1], perr[1]), "\n")
            '''

            # Get the blackbody curve for the estimated parameters
            Fp_fit = self.BB_for_fit(self.wl_bins, popt[0], popt[1])

            # ToDo Question: We don't save snr per wavelength in the new version.. --> Leave out option?
            if (plot_snr == True):
                snr_photon_stat = None

            else:
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



#This is the working one
class ML_Extraction1:
    '''
    The ML extraction class is a signal extraction module based on the maximum likelihood method described by Dannert et
    al. (2022) in LIFE II:  Signal simulation, signal extraction and fundamental exoplanet parameters from single epoch
    observations. The code is largely based on the code for an older Lifesim version by Maurice Ottiger, as described
    in his Master Thesis "Exoplanet Detection Yield Simulations and Signal Analysis Studies for the LIFE Mission" (2020)

    Parameters
    ------
    bus: lifesim bus object
        A bus with an already existing dataset (i.e. that already has simulation data)
    planet number: index (integer)
        Index of the planet in the catalog of the bus of which the signal is to be extracted
    mu: float
        Regularization parameter for the calculation of the cost function J as described in section 3.1 of LIFE II

    Attributes
    -----------
    Most of the attributes are simply abbreviations of the respective objects inherited from the bus. These are:
    - 'bus': This is just the bus
    - 'ins': Instrument module attached to the bus
    - 'exozodi': Exozodi module attached to the bus
    - 'localzodi': Localzodi module attached ot the bus
    - 'transm': Transmission module attached to the bus
    - 'L': Number of wavelength bins as given by the wavelength range and the resolution parameter R []
    - 'min_wl': minimum wavelength captured by the instrument [m]
    - 'max_wl': maximum wavelength captured by the instrument [m]
    - 'wl_bins': np.ndarray of size L, consists of all the wavelength bins (middle wavelength of each bin) in [m]
    - 'wl_bin_widths': np.ndarray of size L, consists of all the widths of the wavelength bins in [m]
    - 'image_size': integer, precision that the image can be resolved to in one dimension in [number of pixels]
    - 'radial_ang_px': integer, precision that the image can be resolved to in the radial coordinate in [number of pixels]
    - 'hfov': np.ndarray of size L, consists of the half field of views of the instrument in each wavelength bin in [radians]
    - 'mu': float, regularization parameter for the calculation of the cost function J as described above []
    - 'planet_number': index of the planet in the catalog of the bus of which the signal is to be extracted []
    - 'single_data_row': pd.series, catalog data from the bus corresponding to the planet described by planet_number

    Two attributes concerning the angular dimensioning of the image:
    - 'planet_azimuth': Angular position of the planet on the image. Contrary to the radial position, which is given by
                        the angular separation, this attribute is not known a priori from the simulation and thus set to
                        0 [degrees] for simplicity. Changing this or making it adaptable would require some modification
                        of the transmission functions to comply with the rest of the code
    - 'n_steps':        Resolution of the angular coordinate. Given as an integer, e.g. 360 corresponds to the angular
                        resolution being split into 360 parts, i.e. one part has a size of 1 degree []

    Some detection limits related to the cost function J with positivity constraints as described in LIFE II section 3.2:
    - 'fivesigmathreshold':     float, 5 sigma certainty []
    - 'foursigmathreshold':     float, 4 sigma certainty []
    - 'threesigmathreshold':    float, 3 sigma certainty []
    - 'twosigmathreshold':      float, 2 sigma certainty []
    '''

    def __init__(self,
                 bus,
                 planet_number,
                 mu=0):
        self.bus = bus
        self.inst = bus.modules['inst']
        self.exozodi = bus.modules['exo']
        self.localzodi = bus.modules['local']
        self.transm = bus.modules['transm']
        self.L = self.bus.data.inst['wl_bins'].size
        self.min_wl = self.bus.data.inst['wl_bin_edges'][0]
        self.max_wl = self.bus.data.inst['wl_bin_edges'][-1]
        self.wl_bins = self.bus.data.inst['wl_bins']
        self.wl_bin_widths = self.bus.data.inst['wl_bin_widths']
        #ToDo question: The image size was 256; I changed it to 512 to make it consistent with the old version in options.py line 109
        self.image_size = self.bus.data.options.other['image_size']
        self.radial_ang_px = int(self.image_size / 2)
        self.hfov = self.bus.data.inst['hfov']


        self.mu = mu
        self.planet_number = planet_number
        self.single_data_row = bus.data.catalog.iloc[planet_number]
        # ToDo: Think about if this is valid
        self.hfov_cost = self.single_data_row['angsep'] * 1.5/3600/180*np.pi

        #this is the position of the planet on the polar map; can be chosen arbitrarily, although if !=0 the transition map must be rolled, see old lifesim
        self.planet_azimuth = 0
        self.n_steps = 360

        #Get the detection limit for J
        self.fivesigmathreshold = get_detection_threshold(L=self.L, sigma=5)


    def get_B_C(self, signals, T):
        '''
        This functions calculates two matrices that aid in the calculation of the cost function J and the estimated
        planet flux based on the received signal and the transmission functions. See Appendix B of LIFE II for an
        explanation of the exact calculations

        :param signals: np.ndarray of shape (L,n_steps) containing the received signals in [photons]
        :param T: np.ndarray of shape (L,radial_ang_px,n_steps) containing the tm_chop transmission factor at each pixel
                on the image []

        :return B: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                as described in Appendix B of LIFE II []
        :return C: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                and the cost function as described in Appendix B of LIFE II []
        '''

        #Get dimensions of transmission map
        (n_l, n_r, n_p) = T.shape

        #Get variance of received signals
        var = np.var(signals, axis=1, ddof=1)

        #Calculate B; in this step, the time series is included by the np.repeat function
        B_vec = (T ** 2).sum(axis=-1) / var[:, np.newaxis]
        B = np.repeat(B_vec[:, :, np.newaxis], n_p, axis=2)

        #Take T twice back to back
        T_exp = np.tile(T, 2)

        #Calculate C; also here, the time series is included
        C = np.empty((n_l, n_r, n_p // 2))
        for i in range(n_p // 2):
            T_i = T_exp[:, :, n_p - i: 2 * n_p - i]
            C[:, :, i] = np.einsum("ij,ikj->ik", signals, T_i)

        # Use antisymmetry of C to speed up calculation
        C = np.concatenate((C, -C), axis=2)
        C = (C.T / var).T


        return B, C


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

        #Define the azimuthal coordinates
        phi_lin = np.linspace(0, 2 * np.pi, self.n_steps, endpoint=False)  # 1D array with azimuthal coordinates
        phi_mat = np.tile(phi_lin, (self.radial_ang_px, 1))  # 2D map with azimuthal coordinates

        #Define the radial coordinates
        theta_lin = np.linspace(0, 1, self.radial_ang_px, endpoint=False)  # 1D array with radial separation coord [radians]
        theta_lin += 1 * 0.5 / self.radial_ang_px  # shift the coordinates to the "center" of the bins
        theta_mat = np.tile(theta_lin, (self.n_steps, 1)).T # 2D array with radial separation coordinates [radians]


        #Define matrices which include the different wl parameters; the radial coordinate must be scaled by the hfov
        phi_mat_wl=np.zeros((self.L,self.radial_ang_px,self.n_steps))
        theta_mat_wl=np.zeros((self.L,self.radial_ang_px,self.n_steps))

        #define the inputs for the transmission function using the wl-independent hfov_cost
        for i in range(self.L):
            phi_mat_wl[i,:,:]=phi_mat
            theta_mat_wl[i, :, :] = theta_mat * self.hfov_cost


        #Define the input parameters for the transmission function
        d_alpha = theta_mat_wl*np.cos(phi_mat_wl)
        d_beta = theta_mat_wl*np.sin(phi_mat_wl)

        #Calcualte the transmission map
        _,_,_,_,tm_chop = self.transm.transmission_map(map_selection = ['tm_chop'],
                         direct_mode = True,
                         d_alpha = d_alpha,
                         d_beta = d_beta,
                         hfov = self.hfov,
                         image_size = self.image_size)

        #Plot the resulting transmission map for the first wl bin
        if (plot==True):
            plt.contour(tm_chop[0,:,:])
            plt.show()


        #Normalize the transmission map to instrument performance (required for compatability with functions written for
        # the old lifesim version
        self.tm_chop = tm_chop * self.single_data_row['int_time']/self.n_steps*self.bus.data.inst['telescope_area']*self.bus.data.inst['eff_tot']*self.wl_bin_widths[:,np.newaxis,np.newaxis]*10**6

        #Calculate the matrices B and C
        self.B, self.C = self.get_B_C(signals, self.tm_chop)

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


    def get_F_estimate(self, B, C, mu=0):
        '''
        This function calculates the estimated flux received by the planet based on the matrices B and C which are
        given as input. See LIFE II Appendix B for an explanation of the calculations

        :param B: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                as described in Appendix B of LIFE II []
        :param C: np.ndarray of shape (L,radial_ang_px,n_steps). matrix used for the calcuation of the estimated flux
                and the cost function as described in Appendix B of LIFE II []
        :param mu: float, regularization parameter []

        :return F: np.ndarray of shape (L,radial_ang_px); estimated flux received by the planet for each wl bin at every
                pixel in the image in [photons]
        :return F_pos: np.ndarray of shape (L,radial_ang_px); positive part of estimated flux received by the planet for
                each wl bin at every pixel in the image in [photons]
        '''

        #Get dimensions of the matrix
        (n_l, n_r, n_p) = C.shape

        #If there is no regularization, the calculation can be simplified
        if mu == 0:
            #ToDo question: Is this element-wise division?
            F = C / B
            F_pos = np.where(F >= 0, F, 0)

        #If regularization is included, F is calculated according to B.6, additionally requiring calc. of the D matrix
        else:
            try:
                #Calculate D^2
                #ToDo: Question: In the old version, D_sqr was divided by wl_bin_widths.mean**4, no idea why. It produces BS here
                Dsqr_mat = self.get_Dsqr_mat(n_l) #/ self.bus.data.inst['wl_bin_widths'].mean() ** 4

                B_diags = np.array([np.diag(B[:, i, 0]) for i in range(n_r)])  # B is phi-independent
                #print(B_diags.shape)
                #print(Dsqr_mat.shape)

                #Calculate the inverse of (B+mu*D^2)
                S = B_diags + mu * Dsqr_mat
                #print(S.shape)
                Sinv = np.linalg.inv(S)

                #Calculate F and F_pos
                F = np.einsum("rlm,mrp ->lrp", Sinv, C)

                F_pos = np.empty_like(F)

                for r in range(n_r):  # r-range
                    for p in range(n_p):  # phi-range
                        # nnls: F = argmin_F(abs(S*F-C))
                        F_pos[:, r, p] = sp.optimize.nnls(S[r], C[:, r, p], maxiter=200)[0]

            #For some values of mu, (B+mu*D^2) can be singular. In this case, calculate F without regularization and
            # print a corresponding warning
            except:
                print('Warning: singular matrix obtained for this value of mu; F is calculated for mu=0')
                F = C / B
                F_pos = np.where(F >= 0, F, 0)

        return F, F_pos



    def cost_func_MAP(self, mu=0, plot_maps=False):
        #ToDo this is from the old new cost_func_MAP function
        #ToDo comment this function
        '''
        This function calculates the estimated planet flux and the cost function based on the transmission map, the
        received signal and (if applicable) the regularization parameter. See LIFE II Appendix B for a derivation
        of the exact calcualtions

        :param mu: float, regularization parameter []
        :param plot_maps: boolean, determines whether to plot the cost map

        :return Js: np.ndarray of shape (L,radial_ang_px,n_steps); cost function calculated at each wavelength for every
                    pixel on the image []
        :return Jmaxs: np.ndarray of size L; maximum cost function value (of all pixels) at every wavelength
        :return theta_maxs: np.ndarray of shape (L,2); (r,phi) coordinates of the maximum cost function value at every
                    wavelength in [radial pixel,azimuthal pixel]
        :return rs: np.ndarray of size L; r coordinate at every wavelength of the maximum cost function value in [pixel]
        :return phis: np.ndarray of size L; phi coordinate at every wavelength of the maximum cost function value in [pixel]
        :return Fps: np.ndarray of shape (l,radial_ang_px,n_steps); contains the estimated planet flux received at each
                    wavelength at every pixel in the image in [photons]
        :return Fp_poss: np.ndarray of shape (l,radial_ang_px,n_steps); contains the positive part of the estimated
                    planet flux received at each wavelength at every pixel in the image in [photons]
        :return sigmas: np.ndarray of shape (l,radial_ang_px,n_steps); contains the uncertainty of the estimated planet
                    flux received at each wavelength at every pixel in the image in [photons]
        '''

        F, F_pos = self.get_F_estimate(self.B, self.C, mu=mu)

        Js = []
        for wl in range(self.L):
            J_wl = F_pos[wl] * self.C[wl]
            Js.append(J_wl)
        Js = np.array(Js)

        #ToDo Question: According to LIFE II (B.8), this should have a minus sign. That fucks it up though
        J = (F_pos * self.C).sum(axis=0)
        self.J = J


        theta_max = np.unravel_index(np.argmax(J, axis=None), J.shape)
        (r, p) = theta_max

        Fp_est = F[:, r, p]
        Fp_est_pos = F_pos[:, r, p]
        sigma_est = self.B[:, r, p] ** (-1 / 2)
        SNR_est = np.sum((Fp_est_pos / sigma_est) ** 2) ** (1 / 2)

        if (plot_maps == True):
            #ToDo: Remove scale factor BS
            j_map = pol_to_cart_map(J, self.image_size,scale_factor=1)
            plot_multi_map(j_map, "Cost Value",
                                        self.hfov_cost*3600000*180/np.pi, "inferno")

        #print(sigma_est)
        return J, Js, theta_max, Fp_est, Fp_est_pos, sigma_est, SNR_est



    def MC_spectrum_extraction(self, n_MC=100, plot=False, ideal=False):
        '''
        This is the core function of the ML_Extraction class that executes the Monte Carlo simulation of the signal
        extraction. In each MC run, the extracted spectrum, snr, sigma, planet positions and cost function values are
        calculated based on the parameters in the bus catalog and random noise generation.

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

        :param n_MC: integer, number of MC runs to perform
        :param plot: boolean; determines whether to show plots throughout the runs (only advised for small n_MC)
        :param ideal: boolean; if True, no noise is included in the extraction

        :return extracted spectra: np.ndarray of shape (n_MC,L); contains the extracted spectra per wavelength bin for
                    each of the MC runs in [photons]
        :return extracted snrs: np.ndarray of size n_MC; contains the extracted snrs for each of the MC runs over the
                    entire wavelength range []
        :return extracted sigmas: np.ndarray of shape (n_MC,L); contains the sigmas for the extracted spectra
                    per wavelength bin for each of the MC runs in [photons]
        :return rss: np.ndarray of size n_MC; contains the extracted angular separations for each of the MC runs
                    in [arcsec]
        :return: phis: np.ndarray of size n_MC; contains the extracted azimuthal positions on the image for each of
                    the MC runs in [degrees]
        :return extracted_Jmaxs_wl: np.ndarray of shape (n_MC,L) containing the maximum values of J at every wavelength
                    for every MC run []
        :return extracted Jmaxs: np.ndarray of size n_MC containing the maximum values of J over all wavelengths
                    for every MC run []
        :return Ts: np.ndarray of size n_MC; extracted planet temperatures of every MC run in [K]
        :return Ts_sigma: np.ndarray of size n_MC; uncertainties of the extracted planet temperatures of every MC run in [K]
        :return Rs: np.ndarray of size n_MC; extracted planet radii of every MC run in [K]
        :return Rs_sigma: np.ndarray of size n_MC; uncertainties of the extracted planet radii of every MC run in [K]
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


        for n in range(n_MC):
            if (plot==True):
                print('MC run:',n)

            #Get the signals (with and without noise) for the specified constellation
            self.signals, self.ideal_signals = self.inst.get_signal(temp_s=self.single_data_row['temp_s'],
                                                         radius_s=self.single_data_row['radius_s'],
                                                         distance_s=self.single_data_row['distance_s'],
                                                         lat_s=self.single_data_row['lat'],
                                                         z = self.single_data_row['z'],
                                                         angsep = self.single_data_row['angsep'],
                                                         flux_planet_spectrum = [self.wl_bins * u.meter,self.single_data_row['planet_flux_use'][0]/3600/self.bus.data.inst['telescope_area']/self.bus.data.inst['eff_tot']/self.wl_bin_widths* u.photon / u.second / (u.meter ** 3)],
                                                         integration_time = self.single_data_row['int_time'],
                                                         phi_n = self.n_steps,
                                                         extraction_mode=True
                                                         )


            #Create the transmission maps and the auxiliary matrices B&C
            if (ideal==True):
                self.cost_func(signals=self.ideal_signals, plot=plot)
            else:
                self.cost_func(signals=self.signals,plot=plot)


            #Get the extracted signals and cost functions
            J, Js, theta_max, Fp_est, Fp_est_pos, sigma_est, SNR_est = self.cost_func_MAP(mu=self.mu, plot_maps=plot)


            #Calculate the maximum of the cost function and its position
            Jmax = J.max()
            (r, phi) = theta_max


            #Calculate the best fit temperature and radius along with the corresponding uncertainties
            popt, pcov, perr = self.get_T_R_estimate(Fp_est, sigma_est, plot_flux=plot, plot_BB=plot,
                                                           plot_snr=plot)



            #Add the quantities from this MC run to the final list
            extracted_spectra.append(Fp_est)
            extracted_snrs.append(SNR_est)
            extracted_sigmas.append(sigma_est)
            extracted_Jmaxs.append(Jmax)

            rss.append(2 * r * self.hfov_cost/self.image_size * 180 * 3600 / np.pi)
            phiss.append(phi*self.n_steps/360)

            Ts.append(popt[0])
            Rs.append(popt[1])
            Ts_sigma.append(perr[0])
            Rs_sigma.append(perr[1])


        #Convert the final lists to arrays
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


        #Plot the extracted spectra and sigmas along with the true blackbody function
        if (plot == True):
            sigma_est_mean = extracted_sigmas.mean(axis=0)
            Fp_est_mean = extracted_spectra.mean(axis=0)

            Fp_BB = None

            #ToDo Question: We don't save snr per wavelength in the new version.. --> Leave out option?
            snr_photon_stat = None

            Fp = black_body(mode='planet',
                       bins=self.wl_bins,
                       width=self.wl_bin_widths,
                       temp=self.single_data_row['temp_p'],
                       radius=self.single_data_row['radius_p'],
                       distance=self.single_data_row['distance_s']) / self.wl_bin_widths*10**-6

            plot_planet_SED_and_SNR(self.wl_bins, Fp, Fp_est_mean, sigma_est_mean,
                                    self.min_wl,
                                    self.max_wl, Fp_BB=Fp_BB,
                                    snr_photon_stat=snr_photon_stat)


        return extracted_spectra, extracted_snrs, extracted_sigmas, extracted_Jmaxs, rss, phiss, Ts, Ts_sigma, Rs, Rs_sigma



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

        #Standard planck function
        fgamma = np.array(fact1 / (np.exp(fact2) - 1.0)) * 10**-6 * np.pi * ((Rp * radius_earth) / (dist_s * m_per_pc)) ** 2

        return fgamma


    def get_T_R_estimate(self, spectra, sigmas, p0=(300,1.), absolute_sigma=True, plot_flux=False,
                         plot_BB=False, plot_snr=False):
        '''
        This function obtains the planet temperature and radius by fitting the input spectra and sigmas to blackbody curves

        :param spectra: np.ndarray of size L; contains the extracted planet flux in each wl-bin in [photons]
        :param sigmas: np.ndarray of size L; contains the sigmas of the extracted planet flux in each wl-bin in [photons]
        :param p0: tuple; initial values for the optimization for T and R in [K,R_earth]
        :param absolute_sigma: boolean; If True, sigma is used in an absolute sense and the estimated covariance
                    reflects these absolute values. If False, only the relative magnitudes of the sigma values matter
        :param plot_flux: boolean; determines whether to plot at all. If True, automatically plots the spectra
        :param plot_BB: boolean; determines whether to plot the fitted blackbody curve. Only applicable if plot_flux=True
        :param plot_snr: boolean; determines whether to plot the SNR based on photon statistics for each wl-bin

        :return popt: np.ndarray of size 2; optimal values for (T,R) such that the sum of the squared residuals to the
                        blackbody curve is minimized. In [T,R_earth]
        :return pcov: np.ndarray of shape (2,2); covariance matrix of the obtained popt, a measure of uncertainty.
                        Dimensions [T^2] and [R_earth^2] on the first and second diagonal element
        :return perr: np.ndarray of size 2; standard deviation of the errors on T and R. In [T,R_earth]
        '''

        #Perform the fit of T and R and calculate the uncertainties
        popt, pcov = sp.optimize.curve_fit(self.BB_for_fit, self.wl_bins, spectra,
                                           sigma=sigmas, p0=p0, absolute_sigma=absolute_sigma)
        perr = np.sqrt(np.diag(pcov))

        #Plot the differnt quantities as defined
        if (plot_flux == True):
            '''
            print('--> true   Tp = %.4f K' % (self.single_data_row['temp_p']))
            print('--> fitted Tp = %.4f +- %.4f K' % (popt[0], perr[0]))
            print('--> true   Rp = %.4f Re' % (self.single_data_row['radius_p']))
            print('--> fitted Rp = %.4f +- %.4f Re' % (popt[1], perr[1]), "\n")
            '''

            #Get the blackbody curve for the estimated parameters
            Fp_fit = self.BB_for_fit(self.wl_bins, popt[0], popt[1])

            #ToDo Question: We don't save snr per wavelength in the new version.. --> Leave out option?
            if (plot_snr == True):
                snr_photon_stat = None

            else:
                snr_photon_stat = None

            if (plot_BB == True):
                Fp_BB = Fp_fit

            else:
                Fp_BB = None

            #Get the ideal blackbody flux of the planet
            Fp = black_body(mode='planet',
                       bins=self.wl_bins,
                       width=self.wl_bin_widths,
                       temp=self.single_data_row['temp_p'],
                       radius=self.single_data_row['radius_p'],
                       distance=self.single_data_row['distance_s']) / self.wl_bin_widths*10**-6

            plot_planet_SED_and_SNR(self.wl_bins, Fp, spectra, sigmas, self.min_wl,
                                    self.max_wl,Fp_BB=Fp_BB, snr_photon_stat=snr_photon_stat)

        return popt, pcov, perr