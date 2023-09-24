import numpy as np
from Extraction_auxiliary import *
from astropy import units as u
import matplotlib.pyplot as plt
import scipy as sp
from lifesim.util.radiation import black_body
from lifesim.core.modules import ExtractionModule
from tqdm import tqdm
import multiprocessing as multiprocessing
import mpmath as mp
from matplotlib.ticker import ScalarFormatter
import random as ran
import signal


class ML_Extraction(ExtractionModule):
    '''
    The ML extraction class is a signal extraction module based on the maximum likelihood method described by Dannert et
    al. (2022) in LIFE II:  Signal simulation, signal extraction and fundamental exoplanet parameters from single epoch
    observations. The code is based on the code for an older Lifesim version by Maurice Ottiger, as described in his
    Master Thesis "Exoplanet Detection Yield Simulations and Signal Analysis Studies for the LIFE Mission" (2020)
    '''

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Name of the module.
        """
        super().__init__(name=name)


    def get_B_C(self, signals):
        '''
        This functions calculates two matrices that aid in the calculation of the cost function J and the estimated
        planet flux based on the received signal and the transmission functions. See Appendix B of LIFE II for an
        explanation of the exact calculations

        Attributes
        --------------
        - self.B: np.ndarray of shape (L,radial_ang_px,n_steps); matrix used for the calculation of the estimated flux
                as described in Appendix B of LIFE II []
        - self.C: np.ndarray of shape (L,radial_ang_px,n_steps); matrix used for the calculation of the estimated flux
                and the cost function as described in Appendix B of LIFE II []
        '''

        # get dimensions of transmission map
        (n_l, n_r, n_p) = self.tm_chop.shape

        # get variance of received signals
        var = np.var(signals, axis=1, ddof=1)

        # calculate B; in this step, the time series is included by the np.repeat function
        B_vec = (self.tm_chop ** 2).sum(axis=-1) / var[:, np.newaxis]
        self.B = np.repeat(B_vec[:, :, np.newaxis], n_p, axis=2)

        # take T twice back to back
        T_exp = np.tile(self.tm_chop, 2)

        # calculate C; also here, the time series is included
        C = np.empty((n_l, n_r, n_p // 2))

        for i in range(n_p // 2):
            T_i = T_exp[:, :, n_p - i: 2 * n_p - i]
            C[:, :, i] = np.einsum("ij,ikj->ik", signals, T_i)

        # use antisymmetry of C to speed up calculation
        C = np.concatenate((C, -C), axis=2)
        self.C = (C.T / var).T

        return


    def get_transm_map(self):
        '''
        This function first calculates the transmission function by defining a polar coordinate system and then calling
        the corresponding function from the transmission module

        :param plot: boolean; determines whether to show plot of the transmission function
        '''

        # define the azimuthal coordinates
        phi_lin = np.linspace(0, 2 * np.pi, self.n_steps, endpoint=False)  # 1D array with azimuthal coordinates
        phi_mat = np.tile(phi_lin, (self.radial_ang_px, 1))  # 2D map with azimuthal coordinates

        # define the radial coordinates
        theta_lin = np.linspace(0, 1, self.radial_ang_px,
                                endpoint=False)  # 1D array with radial separation coord [radians]
        theta_lin += 1 * 0.5 / self.radial_ang_px  # shift the coordinates to the "center" of the bins
        theta_mat = np.tile(theta_lin, (self.n_steps, 1)).T  # 2D array with radial separation coordinates [radians]

        # define matrices which include the different wl parameters; the radial coordinate must be scaled by the hfov
        phi_mat_wl = np.zeros((self.L, self.radial_ang_px, self.n_steps))
        theta_mat_wl = np.zeros((self.L, self.radial_ang_px, self.n_steps))

        for i in range(self.L):
            phi_mat_wl[i, :, :] = phi_mat
            theta_mat_wl[i, :, :] = theta_mat * self.hfov_cost

        # define the input parameters for the transmission function
        d_alpha = theta_mat_wl * np.cos(phi_mat_wl)
        d_beta = theta_mat_wl * np.sin(phi_mat_wl)

        # calculate the transmission map
        _, _, _, _, tm_chop = self.run_socket(s_name='transmission',
                                              method='transmission_map',
                                              map_selection=['tm_chop'],
                                              direct_mode=True,
                                              d_alpha=d_alpha,
                                              d_beta=d_beta,
                                              hfov=self.hfov_cost,
                                              image_size=self.image_size
                                              )

        # normalize the transmission map to instrument performance (required for compatability with functions written
        #   for the old lifesim version
        self.tm_chop = tm_chop * self.single_data_row['int_time'] / self.n_steps * self.data.inst['telescope_area'] \
                       * self.data.inst['eff_tot'] * self.wl_bin_widths[:, np.newaxis, np.newaxis] * 10 ** 6

        return


    def get_F_estimate(self):
        '''
        This function calculates the estimated flux received by the planet based on the matrices B and C.
        See LIFE II Appendix B for an explanation of the calculations

        :param mu: float; regularization parameter []

        :return F: np.ndarray of shape (L,radial_ang_px); total estimated flux (planet plus noise) received for each
                    wl bin at every pixel in the image in [photons]
        :return F_pos: np.ndarray of shape (L,radial_ang_px); positive part of total estimated flux received for each
                    wl bin at every pixel in the image in [photons]
        '''

        # get dimensions of the matrix
        (n_l, n_r, n_p) = self.C.shape

        # if there is no regularization, the calculation can be simplified
        if (self.mu == 0):

            # calculation of the total estimated signal (planet signal plus noise)
            F = self.C / self.B
            F_pos = np.where(F >= 0, F, 0)


        # if regularization is included, F is calculated according to B.6, additionally requiring calculation of D
        else:
            try:
                # calculate D^2
                Dsqr_mat = get_Dsqr_mat(n_l)

                B_diags = np.array([np.diag(self.B[:, i, 0]) for i in range(n_r)])  # B is phi-independent

                # calculate the inverse of (B+mu*D^2)
                S = B_diags + self.mu * Dsqr_mat
                Sinv = np.linalg.inv(S)

                # calculate F and F_pos
                F = np.einsum("rlm,mrp ->lrp", Sinv, self.C)

                F_pos = np.empty_like(F)

                for r in range(n_r):
                    for p in range(n_p):
                        F_pos[:, r, p] = sp.optimize.nnls(S[r], self.C[:, r, p], maxiter=200)[0]

            # for some values of mu, (B+mu*D^2) can be singular. In this case, calculate F without regularization and
            #   print a corresponding warning
            except:
                print('Warning: singular matrix obtained for this value of mu; F is calculated for mu=0')
                F = self.C / self.B
                F_pos = np.where(F >= 0, F, 0)

        return F, F_pos

    # not required
    def single_spectrum_extraction(self):
        pass

    def timeout_handler(self):
        print('timeout triggered')
        return


    def FPF_test(self, planet_number=0, n_runs=1, n_processes=1, filepath=None):
        '''
        This function runs random noise simulations to determine how many false positives will be obtained. It is the
        equivalent of the main_parameter_extraction function in Extraction.py

        :param planet_number: integer; index of the planet in the catalog of the bus of which the signal is to be
                    extracted
        :param n_runs: int; number of runs to perform for each planet []
        :param n_processes: int; number of processes to run in parallel for multi-planet extraction (ignored if
                    single_planet_mode=True) []
        :param filepath: str; path to where the file will be saved
        '''

        # define attributes
        self.L = self.data.inst['wl_bins'].size
        self.min_wl = self.data.inst['wl_bin_edges'][0]
        self.max_wl = self.data.inst['wl_bin_edges'][-1]
        self.wl_bins = self.data.inst['wl_bins']
        self.wl_bin_widths = self.data.inst['wl_bin_widths']
        self.image_size = self.data.options.other['image_size']
        self.radial_ang_px = int(self.image_size / 2)
        self.hfov = self.data.inst['hfov']
        self.mu = 0
        self.filepath = filepath

        self.planet_azimuth = 0
        self.n_steps = 360

        if (n_runs == 1):
            self.plot = True
        else:
            self.plot = False

        self.single_data_row = self.data.catalog.iloc[planet_number]
        self.hfov_cost = self.single_data_row['angsep'] * 1.2 / 3600 / 180 * np.pi


        # create the transmission maps
        self.get_transm_map()

        # get signal (cancel the input signal to zero by multiplying by 0)
        self.noise = self.run_socket(s_name='instrument',
                                method='get_signal',
                                temp_s=self.single_data_row['temp_s'],
                                radius_s=self.single_data_row['radius_s'],
                                distance_s=self.single_data_row['distance_s'],
                                lat_s=self.single_data_row['lat'],
                                z=self.single_data_row['z'],
                                angsep=self.single_data_row['angsep'],
                                flux_planet_spectrum=[self.wl_bins * u.meter,
                                                      0 * self.single_data_row[
                                                          'planet_flux_use'][
                                                          0] / 3600 /
                                                      self.data.inst[
                                                          'telescope_area'] /
                                                      self.data.inst[
                                                          'eff_tot'] / self.wl_bin_widths
                                                      * u.photon / u.second /
                                                      (u.meter ** 3)],
                                integration_time=self.single_data_row['int_time'],
                                phi_n=self.n_steps)


        # begin multiprocessing part
        extracted_Jmaxs_tot = multiprocessing.Manager().list()
        extracted_clusters_tot = multiprocessing.Manager().list()
        n_planets_per_process = int(n_runs/n_processes)

        processes = []


        print('beginning random sampling')
        all_signals = []

        # create random signals
        for i in range(n_processes):
            signals_in_process = []
            for j in range(n_planets_per_process):
                signal = np.random.poisson(lam=self.noise) - np.random.poisson(lam=self.noise)
                signals_in_process.append(signal)
            all_signals.append(signals_in_process)
        print('random sampling completed')

        # execute the multiprocessing
        for i in range(n_processes):
            p = multiprocessing.Process(target=self.execute_multiprocessing, args=[all_signals[i], i,
                                                                   extracted_Jmaxs_tot, extracted_clusters_tot])
            processes.append(p)

            p.start()

        # wait for all processes to finish
        for process in processes:
            process.join(timeout=n_planets_per_process*2)

        print('saving file')

        extracted_Jmaxs_tot = np.array(extracted_Jmaxs_tot)
        extracted_clusters_tot = np.array(extracted_clusters_tot)

        # save results
        np.savetxt(filepath+'changeme_Jmaxs.csv', extracted_Jmaxs_tot, delimiter=',')
        np.savetxt(filepath + 'changeme_clusters.csv', extracted_clusters_tot, delimiter=',')
        print('files saved')

        print('Jmax runs completed')

        return


    def execute_multiprocessing(self, signals_in_process, n_process, extracted_Jmaxs_tot, extracted_clusters_tot):
        '''
        This function is called by each of the parallel running processes and executes the signal extraction for the
        planets in its range

        :param signals_in_process: np.array of size n_planets_per_process; contains the signals that the process
                                    should use [photons]
        :param n_process: int; indicates the process number [0, n_processes] (label-like)
        :param extracted_Jmaxs_tot: list object of the multiprocessing library; to be filled with the Jmax values
                                    extracted []
        :param: extracted_clusters_tot; list object of the multiprocessing library; to be filled with the number of
                                    cluster values extracted []
        '''

        print('Process #', n_process, ' started')

        for j in tqdm(range(len(signals_in_process))):

            single_signal = signals_in_process[j]

            # get B and C
            self.get_B_C(signals=single_signal)

            # get the estimated flux
            F, F_pos = self.get_F_estimate()

            # get the cost function J
            J = (F_pos * self.C).sum(axis=0)
            Jmax = np.max(J)

            j_map = pol_to_cart_map(J, self.image_size)
            clusters = get_clusters(j_map)

            if (self.plot == True):
                # plot the cost function heatmap
                plot_multi_map(j_map, "Cost Value", self.hfov_cost * 3600000 * 180 / np.pi,
                               "inferno", filename_post=None)

                print('Number of clusters:', clusters)

            extracted_Jmaxs_tot.append(Jmax)
            extracted_clusters_tot.append(clusters)

        print('Process #', n_process, ' finished')

        return


    def cluster_test(self, planet_numbers=[0], n_runs=1, filepath=None):
        '''
        This function calculates the number of clusters required to best fit the Jmax cdf function for a list of
        planets. It is the equivalent of the main_parameter_extraction function in Extraction.py

        :param planet_numbers: list of variable length; indices of the planets in the catalog of the bus of which the
                    algorithm is to be run
        :param n_runs: int; number of runs to perform for each planet []
        :param filepath: str; path to where the file will be saved
        '''

        # define attributes
        self.L = self.data.inst['wl_bins'].size
        self.min_wl = self.data.inst['wl_bin_edges'][0]
        self.max_wl = self.data.inst['wl_bin_edges'][-1]
        self.wl_bins = self.data.inst['wl_bins']
        self.wl_bin_widths = self.data.inst['wl_bin_widths']
        self.image_size = self.data.options.other['image_size']
        self.radial_ang_px = int(self.image_size / 2)
        self.hfov = self.data.inst['hfov']
        self.mu = 0
        self.filepath = filepath

        self.planet_azimuth = 0
        self.n_steps = 360

        self.n_runs = n_runs


        # begin multiprocessing part
        n_processes = len(planet_numbers)

        extracted_Jmaxs_tot = []
        extracted_clusters_tot = []
        extracted_angseps_tot = []

        processes = []
        events = []
        res_queue = multiprocessing.Queue()
        num_queue = multiprocessing.Queue()

        for i in range(n_processes):
            e = multiprocessing.Event()
            p = multiprocessing.Process(target=self.execute_cluster_multiprocessing, args=[planet_numbers[i],
                                                                                   res_queue, num_queue, e, i])
            p.start()
            events.append(e)
            processes.append(p)


        # wait for all processes to finish
        for event in events:
            event.wait()


        # get the results from the queues and store them in lists. The numbers list is used to keep track of what
        #   order the queues finished in
        results = []
        numbers = []

        for i in range(n_processes):
            result = res_queue.get()
            number = num_queue.get()
            results.append(result)
            numbers.append(number)


        # add the results to the main list in the correct order
        for i in range(n_processes):
            place_in_queue = int(numbers.index(i))

            extracted_Jmaxs_tot.append(results[place_in_queue][0])
            extracted_clusters_tot.append(results[place_in_queue][1])
            extracted_angseps_tot.append(results[place_in_queue][2])

        extracted_Jmaxs_tot = np.array(extracted_Jmaxs_tot).T
        extracted_clusters_tot = np.array(extracted_clusters_tot).T
        extracted_angseps_tot = np.concatenate((np.array(extracted_angseps_tot), np.array(extracted_angseps_tot)))

        data = np.concatenate((extracted_Jmaxs_tot, extracted_clusters_tot), axis=1)
        columns = [str(angsep) for angsep in extracted_angseps_tot]

        final_df = pd.DataFrame(data, columns=columns)

        final_df.to_csv(filepath+'changeme.csv', index=False)

        print('cluster extraction completed')

        return



    def execute_cluster_multiprocessing(self, planet_number, res, num, event, n_process):
        '''
        This function is called by each of the parallel running processes and executes the signal extraction for the
        planets in its range

        :param planet_number: integer; index of the planet in the catalog of the bus of which the signal is to be
                                        extracted. Only applicable if single_planet_mode=True
        :param res: multiprocessing.Queue; this is where the results are stored
        :param num: multiprocessing.Queue; keeps track of the order in which the processes fill up the results queue
        :param event: multiprocessing.Event; this object is 'set' as soon as the process is complete
        :param n_process: int; indicates the process number [0, n_processes] (label-like)
        '''

        print('Process #', n_process, ' started')

        self.single_data_row = self.data.catalog.iloc[planet_number]
        self.hfov_cost = self.single_data_row['angsep'] * 1.2 / 3600 / 180 * np.pi

        # create the transmission maps
        self.get_transm_map()

        # get signal (cancel the input signal to zero by multiplying by 0)
        self.noise = self.run_socket(s_name='instrument',
                                     method='get_signal',
                                     temp_s=self.single_data_row['temp_s'],
                                     radius_s=self.single_data_row['radius_s'],
                                     distance_s=self.single_data_row['distance_s'],
                                     lat_s=self.single_data_row['lat'],
                                     z=self.single_data_row['z'],
                                     angsep=self.single_data_row['angsep'],
                                     flux_planet_spectrum=[self.wl_bins * u.meter,
                                                           0 * self.single_data_row[
                                                               'planet_flux_use'][
                                                               0] / 3600 /
                                                           self.data.inst[
                                                               'telescope_area'] /
                                                           self.data.inst[
                                                               'eff_tot'] / self.wl_bin_widths
                                                           * u.photon / u.second /
                                                           (u.meter ** 3)],
                                     integration_time=self.single_data_row['int_time'],
                                     phi_n=self.n_steps)

        extracted_Jmaxs = []
        extracted_clusters = []

        for _ in tqdm(range(self.n_runs)):

            signal = np.random.poisson(lam=self.noise) - np.random.poisson(lam=self.noise)

            # get B and C
            self.get_B_C(signals=signal)

            # get the estimated flux
            F, F_pos = self.get_F_estimate()

            # get the cost function J
            J = (F_pos * self.C).sum(axis=0)
            Jmax = np.max(J)

            j_map = pol_to_cart_map(J, self.image_size)
            clusters = get_clusters(j_map)

            extracted_Jmaxs.append(Jmax)
            extracted_clusters.append(clusters)


        # add the process number and the results to the queue and set the event
        num.put(n_process)
        res.put([extracted_Jmaxs, extracted_clusters, self.single_data_row['angsep']])
        event.set()

        print('Process #', n_process, ' finished')

        return