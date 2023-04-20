# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import statistics as st
import os
import matplotlib.pyplot as plt

import timeit

from modules.plotting import plotter
from modules.plotting import plotanalysis
from modules import transmission_generator as tm_gen
from modules import coordinate_transformations as trafo
from modules import constants
from modules.snr_calc import SnrCalc

from models.planet import Planet
from models.instrument import Instrument
from models.disk import Disk
from models.localzodi import Localzodi
from models.star import Star


'''
------------ measurement and analysis simulation ----------------
'''

class Measurement:
    def __init__(self,
                 options,
                 inst,
                 star=None,
                 planets=None,
                 n_steps=360*4):

        self.inst = inst
        self.options = options
        self.star = star
        self.exozodi = Disk(star=self.star,options=options,maspp=self.inst.maspp,maps=True)
        self.localzodi = Localzodi(model=options.lz_model)
        self.planets = planets
        
        #self.inst.adjust_bl_to_HZ(self.star, self.options)
        self.inst.add_transmission_maps(self.options, map_selection="tm3_only")
                
        self.i_p = 0
        self.n_steps = int(n_steps)
        

    def create_signals(self):
        inst = self.inst
        options= self.options
        
        mult_f = (options.t_tot/self.n_steps * inst.telescope_area *
                  inst.wl_bin_widths * inst.eff_tot)
        self.mult_f = mult_f
        
        N3sl = inst.get_stellar_leakage(self.star, options, tm="3")
        N3lz = inst.get_localzodi_leakage(self.localzodi, self.star, options, tm="3")
        N3ez = inst.get_exozodi_leakage(self.exozodi, options, tm="3")
        
        N3 = mult_f * (N3sl + N3lz + N3ez)
        N4 = N3 # equal for symmetric sources
        
        a3 = np.random.poisson(N3, (self.n_steps,len(inst.wl_bins))).T
        a4 = np.random.poisson(N4, (self.n_steps,len(inst.wl_bins))).T        
        
        Sp = []
        tms = []
        for i, planet in enumerate(self.planets):
            Fp = planet.fgamma(inst.wl_bins, inst.wl_bin_edges)
            Fp = np.expand_dims(Fp, axis=1)
            
            # planet.az_deg = 0 #np.random.randint(self.n_steps) #45
            
            tm_curves = tm_gen.transm_curve(bl=inst.bl,
                                            wl=inst.wl_bins,
                                            ang_sep_as=planet.ang_sep,
                                            phi_n=self.n_steps)
            
            tm_curves= np.roll(tm_curves, int(self.n_steps*planet.az_deg//360), axis=-1)
            tms.append(tm_curves * mult_f[:,np.newaxis])

            Sp.append(Fp * tm_curves * mult_f[:,np.newaxis])
            
        Sp = np.array(Sp)
        tms = np.array(tms)
        tms = tms.sum(axis=0)
        signals = Sp.sum(axis=0) + (a3 - a4)
        
        self.signals_ideal = Sp
        self.signals = signals
        
        return signals, tms
    
    def cross_corr(self, az_deg, signal, tm_curve):
        """
        Parameters
        ----------
        az_deg: float
            azimutal position to be evaluated.
        signal: 1D array
            planet signal used for cross correlation at a certain
            wavelength.
        tm_curve: 1D array
            transmission curve at a certain radial position
            and wavelength.
            
        Returns
        -------
        M: float.
            cross correlation of signal and template at
            given location and wavelength.
        """
        template = np.roll(tm_curve, int(self.n_steps*az_deg//360), axis=-1)
        
        signal2 = (signal**2).sum()
        template2 = (template**2).sum()
        
        if signal2*template2 == 0: #put correlation at center to zero
            M = 0
        else:
            M = (signal*template).sum()/np.sqrt(signal2*template2)
            
        return M
    
    def dirty_map(self, wl_bin, hfov_mas, signal):
        """
        Parameters
        ----------
        wl_bin: int
            index of the wavelength bin under consideration.
        hfov_mas: float
            half field-of-view in mas.
        signal: 1D array
            planet signal at wavelength bin wl_bin.
    
        Returns
        -------
        dirty_map: 2D array
            map of cross correlation of signal and template
            in polar coordinates at one wavelength.
        """
        r = self.options.image_size//2
        phi = self.n_steps
    
        hfov = hfov_mas/1000 #hfov in arcsec
    
        rs = np.linspace(0,hfov,r,endpoint=False)
        phis = np.linspace(0,360,phi,endpoint=False)
    
        dirty_map = np.zeros((r, phi))
    
        for i in enumerate(rs):
            tm_curve = tm_gen.transm_curve(bl=self.inst.bl,
                                        wl=self.inst.wl_bins[wl_bin],
                                        ang_sep_as=i[1],
                                        phi_n=self.n_steps)
            for j in enumerate(phis):
                dirty_map[i[0]][j[0]] = self.cross_corr(j[1],signal,tm_curve)
            
        return dirty_map
    
    def plot_dirty_map(self, hfov_mas, wl_bins=[0], integrate=True, ideal=False, plot=True):
        """
        Parameters
        ----------
        hfov_mas: float
            half field-of-view in mas.
        wl_bins: list
            indices of the wavelength bins to integrate
            (only used if integrate=False).
        integrate: bool
            if True, integrate over all wavelength bins.
        ideal: bool
            if True, assume noiseless planet signal.
        plot: bool
            if True, print planet positions and plot.
            
        Returns
        -------
        dm: 2D array
            map of cross correlation of signal and template
            (dirty map) in polar coordinates, integrated over
            all wavelengths.
        dms: 3D array
            dirty maps at all considered wavelengths.
        r_mas: float
            estimated radial planet position in mas.
        phi_deg: float
            estimated azimutal planet position in deg.
        """
        self.create_signals()
        signals = self.signals
        ideals = self.signals_ideal.sum(axis=0)
        if ideal:
            signals = ideals
        
        dms = []
        if integrate:
            n_wl = len(self.inst.wl_bins)
            wl_bins = list(range(0,n_wl))
        
            for i in wl_bins:
                print('\r Processing wavelength bin',str(i+1),'of',str(len(wl_bins)),'...',end='')
                dm = self.dirty_map(i, hfov_mas, signals[i])
                dms.append(dm)
        else:
            for i in wl_bins:
                print('\r Processing wavelength bin',str(i),'...',end='')
                dm = self.dirty_map(i, hfov_mas, signals[i])
                dms.append(dm)
                
        dms = np.array(dms)    
        dms_pos = np.maximum(dms,0)
        dm = np.sqrt((dms_pos**2).sum(axis=0)/len(dms_pos))
        
        theta_max = np.unravel_index(np.argmax(dm, axis=None), dm.shape)
        (r, p) = theta_max

        maspp = 2 * hfov_mas/ self.options.image_size
        
        r_mas = r*maspp
        phi_deg = 360* p / self.n_steps
        
        if plot:
            planet = self.planets[0]
            print("\n--> true planet position: "\
                    f"r = {planet.ang_sep*1000:.0f} mas, phi = {planet.az_deg:.0f} deg "\
                    "\n--> est. planet position: "\
                    f"r = {r_mas:.0f} mas, phi = {phi_deg:.0f} deg")
            if integrate:
                d_map = trafo.pol_to_cart_map(dm, self.options.image_size)
            elif len(wl_bins)==1:
                d_map = trafo.pol_to_cart_map(dms[0], self.options.image_size)
            else:
                d_map = trafo.pol_to_cart_map(dm, self.options.image_size)

            cbar_location = "right"
            fig, ax = plt.subplots(figsize=(5.8, 4.8),dpi=300)

            im=plt.imshow(d_map, cmap = "inferno", origin="lower", extent=[hfov_mas,-hfov_mas,-hfov_mas,hfov_mas])

            ax.set_xticks([-hfov_mas/2, 0, hfov_mas/2])
            ax.set_yticks([-hfov_mas/2, 0, hfov_mas/2])
        
            ax.tick_params(pad=1)
            plt.setp(ax.get_yticklabels(), rotation=90, va='center')
    
            plt.xlabel(r"$\Delta$RA [mas]")
            plt.ylabel(r"$\Delta$Dec [mas]")
                 
            cbar = plt.colorbar(im, pad=0.01)
            cbar.ax.set_ylim(d_map.min(), d_map.max())
            cbar.ax.set_ylabel("Cross correlation", fontsize=12)
  
            plt.show()
    
        return dm, dms, r_mas, phi_deg
    
    def flux_estimate(self, r_mas, phi_deg, ideal = False):
        """
        Parameters
        ----------
        r_mas: float
            radial planet position in mas.
        phi_deg: float
            azimutal planet position in deg.
        ideal: bool
            if True, assume noiseless planet
            signal.
        
        Returns
        -------
        F_est: 1D array
            estimated flux spectrum of the planet.
        sigma_est: 1D array
            estimated standard deviation of flux at 
            each wavelength.
        """
        if ideal == False:
            signals = self.signals
        else:
            signals = self.signals_ideal.sum(axis=0)
            
        mult_f = self.mult_f
        
        tm_curves = tm_gen.transm_curve(bl=self.inst.bl,
                                        wl=self.inst.wl_bins,
                                        ang_sep_as=r_mas/1000,
                                        phi_n=self.n_steps)
        tm_curves = np.roll(tm_curves, int(self.n_steps*phi_deg//360), axis=-1)*mult_f[:,np.newaxis]
        tm2 = (tm_curves**2).sum(axis=-1) #normalisation factor (corresponds to B)

        F_est_rot = signals*tm_curves/tm2[:,np.newaxis]
        F_est = F_est_rot.sum(axis=-1)
        
        var = np.var(signals, axis=1, ddof=1)
        T2 = (tm_curves)**2
        sigma_est = np.sqrt(var/T2.sum(axis=1))
        
        return F_est, sigma_est
        
    """
    def MC_cross_corr(self, n_MC, hfov_mas, wl_bins=[0], integrate=True, ideal=False):
        
        SUBOPTIMAL VERSION.
        Creates signal and performs extraction over n_MC iterations.
        Prints mean and std-deviation of determined planet positions.
        
        Parameters
        ----------
        n_MC: int
            number of iterations.
        hfov_mas: float
            half field-of-view in mas.
        wl_bins: list
            indices of the wavelength bins to integrate
            (only used if integrate=False).
        integrate: bool
            if True, integrate over all wavelength bins.
        ideal: bool
            if True, assume noiseless planet signal.
        
        Returns
        -------
        r: float
            estimated radial planet position in mas.
        phi: float
            estimated azimutal planet position in deg.
          
        rs = []
        phis = []

        for i in range (n_MC):
            print("\r",str(i), end="  ")
            r, phi = self.plot_dirty_map(hfov_mas, wl_bins, integrate, ideal, plot=False)
            rs.append(r)
            phis.append(phi)
    
        rs=np.array(rs)
        r=rs.mean()
        var_r=np.var(rs)
        std_r=np.sqrt(var_r)
    
        phis=np.array(phis)
        phi=phis.mean()
        var_phi=np.var(phis)
        std_phi=np.sqrt(var_phi)

        print('\n Estimated r: %.1f +- %.1f mas'%(r,std_r))
        print('\n Estimated phi: %.1f +- %.1f deg'%(phi,std_phi))
        
        return r, phi
    """
    
    def MC_cross_corr(self, n_MC, hfov_mas, wl_bins=[0], integrate=True, ideal=False):
        """
        Parameters
        ----------
        n_MC: int
            number of iterations.
        hfov_mas: float
            half field-of-view in mas.
        wl_bins: list
            indices of the wavelength bins to integrate
            (only used if integrate=False).
        integrate: bool
            if True, integrate over all wavelength bins.
        ideal: bool
            if True, assume noiseless planet signal.
            
        Returns
        -------
        rs: 1D array
            extracted radial planet positions in mas
            from all n_MC runs.
        phis: 1D array
            extracted azimutal planet positions in deg
            from all n_MC runs.
        rss: 2D array
            rs for all wavelengths separately.
        phiss: 2D array
            phis for all wavelengths separately.
        F_ests: 2D array
            estimated flux spectra of the planet from
            all n_MC runs.
        sigmas: 2D array
            standard deviations of F_ests at every wavelength.
        SNRs: 1D array
            estimated SNR from all n_MC runs.
        cc_maxs: 1D array
            maximal cross-correlation value of dirty map
            from all n_MC runs.
        """
        
        r = self.options.image_size//2
        phi = self.n_steps
    
        hfov = hfov_mas/1000 #hfov in arcsec
        maspp = hfov_mas/r
    
        rs = np.linspace(0,hfov,r,endpoint=False)
        phis = np.linspace(0,360,phi,endpoint=False)
        
        planet = self.planets[0]

        #create all template functions in advance since they are the same for each MC run
        if integrate:
            n_wl = len(self.inst.wl_bins)
            wl_bins = list(range(0,n_wl))

        templates = np.ndarray((len(wl_bins), r, phi, phi))
        for wl in enumerate(wl_bins):
            print('\r Generating templates at wavelength',str(wl[0]+1),'of',str(len(wl_bins)),'...',end='')
            for i in enumerate(rs):
                tm_curve = tm_gen.transm_curve(bl=self.inst.bl,
                                                wl=self.inst.wl_bins[wl[1]],
                                                ang_sep_as=i[1],
                                                phi_n=self.n_steps)
                for j in enumerate(phis):
                    template = np.roll(tm_curve, int(self.n_steps*j[1]//360), axis=-1)
                    templates[wl[0]][i[0]][j[0]] = template
        
        #extraction of planet position, spectrum & std-dev, snr and cost function maximum n_MC times
        rs = []
        phis = []
        rss = []
        phiss = []
        F_ests = []
        sigmas = []
        SNRS = []
        cc_maxs = []
        print('\n')
        for i in range(n_MC):
            print('\r Performing MC run',str(i+1),'of',str(n_MC),'...',end='')
            signals, tms = self.create_signals()
    
            if ideal:
                ideals = self.signals_ideal.sum(axis=0)
                signals = ideals
    
            dms = []
            for wl in enumerate(wl_bins):
                signal = signals[wl[1]]
                signal2 = (signal**2).sum()
            
                dirty_map = np.zeros((r,phi))
                for i in range(r):
                    template2 = (templates[wl[0]][i][0]**2).sum()
                    for j in range(phi):
                        #dirty_map[i][j] = (signal*templates[wl[0]][i][j]*self.mult_f[wl[0]]).sum()
                        if signal2*template2 == 0:
                            dirty_map[i][j] = 0
                        else:
                            dirty_map[i][j] = (signal*templates[wl[0]][i][j]).sum()/np.sqrt(signal2*template2)
                
                dms.append(dirty_map)
            
            dms = np.array(dms)
            dms_pos = np.maximum(dms,0)
            
            Rs = []
            Phis = []
            for i in range(len(wl_bins)):
                dms_pos_i = dms_pos[i]
                #extraction of planet position at different wavelengths
                (R_i, Phi_i) = np.unravel_index(np.argmax(dms_pos_i, axis=None), dms_pos_i.shape)
                Rs.append(R_i)
                Phis.append(Phi_i)
            Rs = np.array(Rs)
            Phis=np.array(Phis)
            Rs_mas = Rs*maspp
            Phis_deg = 360*Phis/self.n_steps
            rss.append(Rs_mas)
            phiss.append(Phis_deg)
            
            dm_MC = np.sqrt((dms_pos**2).sum(axis=0)/len(dms_pos))
            
            #extraction of planet position
            (R,Phi) = np.unravel_index(np.argmax(dm_MC, axis=None), dm_MC.shape)
            R_mas = R*maspp #convert to mas
            Phi_deg = 360*Phi/self.n_steps #convert to deg
            
            rs.append(R_mas)
            phis.append(Phi_deg)
            
            cc_max = dm_MC.max()
            cc_maxs.append(cc_max)
            """
            #Calculate cost function J2, find maximal value
            T = tm_gen.tms_chop_pol(image_size=self.options.image_size, hfov_mas=hfov_mas, small_baseline=self.inst.bl,
                                    wl=self.inst.wl_bins, phi_n=self.n_steps) #shape (n_wl,r,phi)
            T2_vec = (T**2).sum(axis=-1) #shape (n_wl,r)
            T2 = np.repeat(T2_vec[:, :, np.newaxis], phi, axis=2) #B, shape (n_wl,r,phi)
            var = np.var(signals, axis=1, ddof=1) #shape (n_wl)
            var = var[:,np.newaxis,np.newaxis] #shape (n_wl,1,1)
       
            Fp = dms/T2/self.mult_f[:,np.newaxis,np.newaxis] #C/B
            Fppos = np.where(Fp>=0, Fp, 0)
            
            J2 = ((Fppos)*(dms*self.mult_f[:,np.newaxis,np.newaxis]/var)).sum(axis=0) #shape (r,phi)
            Jmaxs.append(J2.max())
            """
            #extract planet flux spectrum, compute std-deviations of the spectrum 
            F_est, sigma_est = self.flux_estimate(R_mas,Phi_deg,ideal)
            F_ests.append(F_est)
            sigmas.append(sigma_est)
            F_est_pos = np.where(F_est>=0,F_est,0)
            
            #compute SNR
            SNR_est = np.sum((F_est_pos/sigma_est)**2)**(1/2)
            SNRS.append(SNR_est)
        
        rs = np.array(rs)
        phis = np.array(phis)
        rss = np.array(rss)
        phiss = np.array(phiss)
        F_ests = np.array(F_ests)
        sigmas = np.array(sigmas)
        SNRS = np.array(SNRS)
        cc_maxs = np.array(cc_maxs)
        
        return rs, phis, rss, phiss, F_ests, sigmas, SNRS, cc_maxs

    
    def cost_func(self, signals, image_size=256, hfov_mas=150):
        '''
        Ref:
            - Mugnier et al (2005): "Data Processing in Nulling Interf... "
            - Thiebaut & Mugnier (2005): "Maximum a posteriori planet
                                        detection and characterization
                                        with a nulling interferometer"
        '''
        self.image_size = int(image_size)
        self.hfov_mas = hfov_mas
        self.maspp = 2 * hfov_mas/ image_size  
        
        T = tm_gen.tms_chop_pol(image_size, hfov_mas, self.inst.bl, self.inst.wl_bins,  self.n_steps)
        self.T = (self.mult_f * T.T).T # normalize transmission map to instrument performance
       
        self.B, self.C = self.get_B_C(signals, self.T)
        pass
    

    def cost_func_MAP(self, mu=0, plot_maps=False):

        F, F_pos = self.get_F_estimate(self.B, self.C, mu=mu)
        
        Js = []
        for wl in range(len(self.inst.wl_bins)):
            J_wl = F_pos[wl]*self.C[wl]
            Js.append(J_wl)
        Js = np.array(Js)
          
        J =  (F_pos * self.C).sum(axis=0)
        self.J = J
        theta_max = np.unravel_index(np.argmax(J, axis=None), J.shape)
        
        (r, p) = theta_max

        Fp_est = F[:,r,p]
        Fp_est_pos = F_pos[:,r,p]
        
        sigma_est = self.B[:,r,p]**(-1/2)

        SNR_est = np.sum((Fp_est_pos/sigma_est)**2)**(1/2)

        if plot_maps:
            planet = self.planets[self.i_p]
            print("--> true planet position: "\
                  f"r = {planet.ang_sep*1000:.0f} mas, phi = {planet.az_deg:.0f} deg "\
                  "\n--> est. planet position: "\
                  f"r = {r*self.maspp:.0f} mas, phi = {360* p / self.n_steps:.0f} deg")
            
            j_map = trafo.pol_to_cart_map(J, self.image_size)
            
            plotanalysis.plot_multi_map(j_map, "Cost Value",
                                self.hfov_mas, "inferno", filename_post="cost_func")
        
        return J, Js, theta_max, Fp_est, Fp_est_pos, sigma_est, SNR_est
        
        
    def get_B_C(self, signals, T):        
        (n_l, n_r, n_p) = T.shape# number of measurement points in time/angle
        var = np.var(signals, axis=1, ddof=1)

        B_vec = (T**2).sum(axis=-1) / var[:,np.newaxis]
        B = np.repeat(B_vec[:, :, np.newaxis], n_p, axis=2)  
        
        T_exp = np.tile(T, 2)
    
        C = np.empty((n_l,n_r, n_p//2))
        for i in range(n_p//2):
            T_i = T_exp[:,:, n_p-i : 2*n_p-i]
            C[:,:,i] = np.einsum("ij,ikj->ik", signals, T_i) 
            
        C = np.concatenate((C,-C), axis=2) # use antisymmetry of C to speed up calculation
        C = (C.T / var).T 
        return B, C
    
        
    def get_Dsqr_mat(self, l):
        '''
        l : int, number of wavelegth bins
        '''
        dif_diag=-2 * np.diag(np.ones(l))
        dif_pre = 1 * np.diag(np.ones(l-1), 1)
        dif_post= 1 * np.diag(np.ones(l-1),-1)
          
        D_mat = dif_diag + dif_pre + dif_post
        D_mat[ 0, :2] = np.array([-1, 1])
        D_mat[-1,-2:] = np.array([ 1,-1])
        
        Dsqr_mat = np.dot(D_mat, D_mat)
        
        return Dsqr_mat
        
    def get_F_estimate(self, B, C, mu=0):
        
        (n_l, n_r, n_p) = C.shape
        
        if mu == 0:
            F = C / B
            F_pos = np.where(F>= 0, F, 0)
        
        else:        
            Dsqr_mat = self.get_Dsqr_mat(n_l) / self.inst.wl_bin_widths.mean() ** 4
        
            B_diags = np.array([np.diag(B[:,i,0]) for i in range(n_r)]) # B is phi-independent
        
            S = B_diags + mu * Dsqr_mat
            
            Sinv = np.linalg.inv(S)
                    
            F = np.einsum("rlm,mrp ->lrp", Sinv, C)
            
            F_pos = np.empty_like(F)
            
            for r in range(n_r): # r-range
                for p in range(n_p): # phi-range
                    #nnls: F = argmin_F(abs(S*F-C))
                    F_pos[:,r,p] = sp.optimize.nnls(S[r], C[:,r,p], maxiter=200)[0]
        
        return F, F_pos
            
        
    def get_T_R_estimate(self, Fp_est, sigma_est, i_p=0, plot_flux=False,
                         plot_BB=False, plot_snr=False, filename_post=None): 

        wl_bins, wl_bin_edges = self.inst.wl_bins, self.inst.wl_bin_edges
        
        popt, pcov = sp.optimize.curve_fit(self.BB_for_fit, wl_bins, Fp_est ,
                                        sigma=sigma_est,
                                        p0=(300, 1.),
                                        absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        
        if plot_flux:
            planet = self.planets[i_p]
            
            print('--> true   Tp = %.4f K'%(planet.Tp))
            print('--> fitted Tp = %.4f +- %.4f K'%(popt[0],perr[0]))
            print('--> true   Rp = %.4f Re'%(planet.Rp))
            print('--> fitted Rp = %.4f +- %.4f Re'%(popt[1],perr[1]),"\n")   
            
            Fp_fit = self.BB_for_fit(wl_bins, popt[0], popt[1])
            
            if plot_snr:
                snrcalc = SnrCalc(planet, self.options, self.inst)
                Fp, snr = snrcalc.predict_SNR(plot=False)
                snr_photon_stat = snr
                
            else:
                snr_photon_stat=None
                
            if plot_BB:
                Fp_BB = Fp_fit
                
            else:
                Fp_BB=None

            Fp = planet.fgamma(wl_bins, wl_bin_edges)
            
            plotter.plot_planet_SED_and_SNR(wl_bins, Fp, Fp_est, sigma_est, self.inst,
                                            Fp_BB=Fp_BB, snr_photon_stat=snr_photon_stat)
        
        return popt, pcov
        
    
    def cost_func_multi_planets(self, signals):
        
        Js, popts, pcovs = [], [], []

        N_p = len(self.planets)
        
        signals_reduced = signals - 0
        
        for i_p in range(N_p):
            self.i_p = i_p
            if i_p < N_p:
                # print("\nPlanet "+str(i_p+1)+":")
                plot_flux = True
                
            else:
                plot_flux = False
                   
            self.cost_func(signals_reduced, image_size=512, hfov_mas=200)    
            
            J, theta_max, Fp_est, Fp_est_pos, sigma_est, _ = self.cost_func_MAP(mu=0)
        
            transm_curve_est = np.roll(self.T, theta_max[1], axis=2)[:,theta_max[0],:]
            
            signal_est = (Fp_est * transm_curve_est.T).T
        
            popt, pcov = self.get_T_R_estimate(Fp_est, sigma_est, i_p=i_p, plot_flux=False)    
    
            popts.append(popt)
            pcovs.append(pcov)
            Js.append(J)
            
            signals_reduced -= np.array(signal_est)
    
        # print("\n...transforming and plotting images...")

        j_maps = [trafo.pol_to_cart_map(Js[i], 2*len(Js[i])) for i in range(len(Js))]

        # plotanalysis.plot_multi_map(j_maps, hfov_mas=self.hfov_mas,
        #                             map_type="Cost Value", show_detections=True,
        #                             #filename_post= "cost_func_multiplanet_no_smooth"
        #                             )
       
        # plotanalysis.plot_TpRp_ext(self.planets, popts, pcovs, save_fig=False)
        
        #self.statistical_analysis()
        self.i_p = 0
        return j_maps


    def BB_for_fit(self, wl, Tp, Rp):
        Ds = self.star.Ds
        fact1 = 2 * constants.c /(wl**4)
        fact2 = (constants.h * constants.c)/(constants.k * Tp * wl)
        fgamma = np.array(fact1/(np.exp(fact2)-1.0)) * 1e-6 *np.pi * (
            (Rp * constants.R_earth) / (Ds * constants.m_per_pc))**2
        return fgamma
    
    
    def MC_spectrum_extraction(self, hfov_mas, n_MC=100, plot=False, ideal=False):
        extracted_spectra = []
        extracted_snr = []
        extracted_sigma = []
        extracted_loc = []
        extracted_Jmaxs = []
        rss = []
        rssC = []
        phiss = []
        
        for n in range(n_MC):
            print("\r",str(n), end="  ")

            signals = self.create_signals()
            print('hi')
            print(signals_ideal[30,:])
            
            if ideal:
                self.cost_func(signals=self.signals_ideal.sum(axis=0), image_size=self.options.image_size, hfov_mas=hfov_mas)
            else:
                self.cost_func(signals=self.signals, image_size=self.options.image_size, hfov_mas=hfov_mas)
            
            J, Js, theta_max, Fp_est, Fp_est_pos, sigma_est, SNR_est  = self.cost_func_MAP(mu=0,plot_maps=False)
            Jmax = J.max()
            
            rs = []
            rsC = []
            phis = []
            for wl in range(len(self.inst.wl_bins)):
                theta = np.unravel_index(np.argmax(Js[wl], axis=None), Js[wl].shape)
                (r,phi) = theta
                rs.append(r)
                phis.append(phi)
                thetaC = np.unravel_index(np.argmax(self.C[wl], axis=None), self.C[wl].shape)
                (rC, phiC) = thetaC
                rsC.append(rC)
            rs = np.array(rs)
            rsC = np.array(rsC)
            phis = np.array(phis)
                        
            extracted_spectra.append(Fp_est)
            extracted_snr.append(SNR_est)
            extracted_sigma.append(sigma_est)
            extracted_loc.append(theta_max)
            extracted_Jmaxs.append(Jmax)
            
            rss.append(rs)
            rssC.append(rsC)
            phiss.append(phis)
                
        self.extracted_spectra = np.array(extracted_spectra) 
        self.extracted_snr = np.array(extracted_snr) 
        self.extracted_sigma = np.array(extracted_sigma) 
        self.extracted_loc = np.array(extracted_loc)
        extracted_Jmaxs = np.array(extracted_Jmaxs)
        rss = np.array(rss)
        rssC = np.array(rssC)
        phiss = np.array(phiss)
        
        if plot:
            sigma_est_mean = self.extracted_sigma.mean(axis=0)
            Fp_est_mean = self.extracted_spectra.mean(axis=0)
        
            planet = self.planets[self.i_p]

            snrcalc = SnrCalc(planet, self.options, self.inst)
            Fp, snr = snrcalc.predict_SNR(plot=False)
      
            snr_photon_stat = snr
            
            Fp = planet.fgamma(self.inst.wl_bins, self.inst.wl_bin_edges)
        
            plotter.plot_planet_SED_and_SNR(self.inst.wl_bins, Fp, Fp_est_mean, sigma_est_mean, self.inst,
                                        Fp_BB=None, snr_photon_stat=snr_photon_stat, filename="mc_flux_extraction")

        return self.extracted_spectra, self.extracted_snr, self.extracted_sigma, self.extracted_loc, rss, rssC, phiss, extracted_Jmaxs
            
        
    def example_simulation(self):
        signals = self.create_signals()
        plotter.plot_transm_sig(self.signals[25], self.signals_ideal[0][25], self.options)
        
        image_size=512
        hfov_mas = 200
        
        self.cost_func(signals, image_size=image_size, hfov_mas=hfov_mas)
        
        J, _, Fp_est, Fp_est_pos, sigma_est, _ = self.cost_func_MAP(mu=0, plot_maps=True)
        
        self.get_T_R_estimate(Fp_est, sigma_est, plot_flux=True, plot_snr=True)

        j1 = trafo.pol_to_cart_map((self.C**2 / self.B).sum(axis=0), image_size)
        j2 = trafo.pol_to_cart_map(J, image_size)
        
        J, _, Fp_est, Fp_est_pos, sigma_est, _ = self.cost_func_MAP(mu=5)
        
        self.get_T_R_estimate(Fp_est, sigma_est, plot_flux=True, plot_snr=True)
        
        j3 = trafo.pol_to_cart_map(J, image_size)
        plotanalysis.plot_multi_map([j2,j3], hfov_mas=hfov_mas, map_type="Cost Value",
                                    filename_post="cost_func_no_smooth_and_smooth")

    
        
        
    """
    Statistical analysis of values in cost maps to proof detection criterion.
    Taken and developed further from pacoASDI paper.
    """
    #Gauss-pdf und -cdf für cross-correlation approach
    def Gauss_pdf(self,x,a,b):
        y = a*np.exp(-x**2/b**2)
        return y
    
    def Gauss_cdf(self,x,b):
        y=(sp.special.erf(x/b)+1)/2
        return y

    def get_j2_pdf(self, j, L):
        fact = sp.special.factorial
        pdf = 0
        for l in range(L):
            pdf += fact(L) / (2**L*fact(l)*fact(L-l)) * sp.stats.chi2.pdf(j, L-l)
        return pdf
        
    def get_j2_cdf(self, j, L):
        fact = sp.special.factorial
        cdf =  1 / 2**L
        for l in range(0,L):
            cdf += fact(L) / (2**L*fact(l)*fact(L-l)) * sp.special.gammainc((L-l)/2, j/2)
        return cdf
    
    def statistical_analysis(self, L=None):
        if L is None:
            L = len(self.inst.wl_bins)
            
        j2 = self.J.flatten()
        j2 = j2[j2!=0]
        
        j1 =  (self.C**2 / self.B).sum(axis=0).flatten()

        eta = np.linspace(0, 300, int(1e5))
        
        eta_ind1 = np.searchsorted(sp.special.gammainc(L/2, eta/2), sp.stats.norm.cdf(5))        
        eta_ind2 = np.searchsorted(self.get_j2_cdf(eta, L), sp.stats.norm.cdf(5))
        
        eta1, eta2 = eta[[eta_ind1, eta_ind2]]
        
        print(f"J'': max value: = {j2.max():.1f}, detection threshold eta = {eta2:.1f} \
              \nJ':  max value: = {j1.max():.1f}, detection threshold eta = {eta1:.1f}")

        j_array = np.linspace(0, int(1.1* eta1), 101)

        pdf_j1 = sp.stats.chi2.pdf(j_array, L)
        pdf_j2 = self.get_j2_pdf(j_array, L = L)
        
        # ----------------              
        plt.figure()
        plt.hist(j1, bins=j_array, density=True,alpha=0.5, label="J' empirical", color="grey", zorder=0)
        plt.hist(j2, bins=j_array, density=True,alpha=0.5, label="J'' empirical", color="mediumblue", zorder=0)
        plt.plot(j_array, pdf_j1, label="J' theoretical", c="red", linewidth=2, zorder=1)
        plt.plot(j_array, pdf_j2, label="J'' theoretical", c="orange", linestyle="-", linewidth=2, zorder=0)
        
        plt.axvline(eta2, 0, 1, color="orange", linestyle="--", linewidth=2)
        plt.axvline(eta1, 0, 1, color="red", linestyle="--", linewidth=2)
        
        plt.grid()
        
        plt.legend(loc="upper right")
        plt.xlabel("Cost value J'/J''", fontsize=12)
        plt.ylabel("Normalized probability density p(J)", fontsize=12)
        filename = os.path.join("plots","analysis_maps", "statisitcal_dist_j"+".pdf")
        plt.savefig(filename, bbox_inches='tight')
        
        plt.show()
        
    