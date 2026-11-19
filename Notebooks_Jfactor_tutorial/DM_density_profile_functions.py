#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:07:09 2024

@author: jaumezuriaga


Script with all the functions needed for calculating the DM density profile parameters

"""

import numpy as np

from scipy.optimize import fsolve
from scipy.interpolate import interp1d

import mpmath as mp


#%% Conversion factors


class constants:
    
    # define constants as class astributes
    
    M_BH_GRAVITY_2021 = 4.297e6 # M_Sun, GRAVITY 2021
    M_BH_sup = M_BH_GRAVITY_2021 + 0.012e6 # M_Sun
    M_BH_inf = M_BH_GRAVITY_2021 - 0.012e6 # M_Sun
        
    eV_to_J = 1.60217662e-19 # 1 ev = 1.6e-19 J
    speed_light =  2.99792458e8 # m/s
    Newton_constant = 6.67428e-11 # N m / kg^2
    M_Sun = 1.98842e30 # kg
    M_Sun_to_GeV = M_Sun*speed_light**2 * (1/eV_to_J)*(1/1e9) # GeV

    kpc_to_cm = 3.08568e21 # 1 kpc = 3.086e21 cm
    kpc_to_m = 3.08568e19 # 1 kpc = 3.086e19 m
    M_Sun_per_kpc3_to_GeV_per_cm3 = M_Sun_to_GeV/kpc_to_cm**3 # (GeV/M_Sun)*(kpc/cm)**3

    R_S = 2.*Newton_constant*M_BH_GRAVITY_2021 * M_Sun/speed_light**2  # m
    R_S_kpc = R_S * (1/kpc_to_m) # kpc

    mass_SgrA =  M_BH_GRAVITY_2021 # Msun, GRAVITY 2021



#%%   DM density profile parameters


class parameters:


    def parameters_computation(params):
        """Compute halo, spike and saturation parameters in place and return params."""
        r_s = params["r_s"]
        rho_sun = params["rho_sun"]
        R_sun = params["R_sun"]
        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]
        R_sat = params["R_sat"]
        rho_sat = params["rho_sat"]
        slope_sat = params["slope_sat"]
        mode = params["Mode"]

        # These keys are required by the original input dictionary.
        _ = params["Arbitrary_Slope"], params["R_spike"], params["theta_spike"]

        uses_nfw_halo = mode in ("NFW", "Adiabatic_Spike", "Stars_Spike")
        if not uses_nfw_halo and mode not in ("Einasto", "Burkert"):
            return params

        # Normalize the halo to the local density, in solar masses per kpc³.
        rho_sun_msun_kpc3 = rho_sun / constants.M_Sun_per_kpc3_to_GeV_per_cm3
        if uses_nfw_halo:
            rho_s = (
                rho_sun_msun_kpc3
                * (R_sun / r_s) ** gamma
                * (1 + (R_sun / r_s) ** alpha) ** ((beta - gamma) / alpha)
            )
        elif mode == "Einasto":
            rho_s = rho_sun_msun_kpc3 * np.exp(
                (2 / alpha) * ((R_sun / r_s) ** alpha - 1)
            )
        else:  # Burkert
            rho_s = rho_sun_msun_kpc3 * (1 + R_sun / r_s) * (1 + (R_sun / r_s) ** 2)
        params["rho_s"] = rho_s

        # The NFW family converts the saturation density to the halo's units.
        if uses_nfw_halo:
            rho_sat = rho_sat / constants.M_Sun_per_kpc3_to_GeV_per_cm3
            params["rho_sat"] = rho_sat
            if mode in ("Adiabatic_Spike", "Stars_Spike"):
                parameters._compute_adiabatic_spike(params, R_sun, gamma, rho_sat)

        # A negative radius disables the plateau at every physical radius.
        if slope_sat is None:
            slope_sat = 0
            params["slope_sat"] = slope_sat

        if R_sat is None and rho_sat is None and slope_sat == 0:
            params["R_sat"] = -2
            params["rho_sat"] = 1
            print("No Saturation Plateau in the profile")

        # Einasto and Burkert always solve for the radius.
        if mode == "Einasto":

            def saturation_residual(r):
                return rho_s * np.exp((-2 / alpha) * ((r / r_s) ** alpha - 1)) - rho_sat

            params["R_sat"] = parameters._find_saturation_radius(saturation_residual)
        elif mode == "Burkert":
            params["R_sat"] = parameters._find_saturation_radius(
                lambda r: profiles.Burkert(r, params) - rho_sat
            )
        elif mode == "NFW" and R_sat is None and rho_sat is not None:
            params["R_sat"] = parameters._find_saturation_radius(
                lambda r: profiles.NFW_gen(r, params) - rho_sat
            )
        elif R_sat is not None and rho_sat is None:
            rho_sat = profiles.rho_DM(R_sat, params)
            params["rho_sat"] = rho_sat

        if mode == "Stars_Spike":
            parameters._compute_stars_spike(params, rho_sat)

        return params

    @staticmethod
    def _find_saturation_radius(saturation_residual):
        """Find where the halo reaches saturation and apply the inner cutoff."""
        R_sat = fsolve(saturation_residual, 1e-10)[0]  # kpc
        minimum_radius = 2 * constants.R_S_kpc
        if R_sat <= minimum_radius:
            R_sat = minimum_radius
        return R_sat

    @staticmethod
    def _compute_adiabatic_spike(params, R_sun, gamma, rho_sat):
        # Preserve the original equality check for array-valued inputs.
        if params["rho_sat"] == None:
            raise ValueError(
                "rho_sat parameter not defined, needed for Adiabatic Spike or Stars_Spike"
            )

        R_spike = parameters.Adiabatic_Spike_calculator(params)  # kpc
        theta_spike = np.arctan(R_spike / R_sun) * 180 / np.pi  # degrees
        rho_halo_R_sp = profiles.NFW_gen(R_spike, params)
        gamma_spike = (9.0 - 2.0 * gamma) / (4.0 - gamma)
        R_ann = R_spike * (rho_halo_R_sp / rho_sat) ** (1 / gamma_spike)  # kpc

        params["R_spike"] = R_spike
        params["theta_spike"] = theta_spike
        params["rho_halo_R_sp"] = rho_halo_R_sp
        params["gamma_Adiabatic_Spike"] = gamma_spike
        params["R_ann"] = R_ann

    @staticmethod
    def _compute_stars_spike(params, rho_sat):
        R_h = 2e-3  # kpc
        gamma_stars = 1.5
        rho_h_norm = profiles.rho_adiabatic_spike(R_h, params)
        R_sat = (rho_sat / rho_h_norm) ** (-1 / gamma_stars) * R_h  # kpc

        params["rho_h_norm"] = rho_h_norm
        params["R_sat"] = R_sat
        params["gamma_stars"] = gamma_stars

        minimum_radius = 2 * constants.R_S_kpc
        if params["R_spike"] < R_h:
            R_h = minimum_radius
        params["R_h"] = R_h

        if R_sat < minimum_radius:
            params["R_sat"] = minimum_radius


    def Adiabatic_Spike_calculator(params):
    
        def alpha_gamma(gamma):
            if gamma < 0.05: # 0.2
                return 0.293 * gamma**(4/9)
            else:   
                array_gamma_for_alpha = 0.05*np.array([1,4,8,12,16,20,24,28,40])   
                alpha_gamma_array = np.array([0.293 * 0.05**(4/9), 0.120, 0.140, 0.142, 0.135, 0.122, 0.103, 0.0818, 0.0177])
                return interp1d(array_gamma_for_alpha,alpha_gamma_array)(gamma)
    
        r_s = params['r_s']
        rho_s = params['rho_s']
        gamma = params['gamma']
        
        R_spike = alpha_gamma(gamma)*r_s*(constants.mass_SgrA/(rho_s*r_s**3))**(1./(3.-gamma)) # kpc
    
        return R_spike # kpc


#%% DM density Halo profiles


class profiles:
    
    def rho_DM(r, params):
        Mode = params['Mode']
        
        if Mode == 'NFW':
            return profiles.rho_NFW_gen(r, params)
        
        elif Mode == 'Einasto':
            return profiles.rho_Einasto(r, params)
        
        elif Mode == 'Burkert':
            return profiles.rho_Burkert(r, params)
        
        elif Mode == 'Adiabatic_Spike':
            return profiles.rho_adiabatic_spike(r, params)
        
        elif Mode == 'Stars_Spike':
            return profiles.rho_stars_spike(r, params)
        
        elif Mode == 'Arbitrary_Spike':
            return profiles.rho_arbitrary_spike(r, params)
    
    
    ## NFW Zhao profile
    
    def NFW_gen(r, params): 
        
        r_s = params['r_s']
        rho_s = params['rho_s']
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']
        
        return rho_s /( (r/r_s)**gamma * (1 + (r/r_s)**alpha )**((beta-gamma)/alpha) )
    
    
    def rho_NFW_gen(r, params):
        R_vir = params['R_vir']
        R_sat = params['R_sat']
        rho_sat = params['rho_sat']
        slope_sat = params['slope_sat']
        
        
        return np.piecewise(r,[2.*constants.R_S_kpc > r,  (2.*constants.R_S_kpc <= r) & (r < R_sat), (R_sat <= r)  & (r < R_vir), (r > R_vir)],\
                                [lambda r: 0.,\
                                 lambda r: rho_sat * (r/R_sat)**(-slope_sat),\
                                 lambda r: profiles.NFW_gen(r, params),\
                                 lambda r: 0.])
            
    ## Einasto
    
    def Einasto(r, params):
        rho_s = params['rho_s']
        r_s = params['r_s']
        alpha = params['alpha']
        
        # numpy.frompyfunc(func, /, nin, nout, *[, identity])
        # Parameters:
        # func: An arbitrary Python function.
        # nin: The number of input arguments.
        # nout: The number of objects returned by func.
        # indetity: object, optional. The value to use for the identity attribute 
        #           of the resulting object. If specified, this is equivalent to setting the 
        #           underlying C identity field to PyUFunc_IdentityValue. If omitted, the
        #           identity is set to PyUFunc_None. Note that this is _not_ equivalent to
        #           setting the identity to None, which implies the operation is reorderable.
        exp_array = np.frompyfunc(mp.exp, 1, 1) 

        
        return rho_s * exp_array( (-2/alpha) * ((r/r_s)**alpha - 1)  )
    
    
    def rho_Einasto(r, params):
        R_vir = params['R_vir']
        R_sat = params['R_sat']
        rho_sat = params['rho_sat']
        slope_sat = params['slope_sat']


        return np.piecewise(r,[2.*constants.R_S_kpc > r,  (2.*constants.R_S_kpc <= r) & (r < R_sat), (R_sat <= r)  & (r < R_vir), (r > R_vir)],\
                                [lambda r: 0.,\
                                 lambda r: rho_sat * (r/R_sat)**(-slope_sat),\
                                 lambda r: profiles.Einasto(r, params),\
                                 lambda r: 0.])
    
    ### Burkert
    
    def Burkert(r, params): 
        
        r_s = params['r_s']
        rho_s = params['rho_s']
        
        return rho_s /( (1 + (r/r_s)) * (1 + (r/r_s)**2 ) )
    
    
    def rho_Burkert(r, params):
        R_vir = params['R_vir']
        R_sat = params['R_sat']
        rho_sat = params['rho_sat']
        slope_sat = params['slope_sat']
        
        
        return np.piecewise(r,[2.*constants.R_S_kpc > r,  (2.*constants.R_S_kpc <= r) & (r < R_sat), (R_sat <= r)  & (r < R_vir), (r > R_vir)],\
                                [lambda r: 0.,\
                                 lambda r: rho_sat * (r/R_sat)**(-slope_sat),\
                                 lambda r: profiles.Burkert(r, params),\
                                 lambda r: 0.])
    
    
    ## Adiabatic Spike profile
    
    def rho_adiabatic_spike(r, params):
        
        R_spike = params['R_spike']
        R_vir = params['R_vir']
        
    
        return np.piecewise(r,[2.*constants.R_S_kpc > r, (2.*constants.R_S_kpc <= r)  & (r < R_spike),(R_spike <= r) & (r < R_vir)],\
                                [lambda r: 0.,\
                                 lambda r: (profiles.rho_adiabatic_spike_sat(r, params)*profiles.rho_adiabatic_spike_inner(r, params)) / (profiles.rho_adiabatic_spike_inner(r, params) + profiles.rho_adiabatic_spike_sat(r, params)),\
                                 lambda r: profiles.NFW_gen(r, params),\
                                 lambda r: 0.])
    
    def rho_adiabatic_spike_inner(r, params):
        
        R_spike = params['R_spike']
        gamma_spike = params['gamma_Adiabatic_Spike']
        rho_halo_R_sp = params['rho_halo_R_sp']
        
        return rho_halo_R_sp*(1 - 2*constants.R_S_kpc/r)**3 * (R_spike/r)**(gamma_spike)
    
    def rho_adiabatic_spike_sat(r, params):
        
        rho_sat = params['rho_sat']
        slope_sat = params['slope_sat']
        R_ann = params['R_ann']
    
        return rho_sat*(r/R_ann)**(-slope_sat)
        
    
    
    ## Stars Spike profile
    
    def rho_stars_spike(r, params):
        R_h = params['R_h']
        gamma_stars = params['gamma_stars']
        R_spike = params['R_spike']
        R_vir = params['R_vir']
        rho_h_norm = params['rho_h_norm']
        R_sat = params['R_sat']
        rho_sat = params['rho_sat']
        slope_sat = params['slope_sat']
    
        return np.piecewise(r,[2.*constants.R_S_kpc > r, (2.*constants.R_S_kpc <= r) & (r < R_sat), (R_sat <= r) & (r < R_h), (R_h <= r) & (r < R_spike), (R_spike <= r) & (r < R_vir)],\
                                [lambda r: 0.,\
                                 lambda r: rho_sat * (r/R_sat)**(-slope_sat),\
                                 lambda r: rho_h_norm * (r/R_h)**(-gamma_stars),\
                                 lambda r: profiles.rho_adiabatic_spike(r, params),\
                                 lambda r: profiles.NFW_gen(r, params),\
                                 lambda r: 0.])
    
    
    
    ## Arbitrary Spike profile
    
    def rho_arbitrary_spike(r, params):
        rho_arbitrary_norm = params['rho_arbitrary_norm']
        Arbitrary_Slope = params['Arbitrary_Slope']
        R_spike = params['R_spike']
        R_vir = params['R_vir']
        R_sat = params['R_sat']
        rho_sat = params['rho_sat']
        slope_sat = params['slope_sat']
    
        return np.piecewise(r,[2.*constants.R_S_kpc > r, (2.*constants.R_S_kpc <= r) & (r < R_sat), (R_sat <= r) & (r < R_spike), (R_spike <= r) & (r < R_vir)],\
                                [lambda r: 0.,\
                                 lambda r: rho_sat * (r/R_sat)**(-slope_sat),\
                                 lambda r: rho_arbitrary_norm * (r/R_spike)**(-Arbitrary_Slope),\
                                 lambda r: profiles.NFW_gen(r, params),\
                                 lambda r: 0.])
    
    
    
