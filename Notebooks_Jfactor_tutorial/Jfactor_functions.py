#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:22:24 2024

@author: jaumezuriaga


Script with all the functions needed for calculating precise J-factors

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mpmath as mp
mp.mp.dps = 35; mp.mp.pretty = True

from astropy.coordinates import angular_separation
import astropy.units as u

from scipy.integrate import simpson, trapezoid as trapz

# We import the script with all the density functions needed
import DM_density_profile_functions as density



#%%

class J_factor:
    
    
    def location_sampler(n_fi, g_lon, g_lat, fov_pixel, n_circles=int(100), circles_is_log = False, first_circle = 1e-12, 
                         center_halo = [0,0]):
        '''
        Point sampling inside the pixel
            
        Parameters
        ----------
        n_fi: int, number of samples in azimutal angle
        g_lon, g_lat: floats, in rad. Position of the center of the pixel
        fov_pixel: float, in rad. Size of the pixel. CLUMPY averages over a circle, so it has a radius defined.
        n_circles: concentric circles where we will compute the j-factors to average. Default is 100.
        circles_is_log: bool, logarithmic sampling of n_circles, default is False
        first_circle: float, in rad. First value of n_circle if  circles_is_log. Default is 1e-12 rad.
        center_halo: [float, float] in rad. Position [glon, glat] of the center of the halo. Default [0,0]
        '''
        
        
        alpha_center = angular_separation(center_halo[0]*u.rad, center_halo[1]*u.rad, g_lon*u.rad, g_lat*u.rad).value # rad
        if (n_circles == 0) and (n_fi == 0):
            return [alpha_center], [[g_lon], [g_lat]]
        
        else:
            if circles_is_log:
                radius_ncircles = 10**np.array(mp.linspace(mp.log10(first_circle), mp.log10(fov_pixel), n_circles))   # rad
            else:
                radius_ncircles = [(i+1)*fov_pixel/(n_circles) for i in range(n_circles)] # rad

            fi_points = mp.linspace(0, 2*mp.pi, n_fi) # rad
            
            alpha_sample = np.ones(len(fi_points))*alpha_center # rad, we put the center len(fi_points) times, needed in Solid_Angle_Integrator
            glon_sample = np.ones(len(fi_points))*g_lon # rad
            glat_sample =  np.ones(len(fi_points))*g_lat # rad
            fi_sample = fi_points # rad
            
            for radius_circle in radius_ncircles:
                x_points = np.array([radius_circle*mp.cos(fi) for fi in fi_points]) + g_lon
                y_points = np.array([radius_circle*mp.sin(fi) for fi in fi_points]) + g_lat
                alpha_points = angular_separation(center_halo[0]*u.rad, center_halo[1]*u.rad, x_points*u.rad, y_points*u.rad).value # rad
    
                fi_sample = np.concatenate((fi_sample, fi_points))
                alpha_sample = np.concatenate((alpha_sample, alpha_points))
                glon_sample = np.concatenate((glon_sample, x_points))
                glat_sample = np.concatenate((glat_sample, y_points))
    
    
            return alpha_sample, fi_sample, [glon_sample, glat_sample], fi_points # rad
        
     
    def Jfactor_to_GeV_cm(J_factor_integrated, is_annih = True):
        # from Msun2 kpc-5 to GeV2 cm-5
        # or from Msun kpc-2 to GeV cm-2
        if is_annih:
            return J_factor_integrated * (density.constants.M_Sun_to_GeV**2) / (density.constants.kpc_to_cm)**5
        else:
            return J_factor_integrated * (density.constants.M_Sun_to_GeV) / (density.constants.kpc_to_cm)**2

        
    def s_rlim_minus(angle, R_sun, R_lim):
        return R_sun*mp.cos(angle) - mp.sqrt(R_lim**2 - R_sun**2*mp.sin(angle)**2)
    
    def s_rlim_plus(angle, R_sun, R_lim):
        return R_sun*mp.cos(angle) + mp.sqrt(R_lim**2 - R_sun**2*mp.sin(angle)**2)


    def s_spike_min_calculator(angle, R_sun, R_spike):
        x_p = R_sun - mp.sqrt(R_spike**2 * (1 + mp.tan(angle)**2) - R_sun**2 * mp.tan(angle)**2)
        x_p = x_p / (1 + mp.tan(angle)**2)
        y_p = x_p * mp.tan(angle)
        s = mp.sqrt(y_p**2 + x_p**2)
        return s


    def s_spike_max_calculator(angle, R_sun, R_spike):
        x_p = R_sun + mp.sqrt(R_spike**2 * (1 + mp.tan(angle)**2) - R_sun**2 * mp.tan(angle)**2)
        x_p = x_p / (1 + mp.tan(angle)**2)
        y_p = x_p * mp.tan(angle)
        s = mp.sqrt(y_p**2 + x_p**2)
        return s
    
    
    def s_spike_med_calculator(angle, R_sun):
        return abs(R_sun*mp.cos(angle))


    def r(s_point, alpha_point, R_sun):
        # distances in kpc and rad
        
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

        # to evaluate mpmath cos, power and multiplication function must yield numpy numbers
        cos_array = np.frompyfunc(mp.cos, 1, 1) 
        pow_array = np.frompyfunc(mp.power, 2, 1) 
        multi_array = np.frompyfunc(mp.fmul, 2, 1) 
        R_sun_times_s_point = multi_array(R_sun, s_point)
        
        return pow_array(pow_array(s_point,2) + pow_array(R_sun,2) - 2*multi_array(R_sun_times_s_point, cos_array(alpha_point)), 0.5)
            

    def los_creator(angle, s_spike_min, s_spike_med, s_spike_max, s_rvir, n_los_inner, n_los_outer, centered = True):
        '''
        Here we are oversampling the l.o.s. This is so we make sure that it works for every kind of profile, no matter how steep it is.
        '''
        
        if s_spike_min != s_spike_max :
        
            # first array from sun to r_spike
            s_array_1 = s_spike_min - 10**np.array(mp.linspace(mp.log10(1e-10), mp.log10(s_spike_min), int(n_los_outer)))
            s_array_1 = s_array_1[::-1]
            # eliminate if first element is negative but small (could be becaouse how we define s_array_1)
            if s_array_1[0] < 0:
                if np.abs(s_array_1[0]) < 1e-7:
                    s_array_1 = s_array_1[1:]
                else:
                    print('-- WARNING l.o.s. first point not correctly sampled - ', angle, ': ', s_array_1[0])
                
            s_array_1 = np.append([0], s_array_1)
        
            # now from r_spike to r_med            
            #s_array_2 = np.array(mp.linspace(s_spike_min, s_spike_med, int(n_los_inner)))
            #s_array_2 = s_array_2[1::]

            # could have the same problem as s_array_1
            s_array_2 = s_spike_med - 10**(np.array(mp.linspace(mp.log10(1e-10), mp.log10(s_spike_med - s_spike_min), int(n_los_inner))))
            s_array_2 = s_array_2[::-1]
            s_array_2 = s_array_2[1::]

            # now from r_med to r_spike
            #s_array_3 = np.array(mp.linspace(s_spike_med, s_spike_max, int(n_los_inner)))
            #s_array_3 = s_array_3[1::]
            
            # could have the same problem as s_array_1
            s_array_3 = s_spike_med + 10**(np.array(mp.linspace(mp.log10(1e-10), mp.log10(s_spike_max-s_spike_med), int(n_los_inner))))
            s_array_3 = s_array_3[1::]

            # last, from r_spike to r_vir
            s_array_4 = s_spike_max + 10**np.array(mp.linspace(mp.log10(1e-8), mp.log10(s_rvir - s_spike_max), int(n_los_outer)))   
            s_array_4 = s_array_4[1::]   
            
            s_array = np.append(s_array_1, s_array_2)
            s_array = np.append(s_array, s_array_3)
            s_array = np.append(s_array, s_array_4)
        else:
            if centered:
                # first array from sun to gc
                #s_array_1 = 10**np.array(mp.linspace(mp.log10(1e-4), mp.log10(s_spike_med), int(n_los_outer)))
                s_array_1 = s_spike_med - 10**np.array(mp.linspace(mp.log10(1e-10), mp.log10(s_spike_med), int(n_los_outer)))
                s_array_1 = s_array_1[::-1]
                # Keep the observer as the first point despite round-off.
                if np.abs(s_array_1[0]) < 1e-7:
                    s_array_1[0] = 0
                elif s_array_1[0] < 0:
                    print('-- WARNING l.o.s. first point not correctly sampled - ', angle, ': ', s_array_1[0])
            
                # second array from gc to r_vir
                # could have the same problem as s_array_1
                s_array_2 = s_spike_med + 10**np.array(mp.linspace(mp.log10(1e-10), mp.log10(-s_spike_med + s_rvir), int(n_los_outer)))
                s_array_2 = s_array_2[1::]        
                s_array = np.append(s_array_1, s_array_2)
            else:
                #quizás esto debería ser lineal?
                # first array from sun to gc plane
                s_array_1 = 10**np.array(mp.linspace(mp.log10(1e-4), mp.log10(s_spike_med), int(n_los_outer)))
                s_array_1 = np.append([0], s_array_1)
                
                # second array from gc plane to r_vir
                s_array_2 = s_spike_med + 10**np.array(mp.linspace(mp.log10(1e-10), mp.log10(-s_spike_med + s_rvir), int(n_los_outer)))
                s_array_2 = s_array_2[1::]        
                s_array = np.append(s_array_1, s_array_2)

        # eliminate duplicates 0
        if s_array[0] == s_array[1]:
            s_array = s_array[1:]
          
        return s_array


    def step1_calculator(angle, angle_lim, R_sun, R_lim, fov, params, n_los_inner, n_los_outer, is_annih = True, centered = True, integrate_los = True):
        # return step1/rho_s**2
        s_spike_med = J_factor.s_spike_med_calculator(angle, R_sun)

        if angle < angle_lim:
            s_spike_min = J_factor.s_spike_min_calculator(angle, R_sun, R_lim)
            s_spike_max = J_factor.s_spike_max_calculator(angle, R_sun, R_lim)
            
        else: # possible oversampling of the data. Depends on how steep is the profile
            if centered:
                if fov < angle_lim:
                    R_fov = R_sun*mp.tan(fov)
                    s_spike_min = J_factor.s_spike_min_calculator(angle, R_sun, R_fov)
                    s_spike_max = J_factor.s_spike_max_calculator(angle, R_sun, R_fov)
                else:
                    s_spike_min = s_spike_max = 0
                # s_spike_max = s_spike_min = s_spike_med sheck si esto está bien!!!!!!!!!!!!!!!
            else:
                s_spike_min = s_spike_max = 0
        #if (s_spike_min == 0) and (s_spike_max == 0) and (s_spike_med < 0):
        #    print('WARNING l.o.s. not correctly sampled - ', angle, ': ', s_spike_min, s_spike_med, s_spike_max)
        if s_spike_min != s_spike_max:
            if (s_spike_med < s_spike_min) or (s_spike_max < s_spike_med):
               print('-- WARNING l.o.s. s_spike_lims not correctly sampled - ', angle, ': ', s_spike_min, s_spike_med, s_spike_max)
        if (s_spike_med < 0) or (s_spike_max < 0) or (s_spike_min < 0):
            print('-- WARNING l.o.s. s_spike_lims negative - ', angle, ': ', s_spike_min, s_spike_med, s_spike_max)

        R_vir = params['R_vir']
        s_rvir = J_factor.s_rlim_plus(angle, R_sun, R_vir)
            
        s_array = J_factor.los_creator(angle, s_spike_min, s_spike_med, s_spike_max, s_rvir, n_los_inner, n_los_outer, centered = centered)
        
        if not np.all(np.diff(s_array) > 0):
            print('-- WARNING l.o.s. not correctly sampled (doesnt grow monotically) - ', angle)
            print('Positions - ')
            positions  = np.where(np.diff(s_array) <= 0)[0]
            for pos in positions:
                print('      ', pos, ' - ', s_array[pos], s_array[pos+1])
            plt.figure()
            plt.plot(s_array)
            if s_spike_min > 0:
                plt.axhline(s_spike_min)
            if s_spike_max > 0:
                plt.axhline(s_spike_max)
            plt.axhline(s_spike_med)
            
            plt.yscale('log')
            plt.title(angle)
            plt.show()
            plt.close()
            
        if not (np.all(s_array >= 0)):
            print('-- WARNING l.o.s. not correctly sampled (negative numbers) - ', angle)
            print('Positions - ')
            positions  = np.where(s_array < 0)[0]
            for pos in positions:
                print('      ', pos, ' - ', s_array[pos])
            plt.figure()
            plt.plot(s_array)
            if s_spike_min > 0:
                plt.axhline(s_spike_min)
            if s_spike_max > 0:
                plt.axhline(s_spike_max)
            plt.axhline(s_spike_med)
            
            plt.yscale('log')
            plt.title(angle)
            plt.show()
            plt.close()

        # To integrate, although we have irregularly spaced data we can still use scipy.integrate.simpson,        
        rho_s = params['rho_s']
        if is_annih:
            rho_array_integral = density.profiles.rho_DM(J_factor.r(s_array, angle, R_sun), params)**2 / (rho_s**2)  # renormalization
        else:
            rho_array_integral = density.profiles.rho_DM(J_factor.r(s_array, angle, R_sun), params) / (rho_s)  # renormalization
            
        if integrate_los:
            # integral over l.o.s of (rho/rho_s)^2
            step1 = trapz(rho_array_integral, x = s_array)
            '''
            if step1 < 0:
                # sometimes the smapling of rho_array_integral is such that we have 1 value going to 0, or that in 1 point inchanges a lot of orders of magnitude
                # this transalates that the simpson integral is negative. Should we use trapz always????????
                # I think that this is a problem of simpson integral around narrow peaks. But how can we detect it more efficiently???
                step1 = trapz(rho_array_integral, x = s_array)
            '''
            return step1
        
        else:
            if is_annih:
                factor = rho_s
            else:
                factor = rho_s**2
                
            return rho_array_integral*factor, s_array




    def Solid_Angle_pixel(fov_pixel): # fov_pixel, rad
        return 2*mp.pi*(1 - mp.cos(fov_pixel)) # sr



    def Solid_Angle_Integrator(y_vals, fi_vals, alpha_vals, locations_vals, fi_array, save_fig_jfac=None):
        ''' Integrate functions in a solid_angle
        for each fi, we integrate in alpha
        fi_array are the fi_values not repeated
        each y_vals has a corresponding fi_vals and alpha_vals. fi_vals are repeated, each fi in fi_array appearing n times
        alpha_vals is not needed
        '''
        
        y_integrated_fi = []
        sin_array = np.frompyfunc(mp.sin, 1, 1)
        for fi in fi_array:
            positions = np.where(fi == fi_vals)[0]
            y_vals_fi = y_vals[positions]
            # alpha_vals_fi = alpha_vals[positions]
            glon_vals_fi = locations_vals[0][positions]
            glat_vals_fi = locations_vals[1][positions]
            glon_center = glon_vals_fi[0] # rad
            glat_center = glat_vals_fi[0] # rad
            # the first element of glon_vals_fi and glat_vals_fi corresponds to the center of the pixel
            distance_to_center = angular_separation(lon1 = glon_vals_fi*u.rad, lat1 = glat_vals_fi*u.rad, 
                                                    lon2 = glon_center*u.rad , lat2 = glat_center*u.rad).value # rad
            distance_to_center_sorted = np.sort(distance_to_center)
            positions_sorted = [np.where(distance == distance_to_center)[0][0] for distance in distance_to_center_sorted]
                        
            # alpha_vals_sorted = alpha_vals_fi[positions_sorted]
            y_vals_sorted = y_vals_fi[positions_sorted]
            y_integrated_fi.append(simpson(y_vals_sorted*sin_array(distance_to_center_sorted), x=distance_to_center_sorted))
            if save_fig_jfac is not None:
                plt.figure(1)
                plt.plot(distance_to_center_sorted, y_vals_sorted*sin_array(distance_to_center_sorted))
                #plt.xscale('log')
                #plt.yscale('log')
                plt.xlabel('Distance Center Pixel (rad)')
                plt.ylabel('step1*sin(alpha) (a.u.)')
                plt.title(str(np.round(np.float64(fi*180/np.pi), 2)) + ' deg')
                plt.grid(ls=':')
                plt.savefig(save_fig_jfac + 'step_1_pixel_' + str(np.round(np.float64(fi*180/np.pi), 2)) + '.pdf', format = 'pdf', bbox_inches = 'tight')
                plt.show()
                plt.close()
        
        y_integrated_solid_angle = simpson(y_integrated_fi, x=fi_array)
        if save_fig_jfac is not None:
            plt.figure(1)
            plt.plot(fi_array, y_integrated_fi)
            #plt.axhline(y_integrated_solid_angle, c = 'grey', ls = '--')
            if glon_center != 0 and glat_center != 0:
                fi_center = mp.atan(glat_center/glon_center) + mp.pi*(mp.atan(glat_center/glon_center) < 0)
                plt.axvline(fi_center, c = 'black', ls = '--')
                plt.axvline(fi_center + mp.pi, c = 'black', ls = '--')
            #plt.yscale('log')
            glon_center = np.float64(glon_center)
            glat_center = np.float64(glat_center)
            plt.title(str([np.round(glon_center*180/np.pi, 2), np.round(glat_center*180/np.pi, 2)]) + ' deg')
            plt.xlabel('Azimutal Angle (rad)')
            plt.ylabel('J-Factor (a.u.)')
            plt.grid(ls=':')
            plt.savefig(save_fig_jfac + 'J-factor_pixel_' + str([np.round(glon_center*180/np.pi, 2), np.round(glat_center*180/np.pi, 2)]) + '.pdf', format = 'pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()

        return y_integrated_solid_angle
        


    def j_factor_pixel_calculator(params, fov_pixel, location, n_los_outer = 100, n_los_inner = 500, 
                                  fov=None,  n_fi = 20, n_circles = 100, center_halo = [0,0],
                                  params_computation = False, per_sr = True, step1_return = False,
                                  save_fig_jfac = None, circles_is_log = True, first_circle = 1e-12):
        """Compute J-factor or J-factor_per_sr inside a pixel.

       This function allows to compute a j-factor for a given position in the sky 
       and averages over the size of the pixel (the pixel must be circular, like CLUMPY). 
       Params should correspond to a spherical DM density profile
    
       Default parameters are to assure, in general, convergence of the integral. In case of cuspier profiles than 
       Adibatic Spike, check convergende with n_los_outer, n_los_inner, n_fi, n_circles, first_circle, circles_is_log
       Parameters
       ----------
       params: dictionary
           List with all the parameters defined, see DM_density_profile_functions
           to see the definion of params.
       fov_pixel: float
           Size of the pixel where we want to compute the J-factor, in deg.
       location: tuple of floats or array like (len=2)
           Location of the pixel where we want to compute the J-factor.
           tuple or array like: (galactic longitude, galactic latitude), in deg.
       n_los_outer, n_los_inner: int, optional
           Number of steps for the l.o.s integrals: 2*n_los_outer + 2*n_los_inner.
           Default is n_los_outer = 100, n_los_inner = 500
       fov: float, optional
           Size of the whole computation, in deg. If None, default fov_pixel*1.01 deg greater than location.
           Default is None.
       n_fi: int, optional
           Number of points arms where the J-factor will be integrated in the pixel (i.e. azimutal sampling of the pixel). 
           Total number of points is n_fi * n_circles. Default is 20.
       n_circles: int, optional
           Number of circles where n_fi will be taken in. Total number of points is n_fi * n_circles. Default is 100. 
        center_halo: [float, float], optional.
            In deg. Position [glon, glat] of the center of the halo. Default [0,0]
       params_computation: bool, optional
           If we need to compute all of the parameters defined in params
           (via density.parameters.parameters_computation(params)). Default is False.
       per_sr: bool, optional
           If we want J-factor_per_sr or J-factor averaged per pixel.
           Default is true.
       step1_return: bool, optional
           If True returns the step1, angle_sample and alpha_sample values as well. Default is False.
      save_fig_jfac: str, optional
           If we want to save intermediate steps to double check dependencies and convergence. Defualt is None
      circles_is_log: bool, optional
           Distribution of n_circles in log scale or linear scale. Default is True. Very sensitive when considering cuspy profiles.
     first_circle: float, optional
           If circles_is_log is set to True, first_circle is the angle (radians) where the sampling begins. If circles_is_log is False,
           the parameter is ignored. Default is 1e-12 rad.
        """
        
        ## We need to integrate now in fov_int and s (l.o.s) at the same time, 
        ## so calculating numerically can be tricky
        
        ## In order to do that:
            # 1: create array of alpha_int around the pixel
            # 2: integrate for each alpha over the l.o.s
            # 3: integrate in alpha_int the final function


        # check if we need to compute the rest of parameters
        if params_computation:
            params = density.parameters.parameters_computation(params)
                
        
        # location of the pixel           
        g_lon = location[0] * mp.pi/180 # rad
        g_lat = location[1] * mp.pi/180 # rad
        
        # location of the halo center
        center_halo = np.array(center_halo) * mp.pi/180 # rad
        
        ## step 1: array of alpha_int around the pixel. 
        ## CLUMPY averages over the center of the pixel, so in this case we will do the same 
        ## to compute the J_factor_per_sr
        
        fov_pixel = fov_pixel * mp.pi / 180 # rad
        if fov is None:
            fov = angular_separation(g_lon*u.rad, g_lat*u.rad, center_halo[0]*u.rad, center_halo[1]*u.rad).value + 1.01*fov_pixel # rad
        else:
            fov = fov*mp.pi/180 # rad
            
        pixel_size = J_factor.Solid_Angle_pixel(fov_pixel) # sr
        
        n_fi = int(n_fi)
        n_circles = int(n_circles)
        # locations within the pixel where we will average the J-factor value
        sample = J_factor.location_sampler(
            n_fi, g_lon, g_lat, fov_pixel, n_circles=n_circles,
            center_halo=center_halo, circles_is_log=circles_is_log,
            first_circle=first_circle,
        )
        alpha_sample = sample[0]
        angle_lim, R_sun, R_lim = J_factor._los_limits(params)
        step1_array = np.array([
            J_factor.step1_calculator(
                alpha, angle_lim, R_sun, R_lim, fov, params, n_los_inner, n_los_outer
            )
            for alpha in alpha_sample
        ])
        return J_factor._integrate_pixel_sample(
            step1_array, sample, params["rho_s"], pixel_size,
            per_sr, step1_return, save_fig_jfac,
        )

    def j_factor_pixels_calculator(
        params, fov_pixel, locations, n_los_outer=100, n_los_inner=500,
        fov=None, n_fi=20, n_circles=100, center_halo=(0, 0),
        params_computation=False, per_sr=True, step1_return=False,
        save_fig_jfac=None, circles_is_log=True, first_circle=1e-12,
        *, n_alpha=1000, alpha_is_log=True,
    ):
        """Compute J-factors for circular pixels using a shared line-of-sight table.

        Locations are (galactic longitude, latitude) pairs in degrees. The
        pixels share a radius (fov_pixel, in degrees) and a spherical halo.
        The optional fov is also shared and is in degrees. By default it is
        derived from the outermost sampled direction.

        n_alpha sets the number of points in the common angular grid.
        alpha_is_log selects a logarithmic grid, with a separate evaluation
        at zero if sampled. This grid is independent of circles_is_log,
        which controls the sampling within each pixel.

        Returns one result per location, in input order. With step1_return
        each result is (J, step1_array, angle_sample, alpha_sample), as in
        j_factor_pixel_calculator, with interpolated line-of-sight values.
        """
        if n_alpha < 2:
            raise ValueError("n_alpha must be at least 2")

        fov_pixel = fov_pixel * mp.pi / 180 # rad
        center_halo = np.array(center_halo) * mp.pi / 180 # rad
        n_fi = int(n_fi)
        n_circles = int(n_circles)
        samples = [
            J_factor.location_sampler(
                n_fi, lon * mp.pi / 180, lat * mp.pi / 180, fov_pixel,
                n_circles=n_circles, center_halo=center_halo,
                circles_is_log=circles_is_log, first_circle=first_circle,
            )
            for lon, lat in locations
        ]
        if not samples:
            return []

        if params_computation:
            params = density.parameters.parameters_computation(params)

        # Find the angular range across the actual samples in all pixels.
        alpha_samples = [sample[0] for sample in samples]
        alpha_min = min(alphas.min() for alphas in alpha_samples)
        alpha_max = max(alphas.max() for alphas in alpha_samples)
        if alpha_min == alpha_max:
            alpha_grid = np.array([alpha_min])
        elif alpha_is_log:
            alpha_min_positive = min(
                alphas[alphas > 0].min(initial=np.inf) for alphas in alpha_samples
            )
            alpha_grid = (
                np.array([alpha_max]) if alpha_min_positive == alpha_max
                else np.geomspace(alpha_min_positive, alpha_max, n_alpha)
            )
            if alpha_min == 0:
                alpha_grid = np.insert(alpha_grid, 0, 0.0)
        else:
            alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha)

        if fov is None:
            fov = alpha_max + 0.01 * fov_pixel
        else:
            fov = fov * mp.pi / 180
        angle_lim, R_sun, R_lim = J_factor._los_limits(params)
        step1_grid = np.array([
            J_factor.step1_calculator(
                alpha, angle_lim, R_sun, R_lim, fov, params, n_los_inner, n_los_outer
            )
            for alpha in alpha_grid
        ], dtype=float)

        pixel_size = J_factor.Solid_Angle_pixel(fov_pixel)
        rho_s = params["rho_s"]
        results = []
        for index, sample in enumerate(samples):
            step1_array = np.interp(sample[0], alpha_grid, step1_grid)
            figure_prefix = save_fig_jfac
            if figure_prefix is not None:
                figure_prefix = f"{figure_prefix}pixel_{index}_"
            results.append(J_factor._integrate_pixel_sample(
                step1_array, sample, rho_s, pixel_size,
                per_sr, step1_return, figure_prefix,
            ))
        return results

    @staticmethod
    def _los_limits(params):
        mode = params["Mode"]
        if mode in ("NFW", "Einasto") or (
            mode == "Burkert" and params.get("R_spike") is None
        ):
            R_lim = 1e-2  # kpc
        else:
            R_lim = params["R_spike"]
        R_sun = params["R_sun"]
        return mp.atan(R_lim / R_sun), R_sun, R_lim

    @staticmethod
    def _integrate_pixel_sample(
        step1_array, sample, rho_s, pixel_size, per_sr, step1_return, save_fig_jfac,
    ):
        alpha_sample, fi_sample, angle_sample, fi_array = sample
        J_pixel = J_factor.Solid_Angle_Integrator(
            step1_array, fi_sample, alpha_sample, angle_sample, fi_array,
            save_fig_jfac=save_fig_jfac,
        ) * rho_s**2
        J_pixel_per_sr = J_pixel / pixel_size
        result = J_factor.Jfactor_to_GeV_cm(J_pixel_per_sr if per_sr else J_pixel)
        if step1_return:
            step1_values = J_factor.Jfactor_to_GeV_cm(step1_array * rho_s**2)
            return result, step1_values, angle_sample, alpha_sample
        return result
