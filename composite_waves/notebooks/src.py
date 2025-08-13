import xarray as xr
import numpy as np
import time
import math

from scipy.interpolate import interp1d,griddata
from scipy.special import gammainc
from scipy.interpolate import griddata, RectBivariateSpline, RegularGridInterpolator
from matplotlib.path import Path
from scipy.ndimage import rotate, gaussian_filter
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter
import numpy.matlib
from scipy.stats.distributions import chi2


g = 9.81 #  acceleration of gravity

# @ g.marechal
# contact: gwendal.marechal@protonmail.com

def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


def gaussian_filter_nan(data, sigma):
    """Apply Gaussian filter while ignoring NaNs."""
    nan_mask = np.isnan(data)  # Mask of NaNs
    filled_data = np.where(nan_mask, np.nanmean(data), data)  # Fill NaNs with mean

    # Apply Gaussian filter
    filtered_data = gaussian_filter(filled_data, sigma=sigma)

    # Restore NaNs
    filtered_data[nan_mask] = np.nan  
    return filtered_data

def eddy_concentration(x_AMEDA, y_AMEDA, x_grid, y_grid):
    """2D histogram of eddies."""
    # Remove NaNs or masked values

    valid = np.isfinite(x_AMEDA) & np.isfinite(y_AMEDA)
    lon_points = x_AMEDA[valid]
    lat_points = y_AMEDA[valid]
    
    # Compute 2D histogram
    heatmap, _, _ = np.histogram2d(lon_points, lat_points, bins=[x_grid, y_grid])

    return heatmap
    
def compute_composite(ds_model, filtered_ds, ds_AMEDA, varname, wave_age_thr, pol, npts_composite, sigma0, \
                      eddy_entension_factor=2.5, n_realizations = 33_000, \
                      filtering = True, rotation = True, time_average = False, already_filtered = False):

    """
    Purpose: Perform the composite analysis of a given variables from numerical outputs
    --------

    Inputs:
    --------
    ds_model: The 3D model dataset (time, longigude, latitude), should be daily sampled
    ds_AMEDA: the AMEDA Atlas
    varname: variable name
    wave_age_thr: the wave age threshold (should be equal to 1.2, ==0 if no partionning needed)
    pol: the polarization of the eddies
    npts_composite: number of point in X and Y for the composite grid
    sigma: the kernel's size for spatial filtering
    eddy_entension_factor: the area's size in the vicinity of the spotted eddies
    n_realizations: a dummy large number to store all eddy outputs
    filtering: Spatial filtering True/False?
    rotation: Rotation of the field according to the wave direction True/False
    time_average: Temporal filtering True/False?
    already_filtered: a flag to filter the 2D map (aims to save computation time)

    Outputs:
    --------
    ds_output: the DataArray with all the eddy, partionized by swell and wind sea
    filtered_ds: the DataArray with the filtered model outputs
    index_swell, index_windsea: the indices of independant eddies spotted by AMEDA 
    """

    
    wave_age_thr = wave_age_thr # put 0 if you want to consider all type of waves
    
    # --- Varname selection
    varname_model_ww3_0 = varname
    varname_model_ww3 = varname
    
    ###########
    # --- temporal filter
    ###########
    if time_average:
         # slightly long process (~ 1 min)
        ds_padded = ds_model.pad(time=90, mode="reflect") 
    
        low_pass = ds_padded[varname_model_ww3].rolling(time=90, center=True).mean()
        ds_filtered = low_pass.sel(time=slice(ds_model.time.values[0], ds_model.time.values[-1]))
        # Compute the high-pass filtered data
        dummy = (ds_model[varname_model_ww3] - ds_filtered)
        ds_model['temp_ave_var'] = dummy
    
        ds_model.temp_ave_var.attrs['long_name'] = ds_padded[varname_model_ww3].attrs['long_name']+' temporaly filtered'
        ds_model.temp_ave_var.attrs['units'] = ds_padded[varname_model_ww3].attrs['units']
        varname_model_ww3 = 'temp_ave_var'
        

    theta = np.linspace(0, 2*np.pi, 100)
    
    x_grid, y_grid = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    new_x_grid, new_y_grid = np.meshgrid(x_grid, y_grid)
    coords = np.vstack([new_x_grid.ravel(), new_y_grid.ravel()])
    # --- Crop a circle
    radius = 1
    circle_lon_unity = radius * np.cos(theta)
    circle_lat_unity = radius * np.sin(theta)
    distances = np.sqrt((new_x_grid - 0)**2 + (new_y_grid - 0)**2)
    circle_mask = distances <= radius # a mask

    # --- For the indices of independant eddies
    index_swell = []
    index_windsea = []
    
    # --- Initialize composite grids
    cpt_swell = 0
    cpt_windsea = 0
    
    Compo_windsea = np.zeros((n_realizations, len(x_grid), len(y_grid))) * np.nan
    Compo_swell = np.zeros((n_realizations, len(x_grid), len(y_grid))) * np.nan

    varname_ww3 = varname_model_ww3 # 'hs', 'fp'

    if filtering and already_filtered is not True : # Spatial filtering over a 3D dataset (time, x, y)
        print('Start Spatial filtering, /!\ 1 min process! /!\ ')
        sigma = (0, sigma0, sigma0) 
        filtered_ds = xr.apply_ufunc(
            gaussian_filter_nan, ds_model,
            input_core_dims=[("time", "latitude", "longitude")],  
            output_core_dims=[("time", "latitude", "longitude")],  
            vectorize=True,
            kwargs={"sigma": sigma},
            dask="parallelized"
        )
    else:
        a = 2
    
        
    for slice_time in tqdm((range(len(ds_model.time.values)))):
    
        ds_ww3_sel = ds_model.isel(time = slice_time) # time variable for ww3  = time
        if filtering:
            ds_ww3_sel_filtered = filtered_ds.isel(time = slice_time) # time variable for ww3  = time
        
        ds_AMEDA = ds_AMEDA.assign_coords(obs=("obs", np.arange(ds_AMEDA.dims["obs"]))) # fix a DataArray issue

        id_polarization = np.where(np.sign(ds_AMEDA.Ro)==pol)[0]
        ds_AMEDA_pola = ds_AMEDA.sel(obs = id_polarization)
    
        id_ameda = np.where(ds_AMEDA_pola.time.values == ds_ww3_sel.time.values)[0]# time variable for AMEDA  = time
        sub_ds_AMEDA = ds_AMEDA_pola.isel(obs = id_ameda) # the obs indices are not continuous
        
        # Load axes for model outputs
        lon_ww3 = ds_ww3_sel.longitude.values
        lat_ww3 = ds_ww3_sel.latitude.values
        
        lon_ww3_1d = lon_ww3[0, :]
        lat_ww3_1d = lat_ww3[:, 0]
        
        lon_ww3_min = ds_ww3_sel.longitude.values.min()
        lon_ww3_max = ds_ww3_sel.longitude.values.max()
                
        lat_ww3_min = ds_ww3_sel.latitude.values.min()
        lat_ww3_max = ds_ww3_sel.latitude.values.max()
    
      
        if filtering:
            variable_model_ww3 = ds_ww3_sel[varname_ww3].values - ds_ww3_sel_filtered[varname_ww3].values # remove large scale features
        else:
            variable_model_ww3 = ds_ww3_sel[varname_ww3].values
    
        dir_wave = ds_ww3_sel.dir.values # For rotation, in degree!
        # dir_wave = ds_ww3_sel.dp.values # For rotation, in degree!

        dir_wnd = ds_ww3_sel.dir_wnd.values # For rotation
        wave_age = ds_ww3_sel.mean_wave_age.values # for partitionning
        
        # for idt in range(len(sub_ds_AMEDA.time.values)): # loop over the eddies spotted
        for idt in range(len(sub_ds_AMEDA.obs.values)):

            #####
            # --- Load AMEDA variables
            #####
            # --- The center of the eddy
            lon_center = sub_ds_AMEDA.x_cen[idt].values 
            lat_center = sub_ds_AMEDA.y_cen[idt].values
            
                    # --- The eddy edges
            lon_max = sub_ds_AMEDA.x_max[idt].values
            lat_max = sub_ds_AMEDA.y_max[idt].values
                
            lon_end = sub_ds_AMEDA.x_end[idt].values
            lat_end = sub_ds_AMEDA.y_end[idt].values
            
            # --- Compute the mean radius associated with Rmax/Rend provided by AMEDA
            max_radius = np.sqrt((lon_max-lon_center)**2\
                                 + (lat_max-lat_center)**2)
            end_radius = np.sqrt((lon_end-lon_center)**2\
                                 + (lat_end-lat_center)**2)
                    
            # --- The mean radius
            mean_radius_max = max_radius.mean()
            mean_radius_end = end_radius.mean()
                
            mean_radius_max_big = mean_radius_max*eddy_entension_factor
            mean_radius_end_big = mean_radius_end*eddy_entension_factor
            
            # ---  Generate circle points using parametric equation
            circle_lon = lon_center + mean_radius_max_big * np.cos(theta)
            circle_lat = lat_center + mean_radius_max_big * np.sin(theta)
            
            # --- Find the minimum and maximum longitude/latitude
            min_lon_circle = circle_lon.min()
            max_lon_circle = circle_lon.max()
            min_lat_circle = circle_lat.min()
            max_lat_circle = circle_lat.max()
            
            id_longitude = np.where((lon_ww3_1d<max_lon_circle)&(lon_ww3_1d>min_lon_circle))[0]
            id_latitude = np.where((lat_ww3_1d<max_lat_circle)&(lat_ww3_1d>min_lat_circle))[0]
            
            sub_lat = lat_ww3_1d[np.nanmin(id_latitude):np.nanmax(id_latitude)]
            sub_lon = lon_ww3_1d[np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            # --- Crop variables, wave direction, and wave age
            cropped_eddy = variable_model_ww3[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            dir_eddy = dir_wave[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                           np.nanmin(id_longitude):np.nanmax(id_longitude)]
                
            dir_wnd_eddy = dir_wnd[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            wave_age_eddy = wave_age[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            cropped_eddy_anom = cropped_eddy
            cropped_eddy_anom = np.nan_to_num(cropped_eddy_anom, nan=0, neginf=0) # Avoid nan and -inf values
            
            # --- Compute distance from the center of the circle (center of the Patch considered)
            dist_lat = abs(np.nanmax(sub_lat) - np.nanmin(sub_lat))
            dist_lon = abs(np.nanmax(sub_lon) - np.nanmin(sub_lon))
            # Center the grid -1/+1 around 0
            # xs = 2 * (sub_lon - np.nanmin(sub_lon)) / (np.nanmax(sub_lon) - np.nanmin(sub_lon)) - 1
            # ys = 2 * (sub_lat - np.nanmin(sub_lat)) / (np.nanmax(sub_lat) - np.nanmin(sub_lat)) - 1

            dx = sub_lon - sub_ds_AMEDA.x_bar[idt].values 
            dy = sub_lat - sub_ds_AMEDA.y_bar[idt].values
            
            # Normalize each axis independently so the contour fits in [-1, 1]
            xs = dx / np.nanmax(np.abs(dx))
            ys = dy / np.nanmax(np.abs(dy))

            # xs = sub_lon - np.nanmean(sub_lon)
            # ys = sub_lat - np.nanmean(sub_lat)
            
            xxs, yys = np.meshgrid(xs, ys)
            
            mean_wave_age = np.nanmean(wave_age_eddy) # mean wave age
            
            if mean_wave_age>wave_age_thr: # for swell
                # Project lon/lat to regular -1/+1 grid
                Compo_unity_swell = RectBivariateSpline(ys, xs, cropped_eddy_anom)(y_grid, x_grid)
                masked_compo_swell = np.where(circle_mask, Compo_unity_swell, np.nan)
                valid_mask_ww3_rotated_swell = ~np.isnan(masked_compo_swell)  # Boolean mask for valid (non-NaN) values
                
                if rotation:
                    theta_rot = np.radians(90 - np.nanmean(dir_eddy) + 180)%(2*np.pi) # the +180 is for waves coming from the left boundary
                    # theta_rot = np.radians(np.nanmean(dir_wnd_eddy)+180)%(2*np.pi)
                else:
                    theta_rot = 0 # no rotation
                # Define the rotation matrix
                R = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], 
                        [np.sin(theta_rot),  np.cos(theta_rot)]])
                
                rotated_coords = R @ coords # Apply rotation
                X_rot, Y_rot = rotated_coords.reshape(2, *new_x_grid.shape)
                # masked_compo[np.isnan(masked_compo)] = 0
                Z_rot = xr.DataArray(masked_compo_swell, coords=[y_grid, x_grid], dims=['y', 'x'])
                
                # Define the interpolation function
                interp_func = RegularGridInterpolator(
                        (y_grid, x_grid), masked_compo_swell, method="linear", bounds_error=False, fill_value=np.nan)
                
                # Flatten the 2D rotated grid and interpolate
                points = np.column_stack((Y_rot.ravel(), X_rot.ravel()))
                Z_rot_interp_swell = interp_func(points).reshape(X_rot.shape)
                
                # Fill the cube of data (dim 0 = eddies spotted)
                Compo_swell[cpt_swell, :,:] = Z_rot_interp_swell
                cpt_swell = cpt_swell + 1
                index_swell.append(sub_ds_AMEDA.obs[idt].values) 

    
            elif mean_wave_age<wave_age_thr: # for wind sea
                 # Project lon/lat to regular -1/+1 grid
                Compo_unity_windsea = RectBivariateSpline(ys, xs, cropped_eddy_anom)(y_grid, x_grid)
                masked_compo_windsea = np.where(circle_mask, Compo_unity_windsea, np.nan)
                valid_mask_ww3_rotated_windsea = ~np.isnan(masked_compo_windsea)  # Boolean mask for valid (non-NaN) values
                 
                if rotation:
                     # theta_rot = np.radians(90 - np.nanmean(dir_eddy))%(2*np.pi)
                     theta_rot = np.radians(np.nanmean(dir_wnd_eddy))%(2*np.pi) # add + 180 if you want the wind bloxing east->west
                else:
                     theta_rot = 0 # no rotation
                     
                 # Define rotation matrix
                R = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], 
                            [np.sin(theta_rot),  np.cos(theta_rot)]])
                rotated_coords = R @ coords # Apply rotation matrix
                X_rot, Y_rot = rotated_coords.reshape(2, *new_x_grid.shape)
                # masked_compo[np.isnan(masked_compo)] = 0
                Z_rot = xr.DataArray(masked_compo_windsea, coords=[y_grid, x_grid], dims=['y', 'x'])
                
                # Define the interpolation function
                interp_func = RegularGridInterpolator(
                    (y_grid, x_grid), masked_compo_windsea, method="linear", bounds_error=False, fill_value=None)
                
                # Flatten the 2D rotated grid and interpolate
                points = np.column_stack((Y_rot.ravel(), X_rot.ravel()))
                Z_rot_interp_windsea = interp_func(points).reshape(X_rot.shape)
                    
                Compo_windsea[cpt_windsea, :,:] = Z_rot_interp_windsea
                cpt_windsea = cpt_windsea + 1
                index_windsea.append(sub_ds_AMEDA.obs[idt].values) 


    ###########
    # ---  Independant eddies
    ###########    
    N_ind_swell = np.isin(index_swell, ds_AMEDA.index_of_first_observation).sum()
    N_ind_windsea = np.isin(index_windsea, ds_AMEDA.index_of_first_observation).sum()
    
    ###########
    # ---  Save the Dataset
    ###########
    cube_composite_wave_swell = xr.DataArray(data=Compo_swell, dims=['n_realization', 'x', 'y'],
                coords=dict(n_realization = np.array(np.arange(0, n_realizations, 1), dtype = int), x=x_grid, y=y_grid))
    
    cube_composite_wave_windsea = xr.DataArray(data=Compo_windsea, dims=['n_realization', 'x', 'y'],
                coords=dict(n_realization = np.array(np.arange(0, n_realizations, 1), dtype = int), x=x_grid, y=y_grid))
        
    
    ds_output = xr.Dataset()
    ds_output['composite_swell'] = cube_composite_wave_swell
    ds_output['composite_windsea'] = cube_composite_wave_windsea
    
    ds_output.x.attrs['units'] = 'Dimensionless'
    ds_output.x.attrs['long_name'] = 'Normalized Distance Y'
    ds_output.y.attrs['units'] = 'Dimensionless'
    ds_output.y.attrs['long_name'] = 'Normalized Distance X'
    ds_output.composite_swell.attrs['units'] = '%s'%str(ds_model[varname_model_ww3].attrs['units'])
    ds_output.composite_swell.attrs['long_name'] = 'Anomaly of %s for swell partition'%str(varname_model_ww3_0) 
    ds_output.composite_windsea.attrs['units'] = '%s'%str(ds_model[varname_model_ww3].attrs['units'])
    ds_output.composite_windsea.attrs['long_name'] = 'Anomaly of %s for wind sea partition'%str(varname_model_ww3_0) 
    ds_output.attrs['variable_name_for_composite'] = varname_model_ww3_0
    ds_output.attrs['independant_eddies_swell'] = str(N_ind_swell)
    ds_output.attrs['independant_eddies_windsea'] = str(N_ind_windsea)
    ds_output.attrs['creator_name'] = 'g.marechal'
    ds_output.attrs['creator_contact'] = 'gwendal.marechal@univ-tlse3.fr'
    ds_output.attrs['time_creation'] = time.ctime()

    return ds_output, filtered_ds



def compute_composite_theo(ds_model, filtered_ds, ds_AMEDA, varname_hs, varname_u, varname_f, wave_age_thr, pol, npts_composite, sigma0, \
                      eddy_entension_factor=2.5, n_realizations = 33_000, \
                      filtering = True, rotation = True, time_average = False, already_filtered = False):

    """
    Purpose: Perform the composite analysis of a given variables from numerical outputs
    --------

    Inputs:
    --------
    ds_model: The 3D model dataset (time, longigude, latitude), should be daily sampled
    ds_AMEDA: the AMEDA Atlas
    varname: variable name
    wave_age_thr: the wave age threshold (should be equal to 1.2, ==0 if no partionning needed)
    pol: the polarization of the eddies
    npts_composite: number of point in X and Y for the composite grid
    sigma: the kernel's size for spatial filtering
    eddy_entension_factor: the area's size in the vicinity of the spotted eddies
    n_realizations: a dummy large number to store all eddy outputs
    filtering: Spatial filtering True/False?
    rotation: Rotation of the field according to the wave direction True/False
    time_average: Temporal filtering True/False?
    already_filtered: a flag to filter the 2D map (aims to save computation time)

    Outputs:
    --------
    ds_output: the DataArray with all the eddy, partionized by swell and wind sea
    filtered_ds: the DataArray with the filtered model outputs
    index_swell, index_windsea: the indices of independant eddies spotted by AMEDA 
    """

    
    wave_age_thr = wave_age_thr # put 0 if you want to consider all type of waves
    
    # --- Varname selection
    varname_model_ww3_0 = varname_hs
    varname_model_ww3 = varname_hs
    ###########
    # --- temporal filter
    ###########
    if time_average:
         # slightly long process (~ 1 min)
        ds_padded = ds_model.pad(time=90, mode="reflect") 
    
        low_pass = ds_padded[varname_model_ww3].rolling(time=90, center=True).mean()
        ds_filtered = low_pass.sel(time=slice(ds_model.time.values[0], ds_model.time.values[-1]))
        # Compute the high-pass filtered data
        dummy = (ds_model[varname_model_ww3] - ds_filtered)
        ds_model['temp_ave_var'] = dummy
    
        ds_model.temp_ave_var.attrs['long_name'] = ds_padded[varname_model_ww3].attrs['long_name']+' temporaly filtered'
        ds_model.temp_ave_var.attrs['units'] = ds_padded[varname_model_ww3].attrs['units']
        varname_model_ww3 = 'temp_ave_var'
        

    theta = np.linspace(0, 2*np.pi, 100)
    
    x_grid, y_grid = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    new_x_grid, new_y_grid = np.meshgrid(x_grid, y_grid)
    coords = np.vstack([new_x_grid.ravel(), new_y_grid.ravel()])
    # --- Crop a circle
    radius = 1
    circle_lon_unity = radius * np.cos(theta)
    circle_lat_unity = radius * np.sin(theta)
    distances = np.sqrt((new_x_grid - 0)**2 + (new_y_grid - 0)**2)
    circle_mask = distances <= radius # a mask

    # --- For the indices of independant eddies
    index_swell = []
    index_windsea = []
    
    # --- Initialize composite grids
    cpt_swell = 0
    cpt_windsea = 0
    
    Compo_windsea = np.zeros((n_realizations, len(x_grid), len(y_grid))) * np.nan
    Compo_swell = np.zeros((n_realizations, len(x_grid), len(y_grid))) * np.nan

    # varname_ww3 = varname_model_ww3 # 'hs', 'fp'

    if filtering and already_filtered is not True : # Spatial filtering over a 3D dataset (time, x, y)
        print('Start Spatial filtering, /!\ 1 min process! /!\ ')
        sigma = (0, sigma0, sigma0) 
        filtered_ds = xr.apply_ufunc(
            gaussian_filter_nan, ds_model,
            input_core_dims=[("time", "latitude", "longitude")],  
            output_core_dims=[("time", "latitude", "longitude")],  
            vectorize=True,
            kwargs={"sigma": sigma},
            dask="parallelized"
        )
    else:
        a = 2
    
        
    for slice_time in tqdm((range(len(ds_model.time.values)))):
    
        ds_ww3_sel = ds_model.isel(time = slice_time) # time variable for ww3  = time
        if filtering:
            ds_ww3_sel_filtered = filtered_ds.isel(time = slice_time) # time variable for ww3  = time
        
        ds_AMEDA = ds_AMEDA.assign_coords(obs=("obs", np.arange(ds_AMEDA.dims["obs"]))) # fix a DataArray issue

        id_polarization = np.where(np.sign(ds_AMEDA.Ro)==pol)[0]
        ds_AMEDA_pola = ds_AMEDA.sel(obs = id_polarization)
    
        id_ameda = np.where(ds_AMEDA_pola.time.values == ds_ww3_sel.time.values)[0]# time variable for AMEDA  = time
        sub_ds_AMEDA = ds_AMEDA_pola.isel(obs = id_ameda) # the obs indices are not continuous
        
        # Load axes for model outputs
        lon_ww3 = ds_ww3_sel.longitude.values
        lat_ww3 = ds_ww3_sel.latitude.values
        
        lon_ww3_1d = lon_ww3[0, :]
        lat_ww3_1d = lat_ww3[:, 0]
        
        lon_ww3_min = ds_ww3_sel.longitude.values.min()
        lon_ww3_max = ds_ww3_sel.longitude.values.max()
                
        lat_ww3_min = ds_ww3_sel.latitude.values.min()
        lat_ww3_max = ds_ww3_sel.latitude.values.max()
    
      
        if filtering:
            #variable_model_ww3 = ds_ww3_sel[varname_ww3].values - ds_ww3_sel_filtered[varname_ww3].values # remove large scale features
            variable_model_ww3_u = ds_ww3_sel[varname_u].values - ds_ww3_sel_filtered[varname_u].values 
            # variable_model_ww3_u = ds_ww3_sel[varname_u].values 
            variable_model_ww3_f = 1/ds_ww3_sel[varname_f].values
            variable_model_ww3_hs = ds_ww3_sel[varname_hs].values

        else:
            variable_model_ww3 = ds_ww3_sel[varname_ww3].values
    
        dir_wave = ds_ww3_sel.dir.values # For rotation, in degree!
        # dir_wave = ds_ww3_sel.dp.values # For rotation, in degree!

        dir_wnd = ds_ww3_sel.dir_wnd.values # For rotation
        wave_age = ds_ww3_sel.mean_wave_age.values # for partitionning
        
        # for idt in range(len(sub_ds_AMEDA.time.values)): # loop over the eddies spotted
        for idt in range(len(sub_ds_AMEDA.obs.values)):

            #####
            # --- Load AMEDA variables
            #####
            # --- The center of the eddy
            lon_center = sub_ds_AMEDA.x_cen[idt].values 
            lat_center = sub_ds_AMEDA.y_cen[idt].values
            
                    # --- The eddy edges
            lon_max = sub_ds_AMEDA.x_max[idt].values
            lat_max = sub_ds_AMEDA.y_max[idt].values
                
            lon_end = sub_ds_AMEDA.x_end[idt].values
            lat_end = sub_ds_AMEDA.y_end[idt].values
            
            # --- Compute the mean radius associated with Rmax/Rend provided by AMEDA
            max_radius = np.sqrt((lon_max-lon_center)**2\
                                 + (lat_max-lat_center)**2)
            end_radius = np.sqrt((lon_end-lon_center)**2\
                                 + (lat_end-lat_center)**2)
                    
            # --- The mean radius
            mean_radius_max = max_radius.mean()
            mean_radius_end = end_radius.mean()
                
            mean_radius_max_big = mean_radius_max*eddy_entension_factor
            mean_radius_end_big = mean_radius_end*eddy_entension_factor
            
            # ---  Generate circle points using parametric equation
            circle_lon = lon_center + mean_radius_max_big * np.cos(theta)
            circle_lat = lat_center + mean_radius_max_big * np.sin(theta)
            
            # --- Find the minimum and maximum longitude/latitude
            min_lon_circle = circle_lon.min()
            max_lon_circle = circle_lon.max()
            min_lat_circle = circle_lat.min()
            max_lat_circle = circle_lat.max()
            
            id_longitude = np.where((lon_ww3_1d<max_lon_circle)&(lon_ww3_1d>min_lon_circle))[0]
            id_latitude = np.where((lat_ww3_1d<max_lat_circle)&(lat_ww3_1d>min_lat_circle))[0]
            
            sub_lat = lat_ww3_1d[np.nanmin(id_latitude):np.nanmax(id_latitude)]
            sub_lon = lon_ww3_1d[np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            # --- Crop variables, wave direction, and wave age
            cropped_eddy_u = variable_model_ww3_u[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            cropped_eddy_f = variable_model_ww3_f[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            cropped_eddy_hs = variable_model_ww3_hs[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            dir_eddy = dir_wave[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                           np.nanmin(id_longitude):np.nanmax(id_longitude)]
                
            dir_wnd_eddy = dir_wnd[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            wave_age_eddy = wave_age[np.nanmin(id_latitude):np.nanmax(id_latitude),\
                       np.nanmin(id_longitude):np.nanmax(id_longitude)]
            
            cropped_eddy_anom_u = cropped_eddy_u
            cropped_eddy_anom_u = np.nan_to_num(cropped_eddy_anom_u, nan=0, neginf=0) # Avoid nan and -inf values

            cropped_eddy_anom_f = cropped_eddy_f
            cropped_eddy_anom_f = np.nan_to_num(cropped_eddy_anom_f, nan=0, neginf=0) # Avoid nan and -inf values

            
            # --- Compute distance from the center of the circle (center of the Patch considered)
            dist_lat = abs(np.nanmax(sub_lat) - np.nanmin(sub_lat))
            dist_lon = abs(np.nanmax(sub_lon) - np.nanmin(sub_lon))
            # Center the grid -1/+1 around 0
            # xs = 2 * (sub_lon - np.nanmin(sub_lon)) / (np.nanmax(sub_lon) - np.nanmin(sub_lon)) - 1
            # ys = 2 * (sub_lat - np.nanmin(sub_lat)) / (np.nanmax(sub_lat) - np.nanmin(sub_lat)) - 1

            dx = sub_lon - sub_ds_AMEDA.x_bar[idt].values 
            dy = sub_lat - sub_ds_AMEDA.y_bar[idt].values
            
            # Normalize each axis independently so the contour fits in [-1, 1]
            xs = dx / np.nanmax(np.abs(dx))
            ys = dy / np.nanmax(np.abs(dy))

            # xs = sub_lon - np.nanmean(sub_lon)
            # ys = sub_lat - np.nanmean(sub_lat)
            
            xxs, yys = np.meshgrid(xs, ys)
            
            mean_wave_age = np.nanmean(wave_age_eddy) # mean wave age
            
            if mean_wave_age>wave_age_thr: # for swell
                # Project lon/lat to regular -1/+1 grid
                Compo_unity_swell_u = RectBivariateSpline(ys, xs, cropped_eddy_anom_u)(y_grid, x_grid)
                Compo_unity_swell_f = RectBivariateSpline(ys, xs, cropped_eddy_anom_f)(y_grid, x_grid)

                masked_compo_swell_u = np.where(circle_mask, Compo_unity_swell_u, np.nan)
                masked_compo_swell_f = np.where(circle_mask, Compo_unity_swell_f, np.nan)
                valid_mask_ww3_rotated_swell_u = ~np.isnan(masked_compo_swell_u)  # Boolean mask for valid (non-NaN) values
                valid_mask_ww3_rotated_swell_f = ~np.isnan(masked_compo_swell_f)  # Boolean mask for valid (non-NaN) values

                
                if rotation:
                    theta_rot = np.radians(90 - np.nanmean(dir_eddy) + 180)%(2*np.pi) # the +180 is for waves coming from the left boundary
                    # theta_rot = np.radians(np.nanmean(dir_wnd_eddy)+180)%(2*np.pi)
                else:
                    theta_rot = 0 # no rotation
                # Define the rotation matrix
                R = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], 
                        [np.sin(theta_rot),  np.cos(theta_rot)]])
                
                R0 = np.array([[np.cos(0), -np.sin(0)], 
                        [np.sin(0),  np.cos(0)]])
                
                rotated_coords = R @ coords # Apply rotation
                rotated_coords0 = R0 @ coords # Apply rotation

                X_rot, Y_rot = rotated_coords.reshape(2, *new_x_grid.shape)
                X_rot0, Y_rot0 = rotated_coords0.reshape(2, *new_x_grid.shape)

                # masked_compo[np.isnan(masked_compo)] = 0
                Z_rot_u = xr.DataArray(masked_compo_swell_u, coords=[y_grid, x_grid], dims=['y', 'x'])
                Z_rot_f = xr.DataArray(masked_compo_swell_f, coords=[y_grid, x_grid], dims=['y', 'x'])

                # Define the interpolation function
                interp_func_f = RegularGridInterpolator(
                        (y_grid, x_grid), masked_compo_swell_f, method="linear", bounds_error=False, fill_value=np.nan)
                
                interp_func_u = RegularGridInterpolator(
                        (y_grid, x_grid), masked_compo_swell_u, method="linear", bounds_error=False, fill_value=np.nan)
                
                # Flatten the 2D rotated grid and interpolate
                points = np.column_stack((Y_rot.ravel(), X_rot.ravel()))
                points0 = np.column_stack((Y_rot0.ravel(), X_rot0.ravel()))
                
                Z_rot_interp_swell_f = interp_func_f(points).reshape(X_rot.shape)
                Z_rot_interp_swell_u = interp_func_u(points0).reshape(X_rot0.shape)

                Hs0 = np.nanmean(cropped_eddy_hs)
                E1 = np.nanmean(cropped_eddy_hs)**2/16
                # Fill the cube of data (dim 0 = eddies spotted)
                # alpha = Z_rot_interp_swell_u*2*np.pi*Z_rot_interp_swell_f/9.81


                
                sigma2 = 2*np.pi*Z_rot_interp_swell_f
                sigma1 =  2*np.pi*np.nanmean(Z_rot_interp_swell_f)
                alpha = (Z_rot_interp_swell_u*sigma1)/9.81
                sigma2_theo = (1-alpha)*sigma1
                c1 = g/sigma1
                c2 = g/sigma2

                cg1 = 1/2 * c1
                cg2 = 1/2 * c2
                E2_theo = E1 * (sigma2/sigma1)*(cg1/(cg2))
                hs2 = 4*np.sqrt(E2_theo)
            
                # Compo_swell[cpt_swell, :,:] =  np.sqrt(Hs0**2*(1-4*alpha))
                Compo_swell[cpt_swell, :,:] =  hs2
                # np.sqrt(hs2)
                                                      
                                                      # (1-4*(Z_rot_interp_swell_u*2*np.pi*Z_rot_interp_swell_f)/9.81))
                cpt_swell = cpt_swell + 1
                index_swell.append(sub_ds_AMEDA.obs[idt].values) 

    
            elif mean_wave_age<wave_age_thr: # for wind sea
                 # Project lon/lat to regular -1/+1 grid
                Compo_unity_ws_u = RectBivariateSpline(ys, xs, cropped_eddy_anom_u)(y_grid, x_grid)
                Compo_unity_ws_f = RectBivariateSpline(ys, xs, cropped_eddy_anom_f)(y_grid, x_grid)

                masked_compo_ws_u = np.where(circle_mask, Compo_unity_ws_u, np.nan)
                masked_compo_ws_f = np.where(circle_mask, Compo_unity_ws_f, np.nan)
                valid_mask_ww3_rotated_ws_u = ~np.isnan(masked_compo_ws_u)  # Boolean mask for valid (non-NaN) values
                valid_mask_ww3_rotated_ws_f = ~np.isnan(masked_compo_ws_f)  # Boolean mask for valid (non-NaN) values

                
                if rotation:
                    theta_rot = np.radians(90 - np.nanmean(dir_eddy) + 180)%(2*np.pi) # the +180 is for waves coming from the left boundary
                    # theta_rot = np.radians(np.nanmean(dir_wnd_eddy)+180)%(2*np.pi)
                else:
                    theta_rot = 0 # no rotation
                # Define the rotation matrix
                R = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], 
                        [np.sin(theta_rot),  np.cos(theta_rot)]])
                
                R0 = np.array([[np.cos(0), -np.sin(0)], 
                        [np.sin(0),  np.cos(0)]])
                
                rotated_coords = R @ coords # Apply rotation
                rotated_coords0 = R0 @ coords # Apply rotation

                X_rot, Y_rot = rotated_coords.reshape(2, *new_x_grid.shape)
                X_rot0, Y_rot0 = rotated_coords0.reshape(2, *new_x_grid.shape)

                # masked_compo[np.isnan(masked_compo)] = 0
                Z_rot_u = xr.DataArray(masked_compo_ws_u, coords=[y_grid, x_grid], dims=['y', 'x'])
                Z_rot_f = xr.DataArray(masked_compo_ws_f, coords=[y_grid, x_grid], dims=['y', 'x'])

                # Define the interpolation function
                interp_func_f = RegularGridInterpolator(
                        (y_grid, x_grid), masked_compo_ws_f, method="linear", bounds_error=False, fill_value=np.nan)
                
                interp_func_u = RegularGridInterpolator(
                        (y_grid, x_grid), masked_compo_ws_u, method="linear", bounds_error=False, fill_value=np.nan)
                
                # Flatten the 2D rotated grid and interpolate
                points = np.column_stack((Y_rot.ravel(), X_rot.ravel()))
                points0 = np.column_stack((Y_rot0.ravel(), X_rot0.ravel()))
                
                Z_rot_interp_ws_f = interp_func_f(points).reshape(X_rot.shape)
                Z_rot_interp_ws_u = interp_func_u(points0).reshape(X_rot0.shape)

                # Hs0 = np.nanmean(cropped_eddy_hs)
                # Fill the cube of data (dim 0 = eddies spotted)


                E1 = np.nanmean(cropped_eddy_hs)**2/16
                # Fill the cube of data (dim 0 = eddies spotted)
                # alpha = Z_rot_interp_swell_u*2*np.pi*Z_rot_interp_swell_f/9.81
                
                sigma2 = 2*np.pi*Z_rot_interp_ws_f
                sigma1 =  2*np.pi*np.nanmean(Z_rot_interp_ws_f)
                alpha = (Z_rot_interp_ws_u*sigma1)/9.81
                sigma2_theo = (1-alpha)*sigma1
                c1 = g/sigma1
                c2 = g/sigma2

                cg1 = 1/2 * c1
                cg2 = 1/2 * c2
                E2_theo = E1 * (sigma2/sigma1)*(cg1/(cg2))
                hs2 = 4*np.sqrt(E2_theo)
                # alpha = Z_rot_interp_ws_u*2*np.pi*Z_rot_interp_ws_f/9.81
                # Compo_windsea[cpt_windsea, :,:] = np.sqrt(Hs0**2*(1-4*alpha))
                Compo_windsea[cpt_windsea, :,:] = hs2

                                                      
                                                      # (1-4*(Z_rot_interp_swell_u*2*np.pi*Z_rot_interp_swell_f)/9.81))
                cpt_windsea = cpt_windsea + 1
                index_windsea.append(sub_ds_AMEDA.obs[idt].values) 


    ###########
    # ---  Independant eddies
    ###########    
    N_ind_swell = np.isin(index_swell, ds_AMEDA.index_of_first_observation).sum()
    N_ind_windsea = np.isin(index_windsea, ds_AMEDA.index_of_first_observation).sum()
    
    ###########
    # ---  Save the Dataset
    ###########
    cube_composite_wave_swell = xr.DataArray(data=Compo_swell, dims=['n_realization', 'x', 'y'],
                coords=dict(n_realization = np.array(np.arange(0, n_realizations, 1), dtype = int), x=x_grid, y=y_grid))
    
    cube_composite_wave_windsea = xr.DataArray(data=Compo_windsea, dims=['n_realization', 'x', 'y'],
                coords=dict(n_realization = np.array(np.arange(0, n_realizations, 1), dtype = int), x=x_grid, y=y_grid))
        
    
    ds_output = xr.Dataset()
    ds_output['composite_swell'] = cube_composite_wave_swell
    ds_output['composite_windsea'] = cube_composite_wave_windsea
    
    ds_output.x.attrs['units'] = 'Dimensionless'
    ds_output.x.attrs['long_name'] = 'Normalized Distance Y'
    ds_output.y.attrs['units'] = 'Dimensionless'
    ds_output.y.attrs['long_name'] = 'Normalized Distance X'
    ds_output.composite_swell.attrs['units'] = '%s'%str(ds_model[varname_model_ww3].attrs['units'])
    ds_output.composite_swell.attrs['long_name'] = 'Anomaly of %s for swell partition'%str(varname_model_ww3_0) 
    ds_output.composite_windsea.attrs['units'] = '%s'%str(ds_model[varname_model_ww3].attrs['units'])
    ds_output.composite_windsea.attrs['long_name'] = 'Anomaly of %s for wind sea partition'%str(varname_model_ww3_0) 
    ds_output.attrs['variable_name_for_composite'] = varname_model_ww3_0
    ds_output.attrs['independant_eddies_swell'] = str(N_ind_swell)
    ds_output.attrs['independant_eddies_windsea'] = str(N_ind_windsea)
    ds_output.attrs['creator_name'] = 'g.marechal'
    ds_output.attrs['creator_contact'] = 'gwendal.marechal@univ-tlse3.fr'
    ds_output.attrs['time_creation'] = time.ctime()

    return ds_output, filtered_ds



    

def bootstrap_uncertainty(ds, varname, npts, n_bootstrap=100):
    """
    Purpose:
    ---------
    Perform bootstrap resampling to quantify the uncertainty of the average.

    Inputs:
    ---------
    data : The input data with shape (number_of_realizations, x, y)
    n_bootstrap : int, optional
        Number of bootstrap resamples (default is 100).
    npts: number of point of the composite axes
        
    Outputs:
    ---------
    mean_map : The mean map averaged along the realizations.
    lower_ci : The lower bound of the confidence interval (e.g., 2.5th percentile).
    upper_ci : The upper bound of the confidence interval (e.g., 97.5th percentile).
    """
    
    # Number of realizations, x, y
    data = ds[varname].values
    n_realizations, x_size, y_size = data.shape
    
    # Pre-allocate output arrays (no need to store all bootstrap means)
    bootstrap_means = np.zeros((n_bootstrap, x_size, y_size), dtype=data.dtype)
    
    # Process in a loop but avoid redundant memory allocation
    for i in tqdm(range(n_bootstrap), desc="Bootstrapping", ncols=80):
        resampled_indices = np.random.choice(n_realizations, size=n_realizations, replace=True)
        bootstrap_means[i] = np.nanmean(data[resampled_indices], axis=0)
    
    # Compute statistics in-place to save memory
    mean_map = np.nanmean(bootstrap_means, axis=0)
    lower_ci, upper_ci = np.nanpercentile(bootstrap_means, [2.5, 97.5], axis=0)

    ##########
    # Save the Dataset
    ##########
    x_grid = np.linspace(-1, 1, npts)
    y_grid = x_grid
    da_mean_map = xr.DataArray(data=mean_map, dims=['x', 'y'],
                coords=dict(x=x_grid, y=y_grid))
    
    da_lower_ci = xr.DataArray(data=lower_ci, dims=['x', 'y'],
                    coords=dict(x=x_grid, y=y_grid))
    
    da_lower_ci = xr.DataArray(data=upper_ci, dims=['x', 'y'],
                    coords=dict(x=x_grid, y=y_grid))  
    
    ds_output = xr.Dataset()
    ds_output['mean_map'] = da_mean_map
    ds_output['lower_ci'] = da_lower_ci
    ds_output['upper_ci'] = da_lower_ci
    
    ds_output.x.attrs['units'] = 'Dimensionless'
    ds_output.x.attrs['long_name'] = 'Normalized Distance Y'
    ds_output.y.attrs['units'] = 'Dimensionless'
    ds_output.y.attrs['long_name'] = 'Normalized Distance X'
    
    ds_output.mean_map.attrs['units'] = '%s'%str(ds.composite_swell.attrs['units'])
    ds_output.mean_map.attrs['long_name'] = 'mean map of the composite'
    
    ds_output.lower_ci.attrs['units'] = '%s'%str(ds.composite_swell.attrs['units'])
    ds_output.lower_ci.attrs['long_name'] = '2.5th percentile map of the composite'
    
    ds_output.upper_ci.attrs['units'] = '%s'%str(ds.composite_swell.attrs['units'])
    ds_output.upper_ci.attrs['long_name'] = '97.5th percentile map of the composite'
    
    ds_output.attrs['creator_name'] = 'g.marechal'
    ds_output.attrs['creator_contact'] = 'gwendal.marechal@univ-tlse3.fr'
    ds_output.attrs['time_creation'] = time.ctime()
    
    return ds_output        

def bin_data(bins, data_binned, data_to_bin):
    """
    Bin 1D data
    """
    
    mean_data_binned = []
    std_data_binned = []
    bin_centered = []
    
    for bin_i in range(len(bins)-1):
        id_bin = np.where((data_binned>bins[bin_i]) & (data_binned<bins[bin_i+1]))[0]
        data_sel = data_to_bin[id_bin]
        mean_data_binned.append(np.nanmean(data_sel))
        std_data_binned.append(np.nanstd(data_sel)/(len(data_sel)**(1/2)))
        print(len(data_sel))
        bin_centered.append((bins[bin_i]+bins[bin_i+1])/2)

    bin_centered = np.array(bin_centered)
    mean_data_binned = np.array(mean_data_binned)
    std_data_binned = np.array(std_data_binned)

    return bin_centered, mean_data_binned, std_data_binned


def two_dimensional_wave_spectrum(field_2D, dx, dy):
    """
    Purpose:
    --------
    Perform the spectral analysis of the 2D MASS SSH with a pixel size of dx and dy
    Inputs:
    --------
    field_2D: The 2D MASS swath (dimensions ~ 2500 m| x 500 m)
    dx: pixel size in x
    dy: pixel size in y
    Outputs:
    --------
    phase_spec: The phase spectrum
    kx, ky the zonal and the meridional wavenumbers
    psd2D: The two-dimensional wave spectrum
    Z_shift: the intermediate wave spectrum
    dkx, dky: The wavenumber bins
    wc2xy: The 2D window correction 
    """
    
    Nx = np.size(field_2D, 0)
    Ny = np.size(field_2D, 1)
    
    #-----------------------------
    ###### hanning
    #-----------------------------
    #1D  windows

    hanningx = 0.5 * (1-np.cos(2*np.pi*np.transpose(np.linspace(0,Nx-1,Nx))/(Nx-1)))
    hanningy = np.ones((1,Ny))*0.5 * (1-np.cos(2*np.pi*np.linspace(0,Ny-1,Ny)/(Ny-1)))

    hanningxy = np.matlib.repmat(hanningx,Ny,1)*np.matlib.repmat(np.transpose(hanningy), 1, Nx)

    # window correction factors
    wc2x = 1/np.mean(hanningx**2)
    wc2y = 1/np.mean(hanningy**2)
    wc2xy  = wc2x * wc2y

#----------------------------
#spatial frequency axis
#----------------------------

    kx_max=(2*np.pi/dx)/2
    ky_max=(2*np.pi/dy)/2

    # step
    dkx=2*np.pi/(dx*(Nx//2)*2) #if odd-value data the max and min
    dky=2*np.pi/(dy*(Ny//2)*2)

    if np.mod(Nx,2)==0:
        #axes (for even-valued data, after fftshift...see example fft in MATLAB)
        kx=np.linspace(-kx_max, kx_max-dkx, Nx)
    else: #odd
        kx=np.linspace(-kx_max, kx_max, Nx)

    if np.mod(Ny,2)==0:
        #axes (for even-valued data, after fftshift...see example fft in MATLAB)
        ky=np.linspace(-ky_max, ky_max-dky, Ny)
    else: #odd
        ky=np.linspace(-ky_max, ky_max, Ny)

    dataW=np.zeros((np.shape(field_2D)))
    
    dkx = kx[3] - kx[2]
    dky = ky[3] - ky[2]
    
    #windowing
    new_data= field_2D - np.mean(field_2D[:])
    dataW = new_data*hanningxy.T
        
    #--------------------------
    # Spectral analysis
    #--------------------------

    #Z_fft = np.fft.fft2(dataW, s=None)
    #Z_shift = np.fft.fftshift(Z_fft)/(Nx*Ny)
    Z_shift = np.fft.fftshift(np.fft.fft2(dataW))/ (Nx*Ny)
        
    # Calculate a 2D power spectrum
    psd2D_0 = np.abs(Z_shift)**2/(dkx*dky)
    phase_spec=np.arctan2(np.imag(Z_shift), np.real(Z_shift))

    psd2D = psd2D_0*wc2xy
    Ekxky = xr.DataArray(psd2D, [('kx', kx),  ('ky', ky)])

    ds = xr.Dataset({'kx': ('kx', kx),
                    'ky': ('ky', ky),
                    })
                    
    ds['Ekxky'] = Ekxky
    ds = ds.set_coords(('kx', 'ky'))

    ds['Ekxky'].attrs['units'] = 'm^2/(rad/m)'

    ds['kx'].attrs['units'] = 'rad/m'
    ds['ky'].attrs['units'] = 'rad/m'

    return ds



def E_kxky_to_Ekth(E_kxky, kx, ky, number_of_dirs = 24):
    """
    Purpose: Interpolate the (kx, ky) spectrum onto directional-wavenumber spectrum
    inputs:
    E_kxky: the 2D wavenumber spectrum
    kx: the horizontal wavenumber (1D)
    ky: the vertical wavenumber (1D)
    
    outputs:
    dirspec: wavenumber-direction spectrum
    dd_new, kk_new: 2D dir-wavenumber axes
    """
    nk = np.size(kx)
    dkx = kx[1]-kx[0]
    k_new = np.linspace(0, np.amax(kx), nk)
    # Number of directions that you want on the polar grid
    nd = number_of_dirs
    dd = 2*np.pi/nd
    # Direction axis on the polar grid ranging from -pi to pi with dd increments
    d_new = np.arange(-np.pi, np.pi, dd)
    # Polar grid to interpolate onto
    # d_new = d_new0[number_of_dirs//2:]

    kk_new, dd_new = np.meshgrid(k_new, d_new)

    dirspec = xr.DataArray(np.zeros(dd_new.shape) * np.nan, [('theta', d_new),  ('K', k_new)])
    
    # Be sure working with a good xarray format
    Ekxky = xr.DataArray(E_kxky.T, dims=['ky', 'kx'], coords={'ky':ky, 'kx':kx})

    for i in range(nd):
        di = dd_new[i][0]
        ki = k_new
        kx_int = xr.DataArray(ki*np.cos(di), dims='K')
        ky_int = xr.DataArray(ki*np.sin(di), dims='K')
        dirspec[i, :] = Ekxky.interp(ky=ky_int, kx=kx_int)

    #here we're multiplying by the jacobian of the transformation from cartesian to polar
    dirspec = dirspec * kk_new 
    
    return dirspec, dd_new, kk_new


def radial_average(data, center=None, nbins=50):
    ny, nx = data.shape

    # Create physical coordinate arrays assuming [-1, 1] range
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)

    # Center in physical units
    if center is None:
        cx = 0.0
        cy = 0.0
    else:
        cx, cy = center

    # Compute true radial distances in physical space
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Flatten arrays and remove NaNs
    r_flat = r.ravel()
    data_flat = data.ravel()
    mask = ~np.isnan(data_flat)
    r_flat = r_flat[mask]
    data_flat = data_flat[mask]

    # Force radial bin edges to span [0, sqrt(2)]
    r_max = np.sqrt(2)
    bin_edges = np.linspace(0, r_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Bin and average
    bin_idx = np.digitize(r_flat, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    tbin = np.bincount(bin_idx, weights=data_flat, minlength=nbins)
    nr = np.bincount(bin_idx, minlength=nbins)
    radialprofile = tbin / np.maximum(nr, 1)

    return bin_centers, radialprofile

    