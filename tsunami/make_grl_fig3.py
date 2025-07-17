import multiprocessing
multiprocessing.set_start_method("fork")
import os
import torch
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.merge import merge
from datasets import SWETsunamiForPlotting
from scipy.interpolate import griddata
import scipy.io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.weight'] = 'bold'
path_pref = os.getcwd()+"/"

#activate conda env topos

## data source ##
### https://topotools.cr.usgs.gov/gmted_viewer/viewer.htm ###
#
# f1 = rasterio.open(path_pref+'geodata/10N090E_20101117_gmted_mea075.tif')
# dta1 = f1.read()
# f2 = rasterio.open(path_pref+'geodata/10N120E_20101117_gmted_mea075.tif')
# dta2 = f2.read()
# f3 = rasterio.open(path_pref+'geodata/10N150E_20101117_gmted_mea075.tif')
# dta3 = f3.read()
# f4 = rasterio.open(path_pref+'geodata/10S090E_20101117_gmted_mea075.tif')
# dta4 = f4.read()
# f5 = rasterio.open(path_pref+'geodata/10S120E_20101117_gmted_mea075.tif')
# dta5 = f5.read()
# f6 = rasterio.open(path_pref+'geodata/10S150E_20101117_gmted_mea075.tif')
# dta6 = f6.read()
# f7 = rasterio.open(path_pref+'geodata/30N090E_20101117_gmted_mea075.tif')
# dta7 = f7.read()
# f8 = rasterio.open(path_pref+'geodata/30N120E_20101117_gmted_mea075.tif')
# dta8 = f8.read()
# f9 = rasterio.open(path_pref+'geodata/30N150E_20101117_gmted_mea075.tif')
# dta9 = f9.read()
# f10 = rasterio.open(path_pref+'geodata/50N090E_20101117_gmted_mea075.tif')
# dta10 = f10.read()
# f11 = rasterio.open(path_pref+'geodata/50N120E_20101117_gmted_mea075.tif')
# dta11 = f11.read()
# f12 = rasterio.open(path_pref+'geodata/50N150E_20101117_gmted_mea075.tif')
# dta12 = f12.read()
# #
# combined_data = merge([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12])
# full_data = combined_data[0][0]
# transform = combined_data[1]
# topo_dataset = rasterio.open(path_pref+'geodata/merged.tif', 'w',driver='GTiff',
#     height=full_data.shape[0],width=full_data.shape[1],count=1,
#     dtype=full_data.dtype,crs='+proj=latlong',transform=transform)
# topo_dataset.write(full_data, 1)
# topo_dataset.close()
topo = rasterio.open(path_pref+'geodata/merged.tif')
topo_data = topo.read()

import geopandas as gpd
# from shapely.geometry import mapping
# from rasterio import mask as msk
#
# def clip_raster(gdf, img):
#     clipped_array, clipped_transform = msk.mask(img, [mapping(gdf.iloc[0].geometry)])
#     clipped_array, clipped_transform = msk.mask(img, [mapping(gdf.iloc[0].geometry)], nodata=(np.amax(clipped_array[0]) + 1))
#     clipped_array[0] = clipped_array[0] + abs(np.amin(clipped_array))
#     value_range = np.amax(clipped_array) + abs(np.amin(clipped_array))
#     return clipped_array, value_range
#
df = gpd.read_file(path_pref+'geodata/ne_10m_admin_0_countries.shp')
#
chi = df.loc[df['ADMIN'] == 'China']
# clipped_array_chi, clipped_transform_chi = msk.mask(topo, [mapping(chi.iloc[0].geometry)])
# china_topography, china_value_range = clip_raster(chi, topo)
# # # # #
jap = df.loc[df['ADMIN'] == 'Japan']
# clipped_array_jap, clipped_transform_jap = msk.mask(topo, [mapping(jap.iloc[0].geometry)])
# japan_topography, japan_value_range = clip_raster(jap, topo)
# # # # #
rus = df.loc[df['ADMIN'] == 'Russia']
# clipped_array_rus, clipped_transform_rus = msk.mask(topo, [mapping(rus.iloc[0].geometry)])
# russia_topography, russia_value_range = clip_raster(rus, topo)
# # # # #
sko = df.loc[df['ADMIN'] == 'South Korea']
# clipped_array_sko, clipped_transform_sko = msk.mask(topo, [mapping(sko.iloc[0].geometry)])
# sko_topography, sko_value_range = clip_raster(sko, topo)
# # # # #
nko = df.loc[df['ADMIN'] == 'North Korea']
# clipped_array_nko, clipped_transform_nko = msk.mask(topo, [mapping(nko.iloc[0].geometry)])
# nko_topography, nko_value_range = clip_raster(nko, topo)
# # # # #
phi = df.loc[df['ADMIN'] == 'Philippines']
# clipped_array_phi, clipped_transform_phi = msk.mask(topo, [mapping(phi.iloc[0].geometry)])
# phi_topography, phi_value_range = clip_raster(phi, topo)
# # # # # #
tai = df.loc[df['ADMIN'] == 'Taiwan']
# clipped_array_tai, clipped_transform_tai = msk.mask(topo, [mapping(tai.iloc[0].geometry)])
# tai_topography, tai_value_range = clip_raster(tai, topo)
# # # # #
vie = df.loc[df['ADMIN'] == 'Vietnam']
# clipped_array_vie, clipped_transform_vie = msk.mask(topo, [mapping(vie.iloc[0].geometry)])
# vie_topography, vie_value_range = clip_raster(vie, topo)
# # # # #
mon = df.loc[df['ADMIN'] == 'Mongolia']
# clipped_array_mon, clipped_transform_mon = msk.mask(topo, [mapping(mon.iloc[0].geometry)])
# mon_topography, mon_value_range = clip_raster(mon, topo)
# mon_topography = mon_topography - np.min(mon_topography)


min_lat = -10
max_lat = 70
min_lon = 90
max_lon = 180
min_lat_plot = -8
max_lat_plot = 65
min_lon_plot = 110
max_lon_plot = 180
nlat , nlon = np.shape(topo_data[0])
lat_vals = np.linspace(max_lat,min_lat,nlat)
lon_vals = np.linspace(min_lon,max_lon,nlon)
min_lat_idx = np.where(lat_vals >= min_lat_plot)[0][-1]
max_lat_idx = np.where(lat_vals >= max_lat_plot)[0][-1]
min_lon_idx = np.where(lon_vals >= min_lon_plot)[0][0]
max_lon_idx = np.where(lon_vals >= max_lon_plot)[0][0]
nlat_plot, nlon_plot = np.shape(topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx])
lat_vals_new = np.linspace(min_lat_plot,max_lat_plot,nlat_plot)
lon_vals_new = np.linspace(min_lon_plot,max_lon_plot,nlon_plot)
dx = (lon_vals_new[1]-lon_vals_new[0])/2.
dy = (lat_vals_new[1]-lat_vals_new[0])/2.
extent = [lon_vals_new[0]-dx, lon_vals_new[-1]+dx, lat_vals_new[0]-dy, lat_vals_new[-1]+dy]
topo_vals = topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]
print("topo_vals_shape:",np.shape(topo_vals))
topo_vals = topo_vals[::80,::80]
lat_vals = lat_vals_new[::80]
lon_vals = lon_vals_new[::80]
print("new topo_vals_shape:",np.shape(topo_vals))


def coord_idx(s, ver_num):
    if ver_num == 34 or ver_num == 0:
        if 0 <= s <= 144:
            return 0
        elif 145 <= s <= 289:
            return 1
        elif 290 <= s <= 434:
            return 2
        elif 435 <= s <= 579:
            return 3
        elif 580 <= s <= 724:
            return 4
        elif 725 <= s <= 869:
            return 5
        elif 870 <= s <= 1014:
            return 6
        else:
            return 7
    else:
        if 0 <= s <= 143:
            return 0
        elif 144 <= s <= 287:
            return 1
        elif 288 <= s <= 431:
            return 2
        elif 432 <= s <= 575:
            return 3
        elif 576 <= s <= 719:
            return 4
        elif 720 <= s <= 863:
            return 5
        elif 864 <= s <= 1007:
            return 6
        else:
            return 7


def training_lons():
    return np.array([136.6180, 139.5560, 139.3290, 138.9350,
                     140.9290, 135.7400, 141.5010, 142.3870])

def training_lats():
    return np.array([33.0700, 28.8560, 28.9320, 29.3840,
                     33.4530, 33.1570, 35.9360, 35.2670])

def unseen_lons():
    return np.array([136.6500,138.2000,138.9000,
                     139.5000,140.2000,140.5000,
                     141.5000,142.5000])

def unseen_lats():
    return np.array([33.1000,31.0000,28.1000,
                     28.8000,29.1000,31.8000,
                     34.2000,36.2000])

### UNSEEN EPI DISTANCES ###
# (136.65,33.10) - 2.78 miles -
# (138.20,31.00) - 119.97 miles -
# (138.90,28.10) - 63.11 miles -
# (139.50,28.80) - 5.144 miles -
# (140.20,29.10) - 42.42 miles -
# (140.50,31.80) - 117.04 miles -
# (141.50,34.20) - 61.02 miles -
# (142.50,36.20) - 64.77 miles -

epi_num = 8
time = 225 # can be 80, 160, 240, or 60, 120, 180, 240 or 75, 150, 225
split = 8020 # can be 9505 or 8020

if time <= 120:
    fname = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_145.mat"
    if split == 9505:
        ver_num = 34
        save_path = os.getcwd() + "/lightning_logs/combined_34_22/recons/"
    else:
        ver_num = 0
        save_path = os.getcwd() + "/lightning_logs/combined_0_2/recons/"
    slice = int(time*6/5) + (epi_num-1)*145 - 1
    print("selected slice from 0-2 hr model: ", slice)
else:
    fname = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_289.mat"
    if split == 9505:
        ver_num = 22
        save_path = os.getcwd() + "/lightning_logs/combined_34_22/recons/"
    else:
        ver_num = 2
        save_path = os.getcwd() + "/lightning_logs/combined_0_2/recons/"
    slice = int((time-120)*6/5) + (epi_num-1)*144 - 1
    print("selected slice from 2-4 hr model: ",slice)


#good for epi 7
min_lat_plot = 2.5
max_lat_plot = 57.5
min_lon_plot = 120
max_lon_plot = 185

#good for epi 3
# min_lat_plot = 0
# max_lat_plot = 60
# min_lon_plot = 115
# max_lon_plot = 175

dx = (lon_vals[1]-lon_vals[0])/2.
dy = (lat_vals[1]-lat_vals[0])/2.
min_lat_idx = np.argsort(np.abs(lat_vals-min_lat_plot))[0]
max_lat_idx = np.argsort(np.abs(lat_vals-max_lat_plot))[0]
min_lon_idx = np.argsort(np.abs(lon_vals-min_lon_plot))[0]
max_lon_idx = np.argsort(np.abs(lon_vals-max_lon_plot))[0]
extent = [lon_vals[min_lon_idx]-dx, lon_vals[max_lon_idx]+dx,
          lat_vals[min_lat_idx]-dy, lat_vals[max_lat_idx]+dy]


out_path = os.getcwd()+"/lightning_logs/version_"+str(ver_num)+"/"
output_im = torch.load(out_path+'tensor_unseen.pt')

true_data, latitude, longitude, mask, max_ht, times, sensors, div = SWETsunamiForPlotting(fname)
true_data *= max_ht
output_im *= max_ht

# NEED INTERPOLATED WAVE_HEIGT, LONGITUDE, LATITUDE, MASK
epi_lons = unseen_lons()
epi_lats = unseen_lats()
epi_idx = coord_idx(slice,ver_num)
tsun_lons_tmp = longitude*(180 / np.pi)
tsun_lats_tmp = latitude*(180 / np.pi)
tsun_lons = tsun_lons_tmp
tsun_lats = tsun_lats_tmp
tsun_lons = tsun_lons[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot) ]
tsun_lats = tsun_lats[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]

int_Lons, int_Lats = np.meshgrid(lon_vals, lat_vals)

epi_lon = epi_lons[epi_idx]
epi_lat = epi_lats[epi_idx]
mins = time
prediction = output_im[slice].cpu().detach().numpy()

true_data_s = true_data[slice][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
prediction = prediction[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]

unstruct_coords = np.array([tsun_lons, tsun_lats]).T
interp_true = np.asarray(
    griddata(unstruct_coords, true_data_s, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
interp_pred = np.asarray(
    griddata(unstruct_coords, prediction, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))

interp_true = interp_true[:,:,0]
interp_pred = interp_pred[:,:,0]
interp_abs_err = np.abs(interp_true-interp_pred)

interp_true[(topo_vals != 0)] = 0.0
interp_pred[(topo_vals != 0)] = 0.0
interp_abs_err[(topo_vals != 0)] = 0.0
min_val = min([np.min(interp_true),np.min(interp_pred),np.min(interp_abs_err)])
max_val = max([np.max(interp_true),np.max(interp_pred),np.max(interp_abs_err)])


width = (extent[1]-extent[0])
height = (extent[3]-extent[2])
aspect = width/height
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(22.8,6))
# fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(34.2,9.0))
from numpy.ma import masked_array
topo_img = masked_array(topo_vals, topo_vals == 0.0)
lons_topo_img = masked_array(int_Lons, topo_vals == 0.0)
lats_topo_img = masked_array(int_Lats, topo_vals == 0.0)
true_tsu_img = masked_array(interp_true, topo_vals != 0.0)
lons_tsu_img = masked_array(int_Lons, topo_vals != 0.0)
lats_tsu_img = masked_array(int_Lats, topo_vals != 0.0)
pred_tsu_img = masked_array(interp_pred, topo_vals != 0.0)
abs_err_img = masked_array(interp_abs_err, topo_vals != 0.0)
shw1 = axs[0].contourf(lons_tsu_img,np.flip(lats_tsu_img),true_tsu_img,levels=50,extent=extent,cmap='bwr')
shw2 = axs[0].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
shw3 = axs[1].contourf(lons_tsu_img,np.flip(lats_tsu_img),pred_tsu_img,levels=50,extent=extent,cmap='bwr')
shw4 = axs[1].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
shw5 = axs[2].contourf(lons_tsu_img,np.flip(lats_tsu_img),abs_err_img,levels=50,extent=extent,cmap='bwr')
shw6 = axs[2].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
shw1.set_clim(min_val,max_val)
shw3.set_clim(min_val,max_val)
shw5.set_clim(min_val,max_val)
import shapely.ops as sops
new_shape = sops.unary_union([el for el in chi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in rus['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in jap['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in nko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in sko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in phi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in tai['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in vie['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
sensor_longs = sensors[:, 0]
sensor_lats = sensors[:, 1]
axs[0].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=60, marker="^",color='yellow',edgecolor='k')
axs[0].scatter(epi_lon,epi_lat,s=120,marker="x",color='k')

# Circle Selected Sensors
axs[0].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=100,facecolors='none', edgecolors='k', linewidth=3)
#axs[0].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=100,facecolors='none', edgecolors='k',linewidth=3)

print("circled sensor 1:",(sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14])
#print("circled sensor 2:",(sensor_longs * (180 / np.pi))[23],(sensor_longs * (180 / np.pi))[23])

axs[0].set_xlim([extent[0], extent[1]])
axs[0].set_ylim([extent[2],extent[3]])
axs[1].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=60, marker="^",color='yellow',edgecolor='k')
axs[1].scatter(epi_lon,epi_lat,s=120,marker="x",color='k')
axs[1].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=100,facecolors='none', edgecolors='k', linewidth=3)
#axs[1].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=100,facecolors='none', edgecolors='k',linewidth=3)


axs[1].set_xlim([extent[0], extent[1]])
axs[1].set_ylim([extent[2],extent[3]])
axs[2].set_xlim([extent[0], extent[1]])
axs[2].set_ylim([extent[2],extent[3]])
# axs[0].set_title(r'\textbf{True ($m$)}')
axs[0].set_title(r'\textbf{True (' + r'$\mathbf{m}$' +r'\textbf{)}')
axs[1].set_title(r'\textbf{Predicted}')
axs[2].set_title(r'\textbf{Absolute Error}')
axs[0].set_aspect(aspect)
axs[1].set_aspect(aspect)
axs[2].set_aspect(aspect)
axs[0].set_ylabel(r'\textbf{Latitude ($^\circ$N)}', labelpad=5)
axs[0].set_xlabel(r'\textbf{Longitude ($^\circ$E)}', labelpad=5)
fig.suptitle(r'\textbf{Epicenter: (}' + r'\textbf{'+str(epi_lon) + r'}'+
             r'\textbf{$^\circ$E, }' + r'\textbf{'+str(epi_lat) + r'}'+
             r'\textbf{$^\circ$N), Time: }'+ r'\textbf{'+ str(mins) + r'}' +
             r'\textbf{ mins}', y=1.0)

value1 = 3
value2 = 5
fig.colorbar(shw1,ax=axs.ravel().tolist(),fraction=0.047, pad=0.04)
axs[0].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
axs[1].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
axs[2].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.savefig(save_path+"h_recon_epi_{}_time_{}_new.png".format(epi_num,time),dpi=400,bbox_inches='tight')


