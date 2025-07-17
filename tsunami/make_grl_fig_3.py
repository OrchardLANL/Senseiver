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
plt.rcParams["font.size"] = 32
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 32

#activate conda env topos

## data source ##
### https://topotools.cr.usgs.gov/gmted_viewer/viewer.htm ###
#
# f1 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N090E_20101117_gmted_mea075.tif')
# dta1 = f1.read()
# f2 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N120E_20101117_gmted_mea075.tif')
# dta2 = f2.read()
# f3 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N150E_20101117_gmted_mea075.tif')
# dta3 = f3.read()
# f4 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S090E_20101117_gmted_mea075.tif')
# dta4 = f4.read()
# f5 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S120E_20101117_gmted_mea075.tif')
# dta5 = f5.read()
# f6 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S150E_20101117_gmted_mea075.tif')
# dta6 = f6.read()
# f7 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N090E_20101117_gmted_mea075.tif')
# dta7 = f7.read()
# f8 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N120E_20101117_gmted_mea075.tif')
# dta8 = f8.read()
# f9 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N150E_20101117_gmted_mea075.tif')
# dta9 = f9.read()
# f10 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N090E_20101117_gmted_mea075.tif')
# dta10 = f10.read()
# f11 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N120E_20101117_gmted_mea075.tif')
# dta11 = f11.read()
# f12 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N150E_20101117_gmted_mea075.tif')
# dta12 = f12.read()
# #
# combined_data = merge([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12])
# full_data = combined_data[0][0]
# transform = combined_data[1]
# topo_dataset = rasterio.open('/Users/emcdugald/tsunseiver/geodata/merged.tif', 'w',driver='GTiff',
#     height=full_data.shape[0],width=full_data.shape[1],count=1,
#     dtype=full_data.dtype,crs='+proj=latlong',transform=transform)
# topo_dataset.write(full_data, 1)
# topo_dataset.close()

topo = rasterio.open(os.getcwd()+'/geodata/merged.tif')
topo_data = topo.read()

import geopandas as gpd
df = gpd.read_file(os.getcwd()+'/geodata/ne_10m_admin_0_countries.shp')
chi = df.loc[df['ADMIN'] == 'China']
jap = df.loc[df['ADMIN'] == 'Japan']
rus = df.loc[df['ADMIN'] == 'Russia']
sko = df.loc[df['ADMIN'] == 'South Korea']
nko = df.loc[df['ADMIN'] == 'North Korea']
phi = df.loc[df['ADMIN'] == 'Philippines']
tai = df.loc[df['ADMIN'] == 'Taiwan']
vie = df.loc[df['ADMIN'] == 'Vietnam']
mon = df.loc[df['ADMIN'] == 'Mongolia']

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
    if ver_num == 122:
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
        elif 1015 <= s <= 1159:
            return 7
        elif 1160 <= s <= 1304:
            return 8
        elif 1305 <= s <= 1449:
            return 9
        else:
            return 10
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
        elif 1008 <= s <= 1151:
            return 7
        elif 1152 <= s <= 1295:
            return 8
        elif 1296 <= s <= 1439:
            return 9
        else:
            return 10


def training_lons():
    return np.array([136.6180, 137.9440, 141.4890, 141.7020,
                     142.2410, 142.4220, 143.0040,
                     143.3780, 143.4350, 143.8990,
                     136.8940])

def training_lats():
    return np.array([33.0700, 33.1670, 34.5520, 36.1940,
                     37.1470, 40.5910, 38.6600,
                     40.1760, 40.8600, 42.0840,
                     33.6830])


def unseen_lons():
    return np.array([140.8270, 141.4590, 142.0500, 142.5420,
                     143.2280, 144.0600, 142.6190,
                     143.4160,135.9050,137.0710,
                     138.0250])

def unseen_lats():
    return np.array([33.3620, 36.5340, 41.0190, 34.7450,
                     39.8690, 40.2320, 37.8120,
                     41.4150, 33.1230, 33.1840,
                     34.1750])


epi_num = 1
save_path = os.getcwd() + "/lightning_logs/combined_122_125/recons/"
t0 = 75
t1 = 150
t2 = 225
t0_ver_num = 122
t1_ver_num = 125
t2_ver_num = 125
t0_fname = "unseen_agg_11_sims_0_time_ss_2_ss_unstruct_ntimes_3179_wd_new_epis_v7_tf_145.mat"
t1_fname = "unseen_agg_11_sims_0_time_ss_2_ss_unstruct_ntimes_3179_wd_new_epis_v7_tf_289.mat"
t2_fname = "unseen_agg_11_sims_0_time_ss_2_ss_unstruct_ntimes_3179_wd_new_epis_v7_tf_289.mat"
t0_slice =  int(t0*6/5) + (epi_num-1)*145 - 1
t1_slice = int((t1-120)*6/5) + (epi_num-1)*144 - 1
t2_slice = int((t2-120)*6/5) + (epi_num-1)*144 - 1


#good for epi 7
# min_lat_plot = 2.5 #0
# max_lat_plot = 57.5 # 60
# min_lon_plot = 120 # 115
# max_lon_plot = 185 # 175

min_lat_plot = 2.5 #0
max_lat_plot = 60 # 60
min_lon_plot = 120 # 115
max_lon_plot = 187.5 # 175

dx = (lon_vals[1]-lon_vals[0])/2.
dy = (lat_vals[1]-lat_vals[0])/2.
min_lat_idx = np.argsort(np.abs(lat_vals-min_lat_plot))[0]
max_lat_idx = np.argsort(np.abs(lat_vals-max_lat_plot))[0]
min_lon_idx = np.argsort(np.abs(lon_vals-min_lon_plot))[0]
max_lon_idx = np.argsort(np.abs(lon_vals-max_lon_plot))[0]
extent = [lon_vals[min_lon_idx]-dx, lon_vals[max_lon_idx]+dx,
          lat_vals[min_lat_idx]-dy, lat_vals[max_lat_idx]+dy]


t0_out_path = os.getcwd()+"/version_"+str(t0_ver_num)+"/"
t0_output_im = torch.load(t0_out_path+'tensor_unseen.pt')
t0_true_data, latitude, longitude, mask, max_ht, times, sensors, div = SWETsunamiForPlotting(t0_fname)
t0_true_data *= max_ht
t0_output_im *= max_ht

t1_out_path = os.getcwd()+"/version_"+str(t1_ver_num)+"/"
t1_output_im = torch.load(t1_out_path+'tensor_unseen.pt')
t1_true_data, latitude, longitude, mask, max_ht, times, sensors, div = SWETsunamiForPlotting(t1_fname)
t1_true_data *= max_ht
t1_output_im *= max_ht

t2_out_path = os.getcwd()+"/version_"+str(t2_ver_num)+"/"
t2_output_im = torch.load(t2_out_path+'tensor_unseen.pt')
t2_true_data, latitude, longitude, mask, max_ht, times, sensors, div = SWETsunamiForPlotting(t2_fname)
t2_true_data *= max_ht
t2_output_im *= max_ht

# NEED INTERPOLATED WAVE_HEIGT, LONGITUDE, LATITUDE, MASK
epi_lons = unseen_lons()
epi_lats = unseen_lats()
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
unstruct_coords = np.array([tsun_lons, tsun_lats]).T

t0_epi_idx = coord_idx(t0_slice,t0_ver_num)
t0_epi_lon = epi_lons[t0_epi_idx]
t0_epi_lat = epi_lats[t0_epi_idx]

t1_epi_idx = coord_idx(t1_slice,t1_ver_num)
t1_epi_lon = epi_lons[t1_epi_idx]
t1_epi_lat = epi_lats[t1_epi_idx]

t2_epi_idx = coord_idx(t2_slice,t2_ver_num)
t2_epi_lon = epi_lons[t2_epi_idx]
t2_epi_lat = epi_lats[t2_epi_idx]


t0_prediction = t0_output_im[t0_slice].cpu().detach().numpy()
t0_true_data_s = t0_true_data[t0_slice][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
t0_prediction = t0_prediction[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
t0_interp_true = np.asarray(
    griddata(unstruct_coords, t0_true_data_s, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
t0_interp_pred = np.asarray(
    griddata(unstruct_coords, t0_prediction, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
t0_interp_true = t0_interp_true[:,:,0]
t0_interp_pred = t0_interp_pred[:,:,0]
t0_interp_abs_err = np.abs(t0_interp_true-t0_interp_pred)
t0_interp_true[(topo_vals != 0)] = 0.0
t0_interp_pred[(topo_vals != 0)] = 0.0
t0_interp_abs_err[(topo_vals != 0)] = 0.0

t1_prediction = t1_output_im[t1_slice].cpu().detach().numpy()
t1_true_data_s = t1_true_data[t1_slice][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
t1_prediction = t1_prediction[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
t1_interp_true = np.asarray(
    griddata(unstruct_coords, t1_true_data_s, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
t1_interp_pred = np.asarray(
    griddata(unstruct_coords, t1_prediction, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
t1_interp_true = t1_interp_true[:,:,0]
t1_interp_pred = t1_interp_pred[:,:,0]
t1_interp_abs_err = np.abs(t1_interp_true-t1_interp_pred)
t1_interp_true[(topo_vals != 0)] = 0.0
t1_interp_pred[(topo_vals != 0)] = 0.0
t1_interp_abs_err[(topo_vals != 0)] = 0.0


t2_prediction = t2_output_im[t2_slice].cpu().detach().numpy()
t2_true_data_s = t2_true_data[t2_slice][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
t2_prediction = t2_prediction[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
t2_interp_true = np.asarray(
    griddata(unstruct_coords, t2_true_data_s, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
t2_interp_pred = np.asarray(
    griddata(unstruct_coords, t2_prediction, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
t2_interp_true = t2_interp_true[:,:,0]
t2_interp_pred = t2_interp_pred[:,:,0]
t2_interp_abs_err = np.abs(t2_interp_true-t2_interp_pred)
t2_interp_true[(topo_vals != 0)] = 0.0
t2_interp_pred[(topo_vals != 0)] = 0.0
t2_interp_abs_err[(topo_vals != 0)] = 0.0


t0_min_val = min([np.min(t0_interp_true),np.min(t0_interp_pred)])
t0_max_val = max([np.max(t0_interp_true),np.max(t0_interp_pred)])
t1_min_val = min([np.min(t1_interp_true),np.min(t1_interp_pred)])
t1_max_val = max([np.max(t1_interp_true),np.max(t1_interp_pred)])
t2_min_val = min([np.min(t2_interp_true),np.min(t2_interp_pred)])
t2_max_val = max([np.max(t2_interp_true),np.max(t2_interp_pred)])

cmin_val = min([t0_min_val,t1_min_val,t2_min_val])
cmax_val = max([t0_max_val,t1_max_val,t2_max_val])
emin_val = min([np.min(t0_interp_abs_err),np.min(t1_interp_abs_err),np.min(t2_interp_abs_err)])
emax_val = max([np.max(t0_interp_abs_err),np.max(t1_interp_abs_err),np.max(t2_interp_abs_err)])


width = (extent[1]-extent[0])
height = (extent[3]-extent[2])
aspect = width/height

fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(16.75,16.25),constrained_layout=True,gridspec_kw={'wspace': 0.01,'hspace': 0.005})
from numpy.ma import masked_array
topo_img = masked_array(topo_vals, topo_vals == 0.0)
lons_topo_img = masked_array(int_Lons, topo_vals == 0.0)
lats_topo_img = masked_array(int_Lats, topo_vals == 0.0)
lons_tsu_img = masked_array(int_Lons, topo_vals != 0.0)
lats_tsu_img = masked_array(int_Lats, topo_vals != 0.0)

t0_true_tsu_img = masked_array(t0_interp_true, topo_vals != 0.0)
t0_pred_tsu_img = masked_array(t0_interp_pred, topo_vals != 0.0)
t0_abs_err_img = masked_array(t0_interp_abs_err, topo_vals != 0.0)


t0_shw1 = axs[0,0].contourf(lons_tsu_img,np.flip(lats_tsu_img),t0_true_tsu_img,levels=50,extent=extent,cmap='ocean')
t0_shw2 = axs[0,0].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t0_shw3 = axs[0,1].contourf(lons_tsu_img,np.flip(lats_tsu_img),t0_pred_tsu_img,levels=50,extent=extent,cmap='ocean')
t0_shw4 = axs[0,1].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t0_shw5 = axs[0,2].contourf(lons_tsu_img,np.flip(lats_tsu_img),t0_abs_err_img,levels=50,extent=extent,cmap='Blues')
t0_shw6 = axs[0,2].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t0_shw1.set_clim(cmin_val,cmax_val)
t0_shw3.set_clim(cmin_val,cmax_val)

t1_true_tsu_img = masked_array(t1_interp_true, topo_vals != 0.0)
t1_pred_tsu_img = masked_array(t1_interp_pred, topo_vals != 0.0)
t1_abs_err_img = masked_array(t1_interp_abs_err, topo_vals != 0.0)

t1_shw1 = axs[1,0].contourf(lons_tsu_img,np.flip(lats_tsu_img),t1_true_tsu_img,levels=50,extent=extent,cmap='ocean')
t1_shw2 = axs[1,0].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t1_shw3 = axs[1,1].contourf(lons_tsu_img,np.flip(lats_tsu_img),t1_pred_tsu_img,levels=50,extent=extent,cmap='ocean')
t1_shw4 = axs[1,1].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t1_shw5 = axs[1,2].contourf(lons_tsu_img,np.flip(lats_tsu_img),t1_abs_err_img,levels=50,extent=extent,cmap='Blues')

t1_shw6 = axs[1,2].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t1_shw1.set_clim(cmin_val,cmax_val)
t1_shw3.set_clim(cmin_val,cmax_val)

t2_true_tsu_img = masked_array(t2_interp_true, topo_vals != 0.0)
t2_pred_tsu_img = masked_array(t2_interp_pred, topo_vals != 0.0)
t2_abs_err_img = masked_array(t2_interp_abs_err, topo_vals != 0.0)

t2_shw1 = axs[2,0].contourf(lons_tsu_img,np.flip(lats_tsu_img),t2_true_tsu_img,levels=50,extent=extent,cmap='ocean')
t2_shw2 = axs[2,0].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t2_shw3 = axs[2,1].contourf(lons_tsu_img,np.flip(lats_tsu_img),t2_pred_tsu_img,levels=50,extent=extent,cmap='ocean')
t2_shw4 = axs[2,1].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t2_shw5 = axs[2,2].contourf(lons_tsu_img,np.flip(lats_tsu_img),t2_abs_err_img,levels=50,extent=extent,cmap='Blues')

t2_shw6 = axs[2,2].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
t2_shw1.set_clim(cmin_val,cmax_val)
t2_shw3.set_clim(cmin_val,cmax_val)


import shapely.ops as sops
new_shape = sops.unary_union([el for el in chi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0,0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0,1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0,2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1,0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1,1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1,2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2,0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2,1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2,2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in rus['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in jap['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in nko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in sko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in phi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in tai['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in vie['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[0, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1, 2].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2, 2].plot(*geom.exterior.xy, c='k', lw=1.0)

sensor_longs = sensors[:, 0]
sensor_lats = sensors[:, 1]

axs[0,0].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=200, marker="^",color='yellow',edgecolor='k')
axs[0,0].scatter(t0_epi_lon,t0_epi_lat,s=800,marker="x",color='r')
axs[0,0].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[1,0].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=200, marker="^",color='yellow',edgecolor='k')
axs[1,0].scatter(t1_epi_lon,t1_epi_lat,s=800,marker="x",color='r')
axs[1,0].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[2,0].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=200, marker="^",color='yellow',edgecolor='k')
axs[2,0].scatter(t2_epi_lon,t2_epi_lat,s=800,marker="x",color='r')
axs[2,0].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[0,0].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[1,0].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[2,0].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=800,facecolors='none', edgecolors='k', linewidth=5)


print("circled sensor 1:",(sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14])
print("circled sensor 2:",(sensor_longs * (180 / np.pi))[23],(sensor_longs * (180 / np.pi))[23])

axs[0,1].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=200, marker="^",color='yellow',edgecolor='k')
axs[0,1].scatter(t0_epi_lon,t0_epi_lat,s=800,marker="x",color='r')
axs[0,1].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[1,1].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=200, marker="^",color='yellow',edgecolor='k')
axs[1,1].scatter(t1_epi_lon,t1_epi_lat,s=800,marker="x",color='r')
axs[1,1].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[2,1].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=200, marker="^",color='yellow',edgecolor='k')
axs[2,1].scatter(t2_epi_lon,t2_epi_lat,s=800,marker="x",color='r')
axs[2,1].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[0,1].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[1,1].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=800,facecolors='none', edgecolors='k', linewidth=5)
axs[2,1].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=800,facecolors='none', edgecolors='k', linewidth=5)

axs[0,0].set_xlim([extent[0], extent[1]])
axs[0,0].set_ylim([extent[2],extent[3]])
axs[1,0].set_xlim([extent[0], extent[1]])
axs[1,0].set_ylim([extent[2],extent[3]])
axs[2,0].set_xlim([extent[0], extent[1]])
axs[2,0].set_ylim([extent[2],extent[3]])
axs[0,1].set_xlim([extent[0], extent[1]])
axs[0,1].set_ylim([extent[2],extent[3]])
axs[1,1].set_xlim([extent[0], extent[1]])
axs[1,1].set_ylim([extent[2],extent[3]])
axs[2,1].set_xlim([extent[0], extent[1]])
axs[2,1].set_ylim([extent[2],extent[3]])
axs[0,2].set_xlim([extent[0], extent[1]])
axs[0,2].set_ylim([extent[2],extent[3]])
axs[1,2].set_xlim([extent[0], extent[1]])
axs[1,2].set_ylim([extent[2],extent[3]])
axs[2,2].set_xlim([extent[0], extent[1]])
axs[2,2].set_ylim([extent[2],extent[3]])
axs[0,0].set_title(r'\textbf{True}')
axs[0,1].set_title(r'\textbf{Predicted}')
axs[0,2].set_title(r'\textbf{Absolute Error}')
axs[0,0].set_aspect(aspect)
axs[0,1].set_aspect(aspect)
axs[0,2].set_aspect(aspect)
axs[1,0].set_aspect(aspect)
axs[1,1].set_aspect(aspect)
axs[1,2].set_aspect(aspect)
axs[2,0].set_aspect(aspect)
axs[2,1].set_aspect(aspect)
axs[2,2].set_aspect(aspect)


fig.canvas.draw()
thickness = 0.025  # colorbar thickness in figure coordinates
gap = 0.01         # gap between colorbars in figure coordinates
bottom_offset = 0.06  # vertical offset below subplots (adjust as needed)

# Get positions of bottom row axes
pos0 = axs[2, 0].get_position()
pos1 = axs[2, 1].get_position()
pos2 = axs[2, 2].get_position()

# Colorbar 1: under columns 0 and 1
cbar1_x0 = pos0.x0
cbar1_x1 = pos1.x1 - gap/2
cbar1_width = cbar1_x1 - cbar1_x0

# Colorbar 2: under column 2
cbar2_x0 = pos2.x0 + gap/2
cbar2_x1 = pos2.x1
cbar2_width = cbar2_x1 - cbar2_x0

# Vertical position for both colorbars
cbar_y = pos0.y0 - bottom_offset

# Create axes for colorbars
cax1 = fig.add_axes([cbar1_x0, cbar_y, cbar1_width, thickness])
cax2 = fig.add_axes([cbar2_x0, cbar_y, cbar2_width, thickness])

# Create colorbars and set labels
cbar1 = fig.colorbar(t0_shw1, cax=cax1, orientation='horizontal')
cbar1.set_label(r'\textbf{Wave Height (' + r'$\mathbf{m}$' + r'\textbf{)}')

cbar2 = fig.colorbar(t0_shw5, cax=cax2, orientation='horizontal',format="%.2f")
cbar2.set_label(r'\textbf{Error}')
##############

lon_ticks = np.linspace(extent[0]+2,extent[1]-2,5)

lat_ticks = np.linspace(extent[2]+2,extent[3]-2,5)

lon_labels = []
for x in lon_ticks:
    if x < 0:
        txt = r'\textbf{' + str(abs(np.round(x, 1))) + '}' + r'\textbf{$^\circ$W}'
        lon_labels.append(txt)
    elif x > 0:
        txt = r'\textbf{' + str(abs(np.round(x, 1))) + '}' + r'\textbf{$^\circ$E}'
        lon_labels.append(txt)
    else:
        lon_labels.append(r"$0^\circ$")

lat_labels = []
for y in lat_ticks:
    if y < 0:
        txt = r'\textbf{' + str(abs(np.round(y, 1))) + '}' + r'\textbf{$^\circ$S}'
        lat_labels.append(txt)
    elif y > 0:
        txt = r'\textbf{' + str(abs(np.round(y, 1))) + '}' + r'\textbf{$^\circ$N}'
        lat_labels.append(txt)
    else:
        lat_labels.append(r"$0^\circ$")


axs[0,0].set_yticks(lat_ticks)
axs[0,0].set_yticklabels(lat_labels)
axs[1,0].set_yticks(lat_ticks)
axs[1,0].set_yticklabels(lat_labels)
axs[2,0].set_yticks(lat_ticks)
axs[2,0].set_yticklabels(lat_labels)
axs[2,0].set_xticks(lon_ticks)
axs[2,0].set_xticklabels(lon_labels,rotation=90)
axs[2,1].set_xticks(lon_ticks)
axs[2,1].set_xticklabels(lon_labels,rotation=90)
axs[2,2].set_xticks(lon_ticks)
axs[2,2].set_xticklabels(lon_labels,rotation=90)


for ax in fig.get_axes():
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


nrows, ncols = axs.shape  # assuming axs is a 2D array


# Optional: adjust tick label font size or rotation
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

for ax in axs.flat:
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.grid(color='k', linestyle='--', linewidth=0.5)

for i in range(nrows):
    for j in range(ncols):
        ax = axs[i, j]
        # Remove y tick labels for all but first column
        if j > 0:
            ax.set_yticklabels([])
            # Optionally, also remove minor ticks
            ax.tick_params(labelleft=False)
        # Remove x tick labels for all but last row
        if i < nrows - 1:
            ax.set_xticklabels([])
            ax.tick_params(labelbottom=False)

for label in cbar2.ax.get_xticklabels():
    label.set_fontsize(16)
#####################################################################

plt.savefig(save_path+"h_recon_epi_{}_t0_{}_new_v3.png".format(epi_num,75),dpi=600,bbox_inches='tight')



