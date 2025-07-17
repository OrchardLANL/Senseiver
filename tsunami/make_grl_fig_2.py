import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.merge import merge
from datasets import SWETsunamiForPlotting3
import geopandas as gpd
import scipy.io as sio
import os

## data source ##
### https://topotools.cr.usgs.gov/gmted_viewer/viewer.htm ###

### UNCOMMENT BELOW CODE TO MAKE merged.tif file ###
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
#
# combined_data = merge([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12])
# full_data = combined_data[0][0]
# transform = combined_data[1]
# topo_dataset = rasterio.open(os.getcwd()+'/geodata/merged.tif', 'w',driver='GTiff',
#     height=full_data.shape[0],width=full_data.shape[1],count=1,
#     dtype=full_data.dtype,crs='+proj=latlong',transform=transform)
# topo_dataset.write(full_data, 1)
# topo_dataset.close()

topo = rasterio.open(os.getcwd()+'/geodata/merged.tif')
topo_data = topo.read()
df = gpd.read_file(os.getcwd()+'/geodata/ne_10m_admin_0_countries.shp')

chi = df.loc[df['ADMIN'] == 'China']
jap = df.loc[df['ADMIN'] == 'Japan']
rus = df.loc[df['ADMIN'] == 'Russia']
sko = df.loc[df['ADMIN'] == 'South Korea']
nko = df.loc[df['ADMIN'] == 'North Korea']
phi = df.loc[df['ADMIN'] == 'Philippines']
tai = df.loc[df['ADMIN'] == 'Taiwan']
vie = df.loc[df['ADMIN'] == 'Vietnam']



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


def unseen_lons_new():
    return np.array([140.8270, 141.4590, 142.0500, 142.5420,
                     143.2280, 144.0600, 142.6190, 143.4160,135.9050,137.0710,138.0250])

def unseen_lats_new():
    return np.array([33.3620, 36.5340, 41.0190, 34.7450,
                     39.8690, 40.2320,37.8120,41.4150,33.1230,33.1840,34.1750])

min_lat = -10
max_lat = 70
min_lon = 90
max_lon = 180

min_lat_plot = 32.5
max_lat_plot = 45
min_lon_plot = 132.5
max_lon_plot = 145
nlat , nlon = np.shape(topo_data[0])
lat_vals = np.linspace(max_lat,min_lat,nlat)
lon_vals = np.linspace(min_lon,max_lon,nlon)
min_lat_idx = np.where(lat_vals >= min_lat_plot)[0][-1]
max_lat_idx = np.where(lat_vals >= max_lat_plot)[0][-1]
min_lon_idx = np.where(lon_vals >= min_lon_plot)[0][0]
max_lon_idx = np.where(lon_vals >= max_lon_plot)[0][0]
nlat_plot, nlon_plot = np.shape(topo_data[0][max_lat_idx:min_lat_idx,:])
lat_vals_new = np.linspace(min_lat_plot,max_lat_plot,nlat_plot)
lon_vals_new = np.linspace(min_lon_plot,max_lon_plot,nlon_plot)
dx = (lon_vals_new[1]-lon_vals_new[0])/2.
dy = (lat_vals_new[1]-lat_vals_new[0])/2.
extent = [lon_vals_new[0]-dx, lon_vals_new[-1]+dx, lat_vals_new[0]-dy, lat_vals_new[-1]+dy]


plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
### PLOT THE EPICENTERS ###
fig , ax = plt.subplots(figsize=(9,9))
top = topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]

from numpy.ma import masked_array
top_img = masked_array(top, top == 0.0)
ocn_img = masked_array(top, top != 0.0)
ax.set_facecolor('deepskyblue')
im1 = ax.imshow(top_img,cmap='summer', extent=extent)
im1 = ax.scatter(training_lons(),training_lats(),color='k',s=125, marker='x',linewidths=4)
im2 = ax.scatter(unseen_lons_new(),unseen_lats_new(),color='r',s=125,marker='x',linewidths=4)
ax.set_xlim([extent[0], extent[1]])
ax.set_ylim([extent[2],extent[3]])
#fig.text(0.05, 0.90, '(b)', fontsize=24, fontweight='bold', va='top')
plt.legend((im1,im2),('Train epicenters','Test epicenters'),loc='upper left')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.savefig(os.getcwd()+"/fig2_test_train_epis.png",bbox_inches='tight',dpi=600)
plt.close()





#########################################


#### PLOT THE SENSORS ####
min_lat_plot = 0
max_lat_plot = 55
min_lon_plot = 115
max_lon_plot = 170
fname_2hr = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_145.mat"
true_data_2hr, latitude_2hr, longitude_2hr, mask_2hr, max_ht_2hr, times_2hr, sensors_2hr, div_2hr = SWETsunamiForPlotting3(fname_2hr)
tsun_lons = longitude_2hr*(180 / np.pi)
tsun_lats = latitude_2hr*(180 / np.pi)
mat_data = sio.loadmat(os.getcwd()+"/Data/tsunami/"+fname_2hr)
sensor_indices = mat_data['sensor_loc_indices']
sens_lons = tsun_lons[sensor_indices[0]]
sens_lats = tsun_lats[sensor_indices[0]]

sens_lons_inner = sens_lons[(min_lon_plot<=sens_lons) & (sens_lons <= max_lon_plot) & (min_lat_plot<=sens_lats) & (sens_lats <= max_lat_plot)]
sens_lats_inner = sens_lats[(min_lon_plot<=sens_lons) & (sens_lons <= max_lon_plot) & (min_lat_plot<=sens_lats) & (sens_lats <= max_lat_plot)]



nlat , nlon = np.shape(topo_data[0])
lat_vals = np.linspace(max_lat,min_lat,nlat)
lon_vals = np.linspace(min_lon,max_lon,nlon)
min_lat_idx = np.where(lat_vals >= min_lat_plot)[0][-1]
max_lat_idx = np.where(lat_vals >= max_lat_plot)[0][-1]
min_lon_idx = np.where(lon_vals >= min_lon_plot)[0][0]
max_lon_idx = np.where(lon_vals >= max_lon_plot)[0][0]
nlat_plot, nlon_plot = np.shape(topo_data[0][max_lat_idx:min_lat_idx,:])
lat_vals_new = np.linspace(min_lat_plot,max_lat_plot,nlat_plot)
lon_vals_new = np.linspace(min_lon_plot,max_lon_plot,nlon_plot)
dx = (lon_vals_new[1]-lon_vals_new[0])/2.
dy = (lat_vals_new[1]-lat_vals_new[0])/2.
extent = [lon_vals_new[0]-dx, lon_vals_new[-1]+dx, lat_vals_new[0]-dy, lat_vals_new[-1]+dy]
top = topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]



plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

top_img = masked_array(top, top == 0.0)
ocn_img = masked_array(top, top != 0.0)

fig , ax = plt.subplots(figsize=(9,9))
ax.set_facecolor('deepskyblue')
im3 = ax.imshow(top_img,cmap='summer', extent=extent)
im3 = ax.scatter(sens_lons_inner,sens_lats_inner,color='k',marker="^",s=150)


min_lat_plot = 32.5
max_lat_plot = 45
min_lon_plot = 132.5
max_lon_plot = 145
import matplotlib.patches as patches
rect = patches.Rectangle((min_lon_plot, min_lat_plot), max_lon_plot-min_lon_plot, max_lat_plot-min_lat_plot,
                        linewidth=4, edgecolor='red', linestyle='--', facecolor='none')
ax.add_patch(rect)
ax.set_xlim([extent[0], extent[1]])
ax.set_ylim([extent[2],extent[3]])
ax.set_ylabel(r'Latitude ($^\circ$N)', labelpad=2.5)
ax.set_xlabel(r'Longitude ($^\circ$E)', labelpad=2.5)
ax.legend(['Dart Buoys'])
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.savefig(os.getcwd()+"/fig2_buoys.png",bbox_inches='tight',dpi=600)
plt.close()
