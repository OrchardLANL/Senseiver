import multiprocessing
multiprocessing.set_start_method("fork")
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import SWETsunamiForPlotting3
from scipy.interpolate import LinearNDInterpolator
from scipy.signal import medfilt
import scipy.io as sio
import sys
import os

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 28
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
path = os.getcwd()+"/virtual_waveforms/"


### EPICENTERS OF UNSEEN DATA ###
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


epi_lons = unseen_lons()
epi_lats = unseen_lats()


ver_num_2hr = 122
ver_num_4hr = 125
fig_dir = "/figs_122_125"
log_dir = "/logs_122_125"
fname_2hr = "unseen_agg_11_sims_0_time_ss_2_ss_unstruct_ntimes_3179_wd_new_epis_v7_tf_145.mat"
fname_4hr = "unseen_agg_11_sims_0_time_ss_2_ss_unstruct_ntimes_3179_wd_new_epis_v7_tf_289.mat"



logfile = open(path+log_dir+"/metrics_v2.out", 'w')
sys.stdout = logfile


type = "unseen"
out_path_2hr = os.getcwd()+"/lightning_logs_0706/version_"+str(ver_num_2hr)+"/"
output_im_2hr = torch.load(out_path_2hr+'tensor'+'_'+str(type)+'.pt').numpy()
out_path_4hr = os.getcwd()+"/lightning_logs_0706/version_"+str(ver_num_4hr)+"/"
output_im_4hr = torch.load(out_path_4hr+'tensor'+'_'+str(type)+'.pt').numpy()
true_data_2hr, latitude_2hr, longitude_2hr, mask_2hr, max_ht_2hr, times_2hr, sensors_2hr, div_2hr = SWETsunamiForPlotting3(fname_2hr)
true_data_2hr *= max_ht_2hr
output_im_2hr *= max_ht_2hr
true_data_4hr, latitude_4hr, longitude_4hr, mask_4hr, max_ht_4hr, times_4hr, sensors_4hr, div_4hr = SWETsunamiForPlotting3(fname_4hr)
true_data_4hr *= max_ht_4hr
output_im_4hr *= max_ht_4hr
tsun_lons = longitude_4hr*(180 / np.pi)
tsun_lats = latitude_4hr*(180 / np.pi)




### COLLECT SIMULATION DATA INTO ONE FOUR HOUR SET ###
intervals_2hr = [(0, 145), (145, 290), (290, 435), (435, 580), (580, 725), (725, 870), (870, 1015),
                         (1015, 1160), (1160, 1305), (1305, 1450), (1450, 1595)]
sim1_2hr = true_data_2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
sim2_2hr = true_data_2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
sim3_2hr = true_data_2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
sim4_2hr = true_data_2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
sim5_2hr = true_data_2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
sim6_2hr = true_data_2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
sim7_2hr = true_data_2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
sim8_2hr = true_data_2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
sim9_2hr = true_data_2hr[intervals_2hr[8][0]:intervals_2hr[8][1]]
sim10_2hr = true_data_2hr[intervals_2hr[9][0]:intervals_2hr[9][1]]
sim11_2hr = true_data_2hr[intervals_2hr[10][0]:intervals_2hr[10][1]]
intervals_4hr = [(0, 144), (144, 288), (288, 432), (432, 576), (576, 720), (720, 864), (864, 1008),
                         (1008, 1152), (1152, 1296), (1296, 1440), (1440, 1584)]
sim1_4hr = true_data_4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
sim2_4hr = true_data_4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
sim3_4hr = true_data_4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
sim4_4hr = true_data_4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
sim5_4hr = true_data_4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
sim6_4hr = true_data_4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
sim7_4hr = true_data_4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
sim8_4hr = true_data_4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
sim9_4hr = true_data_4hr[intervals_4hr[8][0]:intervals_4hr[8][1]]
sim10_4hr = true_data_4hr[intervals_4hr[9][0]:intervals_4hr[9][1]]
sim11_4hr = true_data_4hr[intervals_4hr[10][0]:intervals_4hr[10][1]]
sim1_full = np.concatenate((sim1_2hr,sim1_4hr),axis=0)
sim2_full = np.concatenate((sim2_2hr,sim2_4hr),axis=0)
sim3_full = np.concatenate((sim3_2hr,sim3_4hr),axis=0)
sim4_full = np.concatenate((sim4_2hr,sim4_4hr),axis=0)
sim5_full = np.concatenate((sim5_2hr,sim5_4hr),axis=0)
sim6_full = np.concatenate((sim6_2hr,sim6_4hr),axis=0)
sim7_full = np.concatenate((sim7_2hr,sim7_4hr),axis=0)
sim8_full = np.concatenate((sim8_2hr,sim8_4hr),axis=0)
sim9_full = np.concatenate((sim9_2hr,sim9_4hr),axis=0)
sim10_full = np.concatenate((sim10_2hr,sim10_4hr),axis=0)
sim11_full = np.concatenate((sim11_2hr,sim11_4hr),axis=0)

### COLLECT SENSEIVER RECONS INTO ONE FOUR HOUR SET ###
sens1_2hr = output_im_2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
sens2_2hr = output_im_2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
sens3_2hr = output_im_2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
sens4_2hr = output_im_2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
sens5_2hr = output_im_2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
sens6_2hr = output_im_2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
sens7_2hr = output_im_2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
sens8_2hr = output_im_2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
sens9_2hr = output_im_2hr[intervals_2hr[8][0]:intervals_2hr[8][1]]
sens10_2hr = output_im_2hr[intervals_2hr[9][0]:intervals_2hr[9][1]]
sens11_2hr = output_im_2hr[intervals_2hr[10][0]:intervals_2hr[10][1]]
sens1_4hr = output_im_4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
sens2_4hr = output_im_4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
sens3_4hr = output_im_4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
sens4_4hr = output_im_4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
sens5_4hr = output_im_4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
sens6_4hr = output_im_4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
sens7_4hr = output_im_4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
sens8_4hr = output_im_4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
sens9_4hr = output_im_4hr[intervals_4hr[8][0]:intervals_4hr[8][1]]
sens10_4hr = output_im_4hr[intervals_4hr[9][0]:intervals_4hr[9][1]]
sens11_4hr = output_im_4hr[intervals_4hr[10][0]:intervals_4hr[10][1]]
sens1_full = np.concatenate((sens1_2hr,sens1_4hr),axis=0)
sens2_full = np.concatenate((sens2_2hr,sens2_4hr),axis=0)
sens3_full = np.concatenate((sens3_2hr,sens3_4hr),axis=0)
sens4_full = np.concatenate((sens4_2hr,sens4_4hr),axis=0)
sens5_full = np.concatenate((sens5_2hr,sens5_4hr),axis=0)
sens6_full = np.concatenate((sens6_2hr,sens6_4hr),axis=0)
sens7_full = np.concatenate((sens7_2hr,sens7_4hr),axis=0)
sens8_full = np.concatenate((sens8_2hr,sens8_4hr),axis=0)
sens9_full = np.concatenate((sens9_2hr,sens9_4hr),axis=0)
sens10_full = np.concatenate((sens10_2hr,sens10_4hr),axis=0)
sens11_full = np.concatenate((sens11_2hr,sens11_4hr),axis=0)


### SENSOR INDICES, LOCATIONS AND BATHYMETRY ###
sensors = sensors_2hr*(180/np.pi)
mat_data = sio.loadmat(os.getcwd()+"/Data/tsunami/"+fname_2hr)
sensor_indices = mat_data['sensor_loc_indices']
sens_lons = tsun_lons[sensor_indices[0]]
sens_lats = tsun_lats[sensor_indices[0]]
bathymetry = mat_data['ocn_floor']
xy = np.c_[tsun_lons, tsun_lats]
bath = LinearNDInterpolator(xy, bathymetry[0])

### RESTRICT DATA TO SENSIBLE WINDOW ###
min_lat_plot = 10
max_lat_plot = 45
min_lon_plot = 125
max_lon_plot = 160
in_window_indicator = np.zeros_like(sens_lons)
for i in range(len(sens_lons)):
    if min_lon_plot <= sens_lons[i] <= max_lon_plot and min_lat_plot <= sens_lats[i] <= max_lat_plot:
        in_window_indicator[i] += 1
in_window_sensor_indices = sensor_indices[0][np.where(in_window_indicator==1.0)]
sens_lons_inner = tsun_lons[in_window_sensor_indices]
sens_lats_inner = tsun_lats[in_window_sensor_indices]
sim1_sens_vals_inner = sim1_full[:,in_window_sensor_indices][:,:,0].T
sim2_sens_vals_inner = sim2_full[:,in_window_sensor_indices][:,:,0].T
sim3_sens_vals_inner = sim3_full[:,in_window_sensor_indices][:,:,0].T
sim4_sens_vals_inner = sim4_full[:,in_window_sensor_indices][:,:,0].T
sim5_sens_vals_inner = sim5_full[:,in_window_sensor_indices][:,:,0].T
sim6_sens_vals_inner = sim6_full[:,in_window_sensor_indices][:,:,0].T
sim7_sens_vals_inner = sim7_full[:,in_window_sensor_indices][:,:,0].T
sim8_sens_vals_inner = sim8_full[:,in_window_sensor_indices][:,:,0].T
sim9_sens_vals_inner = sim9_full[:,in_window_sensor_indices][:,:,0].T
sim10_sens_vals_inner = sim10_full[:,in_window_sensor_indices][:,:,0].T
sim11_sens_vals_inner = sim11_full[:,in_window_sensor_indices][:,:,0].T


### MAKE A MATRIX CONSISTING OF DISTANCES BETWEEN PAIRS OF SENSORS ###
sens_distance_arr = np.zeros(shape=(len(sens_lons_inner),len(sens_lons_inner)))
for i in range(len(sens_lons_inner)):
    for j in range(len(sens_lons_inner)):
        if i<j:
            sens_distance_arr[i,j] += np.sqrt((sens_lons_inner[i]-sens_lons_inner[j])**2+(sens_lats_inner[i]-sens_lats_inner[j])**2)

### RETRIEVE SENSOR PAIRS FOR INTERPOLATION METHOD BASED ON PROXIMITY ###
### IE, THE 10 CLOSEST PAIRS IN THE MATRIX DETERMINE THE PAIRS WE WILL CONIDER ###

flat_nonzero = sens_distance_arr[np.nonzero(sens_distance_arr)]
five_smallest_indices_flat = np.argpartition(flat_nonzero, 10)[:10] #twelve here due to redundancy
original_indices = np.nonzero(sens_distance_arr)
five_smallest_indices = tuple(coord[five_smallest_indices_flat] for coord in original_indices)

sensor_pair_locs = np.zeros((10, 2),dtype=object)
for i in range(10):
    sensor_pair_locs[i][0] = (sens_lons_inner[five_smallest_indices[0][i]],sens_lats_inner[five_smallest_indices[0][i]])
    sensor_pair_locs[i][1] = (sens_lons_inner[five_smallest_indices[1][i]], sens_lats_inner[five_smallest_indices[1][i]])


### USING MIDPOINTS TO MAKE VIRTUAL SENSORS ###
virtual_sensor_locs_1 = np.zeros(10,dtype=object)
for i in range(10):
    virtual_sensor_locs_1[i] = (.5*(sensor_pair_locs[i][0][0] + sensor_pair_locs[i][1][0]),.5*(sensor_pair_locs[i][0][1] + sensor_pair_locs[i][1][1]))

inds = [2,5,6,8]
virtual_sensor_locs = np.delete(virtual_sensor_locs_1,inds)

virtual_lons, virtual_lats = zip(*virtual_sensor_locs)
virtual_lons = np.array(virtual_lons)
virtual_lats = np.array(virtual_lats)

### GET REAL SENSOR LOCS THAT DEFINE THE VIRTUAL SENSORS ###
real_sens_loc_indices = np.unique(np.array(five_smallest_indices).flatten())
n_real_sens = len(real_sens_loc_indices)
real_sensor_locs = np.zeros(n_real_sens,dtype=object)
for i in range(n_real_sens):
    real_sensor_locs[i] = (sens_lons_inner[real_sens_loc_indices[i]],sens_lats_inner[real_sens_loc_indices[i]])
real_lons, real_lats = zip(*real_sensor_locs)
real_lons = np.array(real_lons)
real_lats = np.array(real_lats)


### MAKE BATHYMETRY FIG HERE #######################################
import rasterio
import os
topo = rasterio.open(os.getcwd()+'/geodata/merged.tif')
topo_data = topo.read()
import geopandas as gpd
from scipy.interpolate import griddata
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
min_lat_plot = 10
max_lat_plot = 45
min_lon_plot = 116
max_lon_plot = 169
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
topo_vals = topo_vals[::80,::80]
lat_vals = lat_vals_new[::80]
lon_vals = lon_vals_new[::80]
dx = (lon_vals[1]-lon_vals[0])/2.
dy = (lat_vals[1]-lat_vals[0])/2.
min_lat_idx = np.argsort(np.abs(lat_vals-min_lat_plot))[0]
max_lat_idx = np.argsort(np.abs(lat_vals-max_lat_plot))[0]
min_lon_idx = np.argsort(np.abs(lon_vals-min_lon_plot))[0]
max_lon_idx = np.argsort(np.abs(lon_vals-max_lon_plot))[0]
extent = [lon_vals[min_lon_idx]-dx, lon_vals[max_lon_idx]+dx,
          lat_vals[min_lat_idx]-dy, lat_vals[max_lat_idx]+dy]
ocn_floor = bathymetry
tsun_lons_tmp = tsun_lons
tsun_lats_tmp = tsun_lats
tsun_lons_for_bath = tsun_lons_tmp
tsun_lats_for_bath = tsun_lats_tmp
tsun_lons_for_bath = tsun_lons[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot) ]
tsun_lats_for_bath = tsun_lats[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
int_Lons, int_Lats = np.meshgrid(lon_vals, lat_vals)
ocn_floor_s = ocn_floor[0][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
unstruct_coords = np.array([tsun_lons_for_bath, tsun_lats_for_bath]).T
interp_true = np.asarray(
    griddata(unstruct_coords, ocn_floor_s, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
interp_true[(topo_vals != 0)] = 0.0
width = (extent[1]-extent[0])
height = (extent[3]-extent[2])
aspect = width/height
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6.15))
from numpy.ma import masked_array
topo_img = masked_array(topo_vals, topo_vals == 0.0)
lons_topo_img = masked_array(int_Lons, topo_vals == 0.0)
lats_topo_img = masked_array(int_Lats, topo_vals == 0.0)
true_tsu_img = masked_array(interp_true, topo_vals != 0.0)
lons_tsu_img = masked_array(int_Lons, topo_vals != 0.0)
lats_tsu_img = masked_array(int_Lats, topo_vals != 0.0)
shw1 = ax.contourf(lons_tsu_img,np.flip(lats_tsu_img),true_tsu_img,levels=50,extent=extent,cmap='gist_earth')
shw2 = ax.contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
ax.scatter(real_lons,real_lats,s=480,c='w',marker='o', label=r'\textbf{DART}')
ax.scatter(virtual_lons,virtual_lats,s=480,c='k',marker='x', label=r'\textbf{Virtual}')
ax.text(virtual_lons[0]+1,virtual_lats[0]+1,r'$\mathbf{1}$',color='k',fontsize=32)
ax.text(virtual_lons[1]+1,virtual_lats[1]+1,r'$\mathbf{2}$',color='k',fontsize=32)
ax.text(virtual_lons[2]+1,virtual_lats[2]+1,r'$\mathbf{3}$',color='k',fontsize=32)
ax.text(virtual_lons[3]+1,virtual_lats[3]+1,r'$\mathbf{4}$',color='k',fontsize=32)
ax.text(virtual_lons[4]+1,virtual_lats[4]+1,r'$\mathbf{5}$',color='k',fontsize=32)
ax.text(virtual_lons[5]+1,virtual_lats[5]+1,r'$\mathbf{6}$',color='k',fontsize=32)

ax.legend(loc='upper left')
ax.set_xlim([extent[0], extent[1]])
ax.set_ylim([extent[2],extent[3]])
ax.set_ylabel(r'\textbf{Latitude}', labelpad=10)
ax.set_xlabel(r'\textbf{Longitude}', labelpad=10)
cbar = fig.colorbar(shw1,ax=ax,fraction=.047, pad=0.04)
cbar.set_label(r'\textbf{Bathymetry (m)}')
ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ticks = cbar.get_ticks()
ticklabels = [r'\textbf{%g}' % t for t in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
bath_save_dir = os.getcwd()+"/GRL_figs_0706/virtual_waveforms/"
plt.savefig(bath_save_dir+"bath.png",bbox_inches='tight',dpi=400)
#####################################################################


min_lat_plot = 10
max_lat_plot = 45
min_lon_plot = 125
max_lon_plot = 160

### GET LEFT, RIGHT, AND VIRTUAL BATHYMETRY VALUES ###
real_sens_left_bath = np.zeros(6)
for i in range(6):
    bath_lon = tsun_lons[in_window_sensor_indices[five_smallest_indices[0][i]]]
    bath_lat = tsun_lats[in_window_sensor_indices[five_smallest_indices[0][i]]]
    bathpt = bath(bath_lon,bath_lat)
    real_sens_left_bath[i] = bathpt

real_sens_right_bath = np.zeros(6)
for i in range(6):
    bath_lon = tsun_lons[in_window_sensor_indices[five_smallest_indices[1][i]]]
    bath_lat = tsun_lats[in_window_sensor_indices[five_smallest_indices[1][i]]]
    bathpt = bath(bath_lon,bath_lat)
    real_sens_right_bath[i] = bathpt


virtual_sens_bath = np.zeros(6)
for i in range(6):
    vlon = virtual_lons[i]
    vlat = virtual_lats[i]
    bathpt = bath(vlon, vlat)
    virtual_sens_bath[i] = bathpt

print("Real Sensor Pair Locations: ",sensor_pair_locs)
print("Virtual Sensor Locations: ",virtual_sensor_locs)
print("Left Sens Bathymetry: ",real_sens_left_bath)
print("Right Sens Bathymetry: ",real_sens_right_bath)
print("Virtual Sens Bathymetry: ",virtual_sens_bath)

##############################
### GET THE REAL WAVEFORMS ###
all_inner_sens_vals = [sim1_sens_vals_inner,sim2_sens_vals_inner,sim3_sens_vals_inner,sim4_sens_vals_inner,
                       sim5_sens_vals_inner,sim6_sens_vals_inner,sim7_sens_vals_inner,sim8_sens_vals_inner,
                       sim9_sens_vals_inner, sim10_sens_vals_inner, sim11_sens_vals_inner]

all_true_for_interp = [sim1_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim2_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim3_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim4_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim5_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim6_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim7_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim8_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim9_full[:, (tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                    (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim10_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                       sim11_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]
                       ]

all_senseiver_for_interp = [sens1_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens2_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens3_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens4_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens5_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens6_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens7_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens8_full[:,(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)],
                            sens9_full[:, (tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                          (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]
                            ,
                            sens10_full[:, (tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                          (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]
                            ,
                            sens11_full[:, (tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                                          (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]
                            ]

interp_lons = tsun_lons[(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                        (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]

interp_lats = tsun_lats[(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                        (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]

xy_inner = np.c_[interp_lons, interp_lats]

sens_mean_err_all_sims = []
wang_mean_err_all_sims = []
true_max_amplitudes_full = []
sens_max_amplitudes_full = []
wang_max_amplitudes_full = []
true_arrival_times_full = []
sens_arrival_times_full = []
wang_arrival_times_full = []

for sim_num in range(11):
    display_num = sim_num + 1
    print("########## Simulation {} ##########".format(display_num))
    real_sens_left_waveforms = np.zeros(shape=(6,289))
    for i in range(6):
        real_sens_left_waveforms[i] = all_inner_sens_vals[sim_num][five_smallest_indices[0][i]]

    real_sens_right_waveforms = np.zeros(shape=(6,289))
    for i in range(6):
        real_sens_right_waveforms[i] = all_inner_sens_vals[sim_num][five_smallest_indices[1][i]]

    ### GET THE ARRIVAL TIMES FOR REAL WAVEFORMS ###
    real_sens_left_arrival_times = np.zeros(6)
    real_sens_left_arrival_time_indices = np.zeros(6)
    for i in range(6):
        t = 0
        idx = 0
        for val in real_sens_left_waveforms[i]:
            if val < 3e-2:
                t += 50.0
                idx += 1
            else:
                real_sens_left_arrival_times[i] += t
                real_sens_left_arrival_time_indices[i] += idx
                break

    real_sens_right_arrival_times = np.zeros(6)
    real_sens_right_arrival_time_indices = np.zeros(6)
    for i in range(6):
        t = 0
        idx = 0
        for val in real_sens_right_waveforms[i]:
            if val < 3e-2:
                t += 50.0
                idx += 1
            else:
                real_sens_right_arrival_times[i] += t
                real_sens_right_arrival_time_indices[i] += idx
                break

    wang_arrival_times = .5*real_sens_left_arrival_times+.5*real_sens_right_arrival_times

    t_seconds = np.arange(0, 50 * 289, 50)
    t_minutes = t_seconds/60
    wang_arrival_time_indices = np.zeros(6)
    for i in range(6):
        wang_arrival_time_indices[i] += np.argmin(np.abs(wang_arrival_times[i]-t_seconds))

    interpolated_virtual_sens_waveforms = np.zeros(shape=(6,289))
    stop_plot_indices = np.zeros(6)
    for i in range(6):
        l_idx = int(real_sens_left_arrival_time_indices[i])
        r_idx = int(real_sens_right_arrival_time_indices[i])
        v_idx = int(wang_arrival_time_indices[i])
        smaller_idx = min(l_idx,r_idx)
        res = int(289 - v_idx)
        left_wf_shifted = real_sens_left_waveforms[i][smaller_idx:smaller_idx+res]
        right_wf_shifted = real_sens_right_waveforms[i][smaller_idx:smaller_idx+res]
        interpolated_virtual_sens_waveforms[i][v_idx:v_idx+res] = (((.5*left_wf_shifted)*
                                                   ((-real_sens_left_bath[i])**(.25))+
                                                   (.5*right_wf_shifted)*
                                                   ((-real_sens_right_bath[i])**(.25)))
                                                   /((-virtual_sens_bath[i])**(.25)))



    real_virtual_sens_waveforms = np.zeros(shape=(6, 289))
    for i in range(6):
        vlon = virtual_lons[i]
        vlat = virtual_lats[i]
        for j in range(289):
            tsu = LinearNDInterpolator(xy_inner, all_true_for_interp[sim_num][j,:,0])
            real_virtual_sens_waveforms[i,j] = tsu(vlon,vlat)

    senseiver_virtual_sens_waveforms = np.zeros(shape=(6, 289))
    for i in range(6):
        vlon = virtual_lons[i]
        vlat = virtual_lats[i]
        for j in range(289):
            senseiver = LinearNDInterpolator(xy_inner, all_senseiver_for_interp[sim_num][j,:,0])
            senseiver_virtual_sens_waveforms[i,j] = senseiver(vlon,vlat)

    #filter the senseiver recons
    for i in range(6):
        # SAME AS COMBINED PLOT FILTERING v2
        signal = senseiver_virtual_sens_waveforms[i]
        filtered_signal = medfilt(signal, 9)
        filtered_signal = medfilt(filtered_signal, 3)
        senseiver_virtual_sens_waveforms[i] = filtered_signal

    true_arrival_times = np.zeros(6)
    true_arrival_time_indices = np.zeros(6)
    for i in range(6):
        t = 0
        idx = 0
        for val in real_virtual_sens_waveforms[i]:
            if val < 3e-2:
                t += 50.0
                idx += 1
            else:
                true_arrival_times[i] += t
                true_arrival_time_indices[i] += idx
                break

    true_max_amplitudes = np.zeros(6)
    for i in range(6):
        true_max_amplitudes[i] = np.max(np.abs(real_virtual_sens_waveforms[i]))

    sens_arrival_times = np.zeros(6)
    sens_arrival_time_indices = np.zeros(6)
    for i in range(6):
        t = 0
        idx = 0
        for val in senseiver_virtual_sens_waveforms[i]:
            if val < 3e-2:
                t += 50.0
                idx += 1
            else:
                sens_arrival_times[i] += t
                sens_arrival_time_indices[i] += idx
                break

    senseiver_max_amplitudes = np.zeros(6)
    for i in range(6):
        senseiver_max_amplitudes[i] = np.max(np.abs(senseiver_virtual_sens_waveforms[i]))

    wang_max_amplitudes = np.zeros(6)
    for i in range(6):
        wang_max_amplitudes[i] = np.max(np.abs(interpolated_virtual_sens_waveforms[i]))

    print("Left Sens Arrival Times: ", real_sens_left_arrival_times/60)
    print("Right Sens Arrival Times: ", real_sens_right_arrival_times/60)
    print("Virtual Arrival Times (True): ", true_arrival_times/60)
    print("Virtual Arrival Times (Wang): ", wang_arrival_times/60)
    print("Virtual Arrival Times (Sens): ", sens_arrival_times/60)
    print("Virtual Max Amp (True): ", true_max_amplitudes)
    print("Virtual Max Amp (Wang): ", wang_max_amplitudes)
    print("Virtual Max Amp (Sens): ", senseiver_max_amplitudes)
    print("Wang AT MAE: ",np.mean(np.abs(true_arrival_times-wang_arrival_times))/60)
    print("Senseiver AT MAE: ",np.mean(np.abs(true_arrival_times-sens_arrival_times))/60)
    print("Wang MA MAE: ",np.mean(np.abs(true_max_amplitudes-wang_max_amplitudes)))
    print("Senseiver MA MAE: ",np.mean(np.abs(true_max_amplitudes-senseiver_max_amplitudes)))
    true_max_amplitudes_full.append(true_max_amplitudes)
    sens_max_amplitudes_full.append(senseiver_max_amplitudes)
    wang_max_amplitudes_full.append(wang_max_amplitudes)
    true_arrival_times_full.append(true_arrival_times/60)
    sens_arrival_times_full.append(sens_arrival_times/60)
    wang_arrival_times_full.append(wang_arrival_times/60)

    senseiver_mean_arr = []
    wang_mean_arr = []
    times = (5/6)*np.arange(0, 289, 1)



    for i in range(6):
        plt.rcParams["font.size"] = 16
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
        axs[0].plot(times, real_virtual_sens_waveforms[i], c='k')
        axs[0].plot(times, senseiver_virtual_sens_waveforms[i], c='g')
        axs[0].plot(times, interpolated_virtual_sens_waveforms[i], c='r')
        axs[0].legend(['SWE', 'Senseiver', 'LIHFP'])
        txt = r'\textbf{Height at Sensor }' + r'\textbf{' + str(i + 1) + r'}'
        txt2 = r'\textbf{Absolute Error at Sensor }' + r'\textbf{' + str(i + 1) + r'}'
        axs[0].set(xlabel=r"\textbf{Time (minutes)}", ylabel=txt)
        axs[0].set_xlim(0, 240)
        axs[1].plot(times, np.abs(real_virtual_sens_waveforms[i] - senseiver_virtual_sens_waveforms[i]), c='k')
        axs[1].plot(times, np.abs(real_virtual_sens_waveforms[i] - interpolated_virtual_sens_waveforms[i]), c='r')
        axs[1].legend([r'\textbf{Senseiver: MAE = }' + r'\textbf{' + str(
            round(np.mean(np.abs(real_virtual_sens_waveforms[i] - senseiver_virtual_sens_waveforms[i])), 2)) + r'}',
                       r'\textbf{LIHFP: MAE = }' + r'\textbf{' + str(round(
                           np.mean(np.abs(real_virtual_sens_waveforms[i] - interpolated_virtual_sens_waveforms[i])),
                           2)) + r'}'])
        axs[1].set(xlabel=r"\textbf{Time (minutes)}", ylabel=txt2)
        axs[1].set_xlim(0, 240)
        plt.tight_layout()
        plt.savefig(os.getcwd()+"/virtual_waveforms" + fig_dir +
                    "/sim_{}_senslon_{}_senslat_{}_sensnum_{}_v2.png".format(
                        display_num,
                        round(virtual_lons[i], 2),
                        round(virtual_lats[i], 2), i + 1), bbox_inches='tight', dpi=400
                    )
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 3.25))
        ax.plot(times, real_virtual_sens_waveforms[i], c='k')
        ax.plot(times, senseiver_virtual_sens_waveforms[i], c='#0072B2')
        ax.plot(times, interpolated_virtual_sens_waveforms[i], c='#D55E00')
        txt = r'\textbf{Height at Sensor }' + r'\textbf{' + str(i + 1) + r'}'
        txt2 = r'\textbf{Absolute Error at Sensor }' + r'\textbf{' + str(i + 1) + r'}'
        ax.set(xlabel=r"\textbf{Time (minutes)}")
        ax.set_ylabel(txt)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xlim(0, 240)
        ax.legend([r'\textbf{SWE}',
                   r'\textbf{Senseiver: MAE = }' + r'\textbf{' + str(
            round(np.mean(np.abs(real_virtual_sens_waveforms[i] -
                    senseiver_virtual_sens_waveforms[i])), 2)) + r'}',
                       r'\textbf{LIHFP: MAE = }' + r'\textbf{' + str(round(
                           np.mean(np.abs(real_virtual_sens_waveforms[i] -
                        interpolated_virtual_sens_waveforms[i])),
                           2)) + r'}'])
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.tight_layout()
        plt.savefig(os.getcwd()+ "/virtual_waveforms" + fig_dir +
                    "/single_sim_{}_senslon_{}_senslat_{}_sensnum_{}_v2.png".format(
                        display_num,
                        round(virtual_lons[i], 2),
                        round(virtual_lats[i], 2), i + 1), bbox_inches='tight', dpi=400
                    )
        plt.close()


        print("Senseiver Mean Error for sensor {}:".format(i+1),np.mean(np.abs(real_virtual_sens_waveforms[i]-senseiver_virtual_sens_waveforms[i])))
        print("Wang Mean Error for sensor {}:".format(i+1),np.mean(np.abs(real_virtual_sens_waveforms[i]-interpolated_virtual_sens_waveforms[i])))
        senseiver_mean_arr.append(np.mean(np.abs(real_virtual_sens_waveforms[i]-senseiver_virtual_sens_waveforms[i])))
        wang_mean_arr.append(np.mean(np.abs(real_virtual_sens_waveforms[i]-interpolated_virtual_sens_waveforms[i])))
    print("Mean Senseiver Error for Sim {}:".format(display_num),
          np.mean(senseiver_mean_arr))
    print("Mean Wang Error for Sim {}:".format(display_num),
          np.mean(wang_mean_arr))
    sens_mean_err_all_sims.append(np.mean(senseiver_mean_arr))
    wang_mean_err_all_sims.append(np.mean(wang_mean_arr))

print("Senseiver Mean Error for all Sims: ",np.mean(sens_mean_err_all_sims))
print("Wang Mean Error for all Sims: ",np.mean(wang_mean_err_all_sims))
print("Senseiver Error STD for all Sims: ",np.std(sens_mean_err_all_sims))
print("Wang Error STD for all Sims: ",np.std(wang_mean_err_all_sims))



plt.rcParams["font.size"] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20


true_max_amp_mat = np.zeros(shape=(11,6))
for i in range(11):
    for j in range(6):
        true_max_amp_mat[i,j] = true_max_amplitudes_full[i][j]

wang_max_amp_mat = np.zeros(shape=(11,6))
for i in range(11):
    for j in range(6):
        wang_max_amp_mat[i,j] = wang_max_amplitudes_full[i][j]

sens_max_amp_mat = np.zeros(shape=(11,6))
for i in range(11):
    for j in range(6):
        sens_max_amp_mat[i,j] = sens_max_amplitudes_full[i][j]

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(8,4))
im1 = axs[0].imshow(true_max_amp_mat,cmap='bwr')
axs[0].set_title(r'\textbf{True}')
axs[0].set_xlabel(r'\textbf{Virtual Sensors}')
axs[0].set_ylabel(r'\textbf{Simulations}')
im2 = axs[1].imshow(wang_max_amp_mat,cmap='bwr')
axs[1].set_title(r'\textbf{LIHFP}')
im3 = axs[2].imshow(sens_max_amp_mat,cmap='bwr')
axs[2].set_title(r'\textbf{Senseiver}')
pos = axs[2].get_position()
cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
cbar = plt.colorbar(im3, cax=cax)
cbar.set_label(r'\textbf{Max Amplitudes (m)}')


fig.suptitle(r'\textbf{Max Amplitudes}',y=.95)
plt.savefig(os.getcwd()+ "/virtual_waveforms" + fig_dir+
                    "/MaxAmpMat_v2.png",dpi=400)

true_arr_time_mat = np.zeros(shape=(11,6))
for i in range(11):
    for j in range(6):
        true_arr_time_mat[i, j] = true_arrival_times_full[i][j]

wang_arr_time_mat = np.zeros(shape=(11,6))
for i in range(11):
    for j in range(6):
        wang_arr_time_mat[i, j] = wang_arrival_times_full[i][j]

sens_arr_time_mat = np.zeros(shape=(11,6))
for i in range(11):
    for j in range(6):
        sens_arr_time_mat[i, j] = sens_arrival_times_full[i][j]

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(8,4))
im1 = axs[0].imshow(true_arr_time_mat,cmap='bwr')
axs[0].set_title(r'\textbf{True}')
axs[0].set_xlabel(r'\textbf{Virtual Sensors}')
axs[0].set_ylabel(r'\textbf{Simulations}')
im2 = axs[1].imshow(wang_arr_time_mat,cmap='bwr')
axs[1].set_title(r'\textbf{LIHFP}')
im3 = axs[2].imshow(sens_arr_time_mat,cmap='bwr')
axs[2].set_title(r'\textbf{Senseiver}')

pos = axs[2].get_position()
cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
cbar = plt.colorbar(im3, cax=cax)
cbar.set_label(r'\textbf{Arrival Times (mins)}')


fig.suptitle(r'\textbf{Arrival Times}',y=.95)
plt.savefig(os.getcwd()+"/virtual_waveforms/" + fig_dir+
                    "/ArrivalTimeMat_v2.png",dpi=400
        )

wang_ma_err = np.abs(true_max_amp_mat-wang_max_amp_mat)
sens_ma_err = np.abs(true_max_amp_mat-sens_max_amp_mat)
vmin = min(np.nanmin(wang_ma_err), np.nanmin(sens_ma_err))
vmax = max(np.nanmax(wang_ma_err), np.nanmax(sens_ma_err))

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(5,6.15),constrained_layout=True)
im1 = axs[0].imshow(wang_ma_err, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[0].set_ylabel(r'\textbf{LIHFP Avg: }' + r'\textbf{'+str(round(np.mean(wang_ma_err),1))+r'}'r'\textbf{ (m)}')
im2 = axs[1].imshow(sens_ma_err, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[1].set_ylabel(r'\textbf{Senseiver Avg: }'+ r'\textbf{'+str(round(np.mean(sens_ma_err),1))+r'}'+r'\textbf{ (m)}')
pos = axs[1].get_position()
cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', fraction=0.05, pad=0.02)
ticks = np.linspace(vmin, vmax, num=3)
ticklabels = [r'\textbf{%.2f}' % t for t in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
cbar.set_label(r'\textbf{Max Amplitude MAE}')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].yaxis.tick_right()
axs[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axs[1].set_yticks(np.arange(11))
axs[1].set_xticks(np.arange(6))
axs[1].set_yticklabels([r'\textbf{%d}' % (i+1) for i in range(11)])
axs[1].set_xticklabels([r'\textbf{%d}' % (i+1) for i in range(6)])
plt.savefig(os.getcwd()+ "/virtual_waveforms" + fig_dir+
                    "/MaxAmpMatErr_v2.png",bbox_inches='tight',dpi=400)



wang_at_err = np.abs(true_arr_time_mat-wang_arr_time_mat)
sens_at_err = np.abs(true_arr_time_mat-sens_arr_time_mat)
vmin = min(np.nanmin(wang_at_err), np.nanmin(sens_at_err))
vmax = max(np.nanmax(wang_at_err), np.nanmax(sens_at_err))

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(5,6.15),constrained_layout=True)
im1 = axs[0].imshow(wang_at_err, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[0].set_ylabel(r'\textbf{LIHFP Avg: }' + r'\textbf{'+str(round(np.mean(wang_at_err),1))+r'}'r'\textbf{ (mins)}')
im2 = axs[1].imshow(sens_at_err, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[1].set_ylabel(r'\textbf{Senseiver Avg: }' + r'\textbf{'+str(round(np.mean(sens_at_err),1))+r'}'r'\textbf{ (mins)}')
pos = axs[1].get_position()
cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', fraction=0.05, pad=0.02)
ticks = np.linspace(vmin, vmax, num=3)
ticklabels = [r'\textbf{%.2f}' % t for t in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
cbar.set_label(r'\textbf{Arrival Time MAE}')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].yaxis.tick_right()
axs[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axs[1].set_yticks(np.arange(11))
axs[1].set_xticks(np.arange(6))
axs[1].set_yticklabels([r'\textbf{%d}' % (i+1) for i in range(11)])
axs[1].set_xticklabels([r'\textbf{%d}' % (i+1) for i in range(6)])
plt.savefig(os.getcwd()+ "/virtual_waveforms" + fig_dir+
                    "/ArrivalTimeMatErr_v2.png",bbox_inches='tight',dpi=400)

true_max_amplitudes_full = np.array(true_max_amplitudes_full).flatten()
sens_max_amplitudes_full = np.array(sens_max_amplitudes_full).flatten()
wang_max_amplitudes_full = np.array(wang_max_amplitudes_full).flatten()
true_arrival_times_full = np.array(true_arrival_times_full).flatten()
sens_arrival_times_full = np.array(sens_arrival_times_full).flatten()
wang_arrival_times_full = np.array(wang_arrival_times_full).flatten()

print("Senseiver Max Amp MAE for all sims: ",np.mean(np.abs(true_max_amplitudes_full-sens_max_amplitudes_full)))
print("Wang Max Amp MAE for all sims: ",np.mean(np.abs(true_max_amplitudes_full-wang_max_amplitudes_full)))
print("Senseiver Arrival Time MAE for all sims: ",np.mean(np.abs(true_arrival_times_full-sens_arrival_times_full)))
print("Wang Arrival Time MAE for all sims: ",np.mean(np.abs(true_arrival_times_full-wang_arrival_times_full)))
logfile.close()
