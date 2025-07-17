import multiprocessing
multiprocessing.set_start_method("fork")
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import SWETsunamiForPlotting2
from scipy.interpolate import LinearNDInterpolator
import scipy.io as sio
import sys
import os
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

path = os.getcwd()+"/lihfp_figs/"



### EPICENTERS OF UNSEEN DATA ###
def unseen_lons():
    return np.array([136.6500,138.2000,138.9000,
                     139.5000,140.2000,140.5000,
                     141.5000,142.5000])


def unseen_lats():
    return np.array([33.1000,31.0000,28.1000,
                     28.8000,29.1000,31.8000,
                     34.2000,36.2000])


epi_lons = unseen_lons()
epi_lats = unseen_lats()
###   ###


### SIMULATION DATA  ###
fname_2hr = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_145.mat"
fname_4hr = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_289.mat"

train_split = "8020" #9505 or 8020
regularization = "unreg" #reg or unreg

if train_split == "9505" and regularization == "unreg":
    ver_num_2hr = 34
    ver_num_4hr = 22
elif train_split == "9505" and regularization == "reg":
    ver_num_2hr = 35
    ver_num_4hr = 25
elif train_split == "8020" and regularization == "unreg":
    ver_num_2hr = 0
    ver_num_4hr = 2
else: #train_split=8020, regularization=reg
    ver_num_2hr = 1
    ver_num_4hr = 3

logfile = open(path+"/logs/metrics_{}_{}.out".format(train_split,regularization), 'w')
sys.stdout = logfile


type = "unseen"
out_path_2hr = os.getcwd()+"/lightning_logs/version_"+str(ver_num_2hr)+"/"
output_im_2hr = torch.load(out_path_2hr+'tensor'+'_'+str(type)+'.pt').numpy()
out_path_4hr = os.getcwd()+"/lightning_logs/version_"+str(ver_num_4hr)+"/"
output_im_4hr = torch.load(out_path_4hr+'tensor'+'_'+str(type)+'.pt').numpy()
true_data_2hr, latitude_2hr, longitude_2hr, mask_2hr, max_ht_2hr, times_2hr, sensors_2hr, div_2hr = SWETsunamiForPlotting2(fname_2hr)
true_data_2hr *= max_ht_2hr
output_im_2hr *= max_ht_2hr
true_data_4hr, latitude_4hr, longitude_4hr, mask_4hr, max_ht_4hr, times_4hr, sensors_4hr, div_4hr = SWETsunamiForPlotting2(fname_4hr)
true_data_4hr *= max_ht_4hr
output_im_4hr *= max_ht_4hr
tsun_lons = longitude_4hr*(180 / np.pi)
tsun_lats = latitude_4hr*(180 / np.pi)




### COLLECT SIMULATION DATA INTO ONE FOUR HOUR SET ###
intervals_2hr = [(0,145),(145,290),(290,435),(435,580),(580,725),(725,870),(870,1015),(1015,1160)]
sim1_2hr = true_data_2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
sim2_2hr = true_data_2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
sim3_2hr = true_data_2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
sim4_2hr = true_data_2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
sim5_2hr = true_data_2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
sim6_2hr = true_data_2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
sim7_2hr = true_data_2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
sim8_2hr = true_data_2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
intervals_4hr = [(0,144),(144,288),(288,432),(432,576),(576,720),(720,864),(864,1008),(1008,1152)]
sim1_4hr = true_data_4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
sim2_4hr = true_data_4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
sim3_4hr = true_data_4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
sim4_4hr = true_data_4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
sim5_4hr = true_data_4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
sim6_4hr = true_data_4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
sim7_4hr = true_data_4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
sim8_4hr = true_data_4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
sim1_full = np.concatenate((sim1_2hr,sim1_4hr),axis=0)
sim2_full = np.concatenate((sim2_2hr,sim2_4hr),axis=0)
sim3_full = np.concatenate((sim3_2hr,sim3_4hr),axis=0)
sim4_full = np.concatenate((sim4_2hr,sim4_4hr),axis=0)
sim5_full = np.concatenate((sim5_2hr,sim5_4hr),axis=0)
sim6_full = np.concatenate((sim6_2hr,sim6_4hr),axis=0)
sim7_full = np.concatenate((sim7_2hr,sim7_4hr),axis=0)
sim8_full = np.concatenate((sim8_2hr,sim8_4hr),axis=0)

### COLLECT SENSEIVER RECONS INTO ONE FOUR HOUR SET ###
sens1_2hr = output_im_2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
sens2_2hr = output_im_2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
sens3_2hr = output_im_2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
sens4_2hr = output_im_2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
sens5_2hr = output_im_2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
sens6_2hr = output_im_2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
sens7_2hr = output_im_2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
sens8_2hr = output_im_2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
sens1_4hr = output_im_4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
sens2_4hr = output_im_4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
sens3_4hr = output_im_4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
sens4_4hr = output_im_4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
sens5_4hr = output_im_4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
sens6_4hr = output_im_4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
sens7_4hr = output_im_4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
sens8_4hr = output_im_4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
sens1_full = np.concatenate((sens1_2hr,sens1_4hr),axis=0)
sens2_full = np.concatenate((sens2_2hr,sens2_4hr),axis=0)
sens3_full = np.concatenate((sens3_2hr,sens3_4hr),axis=0)
sens4_full = np.concatenate((sens4_2hr,sens4_4hr),axis=0)
sens5_full = np.concatenate((sens5_2hr,sens5_4hr),axis=0)
sens6_full = np.concatenate((sens6_2hr,sens6_4hr),axis=0)
sens7_full = np.concatenate((sens7_2hr,sens7_4hr),axis=0)
sens8_full = np.concatenate((sens8_2hr,sens8_4hr),axis=0)


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
# fig, ax = plt.subplots()
# ax.scatter(sens_lons_inner,sens_lats_inner)
sim1_sens_vals_inner = sim1_full[:,in_window_sensor_indices][:,:,0].T
sim2_sens_vals_inner = sim2_full[:,in_window_sensor_indices][:,:,0].T
sim3_sens_vals_inner = sim3_full[:,in_window_sensor_indices][:,:,0].T
sim4_sens_vals_inner = sim4_full[:,in_window_sensor_indices][:,:,0].T
sim5_sens_vals_inner = sim5_full[:,in_window_sensor_indices][:,:,0].T
sim6_sens_vals_inner = sim6_full[:,in_window_sensor_indices][:,:,0].T
sim7_sens_vals_inner = sim7_full[:,in_window_sensor_indices][:,:,0].T
sim8_sens_vals_inner = sim8_full[:,in_window_sensor_indices][:,:,0].T


### MAKE A MATRIX CONSISTING OF DISTANCES BETWEEN PAIRS OF SENSORS ###
sens_distance_arr = np.zeros(shape=(len(sens_lons_inner),len(sens_lons_inner)))
for i in range(len(sens_lons_inner)):
    for j in range(len(sens_lons_inner)):
        if i<j:
            sens_distance_arr[i,j] += np.sqrt((sens_lons_inner[i]-sens_lons_inner[j])**2+(sens_lats_inner[i]-sens_lats_inner[j])**2)
# fig, ax = plt.subplots()
# ax.imshow(sens_distance_arr)

### RETRIEVE SENSOR PAIRS FOR INTERPOLATION METHOD BASED ON PROXIMITY ###
### IE, THE 10 CLOSEST PAIRS IN THE MATRIX DETERMINE THE PAIRS WE WILL CONIDER ###

flat_nonzero = sens_distance_arr[np.nonzero(sens_distance_arr)]
five_smallest_indices_flat = np.argpartition(flat_nonzero, 10)[:10] #twelve here due to redundancy
# #inds = [2,9]
# #inds = [2,4,5,7]
# #inds = [2,4,5,7,14]
# inds = [2]
# ten_smallest_indices_flat = np.delete(ten_smallest_indices_flat_1,inds)

original_indices = np.nonzero(sens_distance_arr)
five_smallest_indices = tuple(coord[five_smallest_indices_flat] for coord in original_indices)

sensor_pair_locs = np.zeros((10, 2),dtype=object)
for i in range(10):
    sensor_pair_locs[i][0] = (sens_lons_inner[five_smallest_indices[0][i]],sens_lats_inner[five_smallest_indices[0][i]])
    sensor_pair_locs[i][1] = (sens_lons_inner[five_smallest_indices[1][i]], sens_lats_inner[five_smallest_indices[1][i]])
# inds = [3]
# sensor_pair_locs = np.delete(sensor_pair_locs,inds)


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


# ### MAKE BATHYMETRY FIG HERE #######################################
# import rasterio
# import os
# topo = rasterio.open('/Users/emcdugald/tsunseiver/geodata/merged.tif')
# topo_data = topo.read()
# import geopandas as gpd
# from scipy.interpolate import griddata
# df = gpd.read_file('/Users/emcdugald/tsunseiver/geodata/ne_10m_admin_0_countries.shp')
# chi = df.loc[df['ADMIN'] == 'China']
# jap = df.loc[df['ADMIN'] == 'Japan']
# rus = df.loc[df['ADMIN'] == 'Russia']
# sko = df.loc[df['ADMIN'] == 'South Korea']
# nko = df.loc[df['ADMIN'] == 'North Korea']
# phi = df.loc[df['ADMIN'] == 'Philippines']
# tai = df.loc[df['ADMIN'] == 'Taiwan']
# vie = df.loc[df['ADMIN'] == 'Vietnam']
# mon = df.loc[df['ADMIN'] == 'Mongolia']
# min_lat = -10
# max_lat = 70
# min_lon = 90
# max_lon = 180
# min_lat_plot = 10
# max_lat_plot = 45
# min_lon_plot = 116
# max_lon_plot = 169
# nlat , nlon = np.shape(topo_data[0])
# lat_vals = np.linspace(max_lat,min_lat,nlat)
# lon_vals = np.linspace(min_lon,max_lon,nlon)
# min_lat_idx = np.where(lat_vals >= min_lat_plot)[0][-1]
# max_lat_idx = np.where(lat_vals >= max_lat_plot)[0][-1]
# min_lon_idx = np.where(lon_vals >= min_lon_plot)[0][0]
# max_lon_idx = np.where(lon_vals >= max_lon_plot)[0][0]
# nlat_plot, nlon_plot = np.shape(topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx])
# lat_vals_new = np.linspace(min_lat_plot,max_lat_plot,nlat_plot)
# lon_vals_new = np.linspace(min_lon_plot,max_lon_plot,nlon_plot)
# dx = (lon_vals_new[1]-lon_vals_new[0])/2.
# dy = (lat_vals_new[1]-lat_vals_new[0])/2.
# extent = [lon_vals_new[0]-dx, lon_vals_new[-1]+dx, lat_vals_new[0]-dy, lat_vals_new[-1]+dy]
# topo_vals = topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]
# topo_vals = topo_vals[::80,::80]
# lat_vals = lat_vals_new[::80]
# lon_vals = lon_vals_new[::80]
# dx = (lon_vals[1]-lon_vals[0])/2.
# dy = (lat_vals[1]-lat_vals[0])/2.
# min_lat_idx = np.argsort(np.abs(lat_vals-min_lat_plot))[0]
# max_lat_idx = np.argsort(np.abs(lat_vals-max_lat_plot))[0]
# min_lon_idx = np.argsort(np.abs(lon_vals-min_lon_plot))[0]
# max_lon_idx = np.argsort(np.abs(lon_vals-max_lon_plot))[0]
# extent = [lon_vals[min_lon_idx]-dx, lon_vals[max_lon_idx]+dx,
#           lat_vals[min_lat_idx]-dy, lat_vals[max_lat_idx]+dy]
# ocn_floor = bathymetry
# tsun_lons_tmp = tsun_lons
# tsun_lats_tmp = tsun_lats
# tsun_lons_for_bath = tsun_lons_tmp
# tsun_lats_for_bath = tsun_lats_tmp
# tsun_lons_for_bath = tsun_lons[(tsun_lons_tmp >= min_lon_plot) &
#                       (tsun_lons_tmp <= max_lon_plot) &
#                       (tsun_lats_tmp >= min_lat_plot) &
#                       (tsun_lats_tmp <= max_lat_plot) ]
# tsun_lats_for_bath = tsun_lats[(tsun_lons_tmp >= min_lon_plot) &
#                       (tsun_lons_tmp <= max_lon_plot) &
#                       (tsun_lats_tmp >= min_lat_plot) &
#                       (tsun_lats_tmp <= max_lat_plot)]
# int_Lons, int_Lats = np.meshgrid(lon_vals, lat_vals)
# ocn_floor_s = ocn_floor[0][(tsun_lons_tmp >= min_lon_plot) &
#                       (tsun_lons_tmp <= max_lon_plot) &
#                       (tsun_lats_tmp >= min_lat_plot) &
#                       (tsun_lats_tmp <= max_lat_plot)]
# unstruct_coords = np.array([tsun_lons_for_bath, tsun_lats_for_bath]).T
# interp_true = np.asarray(
#     griddata(unstruct_coords, ocn_floor_s, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
# interp_true[(topo_vals != 0)] = 0.0
# width = (extent[1]-extent[0])
# height = (extent[3]-extent[2])
# aspect = width/height
# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(13,8))
# from numpy.ma import masked_array
# topo_img = masked_array(topo_vals, topo_vals == 0.0)
# lons_topo_img = masked_array(int_Lons, topo_vals == 0.0)
# lats_topo_img = masked_array(int_Lats, topo_vals == 0.0)
# true_tsu_img = masked_array(interp_true, topo_vals != 0.0)
# lons_tsu_img = masked_array(int_Lons, topo_vals != 0.0)
# lats_tsu_img = masked_array(int_Lats, topo_vals != 0.0)
# shw1 = ax.contourf(lons_tsu_img,np.flip(lats_tsu_img),true_tsu_img,levels=50,extent=extent,cmap='cool')
# shw2 = ax.contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
# ax.scatter(real_lons,real_lats,s=120,c='w',marker='o', label='DART')
# ax.scatter(virtual_lons,virtual_lats,s=120,c='k',marker='x', label='Virtual')
#
# #ax.scatter(virtual_lons[4],virtual_lats[4],s=250,facecolors='none', edgecolors='r',linewidth=5)
#
# ax.text(virtual_lons[0]+1,virtual_lats[0]+1,'1',color='k', weight='bold',fontsize=32)
# ax.text(virtual_lons[1]+1,virtual_lats[1]+1,'2',color='k', weight='bold',fontsize=32)
# ax.text(virtual_lons[2]+1,virtual_lats[2]+1,'3',color='k', weight='bold',fontsize=32)
# ax.text(virtual_lons[3]+1,virtual_lats[3]+1,'4',color='k', weight='bold',fontsize=32)
# ax.text(virtual_lons[4]+1,virtual_lats[4]+1,'5',color='k', weight='bold',fontsize=32)
# ax.text(virtual_lons[5]+1,virtual_lats[5]+1,'6',color='k', weight='bold',fontsize=32)
#
# ax.legend(loc='upper left')
# ax.set_xlim([extent[0], extent[1]])
# ax.set_ylim([extent[2],extent[3]])
# #ax.set_aspect(aspect)
# ax.set_ylabel(r'\textbf{Latitude}', labelpad=10)
# ax.set_xlabel(r'\textbf{Longitude}', labelpad=10)
# fig.colorbar(shw1,ax=ax,fraction=.047, pad=0.04)
# ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
# bath_save_dir = "/Users/emcdugald/sparse_sens_tsunami/GRL_figs/virtual_waveforms/"
# #plt.savefig(bath_save_dir+"bath_4.png",bbox_inches='tight',dpi=400)
# plt.savefig(bath_save_dir+"bath_4.png",bbox_inches='tight',dpi=400)
# #####################################################################


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
                       sim5_sens_vals_inner,sim6_sens_vals_inner,sim7_sens_vals_inner,sim8_sens_vals_inner]

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
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]]

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
                                   (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]]

interp_lons = tsun_lons[(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                        (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]

interp_lats = tsun_lats[(tsun_lons >= min_lon_plot) & (tsun_lons <= max_lon_plot) &
                        (tsun_lats >= min_lat_plot) & (tsun_lats <= max_lat_plot)]

xy_inner = np.c_[interp_lons, interp_lats]

sens_mean_err_all_sims = []
wang_mean_err_all_sims = []
# sens_mean_at_err_all_sims = []
# wang_mean_at_err_all_sims = []
# sens_mean_ma_err_all_sims = []
# wang_mean_ma_err_all_sims = []
true_max_amplitudes_full = []
sens_max_amplitudes_full = []
wang_max_amplitudes_full = []
true_arrival_times_full = []
sens_arrival_times_full = []
wang_arrival_times_full = []

for sim_num in range(8):
#for sim_num in [2,4,6,7]:
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
            if val < 1e-3:
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
            if val < 1e-3:
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

    for i in range(6):
        fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(8,8))
        axs[0].plot(t_minutes,real_sens_left_waveforms[i])
        axs[1].plot(t_minutes,real_sens_right_waveforms[i])
        axs[2].plot(t_minutes,interpolated_virtual_sens_waveforms[i])
        axs[2].set(xlabel=r"\textbf{Time (minutes)}", ylabel=r"\textbf{Wave Height (m)}")
        axs[0].set_title(r'\textbf{Real Waveform A}')
        axs[1].set_title(r'\textbf{Real Waveform B}')
        axs[2].set_title(r'\textbf{Virtual Waveform (via LIHFP)}')
        axs[0].set_xlim(0, 240)
        axs[1].set_xlim(0, 240)
        axs[2].set_xlim(0, 240)
        fig.suptitle(r'\textbf{Epicenter: (}' + r'\textbf{' + str(round(epi_lons[sim_num], 2)) + r'}' +
                     r'\textbf{$^\circ$E, }' + r'\textbf{' + str(round(epi_lats[sim_num], 2)) + r'}' +
                     r'\textbf{$^\circ$N), '+r'\textbf{Virtual Location }'+ r'\textbf{' +str(i+1) + r'}')
        plt.tight_layout(pad=1.0)
        plt.savefig("/Users/emcdugald/sparse_sens_tsunami/"
                    "GRL_figs/virtual_waveforms/figs_v4/"
                    "LIHFP_split_{}_sim_{}_reg_{}_sensnum_{}.png".format(
            train_split, display_num,regularization, i + 1), bbox_inches='tight'
        )
        plt.close()



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

    true_arrival_times = np.zeros(6)
    true_arrival_time_indices = np.zeros(6)
    for i in range(6):
        t = 0
        idx = 0
        for val in real_virtual_sens_waveforms[i]:
            if val < 1e-3:
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
            if val < 1e-3:
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
    for i in range(6):
        times = (5/6)*np.arange(0, 289, 1)

        fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(9,8))

        axs[0].plot(times,real_virtual_sens_waveforms[i], c='k')
        axs[0].plot(times,senseiver_virtual_sens_waveforms[i], c='g')
        axs[0].plot(times,interpolated_virtual_sens_waveforms[i],c='r')
        axs[0].legend(['SWE', 'Senseiver', 'LIHFP'])
        # txt = r'\textbf{Wave Height at Virtual Sensor (}' + r'\textbf{' + str(
        #             round(virtual_lons[i],2)) + r'}' + r'\textbf{$^\circ$E, }' + r'\textbf{' + str(
        #             round(virtual_lats[i],2)) + r'}' + r'\textbf{$^\circ$N)}'
        txt = r'\textbf{Wave Height at Virtual Sensor }'+ r'\textbf{' + str(i+1) + r'}'
        axs[0].set_title(txt)
        #axs[0].set_title('Wave Height(m) at ({}$^\circ$E, {}$^\circ$N)'.format(round(virtual_lons[i],2),round(virtual_lats[i],2)))
        axs[0].set(xlabel=r"\textbf{Time (minutes)}", ylabel=r"\textbf{Wave Height(m)}")
        axs[0].set_xlim(0,240)


        axs[1].plot(times,np.abs(real_virtual_sens_waveforms[i] - senseiver_virtual_sens_waveforms[i]), c='k')
        axs[1].plot(times,np.abs(real_virtual_sens_waveforms[i] - interpolated_virtual_sens_waveforms[i]), c='r')
        axs[1].legend([r'\textbf{Senseiver: MAE = }' + r'\textbf{' + str(round(np.mean(np.abs(real_virtual_sens_waveforms[i] - senseiver_virtual_sens_waveforms[i])),2)) + r'}',
                       r'\textbf{LIHFP: MAE = }' + r'\textbf{' + str(round(np.mean(np.abs(real_virtual_sens_waveforms[i] - interpolated_virtual_sens_waveforms[i])),2)) + r'}'])
        axs[1].set_title(r"\textbf{Senseiver vs LIHFP Absolute Error}")
        axs[1].set(xlabel=r"\textbf{Time (minutes)}", ylabel=r"$\mathbf{|h-\hat{h}|}$")
        axs[1].set_xlim(0, 240)
        fig.suptitle(r'\textbf{Epicenter: (}' + r'\textbf{' + str(round(epi_lons[sim_num],3)) + r'}' +
                     r'\textbf{$^\circ$E, }' + r'\textbf{' + str(round(epi_lats[sim_num],3)) + r'}' +
                     r'\textbf{$^\circ$N)', y=1.0)



        plt.tight_layout()
        plt.savefig(path+"/figs/"+
                    "split_{}_sim_{}_senslon_{}_senslat_{}_reg_{}_sensnum_{}.png".format(
            train_split,display_num,
            round(virtual_lons[i],2),
            round(virtual_lats[i],2),
            regularization,i+1)
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


true_max_amp_mat = np.zeros(shape=(8,6))
for i in range(8):
    for j in range(6):
        true_max_amp_mat[i,j] = true_max_amplitudes_full[i][j]

wang_max_amp_mat = np.zeros(shape=(8,6))
for i in range(8):
    for j in range(6):
        wang_max_amp_mat[i,j] = wang_max_amplitudes_full[i][j]

sens_max_amp_mat = np.zeros(shape=(8,6))
for i in range(8):
    for j in range(6):
        sens_max_amp_mat[i,j] = sens_max_amplitudes_full[i][j]

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(16,8))
im1 = axs[0].imshow(true_max_amp_mat)
axs[0].set_title(r'\textbf{True}')
axs[0].set_xlabel(r'\textbf{Virtual Sensors}')
axs[0].set_ylabel(r'\textbf{Simulations}')
im2 = axs[1].imshow(wang_max_amp_mat)
axs[1].set_title(r'\textbf{LIHFP}')
im3 = axs[2].imshow(sens_max_amp_mat)
axs[2].set_title(r'\textbf{Senseiver}')
#fig.colorbar(im1,ax=axs.ravel().tolist(),fraction=0.047, pad=0.04)
pos = axs[2].get_position()
cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
cbar = plt.colorbar(im3, cax=cax)


fig.suptitle(r'\textbf{Max Amplitudes}',y=.95)
plt.savefig(path+"/figs/"+"split_{}_reg_{}_MaxAmpMat.png".format(
            train_split,
            regularization)
        )

true_arr_time_mat = np.zeros(shape=(8,6))
for i in range(8):
    for j in range(6):
        true_arr_time_mat[i, j] = true_arrival_times_full[i][j]

wang_arr_time_mat = np.zeros(shape=(8,6))
for i in range(8):
    for j in range(6):
        wang_arr_time_mat[i, j] = wang_arrival_times_full[i][j]

sens_arr_time_mat = np.zeros(shape=(8,6))
for i in range(8):
    for j in range(6):
        sens_arr_time_mat[i, j] = sens_arrival_times_full[i][j]

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(16,8))
im1 = axs[0].imshow(true_arr_time_mat)
axs[0].set_title(r'\textbf{True}')
axs[0].set_xlabel(r'\textbf{Virtual Sensors}')
axs[0].set_ylabel(r'\textbf{Simulations}')
im2 = axs[1].imshow(wang_arr_time_mat)
axs[1].set_title(r'\textbf{LIHFP}')
im3 = axs[2].imshow(sens_arr_time_mat)
axs[2].set_title(r'\textbf{Senseiver}')
#fig.colorbar(im1,ax=axs.ravel().tolist(),fraction=0.047, pad=0.04)

pos = axs[2].get_position()
cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
cbar = plt.colorbar(im3, cax=cax)

fig.suptitle(r'\textbf{Arrival Times}',y=.95)
plt.savefig(path+"/figs/"+
                    "split_{}_reg_{}_ArrivalTimeMat.png".format(
            train_split,
            regularization)
        )

wang_ma_err = np.abs(true_max_amp_mat-wang_max_amp_mat)
sens_ma_err = np.abs(true_max_amp_mat-sens_max_amp_mat)


fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(13,8))
im1 = axs[0].imshow(wang_ma_err)
axs[0].set_title(r'\textbf{LIHFP MAE: }' + r'\textbf{'+str(round(np.mean(wang_ma_err),2))+r'}'r'\textbf{ (m)}')
axs[0].set_xlabel(r'\textbf{Virtual Sensor Number}')
axs[0].set_ylabel(r'\textbf{Simulation Number}')
im2 = axs[1].imshow(sens_ma_err)
axs[1].set_title(r'\textbf{Senseiver MAE: }'+ r'\textbf{'+str(round(np.mean(sens_ma_err),2))+r'}'+r'\textbf{ (m)}')
fig.suptitle(r'\textbf{Max Amplitude MAE (all simulations)}',y=.95)

pos = axs[1].get_position()
cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
cbar = plt.colorbar(im2, cax=cax)
plt.subplots_adjust(wspace=.1)
#plt.tight_layout()
#fig.colorbar(im1,ax=axs.ravel().tolist(),fraction=0.05, pad=0.04)
plt.savefig(path+"/figs/"+
                    "split_{}_reg_{}_MaxAmpMatErr.png".format(
            train_split,
            regularization),bbox_inches='tight')



wang_at_err = np.abs(true_arr_time_mat-wang_arr_time_mat)
sens_at_err = np.abs(true_arr_time_mat-sens_arr_time_mat)

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(13,8))
im1 = axs[0].imshow(wang_at_err)
axs[0].set_title(r'\textbf{LIHFP MAE: }' + r'\textbf{'+str(round(np.mean(wang_at_err),2))+r'}'r'\textbf{ (mins)}')
axs[0].set_xlabel(r'\textbf{Virtual Sensor Number}')
axs[0].set_ylabel(r'\textbf{Simulation Number}')
im2 = axs[1].imshow(sens_at_err)
axs[1].set_title(r'\textbf{Senseiver MAE: }' + r'\textbf{'+str(round(np.mean(sens_at_err),2))+r'}'r'\textbf{ (mins)}')
fig.suptitle(r'\textbf{Arrival Time MAE (all simulations)}',y=.95)

pos = axs[1].get_position()
cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
cbar = plt.colorbar(im2, cax=cax)
plt.subplots_adjust(wspace=.1)
#plt.tight_layout()
#fig.colorbar(im1,ax=axs.ravel().tolist(),fraction=0.05, pad=0.04)
plt.savefig(path+"/figs/"+
                    "split_{}_reg_{}_ArrivalTimeMatErr.png".format(
    train_split,regularization),bbox_inches='tight')


#make wang max amp colormap
#make sens max amp colormap
#make true at colormap
#make wang at colormap
#make sens at colormap
#make sens ma err cmap
#make wang ma err cmap
#make sens at err cmap
#make wang at err cmap

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

fig, ax = plt.subplots(figsize = (7,6))
ax.plot(true_max_amplitudes_full,color='k')
ax.plot(sens_max_amplitudes_full,color='r')
ax.plot(wang_max_amplitudes_full,color='g')
ax.legend(['True MA', 'Sens MA', 'LIHFP MA'])
plt.tight_layout()
plt.savefig(path+"/figs/"+
                    "split_{}_reg_{}_MaxAmps.png".format(
            train_split,
            regularization)
        )

fig, ax = plt.subplots(figsize = (7,6))
ax.plot(np.abs(true_max_amplitudes_full-wang_max_amplitudes_full),color='k')
ax.plot(np.abs(true_max_amplitudes_full-sens_max_amplitudes_full),color='r')
ax.legend(['Wang MA Err', 'Sens MA Err'])
plt.tight_layout()
plt.savefig(path+"/figs/"+
                    "split_{}_reg_{}_MaxAmpsErrs.png".format(
            train_split,
            regularization)
        )


fig, ax = plt.subplots(figsize = (7,6))
ax.plot(true_arrival_times_full,color='k')
ax.plot(sens_arrival_times_full,color='r')
ax.plot(wang_arrival_times_full,color='g')
ax.legend(['True AT', 'Sens AT', 'LIHFP AT'])
plt.tight_layout()
plt.savefig(path+"/figs/"+
                    "split_{}_reg_{}_ArrivalTimes.png".format(
            train_split,
            regularization)
        )

fig, ax = plt.subplots(figsize = (7,6))
ax.plot(np.abs(true_arrival_times_full-wang_arrival_times_full),color='k')
ax.plot(np.abs(true_arrival_times_full-sens_arrival_times_full),color='r')
ax.legend(['Wang AT Err', 'Sens AT Err'])
plt.tight_layout()
plt.savefig(path+"/figs/"+
                    "split_{}_reg_{}_ArrivalTimesErrs.png".format(
            train_split,
            regularization)
        )