import numpy as np
import os
import scipy.io as sio



def SWETsunamiForPlotting(fname):
    fpath = os.getcwd()+"/Data/tsunami/"+fname
    data = sio.loadmat(fpath)
    wave_height = data['zt']
    max_wave_height = np.max(np.abs(wave_height))
    wave_height /= max_wave_height
    longitude = data['longitude'][0]
    latitude = data['latitude'][0]
    mask = data['ismask'][0]
    times = data['data_times']
    sensors = data['sensor_locs']
    div = data['du']
    return wave_height, latitude, longitude, mask, max_wave_height, times, sensors, div


def SWETsunamiWdiv(fname):
    fpath = os.getcwd()+"/Data/tsunami/"+fname
    data = sio.loadmat(fpath)
    wave_height = data['zt']
    wave_height /= np.max(np.abs(wave_height))
    longitude = data['longitude'][0]
    latitude = data['latitude'][0]
    ocn_floor = data['ocn_floor'][0]
    ocn_floor /= np.max(np.abs(ocn_floor))
    mask = data['ismask'][0]
    divu = data['du']
    divu /= np.max(np.abs(wave_height))
    if 'agg' in fname:
        time_idx = np.cumsum(data['data_times'])
        return wave_height, latitude, longitude, ocn_floor, divu, mask, time_idx
    else:
        return wave_height, latitude, longitude, ocn_floor, divu, mask


def SWETsunamiForPlotting2(fname):
    fpath = "/Users/emcdugald/sparse_sens_tsunami/Data/tsunami/"+fname
    data = sio.loadmat(fpath)
    wave_height = data['zt']
    max_wave_height = np.max(np.abs(wave_height))
    wave_height /= max_wave_height
    longitude = data['longitude'][0]
    latitude = data['latitude'][0]
    mask = data['ismask'][0]
    times = data['data_times']
    sensors = data['sensor_locs']
    div = data['du']
    return wave_height, latitude, longitude, mask, max_wave_height, times, sensors, div
    