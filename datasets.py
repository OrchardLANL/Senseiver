import numpy as np
import h5py
import pickle



def NOAA():
   
    f = h5py.File('Data/NOAA/sst_weekly.mat','r') 
    sst = np.nan_to_num( np.array(f['sst']) )
    
    num_frames = 1914

    sea = np.zeros((num_frames,180,360,1))
    for t in range(num_frames):
        sea[t,:,:,0] = sst[t,:].reshape(180,360,order='F')
    sea /= sea.max()
    return sea



def pipe():
    
   with open("Data/Turbulent/ch_2Dxysec.pickle", 'rb') as f:
       pipe = pickle.load(f)
       pipe /= np.abs(pipe).max()
   return pipe


def cylinder():
    
    with open('Data/Cylinder/Cy_Taira.pickle', 'rb') as f:
        cyl = pickle.load(f)/11.0960
    return cyl
    

def plume():
    with h5py.File('Data/Plume/concentration.h5', "r") as f:
        plume_3D = f['cs']
        plume_3D = np.array(plume_3D)
        plume_3D /= plume_3D.max()
    return plume_3D


def porous():
    with h5py.File('Data/Pore/rho_1.h5', "r") as f:
        pore = f['rho'][:]
    return pore
    
def isotropic3D():
    with h5py.File('Isotropic/scalarHIT_fields100.h5', "r") as f:
        return np.array(f['fields'])
