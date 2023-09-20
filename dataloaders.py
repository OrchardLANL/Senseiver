import os

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch


from datasets import cylinder, NOAA, pipe, plume, porous

from sensor_loc import ( cylinder_16_sensors, 
                         cylinder_8_sensors, 
                         cylinder_4_sensors, 
                         cylinder_4BC_sensors,
                         sea_n_sensors,
                         sensors_3D
                         )



import datetime
from positional import PositionalEncoder



from torch.utils.data import DataLoader,Dataset






def load_data(dataset_name, num_sensors, seed=123):
    
    if dataset_name == 'cylinder':
        data = cylinder()
        
        if num_sensors == 16:
            x_sens, y_sens = cylinder_16_sensors()
            
        if num_sensors == 8:
            x_sens, y_sens = cylinder_8_sensors()
            
        if num_sensors == 4:
            x_sens, y_sens = cylinder_4_sensors()
            
        if num_sensors == 4444:
            x_sens, y_sens = cylinder_4BC_sensors()
            
    elif dataset_name == 'sea':
        data = NOAA()
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
        
    elif dataset_name == 'pipe':
       data = pipe()
       x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
       
    elif dataset_name == 'plume':
        data = plume()
        data = data[None,:,:,:,None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)
        
    elif dataset_name == 'pore':
        data = porous()
        data = data[:,:,:,:,None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)
       
    else:
        #raise NameError('Unknown dataset')
        print(f'The dataset_name {dataset_name} was not provided\n')
        print('************WARNING************')
        print('*******************************\n')
        print('Creating a dummy dataset\n')
        print('************WARNING************')
        print('*******************************\n')
        data = np.random.rand(1000,150,75,1)
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)
        
    print(f'Data size {data.shape}\n')
    return torch.as_tensor( data, dtype=torch.float ), x_sens, y_sens
    
    
    
def senseiver_dataloader(data_config, num_workers=0):
    return DataLoader( senseiver_loader(data_config), batch_size=None, 
                       pin_memory=True, 
                       shuffle = True,
                       num_workers=4
                     )
    

class senseiver_loader(Dataset):
    
    def __init__(self,  data_config):
    
        data_name   = data_config['data_name']
        num_sensors = data_config['num_sensors']
        seed        = data_config['seed']
        
        self.data_name = data_name
        self.data, x_sens, y_sens = load_data(data_name, num_sensors, seed)
        
        total_frames, *image_size, im_ch = self.data.shape
        
        data_config['total_frames'] = total_frames
        data_config['image_size']   = image_size
        data_config['im_ch']        = im_ch
        
        self.training_frames = data_config['training_frames']
        self.batch_frames    = data_config['batch_frames'] 
        self.batch_pixels    = data_config['batch_pixels'] 
        
        
        num_batches = int(self.data.shape[1:].numel()*self.training_frames/(
                                            self.batch_frames*self.batch_pixels))
        
        assert num_batches>0
        
        print(f'{num_batches} Batches of data per epoch\n')
        data_config['num_batches'] = num_batches
        self.num_batches = num_batches
        
        if data_config['consecutive_train']:
            self.train_ind = torch.arange(0,self.training_frames)
        else:
            if seed:
                torch.manual_seed(seed)
            self.train_ind = torch.randperm(self.data.shape[0])[:self.training_frames]
            #print(self.train_ind)
            
        if self.batch_frames > self.training_frames:
            print('Warning: batch_frames bigger than num training samples')
            self.batch_frames = self.training_frames
            
        # sensor coordinates
        sensors = np.zeros(self.data.shape[1:-1])
        
        if len(sensors.shape) == 2:
            sensors[x_sens,y_sens] = 1
        elif len(sensors.shape) == 3: # 3D images
            sensors[x_sens,y_sens[0],y_sens[1]] = 1
            
        self.sensors,*_ = np.where(sensors.flatten()==1)
        
        # sine-cosine positional encodings
        self.pos_encodings = PositionalEncoder(self.data.shape[1:],data_config['space_bands'])
        
        self.indexed_sensors  = self.data.flatten(start_dim=1, end_dim=-2)[:,self.sensors,]
        self.sensor_positions = self.pos_encodings[self.sensors,]
        self.sensor_positions = self.sensor_positions[None,].repeat_interleave(
                                                    self.batch_frames, axis=0)
        # get non-zero pixels
        self.pix_avail = self.data.flatten(start_dim=1, end_dim=-2)[0,:,0].nonzero()[:,0]
        
        if seed:
            torch.manual_seed(datetime.datetime.now().microsecond) # reset seed
            
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        frames = self.train_ind[ torch.randperm( self.training_frames) ][:self.batch_frames]
        pixels = self.pix_avail[ torch.randperm(*self.pix_avail.shape) ][:self.batch_pixels]
        
        sensor_values = self.indexed_sensors[frames,]
        sensor_values = torch.cat([sensor_values,self.sensor_positions], axis=-1)
        
        # moving sensors
        if self.data_name == 'pipe':
            rnd_sensor_num = (40+300*torch.abs( torch.randn(1))).type(torch.int)
            rnd_sensor_ind = torch.randperm(6144)[:rnd_sensor_num]
            sensor_values  = sensor_values[:,rnd_sensor_ind,:]
            
        coords = self.pos_encodings[pixels,][None,]
        coords = coords.repeat_interleave(self.batch_frames, axis=0)
        
        field_values = self.data.flatten(start_dim=1, end_dim=-2)[frames,][:,pixels,]
        
        return sensor_values, coords, field_values
        
     
    
