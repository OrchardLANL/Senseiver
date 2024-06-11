import math
import torch
from sensor_loc import (cylinder_8_sensors, sea_n_sensors)
from datasets import (cylinder,NOAA)
from einops import rearrange, repeat



def interp2d_single(data2: torch.Tensor, x_sens: torch.Tensor, y_sens: torch.Tensor, edist: torch.Tensor) -> torch.Tensor:
    xx = torch.arange(0, data2.shape[1], 1)
    xy = torch.arange(0, data2.shape[2], 1)
    itp = torch.empty(data2.shape[0],len(x_sens), 1)
    itp_edist = torch.empty(len(x_sens))

    for i, (x_sens, y_sens) in enumerate(zip(x_sens,y_sens)):
        idx_xx = torch.sum(torch.ge(x_sens[None], xx[None,:]), 1) -1
        idx_xy = torch.sum(torch.ge(y_sens[None], xy[None,:]), 1) -1
        idx_xx = torch.clamp(idx_xx, 0, data2.shape[1] - 2) 
        idx_xy = torch.clamp(idx_xy, 0, data2.shape[2] - 2) 
        x0 = xx[idx_xx]
        x1 = xx[idx_xx+1]
        y0 = xy[idx_xy]
        y1 = xy[idx_xy+1]
        wa = (x1-x_sens) * (y1-y_sens)
        wb = (x1-x_sens) * (y_sens-y0)
        wc = (x_sens-x0) * (y1-y_sens)
        wd = (x_sens-x0) * (y_sens-y0)
        
        Ia_edist = edist[idx_xx, idx_xy ]
        Ib_edist = edist[idx_xx, idx_xy+1 ]
        Ic_edist = edist[idx_xx+1, idx_xy ]
        Id_edist = edist[idx_xx+1, idx_xy+1 ]

        itp_edist[i] = ((Ia_edist*wa + Ib_edist*wb+ Ic_edist*wc + Id_edist*wd)/((x1-x0)*(y1-y0)))
        
        if edist[idx_xx, idx_xy]!=0.:
            Ia = data2[ :, idx_xx, idx_xy ]
        elif edist[idx_xx+1, idx_xy]!=0.:
            Ia = data2[ :, idx_xx+1, idx_xy ]
        elif edist[idx_xx, idx_xy+1]!=0.:
            Ia = data2[ :, idx_xx, idx_xy+1 ]
        else:
            Ia = data2[ :, idx_xx+1, idx_xy+1 ]
            
        if edist[idx_xx, idx_xy+1]!=0.:
            Ib = data2[ :, idx_xx, idx_xy+1 ]
        elif edist[idx_xx+1, idx_xy+1]!=0.:
            Ib = data2[ :, idx_xx+1, idx_xy+1 ]
        elif edist[idx_xx, idx_xy]!=0.:
            Ib = data2[ :, idx_xx, idx_xy ]
        else:
            Ib = data2[ :, idx_xx+1, idx_xy ]
            
        if edist[idx_xx+1, idx_xy]!=0.:
            Ic = data2[ :, idx_xx+1, idx_xy ]
        elif edist[idx_xx+1, idx_xy+1]!=0.:
            Ic = data2[ :, idx_xx+1, idx_xy+1 ]
        elif edist[idx_xx, idx_xy]!=0.:
            Ic = data2[ :, idx_xx, idx_xy ]
        else:
            Ic = data2[ :, idx_xx, idx_xy+1 ]
            
        if edist[idx_xx+1, idx_xy+1]!=0.:
            Id = data2[ :, idx_xx+1, idx_xy+1 ]
        elif edist[idx_xx+1, idx_xy]!=0.:
            Id = data2[ :, idx_xx+1, idx_xy ]
        elif edist[idx_xx, idx_xy+1]!=0.:
            Id = data2[ :, idx_xx, idx_xy+1 ]
        else:
            Id = data2[ :, idx_xx, idx_xy ]
            
        itp[:,i,:] = ((Ia*wa + Ib*wb+ Ic*wc + Id*wd)/((x1-x0)*(y1-y0)))[:,:,0]
    return itp, itp_edist


def load_data(dataset_name, num_sensors, seed=123):
    
    if dataset_name == 'cylinder':
        data = cylinder()
            
        if num_sensors == 8:
            x_sens, y_sens = cylinder_8_sensors()        
        
    elif dataset_name == 'sea':
        data = NOAA()
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)  
        
    print(f'Data size {data.shape}\n')
    return torch.as_tensor( data, dtype=torch.float ), x_sens, y_sens

def PositionalEncoder_sensors(x_sens,y_sens,image_shape,num_frequency_bands,max_frequencies=None):
    
    *spatial_shape, _ = image_shape

    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]
    pos = torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape)) 
    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]
    frequency_sensors = []
    pos = torch.stack((x_sens/((spatial_shape[0]-1)/2)-1,y_sens/((spatial_shape[1]-1)/2)-1),dim=len(spatial_shape)-1)
    for i, frequencies_i in enumerate(frequencies):
        frequency_sensors.append(pos[..., i:i+1].to('cpu') * frequencies_i[None, ...].to('cpu'))
    encodings.extend([torch.sin(math.pi * frequency_sensor) for frequency_sensor in frequency_sensors])
    encodings.extend([torch.cos(math.pi * frequency_sensor) for frequency_sensor in frequency_sensors])
    enc = torch.cat(encodings, dim=-1)
    
    # flatten encodings along spatial dimensions
    enc = rearrange(enc, "... c -> (...) c")
    return enc


