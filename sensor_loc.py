import numpy as np
import torch





def cylinder_4BC_sensors():
    
    
    coords = []
    for count, i in enumerate( range(0, 112, 28) ):
        if count==0:
            continue
        coords.append([5,  i])
        coords.append([187,i])

    coords = np.array( coords )

    coords = np.flip( coords, axis=1)
    
    
    return coords[[0,1,4,5],0], coords[[0,1,4,5],1]


def cylinder_8_sensors():
    
    coords = np.array( [ [76,71],  [175,69],  [138,49],                   
                         [41, 56], [141,61] ,[30,41],  
                         [177,40], [80,55] ] )
    
    coords = np.flip( coords, axis=1)
    
    return coords[:,0], coords[:,1]


def cylinder_16_sensors():
    
    coords = np.array( [ [76,71],  [175,69], [138,49],                   
                         [41, 56], [141,61], [30,41],  
                         [177,40], [80,55],  [60,41], [70,60],
                         [100,60], [120,51], [160,80],[165,50],
                         [180,60], [30,70] ] )
    
    coords = np.flip( coords, axis=1)
    
    return coords[:,0], coords[:,1]






def sea_n_sensors(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    
    im = np.copy(data[0,]).squeeze()
    
    print('Picking up sensor locations \n')
    coords = []
    
    for n in range(n_sensors):
        while True:
            new_x = np.random.randint(0,data.shape[1],1)[0]
            new_y = np.random.randint(0,data.shape[2],1)[0]
            if im[new_x,new_y] != 0:
                coords.append([new_x,new_y])
                im[new_x,new_y] = 0
                break
    coords = np.array(coords)  
    return coords[:,0], coords[:,1]
                


def sensors_3D(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    
    im = np.copy(data[0,]).squeeze()
    
    print('Picking up sensor locations \n')
    coords = []
    
    for n in range(n_sensors):
        while True:
            new_x = np.random.randint(0,data.shape[1],1)[0]
            new_y = np.random.randint(0,data.shape[2],1)[0]
            new_z = np.random.randint(0,data.shape[3],1)[0]
            if im[new_x,new_y,new_z] != 0:
                coords.append([new_x,new_y,new_z])
                im[new_x,new_y,new_z] = 0
                break
    coords = np.array(coords)  
    return coords[:,0], coords[:,1], coords[:,2]
                
        
        
    
    

        
        
    
    
    
    
    
    