import numpy as np
import pickle as pk

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from tqdm import tqdm as bar

from model import Encoder, Decoder
                    


class Senseiver(pl.LightningModule):
    def __init__(self,**kwargs):
    
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        
        
        pos_encoder_ch = self.hparams.space_bands*len(self.hparams.image_size)*2
        
        self.encoder = Encoder(
            input_ch = self.hparams.im_ch+pos_encoder_ch,
            preproc_ch = self.hparams.enc_preproc_ch,
            num_latents = self.hparams.num_latents,
            num_latent_channels = self.hparams.enc_num_latent_channels,
            num_layers = self.hparams.num_layers,
            num_cross_attention_heads = self.hparams.num_cross_attention_heads,
            num_self_attention_heads = self.hparams.enc_num_self_attention_heads,
            num_self_attention_layers_per_block = self.hparams.num_self_attention_layers_per_block,
            dropout = self.hparams.dropout,
        )
        
       
        self.decoder_1 = Decoder(
            ff_channels = pos_encoder_ch,
            preproc_ch = self.hparams.dec_preproc_ch,  # latent bottleneck
            num_latent_channels = self.hparams.dec_num_latent_channels,  # hyperparam
            latent_size = self.hparams.latent_size,  # collapse from n_sensors to 1
            num_output_channels = self.hparams.im_ch,
            num_cross_attention_heads = self.hparams.dec_num_cross_attention_heads,
            dropout = self.hparams.dropout,
        )

        
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')
        
        
        
    def forward(self, sensor_values, query_coords):
        
        out = self.encoder(sensor_values)
        return self.decoder_1(out, query_coords)

    def training_step(self,batch, batch_idx):
        
        sensor_values, coords, field_values = batch
        
        # forward
        pred_values = self(sensor_values, coords)
        
        # loss
        loss = F.mse_loss(pred_values, field_values, reduction='sum')
        
        self.log("train_loss", loss/field_values.numel(), 
                 on_step=True, on_epoch=True,prog_bar=True, logger=True,
                 batch_size=1)
        
        return loss
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)





    def test(self, dataloader, num_pix=1024, split_time=0):
        
        #dataloader.dataset.data = dataloader.dataset.data[2382:2383,] 
        im_num, *im_size, im_ch = dataloader.dataset.data.shape
        
        im_pix    = np.prod(im_size)
        pixels    = np.arange( 0, im_pix, num_pix )
        output_im = torch.zeros(im_num, im_pix, im_ch)
        
        # split the time steps to avoid OOM errors
        if im_num==1:
            times = [0,1]
        else:
            times = np.linspace(0, im_num, split_time, dtype=int)
        
        # data
        im = dataloader.dataset.data
        sensors = dataloader.dataset.sensors
        pos_encodings = dataloader.dataset.pos_encodings
        
        
        t = 0
        for t_start in bar( times[1:] ):
            dt = t_start-t
            for pix in bar( pixels ):
                
                coords = pos_encodings[pix:pix+num_pix,][None,]
                coords = coords.repeat_interleave(dt, axis=0)
                
                sensor_values = im.flatten(start_dim=1, end_dim=-2)[t:t_start,sensors]
                
                sensor_positions = pos_encodings[sensors,][None,]
                sensor_positions = sensor_positions.repeat_interleave(sensor_values.shape[0], axis=0)
                
                sensor_values = torch.cat([sensor_values,sensor_positions], axis=-1)
           
                out = self(sensor_values, coords)
            
                output_im[t:t_start,pix:pix+num_pix] = out
            t += dt
            
        output_im = output_im.reshape(-1, *im_size, im_ch)
        output_im[dataloader.dataset.data==0]=0
        
        return output_im
    

    
    
    def histogram(self, path):
            import pickle
            
            results = dict()
            with torch.no_grad():
                
                self.im_num = 500
                self.im = self.im[:self.im_num]
                
                
                pixels = np.arange( 0, self.im_pix)
                coords = self.pos_encoder.position_encoding[:,][None,]
                
                for seed in bar( [123,1234,12345,9876,98765,666,777,11111] ):
                    results[str(seed)] = {}
                    for num_of_sensors in [25,50,100,150,200,250,500,750]:
                        
                        torch.manual_seed(seed)
                        rnd_sensor_ind = torch.randperm( 6144 )[:num_of_sensors]
                    
                        pred = torch.zeros(self.im_num, self.im_pix, 1) 
                        
                        sensor_positions = self.pos_encoder.position_encoding[self.sensors[rnd_sensor_ind],][None,]
                        for pix in range(self.im_num):
                            
                            sensor_values = self.im.flatten(start_dim=1, end_dim=2)[pix:pix+1,self.sensors[rnd_sensor_ind]]
                            sensor_values = torch.cat([sensor_values,sensor_positions], axis=-1)
                            pred[pix,:] = self(sensor_values, coords)
                            
                        pred = pred.reshape(-1, *self.im_dims, self.im_ch)
                        e = (self.im.cpu()-pred).norm(p=2, dim=(1,2))/(self.im.cpu()).norm(p=2, dim=(1,2))
                        results[str(seed)][str(num_of_sensors)] = e.mean()
                    print(results)
            with open(f'{path}/errors.pk', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
    
    
    
    
    
    
    
    
    
