#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt


            
            
def plot_cs(model,output_im):
    
    
    i = 600
    
    true = model.im


    plt.figure(dpi=300);
    plt.subplot(3,1,1);
    plt.imshow(true[i,:,:,:], cmap='seismic_r');plt.colorbar(); 
    
    plt.subplot(3,1,2);
    plt.imshow(output_im[i,:,:,:],cmap='seismic_r');plt.colorbar(); 
    
    plt.subplot(3,1,3);plt.imshow(true[i,:,:,:]-output_im[i,:,:,:], cmap='coolwarm');
    plt.colorbar();
    
    plt.tight_layout()
    
    
    plt.suptitle(i)
