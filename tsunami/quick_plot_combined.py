from glob import glob as gb

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from s_parser import parse_args
from dataloaders import senseiver_dataloader
from network_light import Senseiver

from combined_plot import plot_all_ts_tsu_japan_unstruct

import multiprocessing
import os

num_gpus = torch.cuda.device_count()
print("Num Devices:", num_gpus)
print("GPU Info:", [torch.cuda.get_device_name(i) for i in range(num_gpus)])

# arg parser
data_config, encoder_config, decoder_config = parse_args()

multiprocessing.set_start_method("fork")

# load the simulation data and create a dataloader
dataloader1 = senseiver_dataloader(data_config, num_workers=4)

data_config2 = data_config
data_config2['data_key'] = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_289.mat"
dataloader2 = senseiver_dataloader(data_config2, num_workers=4)


model_num1 = encoder_config['load_model_num']
if model_num1 == 34:
    model_num2 = 22
else:
    model_num2 = 2


print(f'Loading {model_num1} ...')

model_loc1 = gb(f"lightning_logs/version_{model_num1}/checkpoints/*.ckpt")[0]
# Use the below commented code if using on HPC
# model = Senseiver.load_from_checkpoint(model_loc,
#                                        **encoder_config,
#                                        **decoder_config,
#                                        **data_config)
model1 = Senseiver.load_from_checkpoint(model_loc1, map_location=torch.device('cpu'),
                                       **encoder_config,
                                       **decoder_config,
                                       **data_config)

print(f'Loading {model_num2} ...')

model_loc2 = gb(f"lightning_logs/version_{model_num2}/checkpoints/*.ckpt")[0]
# Use the below commented code if using on HPC
# model = Senseiver.load_from_checkpoint(model_loc,
#                                        **encoder_config,
#                                        **decoder_config,
#                                        **data_config)
model2 = Senseiver.load_from_checkpoint(model_loc2, map_location=torch.device('cpu'),
                                        **encoder_config,
                                        **decoder_config,
                                        **data_config2)


name = 'tensor.pt'
unseen_flag = data_config['unseen_flag']
mat_data1 = data_config['data_key']

with torch.no_grad():
    if data_config['unseen_flag'] == True:
        output_im1 = torch.load(os.getcwd()+"/lightning_logs/version_"+str(model_num1)+'/tensor'+'_unseen'+'.pt')
        output_im2 = torch.load(os.getcwd()+"/lightning_logs/version_"+str(model_num2)+'/tensor'+'_unseen'+'.pt')
    else:
        output_im1 = torch.load(os.getcwd() + "/lightning_logs/version_" + str(model_num1) + '/tensor' + '_training' + '.pt')
        output_im2 = torch.load(os.getcwd() + "/lightning_logs/version_" + str(model_num2) + '/tensor' + '_training' + '.pt')

if encoder_config['load_model_num'] == 34:
    path = "/Users/emcdugald/sparse_sens_tsunami/lightning_logs/combined_34_22/"
else:
    path = "/Users/emcdugald/sparse_sens_tsunami/lightning_logs/combined_0_2/"

if unseen_flag:
    name = 'tensor_unseen.pt'
    type = 'unseen'
else:
    name = 'tensor_training.pt'
    type = 'training'
plot_all_ts_tsu_japan_unstruct(dataloader1,dataloader2, output_im1, output_im2, mat_data1, type, path)


