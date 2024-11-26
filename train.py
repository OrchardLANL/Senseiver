import numpy as np
import matplotlib.pyplot as plt

from glob import glob as gb

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from s_parser import parse_args
from dataloaders import senseiver_dataloader
from network_light import Senseiver

from plot import plot_cs


def main(data_config, encoder_config, decoder_config):

    # load the simulation data and create a dataloader
    dataloader = senseiver_dataloader(data_config, num_workers=4)

    # instantiate new Senseiver
    model = Senseiver(
        **encoder_config,
        **decoder_config,
        **data_config
    )

    # load model (if requested)
    if encoder_config['load_model_num']:
        model_num = encoder_config['load_model_num']
        print(f'Loading {model_num} ...')

        model_loc = gb(f"lightning_logs/version_{model_num}/checkpoints/*.ckpt")[0]
        model = model.load_from_checkpoint(model_loc,
                                           **encoder_config,
                                           **decoder_config,
                                           **data_config)
    else:
        model_loc = None

    if not data_config['test']:
        # callbacks
        cbs = [ModelCheckpoint(monitor="train_loss", filename="train-{epoch:02d}",
                               every_n_epochs=10, save_on_train_epoch_end=True),
               EarlyStopping(monitor="train_loss", check_finite=False, patience=100)]

        trainer = Trainer(max_epochs=-1,
                          callbacks=cbs,
                          accumulate_grad_batches=data_config['accum_grads'],
                          log_every_n_steps=data_config['num_batches'],
                          )

        trainer.fit(model, dataloader,
                    ckpt_path=model_loc
                    )

    else:
        if data_config['gpu_device']:
            device = data_config['gpu_device'][0]
            model = model.to(f"cuda:{device}")

            model = model.to(f"cuda:{data_config['gpu_device'][0]}")
            dataloader.dataset.data = torch.as_tensor(
                dataloader.dataset.data).to(f"cuda:{device}")
            dataloader.dataset.sensors = torch.as_tensor(
                dataloader.dataset.sensors).to(f"cuda:{device}")
            dataloader.dataset.pos_encodings = torch.as_tensor(
                dataloader.dataset.pos_encodings).to(f"cuda:{device}")

        path = model_loc.split('checkpoints')[0]

        with torch.no_grad():
            output_im = model.test(dataloader, num_pix=2048, split_time=10)
        torch.save(output_im, f'{path}/res.torch')


if __name__ == "__main__":
    # arg parser
    data_config, encoder_config, decoder_config = parse_args()

    main(
        data_config=data_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
