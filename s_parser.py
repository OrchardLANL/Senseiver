import os
import argparse
import torch


# function to parse boolean args
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
        
        


def parse_args():

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Senseiver")
    
    # Data
    parser.add_argument("--data_name", default=None, type=str)
    parser.add_argument("--num_sensors", default=8, type=int)
    parser.add_argument("--gpu_device", default=0, type=int)
    parser.add_argument("--training_frames", default=1000, type=int)
    parser.add_argument("--consecutive_train", default=False, type=str2bool)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--batch_frames", default=64, type=int)
    parser.add_argument("--batch_pixels", default=2048, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--accum_grads", default=None, type=int)
    
    # Positional Encodings
    parser.add_argument("--space_bands", default=32, type=int)
    
    # Checkpoints
    parser.add_argument("--load_model_num", default=None, type=int)
    parser.add_argument("--test", default=False, type=str2bool)
    
    # Encoder
    parser.add_argument("--enc_preproc_ch", default=64, type=int)
    parser.add_argument("--num_latents",   default=4,     type=int)
    parser.add_argument("--enc_num_latent_channels", default=16, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--num_cross_attention_heads", default=2, type=int)
    parser.add_argument("--enc_num_self_attention_heads", default=2, type=int)
    parser.add_argument("--num_self_attention_layers_per_block", default=3, type=int)
    parser.add_argument("--dropout", default=0.00, type=float)
    
    # Decoder
    parser.add_argument("--dec_preproc_ch", default=None, type=int)
    parser.add_argument("--dec_num_latent_channels", default=16, type=int)
    parser.add_argument("--dec_num_cross_attention_heads", default=1, type=int)
    
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        accelerator = "gpu"
        gpus = [args.gpu_device]
    else:
        accelerator = "cpu"
        gpus = None
        
    # Assign the args
    data_config = dict(data_name = args.data_name,
                       num_sensors = args.num_sensors,
                       gpu_device=None if accelerator == 'cpu' else gpus,
                       accelerator = accelerator,
                       training_frames = args.training_frames,
                       consecutive_train = args.consecutive_train,
                       seed = args.seed,
                       batch_frames = args.batch_frames,
                       batch_pixels = args.batch_pixels,
                       lr=args.lr,
                       accum_grads = args.accum_grads,
                       test = args.test,
                       space_bands=args.space_bands,
                       )

    
    
    encoder_config = dict(load_model_num=args.load_model_num,
                          enc_preproc_ch=args.enc_preproc_ch,  # expand input dims
                          num_latents=args.num_latents,     # "seq" latent
                          enc_num_latent_channels=args.enc_num_latent_channels,  # channels [b,seq,chan]
                          num_layers=args.num_layers,
                          num_cross_attention_heads=args.num_cross_attention_heads,
                          enc_num_self_attention_heads=args.enc_num_self_attention_heads,
                          num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
                          dropout=args.dropout,
                          )


    decoder_config = dict(dec_preproc_ch=args.dec_preproc_ch,  # latent bottleneck
                          dec_num_latent_channels=args.dec_num_latent_channels,  # hyperparam
                          latent_size=1,  # collapse from n_sensors to 1 observation
                          dec_num_cross_attention_heads=args.dec_num_cross_attention_heads
                          )
    
    
    return data_config, encoder_config, decoder_config  