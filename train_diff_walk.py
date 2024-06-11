import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm as bar
from s_parser import parse_args
from positional import PositionalEncoder
from network_light import Senseiver
from functions_diff_walk import load_data, interp2d_single, PositionalEncoder_sensors
from scipy.ndimage.morphology import distance_transform_edt

# Parse configurations
data_config, encoder_config, decoder_config = parse_args()
data_name = data_config['data_name']
num_sensors = data_config['num_sensors']
seed = data_config['seed']

# Determine the device
if data_config['accelerator'] == 'gpu':
    device = f"cuda:{data_config['gpu_device'][0]}"
elif data_config['accelerator'] == 'mps':
    device = "mps"
else:
    device = "cpu"

# Load data and sensors
data, x_sens, y_sens = load_data(data_name, num_sensors, seed)
edist = torch.Tensor(distance_transform_edt(data[0,:,:,0])).unsqueeze(-1)


total_frames, *image_size, im_ch = data.shape
data_config.update({
    'total_frames': total_frames,
    'image_size': image_size,
    'im_ch': im_ch
})

training_frames = data_config['training_frames']
batch_frames = data_config['batch_frames']
batch_pixels = data_config['batch_pixels']

# Calculate number of batches per epoch
num_batches = int(data[1:].numel() * training_frames / (batch_frames * batch_pixels))
print(f'{num_batches} Batches of data per epoch\n')
data_config['num_batches'] = num_batches

# Determine training indices
if data_config['consecutive_train']:
    train_ind = torch.arange(0, training_frames)
    print(train_ind)
else:
    if seed:
        torch.manual_seed(seed)
    train_ind = torch.randperm(data.shape[0])[:training_frames]
    print('non_consec')
    print(train_ind)

# Check batch frame size
if batch_frames > training_frames:
    print('Warning: batch_frames bigger than num training samples')
    batch_frames = training_frames

# Initialize sensors
sensors = np.zeros(data.shape[1:-1])
if len(sensors.shape) == 2:
    sensors[x_sens, y_sens] = 1
elif len(sensors.shape) == 3:
    sensors[x_sens, y_sens[0], y_sens[1]] = 1

sensors, *_ = np.where(sensors.flatten() == 1)

# Generate positional encodings
pos_encodings = PositionalEncoder_sensors(
    torch.Tensor(x_sens), 
    torch.Tensor(y_sens), 
    data.shape[1:], 
    data_config['space_bands']
)

indexed_sensors = data.flatten(start_dim=1, end_dim=-2)[:, sensors]
sensor_positions = pos_encodings[None,].repeat_interleave(batch_frames, axis=0)

# Get non-zero pixels
pix_avail = data.flatten(start_dim=1, end_dim=-2)[0, :, 0].nonzero()[:, 0]

if seed:
    torch.manual_seed(datetime.datetime.now().microsecond)

# Initialize model
model = Senseiver(
    **encoder_config,
    **decoder_config,
    **data_config
)

model.x_sens = torch.nn.Parameter(torch.Tensor((x_sens)), requires_grad=True)
model.y_sens = torch.nn.Parameter(torch.Tensor((y_sens)), requires_grad=True)

model_loc = "train-model.pth" if data_config['test'] else None
if model_loc:
    model.load_state_dict(torch.load(model_loc)['model_state_dict'])
model.to(device)


if not data_config['test']:
    parameters = [
        {'params': model.x_sens, 'lr': 0.25},  
        {'params': model.y_sens, 'lr': 0.25}
    ]
    
    other_parameters = [
        param for param in model.parameters() 
        if param.size() != model.x_sens.size() and param.size() != model.y_sens.size()
    ]
    
    parameters += [{'params': other_parameters}]
    optimizer = torch.optim.Adam(parameters, lr=data_config['lr'])
    
    pos_encodings_all = PositionalEncoder(data.shape[1:], data_config['space_bands'])
    loss_best = float('inf')
    check = torch.zeros(num_sensors)
    check_edist_prev = torch.ones(num_sensors)

    # Training loop
    for epoch in range(100000):
        model.train()
        frames = train_ind[torch.randperm(training_frames)][:batch_frames]
        pixels = pix_avail[torch.randperm(*pix_avail.shape)][:batch_pixels]
        
        sensor_values = torch.cat(
            [indexed_sensors[frames], sensor_positions], axis=-1)
        
        coords = pos_encodings_all[pixels][None,].repeat_interleave(batch_frames, axis=0)
        field_values = data.flatten(start_dim=1, end_dim=-2)[frames][:, pixels]
        
        sensor_values_i, coords_i, field_values_i = sensor_values.to(device), coords.to(device), field_values.to(device)

        pred_values = model(sensor_values_i, coords_i)
        loss = F.mse_loss(pred_values, field_values_i, reduction='sum')

        loss.backward()
        model.x_sens.data.clamp_(0, data.shape[1])
        model.y_sens.data.clamp_(0, data.shape[2])

        pos_encodings = PositionalEncoder_sensors(model.x_sens, model.y_sens, data.shape[1:], data_config['space_bands'])
        indexed_sensors, itp_edist = interp2d_single(
            data, model.x_sens.cpu(), model.y_sens.cpu(), edist
        )
        sensor_positions = pos_encodings[None,].repeat_interleave(batch_frames, axis=0)

        threshold = 2
        if epoch > 1:
            if torch.any(itp_edist <= threshold):
                check_temp = check.clone()
                check[itp_edist <= threshold] = 1
                check_temp = check - check_temp
                check_edist = torch.ones(num_sensors)
                check_edist[itp_edist <= threshold] = itp_edist[itp_edist <= threshold]
                check_temp[check_edist <= check_edist_prev] = 1
                check_temp[itp_edist == 0] = 1
                check_edist_prev = check_edist.clone()
                model.x_sens.grad[check_temp == 1] = -model.x_sens.grad[check_temp == 1]
                model.y_sens.grad[check_temp == 1] = -model.y_sens.grad[check_temp == 1]
                optimizer.state_dict()['state'][0]['exp_avg'][check_temp == 1] = -optimizer.state_dict()['state'][0]['exp_avg'][check_temp == 1]
                optimizer.state_dict()['state'][1]['exp_avg'][check_temp == 1] = -optimizer.state_dict()['state'][1]['exp_avg'][check_temp == 1]
            else:
                check = torch.zeros(num_sensors)

        optimizer.step()
        optimizer.zero_grad()

        print(f'Train Epoch {epoch + 1}, Loss {loss}')
        if torch.any(itp_edist == 0):
            print('Error sensors inland!')
        else:
            if loss_best > loss:
                print(f'Saving model at epoch {epoch}')
                torch.save({
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'train-model.pth')
                loss_best = loss

else:
    pos_encodings = PositionalEncoder_sensors(model.x_sens, model.y_sens, data.shape[1:], data_config['space_bands'])
    indexed_sensors, _ = interp2d_single(data, model.x_sens.cpu(), model.y_sens.cpu(), torch.Tensor(edist))
    sensor_positions = pos_encodings[None,].repeat_interleave(batch_frames, axis=0)
    data = torch.as_tensor(data).to(device)
    pos_encodings = torch.as_tensor(pos_encodings).to(device)

    with torch.no_grad():
        im_num, *im_size, im_ch = data.shape
        num_pix = 2048
        split_time = 10
        im_pix = np.prod(im_size)
        pixels = np.arange(0, im_pix, num_pix)
        output_im = torch.zeros(im_num, im_pix, im_ch)
        times = [0, 1] if im_num == 1 else np.linspace(0, im_num, split_time, dtype=int)

        pos_encodings_all = PositionalEncoder(data.shape[1:], data_config['space_bands'])
        t = 0
        for t_start in bar(times[1:]):
            dt = t_start - t
            for pix in bar(pixels):
                coords = pos_encodings_all[pix:pix + num_pix][None,].repeat_interleave(dt, axis=0)
                sensor_values = indexed_sensors[t:t_start].to(device)
                sensor_positions = pos_encodings[None,].repeat_interleave(sensor_values.shape[0], axis=0)
                sensor_values = torch.cat([sensor_values, sensor_positions], axis=-1)
                out = model(sensor_values.to(device), coords.to(device))
                output_im[t:t_start, pix:pix + num_pix] = out
            t += dt
        output_im = output_im.reshape(-1, *im_size, im_ch)
        output_im[data == 0] = 0

