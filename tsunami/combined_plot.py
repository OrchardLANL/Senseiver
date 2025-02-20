#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"]=16
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

def training_lons():
    return np.array([136.6180, 139.5560, 139.3290, 138.9350, 140.9290, 135.7400, 141.5010, 142.3870])

def training_lats():
    return np.array([33.0700, 28.8560, 28.9320, 29.3840, 33.4530, 33.1570, 35.9360, 35.2670])

def unseen_lons():
    return np.array([136.6500, 140.2000, 138.9000, 139.5000])

def unseen_lats():
    return np.array([33.1000, 29.1000, 28.1000, 28.8000])


def unseen_lons_new():
    return np.array([136.6500,138.2000,138.9000,
                     139.5000,140.2000,140.5000,
                     141.5000,142.5000])

def unseen_lats_new():
    return np.array([33.1000,31.0000,28.1000,
                     28.8000,29.1000,31.8000,
                     34.2000,36.2000])



def plot_err_for_each_epi(means, times, dta_type, path, plot_pts):
    time_idxs = np.insert(times,0,0)
    pts = np.sort(plot_pts)

    if dta_type=='training':
        lons = training_lons()
        lats = training_lats()
    else:
        lons = unseen_lons_new()
        lats = unseen_lats_new()

    j=0
    for i in range(1, len(time_idxs)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.2,6))
        emean = means[int(time_idxs[i - 1]):int(time_idxs[i])]
        epi_lon = lons[j]
        epi_lat = lats[j]
        if dta_type == 'training':
            start_plt_pts = (i - 1) * (np.shape(emean)[0])
            end_plt_pts = (i) * (np.shape(emean)[0])
            plot_points = pts[(pts > start_plt_pts) & (pts < end_plt_pts)] - start_plt_pts
            ax.plot(emean)
            ax.scatter(plot_points*(5./6.), torch.tensor(emean)[plot_points], c='r', marker='o', s=5)
            ax.title.set_text(f'Average Error = {torch.tensor(emean).mean():0.3}')
            #ax.set(xlabel=r"\textbf{Time Steps}", ylabel="$\mathbf{|true-pred|/\max(|true|)}$")

            ax.set_xlabel(r"\textbf{Time Steps}")
            ax.set_ylabel(r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$", rotation='horizontal',
                          horizontalalignment='right', verticalalignment='center',fontsize=16)


            ax.legend(['all data', 'training data', '.10 threshold', '.05 threshold'])
        else:
            plot_pts = np.arange(0,len(emean),1)*(5./6.)
            plot_pts = plot_pts
            ax.plot(plot_pts,emean)

            arr = np.array(emean)
            condition = arr < 0.10
            for i in range(len(arr)):
                if condition[i] and np.all(condition[i:]):  # Check if all subsequent elements are also less than 0.10
                    lt_idx = i
                    break
            else:
                lt_idx = None  # If no such index exists

            text = r'\textbf{Average Error = }' + r'\textbf{'+str(round(torch.tensor(emean).mean().item(),3)) + r'}'
            ax.title.set_text(text)
            # ax.title.set_text(f'Average Error = {torch.tensor(emean).mean():0.3}')
            ax.scatter(plot_pts[90],emean[90],s=75,marker="x",color='r')
            ax.scatter(plot_pts[180], emean[180], s=75, marker="x", color='r')
            ax.scatter(plot_pts[270], emean[270], s=75, marker="x", color='r')
            ax.scatter(plot_pts[lt_idx], emean[lt_idx], s=75, marker="x", color='g')
            ax.set_xlim(0,240)
            ax.set_ylim(0,np.max(emean))
            # ax.set(xlabel="Time (minutes)", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
            # ax.legend(['_',
            #            r'Error at $75$ mins: ${}$'.format(round(emean[90],3)),
            #            r'Error at $150$ mins: ${}$'.format(round(emean[180],3)),
            #            r'Error at 225 mins: ${}$'.format(round(emean[270],3)),
            #           r'$10\%$ threshold time: ${}$ mins'.format(round(plot_pts[lt_idx],1))])
            #ax.set(xlabel=r"\textbf{Time (minutes)}", ylabel=r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$")
            ax.set_xlabel(r"\textbf{Time (minutes)}")
            ax.set_ylabel(r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$", rotation='horizontal',
                          horizontalalignment='right', verticalalignment='center',fontsize=16)

            ax.legend(['_',
                       r'\textbf{Error at $75$ mins: ' + r'\textbf{' + str(round(emean[90],3)) + r'}',
                       r'\textbf{Error at $150$ mins: ' + r'\textbf{' + str(round(emean[180], 3)) + r'}',
                       r'\textbf{Error at $225$ mins: ' + r'\textbf{' + str(round(emean[270], 3)) + r'}',
                       r'\textbf{$10\%$ threshold time: }' + r'\textbf{' +str(round(plot_pts[lt_idx], 1)) + r'}' + r'\textbf{ mins}'])
        j += 1


        plt.tight_layout()
        if path:
            plt.savefig(path + '/all_ts_{}'.format(dta_type)+"_epi_{}_{}".format(epi_lon,epi_lat)+"_new.png",dpi=400,bbox_inches='tight')
            plt.close()



def plot_timeseries_for_each_epi_unstruct(true,pred,times,sens_idxs,sens_locs,dta_type,path):
    time_idxs = np.insert(times, 0, 0)

    if dta_type == 'training':
        lons = training_lons()
        lats = training_lats()
    else:
        lons = unseen_lons_new()
        lats = unseen_lats_new()


    for i in range(1, len(time_idxs)):
        j = 0
        for idx in sens_idxs[0]:
            true_ts = true[int(time_idxs[i - 1]):int(time_idxs[i]), idx]
            pred_ts = pred[int(time_idxs[i - 1]):int(time_idxs[i]), idx]
            if np.abs(true_ts).max() >= 5e-2:
                times = np.arange(0,len(true_ts),1)*(5/6)
                times = times
                fig, ax = plt.subplots(figsize=(7.2,6))
                ax.plot(times,true_ts, c='g')
                ax.plot(times,pred_ts, c='k')

                lon_deg = sens_locs[j][0] * 180 / np.pi
                lat_deg = sens_locs[j][1] * 180 / np.pi
                ax.set_xlim(0, 240)

                # ax.title.set_text('Wave Height Time Series at sensor ({}$^\circ$E, {}$^\circ$N)'.format(round(lon_deg,1), round(lat_deg,1)))
                # ax.set(xlabel="Time (minutes)", ylabel="Wave Height (m)")
                # ax.legend(['True Height', 'Predicted Height'])
                txt = r'\textbf{Wave Height at Sensor (}' + r'\textbf{' + str(
                    round(lon_deg, 1)) + r'}' + r'\textbf{$^\circ$E, }' + r'\textbf{' + str(
                    round(lat_deg, 1)) + r'}' + r'\textbf{$^\circ$N)}'
                # txt = r'\textbf{Wave Height Time Series at Sensor (}' + r'\textbf{' + str(round(lon_deg,1))+r'}' + r'\textbf{$^\circ$E}, '
                # + r'\textbf{'+str(round(lat_deg,1))+r'}'+r'\textbf{$^\circ$N)}'
                ax.title.set_text(txt)
                ax.set(xlabel=r"\textbf{Time (minutes)}", ylabel=r"\textbf{Wave Height (m)}")
                ax.legend([r'\textbf{True Height}', r'\textbf{Predicted Height}'])

                plt.tight_layout()

                plt.savefig(path + 'timeseries/{}'.format(dta_type)
                + "/sens_{}_{}_epi_{}_{}".format(round(lon_deg,3),round(lat_deg,3),round(lons[i-1],3),round(lats[i-1],3)) + "_new.png",dpi=400,bbox_inches='tight')
                plt.close()

            j += 1




def plot_div_err_for_each_epi(means, swemeans, times, dta_type, path):
    for k in range(len(times)):
        times[k] -= (k+1)
    time_idxs = np.insert(times,0,0)

    if dta_type=='training':
        lons = training_lons()
        lats = training_lats()
    else:
        lons = unseen_lons_new()
        lats = unseen_lats_new()

    j=0
    for i in range(1, len(time_idxs)):
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7.2,6))
        emean = means[int(time_idxs[i - 1])+8:int(time_idxs[i])]
        swe_emean = swemeans[int(time_idxs[i - 1])+8:int(time_idxs[i])]
        times = np.arange(0,len(emean),1)*(5/6)
        times = times
        epi_lon = lons[j]
        epi_lat = lats[j]
        ax.plot(times, emean, c='g')
        ax.plot(times, swe_emean, c='k')
        # ax.title.set_text(f'Average $h_t$ error = {torch.tensor(emean).mean():0.3}')
        # ax.set(xlabel="Time (minutes)", ylabel=r"$\frac{|h_t-\hat{h_t}|}{\max(|h|)}$")
        # ax.legend(['Senseiver Error', 'SWE Error'])
        txt = r'\textbf{Average }' +r'$\mathbf{h_t}$' + r'\textbf{ Error = }' + r'\textbf{'+str(round(torch.tensor(emean).mean().item(),3)) + r'}'
        ax.title.set_text(txt)
        #ax.set(xlabel=r"\textbf{Time (minutes)}", ylabel=r"$\mathbf{\frac{|h_t-\hat{h_t}|}{\max(|h|)}}$")
        ax.set_xlabel(r"\textbf{Time (minutes)}")
        ax.set_ylabel(r"$\mathbf{\frac{|h_t-\hat{h_t}|}{\max(|h|)}}$",
                      rotation='horizontal', horizontalalignment='right',
                      verticalalignment='center',fontsize=16)
        ax.legend([r'\textbf{Senseiver Error}', r'\textbf{SWE Error}'])
        ax.set_xlim(0,240)

        j += 1
        plt.tight_layout()
        if path:
            plt.savefig(path + '/all_ts_div_err_{}'.format(dta_type)+"_epi_{}_{}".format(epi_lon,epi_lat)+"_new.png",dpi=400,bbox_inches='tight')
            plt.close()

def plot_all_ts_tsu_japan_unstruct(dataloader1,dataloader2,pred1, pred2, matname, data_type,path=None):
    with (torch.no_grad()):

        true2hr = dataloader1.dataset.data.to('cpu')
        true4hr = dataloader2.dataset.data.to('cpu')
        pred2hr = pred1.to('cpu')
        pred4hr = pred2.to('cpu')
        mask = dataloader1.dataset.mask
        times2hr = true2hr.size()[0]
        times4hr = true4hr.size()[0]
        full_mask_2hr = mask.repeat([times2hr,1])
        full_mask_4hr = mask.repeat([times4hr,1])
        true_masked_2hr = torch.where((full_mask_2hr == 0), true2hr[..., 0], 0)
        pred_masked_2hr = torch.where((full_mask_2hr == 0), pred2hr[...,0], 0)
        true_masked_4hr = torch.where((full_mask_4hr == 0), true4hr[..., 0], 0)
        pred_masked_4hr = torch.where((full_mask_4hr == 0), pred4hr[..., 0], 0)
        intervals_2hr = [(0, 145), (145, 290), (290, 435), (435, 580), (580, 725), (725, 870), (870, 1015),
                         (1015, 1160)]
        sim1_2hr = true_masked_2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
        sim2_2hr = true_masked_2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
        sim3_2hr = true_masked_2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
        sim4_2hr = true_masked_2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
        sim5_2hr = true_masked_2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
        sim6_2hr = true_masked_2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
        sim7_2hr = true_masked_2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
        sim8_2hr = true_masked_2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
        intervals_4hr = [(0, 144), (144, 288), (288, 432), (432, 576), (576, 720), (720, 864), (864, 1008),
                         (1008, 1152)]
        sim1_4hr = true_masked_4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
        sim2_4hr = true_masked_4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
        sim3_4hr = true_masked_4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
        sim4_4hr = true_masked_4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
        sim5_4hr = true_masked_4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
        sim6_4hr = true_masked_4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
        sim7_4hr = true_masked_4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
        sim8_4hr = true_masked_4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
        sim1_full = np.concatenate((sim1_2hr, sim1_4hr), axis=0)
        sim2_full = np.concatenate((sim2_2hr, sim2_4hr), axis=0)
        sim3_full = np.concatenate((sim3_2hr, sim3_4hr), axis=0)
        sim4_full = np.concatenate((sim4_2hr, sim4_4hr), axis=0)
        sim5_full = np.concatenate((sim5_2hr, sim5_4hr), axis=0)
        sim6_full = np.concatenate((sim6_2hr, sim6_4hr), axis=0)
        sim7_full = np.concatenate((sim7_2hr, sim7_4hr), axis=0)
        sim8_full = np.concatenate((sim8_2hr, sim8_4hr), axis=0)
        true_masked = np.concatenate((sim1_full,sim2_full,sim3_full,sim4_full,
                                      sim5_full,sim6_full,sim7_full,sim8_full),axis=0)

        sens1_2hr = pred_masked_2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
        sens2_2hr = pred_masked_2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
        sens3_2hr = pred_masked_2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
        sens4_2hr = pred_masked_2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
        sens5_2hr = pred_masked_2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
        sens6_2hr = pred_masked_2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
        sens7_2hr = pred_masked_2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
        sens8_2hr = pred_masked_2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
        sens1_4hr = pred_masked_4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
        sens2_4hr = pred_masked_4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
        sens3_4hr = pred_masked_4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
        sens4_4hr = pred_masked_4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
        sens5_4hr = pred_masked_4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
        sens6_4hr = pred_masked_4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
        sens7_4hr = pred_masked_4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
        sens8_4hr = pred_masked_4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
        sens1_full = np.concatenate((sens1_2hr, sens1_4hr), axis=0)
        sens2_full = np.concatenate((sens2_2hr, sens2_4hr), axis=0)
        sens3_full = np.concatenate((sens3_2hr, sens3_4hr), axis=0)
        sens4_full = np.concatenate((sens4_2hr, sens4_4hr), axis=0)
        sens5_full = np.concatenate((sens5_2hr, sens5_4hr), axis=0)
        sens6_full = np.concatenate((sens6_2hr, sens6_4hr), axis=0)
        sens7_full = np.concatenate((sens7_2hr, sens7_4hr), axis=0)
        sens8_full = np.concatenate((sens8_2hr, sens8_4hr), axis=0)
        pred_masked = np.concatenate((sens1_full,sens2_full,sens3_full,sens4_full,
                                      sens5_full,sens6_full,sens7_full,sens8_full),axis=0)


        maxes = [np.abs(true_masked[i, :]).max() for i in range(len(true_masked))]
        abs_errs = [np.abs(true_masked[i, :] - pred_masked[i, :]) for i in
                    range(len(true_masked))]
        emax = [(abs_errs[i] / maxes[i]).max().item() for i in range(len(abs_errs))]
        ratios = [(abs_errs[i] / maxes[i]) for i in range(len(abs_errs))]
        emean = [ratios[i].flatten()[np.where(np.abs(true_masked[i].flatten()) > 1e-4)].mean().item() for i in range(len(abs_errs))]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
        plot_points_2hr = dataloader1.dataset.train_ind
        plot_points_4hr = dataloader2.dataset.train_ind
        plot_points_4hr = plot_points_4hr + 1160
        plot_points = torch.cat((plot_points_2hr,plot_points_4hr))

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        frame_ids_2hr = dataloader1.dataset.time_idx.astype(int)
        frame_ids_4hr = dataloader2.dataset.time_idx.astype(int)
        frame_ids = frame_ids_2hr + frame_ids_4hr - 289

        if data_type == 'training':
            ax.plot(emean)
            ax.scatter(plot_points, torch.tensor(emean)[plot_points.long()], c='r', marker='o', s=5)

            text = r'\textbf{Average Error = }' + r'\textbf{' + str(round(torch.tensor(emean).mean().item(), 3)) + r'}'
            ax.title.set_text(text)
            #ax.set(xlabel=r"\textbf{Frame Index}", ylabel=r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$")
            ax.set_xlabel(r"\textbf{Frame Index}")
            ax.set_ylabel(r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$", rotation='horizontal',
                          horizontalalignment='right', verticalalignment='center',fontsoze=16)


            ax.vlines(frame_ids, 0, np.max(emean), color='g')
            ax.legend([r'\textbf{all data}', r'\textbf{training data}',r'\textbf{epi split}'])
            # ax.title.set_text(f'$\epsilon$ = {torch.tensor(emean).nanmean():0.3}')
            # ax.set(xlabel="Frame Index", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
            # ax.vlines(frame_ids, 0, np.max(emean), color='g')
            # ax.legend(['all data', 'training data', 'epi split'])

        else:
            ax.plot(emean)
            text = r'\textbf{Average Error = }' + r'\textbf{' + str(round(torch.tensor(emean).mean().item(), 3)) + r'}'
            ax.title.set_text(text)
            #ax.set(xlabel=r"\textbf{Frame Index}", ylabel=r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$")

            ax.set_xlabel(r"\textbf{Frame Index}")
            ax.set_ylabel(r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$", rotation='horizontal',
                          horizontalalignment='right', verticalalignment='center',fontsize=16)


            ax.vlines(frame_ids, 0, np.max(emean), color='g')
            ax.legend([r'\textbf{all data}', r'\textbf{epi split}'])
            # ax.title.set_text(f'$\epsilon$ = {torch.tensor(emean).nanmean():0.3}')
            # ax.set(xlabel="Frame Index", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
            # ax.vlines(frame_ids, 0, np.max(emean), color='g')
            # ax.legend(['all data', 'epi split'])

        plt.tight_layout()
        if path:
            plt.savefig(path + '/all_ts_{}'.format(data_type)+"_new.png")
            plt.close()

        print(f'The mean mean err is {torch.tensor(emean).nanmean():0.3}')
        print(f'The mean max err is {torch.tensor(emax).nanmean():0.3}')

        sim_times = dataloader1.dataset.time_idx + dataloader2.dataset.time_idx
        plot_err_for_each_epi(emean, sim_times, data_type, path, plot_points)

        mat_data = sio.loadmat(os.getcwd()+"/Data/tsunami/"+matname)
        sensor_loc_indices = mat_data['sensor_loc_indices']
        sensor_locs = mat_data['sensor_locs']

        if not os.path.exists(path + 'timeseries/{}'.format(data_type)):
            if not os.path.exists(path + 'timeseries/'):
                os.mkdir(path + 'timeseries/')
            os.mkdir(path + 'timeseries/{}'.format(data_type))

        j = 0
        for idx in sensor_loc_indices[0]:
            true_ts = true_masked[:,idx]
            pred_ts = pred_masked[:,idx]

            if np.abs(true_ts).max() >= 1e-1:
                fig, ax = plt.subplots(figsize=(14,6))
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                ax.plot(true_ts,c='g')
                ax.plot(pred_ts,c='k')
                ax.plot(np.abs(true_ts - pred_ts), c='r')

                lon_deg = sensor_locs[j][1] * 180 / np.pi - 180
                lat_deg = sensor_locs[j][0] * 180 / np.pi
                txt = r'\textbf{Wave Height at Sensor (}' + r'\textbf{' + str(
                    round(lon_deg, 1)) + r'}' + r'\textbf{$^\circ$E, }' + r'\textbf{' + str(
                    round(lat_deg, 1)) + r'}' + r'\textbf{$^\circ$N)}'
                ax.title.set_text(txt)
                ax.set(xlabel=r"\textbf{Frame Index}", ylabel=r"\textbf{Normalized Wave Height (m)}")
                ax.legend([r'\textbf{True Height}', r'\textbf{Predicted Height}', r'\textbf{Absolute Error}'])

                # ax.title.set_text('Wave Height Time Series at {}, {}'.format(round(lon_deg), round(lat_deg)))
                # ax.set(xlabel="Frame Index", ylabel="Normalized Wave Height (m)")
                # ax.legend(['True Height', 'Predicted Height', 'Absolute Error'])

                plt.tight_layout()

                plt.savefig(path + 'timeseries/{}'.format(data_type)+"/sens_{}_{}_all_epis".format(round(lon_deg),round(lat_deg))+"_new.png")
                plt.close()

            j += 1

        plot_timeseries_for_each_epi_unstruct(true_masked, pred_masked, sim_times, sensor_loc_indices, sensor_locs, data_type, path)

        ### PHYSICAL CONSISTENCY PLOTS ###
        n_sims = len(dataloader1.dataset.time_idx)
        train_ind_start = np.append(0, dataloader1.dataset.time_idx[0:n_sims - 1])
        for i in range(8):
            train_ind_start[i] += 144*i
        combined_times = dataloader1.dataset.time_idx + dataloader2.dataset.time_idx
        train_inds = torch.concatenate([torch.range(train_ind_start[i], combined_times[i] - 2) for i in range(8)])
        curr_inds = torch.add(train_inds, 1).long()
        prev_inds = train_inds.long()

        truediv_2hr = dataloader1.dataset.divu.to('cpu')
        truediv_4hr = dataloader2.dataset.divu.to('cpu')
        sim1div_2hr = truediv_2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
        sim2div_2hr = truediv_2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
        sim3div_2hr = truediv_2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
        sim4div_2hr = truediv_2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
        sim5div_2hr = truediv_2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
        sim6div_2hr = truediv_2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
        sim7div_2hr = truediv_2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
        sim8div_2hr = truediv_2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
        sim1div_4hr = truediv_4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
        sim2div_4hr = truediv_4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
        sim3div_4hr = truediv_4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
        sim4div_4hr = truediv_4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
        sim5div_4hr = truediv_4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
        sim6div_4hr = truediv_4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
        sim7div_4hr = truediv_4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
        sim8div_4hr = truediv_4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
        sim1div_full = np.concatenate((sim1div_2hr, sim1div_4hr), axis=0)
        sim2div_full = np.concatenate((sim2div_2hr, sim2div_4hr), axis=0)
        sim3div_full = np.concatenate((sim3div_2hr, sim3div_4hr), axis=0)
        sim4div_full = np.concatenate((sim4div_2hr, sim4div_4hr), axis=0)
        sim5div_full = np.concatenate((sim5div_2hr, sim5div_4hr), axis=0)
        sim6div_full = np.concatenate((sim6div_2hr, sim6div_4hr), axis=0)
        sim7div_full = np.concatenate((sim7div_2hr, sim7div_4hr), axis=0)
        sim8div_full = np.concatenate((sim8div_2hr, sim8div_4hr), axis=0)
        true_div = np.concatenate((sim1div_full, sim2div_full, sim3div_full, sim4div_full,
                                      sim5div_full, sim6div_full, sim7div_full, sim8div_full), axis=0)

        true2hr = dataloader1.dataset.data.to('cpu')
        true4hr = dataloader2.dataset.data.to('cpu')
        pred2hr = pred1.to('cpu')
        pred4hr = pred2.to('cpu')
        sim1_2hr = true2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
        sim2_2hr = true2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
        sim3_2hr = true2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
        sim4_2hr = true2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
        sim5_2hr = true2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
        sim6_2hr = true2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
        sim7_2hr = true2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
        sim8_2hr = true2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
        sim1_4hr = true4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
        sim2_4hr = true4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
        sim3_4hr = true4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
        sim4_4hr = true4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
        sim5_4hr = true4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
        sim6_4hr = true4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
        sim7_4hr = true4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
        sim8_4hr = true4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
        sim1_full = np.concatenate((sim1_2hr, sim1_4hr), axis=0)
        sim2_full = np.concatenate((sim2_2hr, sim2_4hr), axis=0)
        sim3_full = np.concatenate((sim3_2hr, sim3_4hr), axis=0)
        sim4_full = np.concatenate((sim4_2hr, sim4_4hr), axis=0)
        sim5_full = np.concatenate((sim5_2hr, sim5_4hr), axis=0)
        sim6_full = np.concatenate((sim6_2hr, sim6_4hr), axis=0)
        sim7_full = np.concatenate((sim7_2hr, sim7_4hr), axis=0)
        sim8_full = np.concatenate((sim8_2hr, sim8_4hr), axis=0)
        true = np.concatenate((sim1_full, sim2_full, sim3_full, sim4_full,
                                      sim5_full, sim6_full, sim7_full, sim8_full), axis=0)
        sens1_2hr = pred2hr[intervals_2hr[0][0]:intervals_2hr[0][1]]
        sens2_2hr = pred2hr[intervals_2hr[1][0]:intervals_2hr[1][1]]
        sens3_2hr = pred2hr[intervals_2hr[2][0]:intervals_2hr[2][1]]
        sens4_2hr = pred2hr[intervals_2hr[3][0]:intervals_2hr[3][1]]
        sens5_2hr = pred2hr[intervals_2hr[4][0]:intervals_2hr[4][1]]
        sens6_2hr = pred2hr[intervals_2hr[5][0]:intervals_2hr[5][1]]
        sens7_2hr = pred2hr[intervals_2hr[6][0]:intervals_2hr[6][1]]
        sens8_2hr = pred2hr[intervals_2hr[7][0]:intervals_2hr[7][1]]
        sens1_4hr = pred4hr[intervals_4hr[0][0]:intervals_4hr[0][1]]
        sens2_4hr = pred4hr[intervals_4hr[1][0]:intervals_4hr[1][1]]
        sens3_4hr = pred4hr[intervals_4hr[2][0]:intervals_4hr[2][1]]
        sens4_4hr = pred4hr[intervals_4hr[3][0]:intervals_4hr[3][1]]
        sens5_4hr = pred4hr[intervals_4hr[4][0]:intervals_4hr[4][1]]
        sens6_4hr = pred4hr[intervals_4hr[5][0]:intervals_4hr[5][1]]
        sens7_4hr = pred4hr[intervals_4hr[6][0]:intervals_4hr[6][1]]
        sens8_4hr = pred4hr[intervals_4hr[7][0]:intervals_4hr[7][1]]
        sens1_full = np.concatenate((sens1_2hr, sens1_4hr), axis=0)
        sens2_full = np.concatenate((sens2_2hr, sens2_4hr), axis=0)
        sens3_full = np.concatenate((sens3_2hr, sens3_4hr), axis=0)
        sens4_full = np.concatenate((sens4_2hr, sens4_4hr), axis=0)
        sens5_full = np.concatenate((sens5_2hr, sens5_4hr), axis=0)
        sens6_full = np.concatenate((sens6_2hr, sens6_4hr), axis=0)
        sens7_full = np.concatenate((sens7_2hr, sens7_4hr), axis=0)
        sens8_full = np.concatenate((sens8_2hr, sens8_4hr), axis=0)
        pred = np.concatenate((sens1_full, sens2_full, sens3_full, sens4_full,
                                      sens5_full, sens6_full, sens7_full, sens8_full), axis=0)



        pred_dhdt = (pred[curr_inds,:,:]-pred[prev_inds,:,:])/50
        swe_dhdt = (true[curr_inds,:,:]-true[prev_inds,:,:])/50
        true_div_prev = true_div[prev_inds,:,:]
        true_div_curr = true_div[curr_inds,:,:]
        true_div = -.5*(true_div_prev+true_div_curr)
        times = true_div.shape[0]
        full_mask = mask.repeat([times, 1])
        true_div_masked = torch.where((full_mask == 0), torch.from_numpy(true_div[..., 0]), 0)
        pred_dhdt_masked = torch.where((full_mask == 0), torch.from_numpy(pred_dhdt[..., 0]), 0)
        swe_dhdt_masked = torch.where((full_mask == 0), torch.from_numpy(swe_dhdt[..., 0]), 0)
        maxes = [true_div_masked[i, :].abs().max() for i in range(len(true_div_masked))]
        abs_errs = [(true_div_masked[i, :] - pred_dhdt_masked[i, :]).abs() for i in
                    range(len(true_div_masked))]
        ratios = [(abs_errs[i] / maxes[i]) for i in range(len(abs_errs))]
        emean = [ratios[i].flatten()[torch.where(ratios[i].flatten().abs() > 1e-8)].mean() for i in
                 range(len(abs_errs))]
        swe_abs_errs = [(true_div_masked[i, :] - swe_dhdt_masked[i, :]).abs() for i in
                    range(len(true_div_masked))]
        swe_emax = [(swe_abs_errs[i] / maxes[i]).max().item() for i in range(len(swe_abs_errs))]
        swe_ratios = [(swe_abs_errs[i] / maxes[i]) for i in range(len(swe_abs_errs))]
        swe_emean = [swe_ratios[i].flatten()[torch.where(torch.from_numpy(true_masked[i]).flatten().abs() > 1e-4)].mean() for i in
                 range(len(abs_errs))]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.plot(emean)
        ax.scatter(range(len(swe_emean)),swe_emean, c='r',s=.5)
        txt = r'\textbf{Average Error = }' + r'\textbf{' + str(round(torch.tensor(emean).mean().item(), 3)) + r'}'
        ax.title.set_text(txt)
        #ax.set(xlabel=r"\textbf{Frame Index}", ylabel=r"$\mathbf{\frac{|h-\hat{h}|}{\max(|h|)}}$")
        ax.set_xlabel(r"\textbf{Frame Index}")
        ax.set_ylabel(r"$\mathbf{\frac{|h_t-\hat{h_t}|}{\max(|h_t|)}}$", rotation='horizontal',
                      horizontalalignment='right', verticalalignment='center',fontsize=16)
        ax.vlines(frame_ids, 0, max(np.max(swe_emean),np.max(emean)), color='g')
        ax.legend([r'\textbf{Senseiver Err}', r'\textbf{SWE Err}', r'\textbf{epi split}'])
        # ax.title.set_text(f'$\epsilon$ = {torch.tensor(emean).mean():0.3}')
        # ax.set(xlabel="Frame Index", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
        # ax.vlines(frame_ids, 0, max(np.max(swe_emean), np.max(emean)), color='g')
        # ax.legend(['Senseiver Err', 'SWE Err', 'epi split'])

        plt.tight_layout()
        if path:
            plt.savefig(path + '/divergence_all_ts_{}'.format(data_type) + "_new.png")
            plt.close()

        print(f'The mean mean div err is {torch.tensor(emean).mean():0.3}')
        print(f'The mean max div err is {torch.tensor(swe_emax).mean():0.3}')

        plot_div_err_for_each_epi(emean, swe_emean, sim_times, data_type, path)





