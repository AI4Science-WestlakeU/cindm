#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from collections import OrderedDict
import datetime
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import pandas as pd
import gc
pd.options.display.max_rows = 1500
pd.options.display.max_columns = 200
pd.options.display.width = 1000
pd.set_option('max_colwidth', 400)
import pdb
import pickle
import pprint as pp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from deepsnap.batch import Batch as deepsnap_Batch

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from le_pde.argparser import arg_parse
from le_pde.datasets.load_dataset import load_data
from le_pde.models import load_model
from le_pde.pytorch_net.util import groupby_add_keys, filter_df, get_unique_keys_df, Attr_Dict, Printer, get_num_params, get_machine_name, pload, pdump, to_np_array, get_pdict, reshape_weight_to_matrix, ddeepcopy as deepcopy, plot_vectors, record_data, filter_filename, Early_Stopping, str2bool, get_filename_short, print_banner, plot_matrices, get_num_params, init_args, filter_kwargs, to_string, COLOR_LIST
from le_pde.utils import update_legacy_default_hyperparam, EXP_PATH, deepsnap_to_pyg, LpLoss, to_cpu, to_tuple_shape, parse_multi_step, loss_op, get_device, get_data_next_step
from utils import compute_pressForce
#from le_pde.utils import deepsnap_to_pyg, LpLoss, to_cpu, to_tuple_shape, parse_multi_step, loss_op, get_device, get_data_next_step

device = torch.device("cuda:7")
p = Printer()


# ## 1. Functions:

# In[ ]:


def plot_learning_curve(data_record):
    x_axis = np.arange(len(data_record["train_loss"]))
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(x_axis, data_record["train_loss"], label="train")
    plt.plot(x_axis, data_record["val_loss"], label="val")
    plt.plot(x_axis, data_record["test_loss"], label="test")
    plt.legend()
    plt.subplot(1,2,2)
    plt.semilogy(x_axis, data_record["train_loss"], label="train")
    plt.semilogy(x_axis, data_record["val_loss"], label="val")
    plt.semilogy(x_axis, data_record["test_loss"], label="test")
    plt.legend()
    plt.show()
    


# ## 2. Load Data:

# In[ ]:


EXP_PATH = "./results/"

isplot = True
all_hash = [
    # "0LVoHLHQ_ampere4",
    # "zDOCitP9_ampere4",
    # "6en0gt6G_turing1",
    # "zHQu3EKe_turing2",
    # "2okNCadZ_turing3",
    # "I6EepBQI_turing3",
    # "clnAWVnz_hyperturing1",
    # "YDHgg+il_turing3",
    # "HD2hmsb+_turing3",
    # "krep6ZNu_turing2",
    "QvUQ8aaL_turing2",
]
hash_str = all_hash[0]
dirname = EXP_PATH + "naca_ellipse_2023-04-30/"
filename = filter_filename(dirname, include=hash_str)
if len(filename) == 0:
    raise

try:
    data_record = pload(dirname + filename[0])
except Exception as e:
    print(f"error {e}")
    # continue
    raise
if isplot:
    plot_learning_curve(data_record)
args = init_args(update_legacy_default_hyperparam(data_record["args"]))
args.filename = filename
# model = load_model(data_record["best_model_dict"], device=device)
model = load_model(data_record["model_dict"][-1], device=device)
model.eval()
p.print(filename, banner_size=100)

# Load test dataset:
args_test = deepcopy(args)
if args.temporal_bundle_steps == 1:
    if args.dataset in ["fno", "fno-2", "fno-3"]:
        args_test.multi_step = "20"
    elif args.dataset in ["fno-1"]:
        args_test.multi_step = "40"
    elif args.dataset in ["fno-4"]:
        args_test.multi_step = "10"
    elif args.dataset in ["naca_ellipse_lepde"]:
        args_test.multi_step = "1"
        args_test.latent_multi_step="1"
    else:
        raise
else:
    pass
args_test.batch_size = 1
args_test.is_test_only=True

(dataset_train_val, dataset_test), (train_loader, val_loader, test_loader) = load_data(args_test)
test_loader = DataLoader(dataset_test, num_workers=0, collate_fn=deepsnap_Batch.collate(),
                         batch_size=1, shuffle=False, drop_last=False)


# In[ ]:


normalization_filename = os.path.join("./dataset/naca_ellipse/training_trajectories/", "normalization_max_min.p")
normdict = pickle.load(open(normalization_filename, "rb"))
x_max = normdict["x_max"]
x_min = normdict["x_min"]
y_max = normdict["y_max"]
y_min = normdict["y_min"]
p_max = normdict["p_max"]
p_min = normdict["p_min"]


# In[ ]:


from utils import update_static_masks, reconstruct_boundary

double_masks = []
double_offsets = []
for simnum in range(20):
    mask_list = []
    offset_list = []
    for bd_index in range(2):
        bd = np.load("./np_boundary_multiple_double/sim_{:06d}/boundary_{:06d}.npy".format(simnum, bd_index)).transpose()
        torchbd = torch.tensor(bd, dtype=torch.float32)

        mask, offset = update_static_masks(torchbd)
        mask_list.append(mask)
        offset_list.append(offset)

    double_masks.append(mask_list[0] + mask_list[1])
    double_offsets.append(offset_list[0] + offset_list[1])

for data in test_loader:
    break
testdata = data.clone()


# In[ ]:


# data_list = []

# for double_sim in range(20):
#     a = deepsnap_Batch
#     batch, _ = a._init_batch_fields(testdata.keys, [])
#     batch.batch = testdata.batch.clone()
#     batch.compute_func = testdata.compute_func
#     batch.directed = testdata.directed.detach().clone()
#     batch.dyn_dims = testdata.dyn_dims
#     batch.edge_attr = testdata.edge_attr
#     batch.edge_index = {('n0','0','n0'): testdata.edge_index[('n0','0','n0')].detach().clone()}
#     batch.edge_label_index = {('n0','0','n0'): testdata.edge_label_index[('n0','0','n0')].detach().clone()}
#     batch.grid_keys = testdata.grid_keys
#     batch.mask = {"n0": torch.where(double_masks[double_sim]==0, True, False).detach()}

#     dbmask = double_masks[double_sim].reshape(-1, 1, 1)
#     dboffset = double_offsets[double_sim].reshape(-1, 1, 2)
#     dbstatic_grid = torch.cat([torch.cat([dbmask, dboffset], -1) for _ in range(4)], -2)

#     x_velo = torch.FloatTensor(np.stack([np.transpose(np.load("./dataset_naca_ellipse_multiple/training_trajectories/sim_{:06d}/velocity_{:06d}.npy".format(double_sim, j)), (1,2,0)) for j in range(0, 15, 4)], -2))  # [rows, cols, input_steps, 2]
#     x_velo[...,0] = (torch.clamp((x_velo[...,0] - x_min) / (x_max - x_min), 0, 1) - 0.5) * 2
#     x_velo[...,1] = (torch.clamp((x_velo[...,1] - y_min) / (y_max - y_min), 0, 1) - 0.5) * 2
#     x_velo[torch.isnan(x_velo)] = 0
#     # Pressure for input
#     x_pressure = torch.FloatTensor(np.stack([np.load("./dataset_naca_ellipse_multiple/training_trajectories/sim_{:06d}/pressure_{:06d}.npy".format(double_sim, j)) for j in range(0, 15, 4)], -1))[..., None]  # [rows, cols, input_steps, 1]
#     x_pressure = (torch.clamp((x_pressure - p_min) / (p_max - p_min), 0, 1) - 0.5) * 2
#     x_pressure[torch.isnan(x_pressure)] = 0
#     # Concatenate inputs
#     x_feature = torch.cat((x_velo, x_pressure), -1).reshape(-1, 4, 3)

#     batch.node_feature = {"n0": torch.cat((dbstatic_grid, x_feature), -1).detach()}
#     batch.node_label = {"n0": testdata.node_label["n0"].detach().clone()}
#     batch.node_label_index = {"n0": testdata.node_label_index["n0"].detach().clone()}
#     batch.node_pos = {"n0": testdata.node_pos["n0"].detach().clone()}
#     batch.original_shape = testdata.original_shape
#     batch.param = {"n0": testdata.param["n0"]}
#     batch.params = testdata.params
#     batch.part_keys = testdata.part_keys
#     batch.task = testdata.task
#     data_list.append(batch) 


# In[ ]:





# ## 3. inverse optimization with LEPDE

# In[ ]:


EXP_PATH = "./results/"

isplot = True
all_hash = [
    # "0LVoHLHQ_ampere4",
    # "zDOCitP9_ampere4",
    # "6en0gt6G_turing1",
    # "zHQu3EKe_turing2",
    # "2okNCadZ_turing3",
    # "I6EepBQI_turing3",
    # "clnAWVnz_hyperturing1",
    # "YDHgg+il_turing3",
    # "HD2hmsb+_turing3",
    # "krep6ZNu_turing2",
    #"QvUQ8aaL_turing2", #<---- most recent
    # "YpkNljy1_whdeng",
    "yAlVxifp_whdeng",

]
hash_str = all_hash[0]
# dirname = EXP_PATH + "naca_ellipse_2023-09-27/"
dirname = EXP_PATH + "naca_ellipse_2023-11-14/"
filename = filter_filename(dirname, include=hash_str)
if len(filename) == 0:
    raise

try:
    data_record = pload(dirname + filename[0])
except Exception as e:
    print(f"error {e}")
    # continue
    raise
if isplot:
    plot_learning_curve(data_record)
args = init_args(update_legacy_default_hyperparam(data_record["args"]))
args.filename = filename
# model = load_model(data_record["best_model_dict"], device=device)
model = load_model(data_record["model_dict"][-1], device=device)
model.eval()
p.print(filename, banner_size=100)

# Load test dataset:
args_test = deepcopy(args)
if args.temporal_bundle_steps == 1:
    if args.dataset in ["fno", "fno-2", "fno-3"]:
        args_test.multi_step = "20"
    elif args.dataset in ["fno-1"]:
        args_test.multi_step = "40"
    elif args.dataset in ["fno-4"]:
        args_test.multi_step = "10"
    elif args.dataset in ["naca_ellipse_lepde"]:
        args_test.multi_step = "1"
        args_test.latent_multi_step="1"
    else:
        raise
else:
    pass
args_test.batch_size = 1
args_test.is_test_only=True



# In[ ]:


from diffusion_2d_boundary_mask import ForceUnet
force_model = ForceUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=4
)
force_model.load_state_dict(torch.load("./dataset/epoch_12.pth"))
force_model.to(device)
print("ok")


# In[ ]:


from le_pde.utils import get_data_next_step_with_static
from matplotlib.backends.backend_pdf import PdfPages
from utils import compute_pressForce, compute_orthonormal, linear_transform, update_data
        
optim_iter = 100

prerollout = 0
one_period = 6
vis_prerollout = False

is_bdloss = True
if is_bdloss:
    mean_state = torch.tensor(np.load("./dataset/naca_ellipse/states_mean.npy")).to(device)
    mean_bd =  torch.tensor(np.load("./dataset/naca_ellipse/bd_mean.npy")).to(device)
    range_state =  torch.tensor(100.50).to(device)
    range_bd = torch.tensor(7.870).to(device)
    relu = nn.ReLU()
# In[ ]:


for testnum in range(10):
    print("testnum: ", testnum)

    data_list = []
    for double_sim in range(20):
        a = deepsnap_Batch
        batch, _ = a._init_batch_fields(testdata.keys, [])
        batch.batch = testdata.batch.clone()
        batch.compute_func = testdata.compute_func
        batch.directed = testdata.directed.detach().clone()
        batch.dyn_dims = testdata.dyn_dims
        batch.edge_attr = testdata.edge_attr
        batch.edge_index = {('n0','0','n0'): testdata.edge_index[('n0','0','n0')].detach().clone()}
        batch.edge_label_index = {('n0','0','n0'): testdata.edge_label_index[('n0','0','n0')].detach().clone()}
        batch.grid_keys = testdata.grid_keys
        batch.mask = {"n0": torch.where(double_masks[double_sim]==0, True, False).detach()}

        dbmask = double_masks[double_sim].reshape(-1, 1, 1)
        dboffset = double_offsets[double_sim].reshape(-1, 1, 2)
        dbstatic_grid = torch.cat([torch.cat([dbmask, dboffset], -1) for _ in range(4)], -2)

        x_velo = torch.FloatTensor(np.stack([np.transpose(np.load("./dataset_naca_ellipse_multiple_double/training_trajectories/sim_{:06d}/velocity_{:06d}.npy".format(double_sim, j)), (1,2,0)) for j in range(0, 15, 4)], -2))  # [rows, cols, input_steps, 2]
        x_velo[...,0] = (torch.clamp((x_velo[...,0] - x_min) / (x_max - x_min), 0, 1) - 0.5) * 2
        x_velo[...,1] = (torch.clamp((x_velo[...,1] - y_min) / (y_max - y_min), 0, 1) - 0.5) * 2
        x_velo[torch.isnan(x_velo)] = 0
        # Pressure for input
        x_pressure = torch.FloatTensor(np.stack([np.load("./dataset_naca_ellipse_multiple_double/training_trajectories/sim_{:06d}/pressure_{:06d}.npy".format(double_sim, j)) for j in range(0, 15, 4)], -1))[..., None]  # [rows, cols, input_steps, 1]
        x_pressure = (torch.clamp((x_pressure - p_min) / (p_max - p_min), 0, 1) - 0.5) * 2
        x_pressure[torch.isnan(x_pressure)] = 0
        # Concatenate inputs
        x_feature = torch.cat((x_velo, x_pressure), -1).reshape(-1, 4, 3)

        batch.node_feature = {"n0": torch.cat((dbstatic_grid, x_feature), -1).detach()}
        batch.node_label = {"n0": testdata.node_label["n0"].detach().clone()}
        batch.node_label_index = {"n0": testdata.node_label_index["n0"].detach().clone()}
        batch.node_pos = {"n0": testdata.node_pos["n0"].detach().clone()}
        batch.original_shape = testdata.original_shape
        batch.param = {"n0": testdata.param["n0"]}
        batch.params = testdata.params
        batch.part_keys = testdata.part_keys
        batch.task = testdata.task
        data_list.append(batch) 
        
    datanum = 0
    for data in data_list:
        print("datanum: ", datanum)
        data.to(device)  
        optim_param = data.node_feature["n0"].detach().clone()
        optim_param.requires_grad=True

        testdata = data.clone()
        static_data = data.clone()
        testdata.node_feature["n0"] = optim_param
        static_data.node_feature["n0"] = optim_param[..., :3]

        optimizer = torch.optim.Adam([optim_param], lr=0.0001)

        list_force = []
        list_drag_force = []
        final_results = []
        for oiter in range(optim_iter):
            # print(oiter)
            total_x_force = 0
            total_y_force = 0
            bd_loss = 0

            testdata.node_feature["n0"] = optim_param
            static_data.node_feature["n0"] = optim_param[..., :3]

            if oiter == (optim_iter - 1):
                force_list = []

            for kk in range(prerollout+one_period):
                # print(kk)
                if oiter % 50 == 49 and kk == 0 and vis_prerollout:
                    print("kk = 0")
                    fig, ax = plt.subplots(figsize=(4,4), ncols=1)
                    ax.imshow(torch.nn.functional.pad(((((pred["n0"].reshape(62, 62, 1, 3)[...,0,-1])/2) + 0.5) * (p_max-p_min)) + p_min, ((1,3,1,3))).detach().cpu().numpy(), cmap='viridis',
                             aspect='auto',
                             origin='lower')
                    plt.show()

                    #testdata, pred = get_data_next_step(model, testdata, use_grads=False, return_data=True, is_y_diff=False, is_rollout=True)
                    # pdb.set_trace()
                    press = ((((pred["n0"].reshape(62, 62, 1, 3)[...,0,-1])/2) + 0.5) * (p_max-p_min)) + p_min
                    # press = ((((testdata.node_label["n0"][:,-1,-1].reshape(62,62)/2) + 0.5) * (p_max-p_min)) + p_min) #.detach().cpu()
                    # raw_bound = (((testdata.param["n0"].reshape(40,2)/2) + 0.5) * 62) + 0

                    length, nx, ny, cen = compute_orthonormal(torch.tensor(raw_bound))
                    cen = cen.to(device)

                if oiter % 50 == 49 and kk == prerollout and vis_prerollout:
                    # print("kk = " + str(prerollout))
                    fig, ax = plt.subplots(figsize=(4,4), ncols=1)
                    ax.imshow(torch.nn.functional.pad(press, ((1,3,1,3))).detach().cpu().numpy(), cmap='viridis',
                             aspect='auto',
                             origin='lower')
                    plt.show()
                    # print(cen.shape, raw_bound.shape)

                if kk >= prerollout:
                    # testdata, pred = get_data_next_step(model, testdata, use_grads=False, return_data=True, is_y_diff=False)
                    testdata, pred = get_data_next_step_with_static(model, testdata, static_data=static_data, use_grads=False, return_data=True, is_y_diff=False)
                    # pdb.set_trace()
                    if oiter == optim_iter - 1:
                        try:    
                            os.makedirs("./optimized_traj_lepde_BP_twobds_rebuttal_na/test_{:06d}/sim_{:06d}".format(testnum, datanum))
                        except Exception:
                            pass   
                        with open('./optimized_traj_lepde_BP_twobds_rebuttal_na/test_{:06d}/sim_{:06d}/feature_{:06d}.npy'.format(testnum, datanum, kk), 'wb') as f:
                            np.save(f, testdata.node_feature["n0"].detach().cpu().numpy())

                    input_press = ((((pred["n0"].reshape(62, 62, 1, 3)[...,-1:])/2) + 0.5) * (p_max-p_min)) + p_min

                    input_node_feature = torch.cat([input_press, testdata.node_feature["n0"][:,-2:-1,:3].reshape(62, 62, 1, 3)], -1).reshape(62, 62, 1, -1)
                    input_node_feature = torch.permute(input_node_feature, (2, 3, 0, 1))
                    data_pad = torch.zeros(1, 4, 64, 64).to(input_node_feature.device)
                    data_pad[ :, :, 1:-1, 1:-1] = input_node_feature
                    input_node_feature = data_pad

                    x_force, y_force = force_model(input_node_feature)[0]
                    if oiter == optim_iter - 1:
                        force_list.append(torch.stack([x_force, y_force], - 1))

                    # press = ((((testdata.node_label["n0"][:,-1,-1].reshape(62,62)/2) + 0.5) * (p_max-p_min)) + p_min) #.detach().cpu()
                    # raw_bound = (((testdata.param["n0"].reshape(40,2)/2) + 0.5) * 62) + 0
                    # lin_press = linear_transform(torch.nn.functional.pad(press, (1,3,1,3)), cen)
                    # x_force, y_force = compute_pressForce(torch.nn.functional.pad(press, (1,3,1,3)), raw_bound)

                    total_x_force += x_force
                    total_y_force += y_force
                    if is_bdloss:
                        # pdb.set_trace()
                        # pdb.set_trace()
                        diff_press = (((((testdata.node_feature["n0"][:,-1,5])/2) + 0.5) * (p_max-p_min)) + p_min - mean_state.reshape(-1,3)[:, 0])
                        diff_velx = (((((testdata.node_feature["n0"][:,-1,3])/2) + 0.5) * (x_max-x_min)) + x_min - mean_state.reshape(-1,3)[:, 1])
                        diff_vely = (((((testdata.node_feature["n0"][:,-1,4])/2) + 0.5) * (y_max-y_min)) + y_min - mean_state.reshape(-1,3)[:, 2])
                        bd_loss += relu(torch.stack([diff_press, diff_velx, diff_vely]).norm() - 0.5*range_state)
                        bd_loss += relu((testdata.node_feature["n0"][:,-1,:3] - mean_bd.reshape(-1,3)).norm() - 0.5*range_bd)


            # if oiter == optim_iter - 1:
            #     with open("./optimized_traj_lepde_BP/sim_{:06d}/raw_force.npy".format(datanum), 'wb') as f:
            #         np.save(f, torch.stack(force_list, 0).detach().cpu().numpy())


            total_x_force = total_x_force/one_period
            total_y_force = total_y_force/one_period

            list_force.append(-total_y_force.item())
            list_drag_force.append(total_x_force.item())

            if is_bdloss:
                output = torch.abs(total_x_force) - total_y_force + bd_loss
            else:
                output = torch.abs(total_x_force) - total_y_force
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            #aft_bd = torch.cat((const_variable, opt_variable), 0).transpose(1,0).flatten()[None,:].reshape(40,2).clone()

            ### Visualization of updated boundary """
            if oiter % 50 == 49:
            # if True:
            # if oiter % 10 == 9:
                print("iteration: ", oiter)
                # fig, ax = plt.subplots(figsize=(18,3), ncols=6)
                # # mappable0 = ax[0].plot(bd[:,0], bd[:,1])
                # # nx = nx.detach().cpu()
                # # ny = ny.detach().cpu()
                # # cen = cen.detach().cpu()
                # # lin_press = lin_press.cpu()
                # # # print(cen.device, normals.device, lin_press.device)
                # # normals = torch.stack((lin_press*nx,lin_press*ny), -1)
                # # for i in range(40):
                # #     rel_normals = cen[i,:] + normals[i,:]
                # #     ax[0].plot((cen[i,0].numpy(), rel_normals[0].detach().numpy()), (cen[i,1].numpy(), rel_normals[1].detach().numpy()))
                # # ax[0].set_xlim(24, 35)
                # # ax[0].set_ylim(32, 43)
                # mappable1 = ax[1].imshow(testdata.node_feature["n0"][:,-1,0].reshape(62,62).detach().cpu().numpy(), cmap='viridis',
                #                          aspect='auto',
                #                          origin='lower')
                # fig.colorbar(mappable1, ax=ax[1])
                # vis_offsetmask = torch.where(testdata.node_feature["n0"][:,-1,1]!=0, 1, 0)
                # mappable2 = ax[2].imshow(testdata.node_feature["n0"][:,-1,1].reshape(62,62).detach().cpu().numpy(), cmap='viridis',
                #                          aspect='auto',
                #                          origin='lower')        
                # fig.colorbar(mappable2, ax=ax[2])
                # mappable3 = ax[3].imshow(testdata.node_feature["n0"][:,-1,2].reshape(62,62).detach().cpu().numpy(), cmap='viridis',
                #                          aspect='auto',
                #                          origin='lower')        
                # fig.colorbar(mappable3, ax=ax[3])
                # mappable4 = ax[4].plot(np.array(list_force)[0::5])
                # mappable5 = ax[5].plot(np.array(list_drag_force)[0::5])
                # # pdf.savefig()
                # plt.show()
                # # print(opt_mask)
                # # print("")
                # # print(opt_offset)

        datanum += 1

        # pdf.close()


# In[ ]:




