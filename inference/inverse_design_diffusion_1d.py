#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

import argparse
from collections import OrderedDict
import datetime
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import gc
import pdb
import pickle
import pprint as pp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import grad
from torch_geometric.data.dataloader import DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from cindm.utils import compute_pressForce,caculate_confidence_interval
from tqdm import tqdm
import matplotlib.backends.backend_pdf
import pprint as pp

import sys, os
import ast
from cindm.data.nbody_dataset import NBodyDataset
from cindm.model.diffusion_1d import TemporalUnet1D, GaussianDiffusion1D
from cindm.utils import p, get_item_1d, eval_simu, simulation, to_np_array, make_dir, pdump, pload
device = torch.device("cuda:0")
import cindm.filepath as filepath

# In[ ]:


import argparse

parser = argparse.ArgumentParser(description='Analyze the trained model')

parser.add_argument('--exp_id', default='inv_design', type=str, help='experiment folder id')
parser.add_argument('--date_time', default='09-23', type=str, help='date for the experiment folder')
parser.add_argument('--dataset', default='nbody-2', type=str, help='dataset to evaluate')

parser.add_argument('--model_type', default='temporal-unet1d', type=str, help='model type.')
parser.add_argument('--model_name', default='basic-model', type=str, help='model type.')
parser.add_argument('--conditioned_steps', default=4, type=int, help='conditioned steps')
parser.add_argument('--rollout_steps', default=20, type=int, help='rollout steps')
parser.add_argument('--time_interval', default=4, type=int, help='time interval')

parser.add_argument('--val_batch_size', default=1000, type=int, help='batch size for validation')
parser.add_argument('--is_test', default=True, type=bool,help='flag for testing')
parser.add_argument('--sample_steps', default=1000, type=int, help='sample steps')
parser.add_argument('--num_features', default=4, type=int,
                    help='in original datset,every data have 4 features,and processed datset just have 11 features ')

parser.add_argument('--dataset_path', default=filepath.current_wp+"/dataset/nbody_dataset", type=str,
                    help='the path to load dataset')

parser.add_argument('--n_composed', default=0, type=int,
                    help='how many prediction to be composed')
parser.add_argument('--compose_start_step', default=10, type=int,
                    help='Starting step of composition.')
parser.add_argument('--compose_n_bodies', default=2, type=int,
                    help='Number of total bodies.')

parser.add_argument('--design_guidance', type=str,
                    help='string for list of design_guidance')
parser.add_argument('--compose_mode', default="mean", type=str,
                    help='"mean" or "noise_sum"')
parser.add_argument('--design_fn_mode', default="L2", type=str,
                    help='Choose from "L2" and "L2square".')
parser.add_argument('--design_coef', default="0.05", type=str,
                    help='Coefficient for the design_fn')
parser.add_argument('--consistency_coef', default="0.05", type=str,
                    help='Coefficient for the consistency regularization')
parser.add_argument('--Unet_dim', default=64, type=int,
                    help='dim of Unet')
parser.add_argument('--initialization_mode', default=0, type=int,
                    help='in wgich mode to iniatialize cond: 0. random noise;1. data;2. data+random noise ')
parser.add_argument('--num_batchs', default=1, type=int,
                    help='number of batchs ')
parser.add_argument('--batch_size_list', default="[50]", type=str,
                    help='the list of different batch_size ')
parser.add_argument('--sample_steps_list', default="[1000]", type=str,
                    help='the list of sample steps ')
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args([])
    args.exp_id = "test"
    args.n_composed = 2
    args.compose_n_bodies = 4
    args.compose_mode = "mean-inside"
    args.design_fn_mode = "L2"
    args.design_coef = "0.4"
    args.consistency_coef = "0.2"

    args.compose_start_step = 10
    args.val_batch_size = 50
    args.model_name = "Diffusion_cond-0_rollout-24_bodies-2"
    args.model_name = "Diffusion_cond-0_rollout-24_bodies-2_more_collision"
    args.sample_steps = 1000
    design_guidance_list = [
        # "standard-recurrence-5",
        # "standard-alpha-recurrence-5",
        # "universal-forward-recurrence-5",
        # "universal-backward-recurrence-5",
        "standard-recurrence-10",
        # "standard-alpha-recurrence-10",
        # "universal-backward-pure-recurrence-10",
        # "universal-forward-pure-recurrence-10",
        # "universal-backward-recurrence-10",
        # "universal-forward-recurrence-10",
        # "standard",
        # "standard-alpha",
        # "universal-forward",
        # "universal-backward",
    ]
    args.design_guidance = ",".join(design_guidance_list)
    is_jupyter = True
except:
    args = parser.parse_args()
    is_jupyter = False
if args.model_name == "basic_model":
    args.rollout_steps = 20
elif args.model_name == "single_step_model":
    args.rollout_steps = 4
elif args.model_name in ["Diffusion_cond-0_rollout-24_bodies-2",
                         "Diffusion_cond-0_rollout-24_bodies-2_more_collision",
                        ]:
    args.rollout_steps = 24
    args.conditioned_steps = 0
elif args.model_name in ["Diffusion_cond-0_rollout-44_bodies-2",
                         "Diffusion_cond-0_rollout-44_bodies-2_Unet_dim-96",
                        ]:
    args.rollout_steps = 44
    args.conditioned_steps = 0
else:
    raise

# ## Load model and dataset:

# In[ ]:


model = TemporalUnet1D(
    horizon=args.conditioned_steps + args.rollout_steps,### horizon Maybe match the time_steps
    transition_dim=2*args.num_features, #n_bodies = 2, this matches num_bodies*nun_feactures
    cond_dim=False,
    dim=args.Unet_dim,
    dim_mults=(1, 2, 4, 8),
    attention=True,
)
diffusion = GaussianDiffusion1D(
    model,
    image_size = args.rollout_steps,
    conditioned_steps=args.conditioned_steps,
    timesteps=1000,           # number of steps
    sampling_timesteps=args.sample_steps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type='l1',           # L1 or L2
).to(device)
model_checkpoint = torch.load(f"checkpoint_path/{args.model_name}.pt")
diffusion.load_state_dict(model_checkpoint["model"])

dataset = NBodyDataset(
    dataset=f"nbody-2",
    input_steps=args.conditioned_steps,
    output_steps=args.rollout_steps+args.n_composed*args.compose_start_step,
    time_interval=4,
    is_y_diff=False,
    is_train=not args.is_test,
    is_testdata=False,
    dataset_path=args.dataset_path
)
dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6)

for data in dataloader:
    break
if args.model_name not in ["Diffusion_cond-0_rollout-24_bodies-2", "Diffusion_cond-0_rollout-24_bodies-2_more_collision","Diffusion_cond-0_rollout-44_bodies-2","Diffusion_cond-0_rollout-44_bodies-2_Unet_dim-96"]:
    cond = get_item_1d(data, "x").to(device)
else:
    cond = None
    initial_state_overwrite = get_item_1d(data, "y").to(device)[:,:4]
y_gt = get_item_1d(data, "y")
output_steps = args.rollout_steps+args.n_composed*args.compose_start_step
# pdb.set_trace()

# ## Design:

# In[ ]:


# Define objective:
def get_design_fn(pos_target, last_n_step, gamma=2, coef=100, time_consistency_coef=0, design_fn_mode="L2"):
    assert len(pos_target.shape) == 1
    def point_objective(pos):
        """pos: [B, steps, n_bodies*4]"""
        n_bodies = pos.shape[-1] // 4
        if design_fn_mode == "L2":
            assert gamma == 2
            loss = torch.stack([(((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)).mean(-1).sum(0) for jj in range(n_bodies)]).sum()
        elif design_fn_mode == "L2square":
            assert gamma == 2
            loss = torch.stack([((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1).mean(-1).sum(0) for jj in range(n_bodies)]).sum()
        else:
            raise
        loss_total = loss * coef
        if time_consistency_coef > 0:
            indices = torch.cat([torch.arange(ii*4,ii*4+2) for ii in range(n_bodies)])
            loss_total = loss_total + (pos[:,1:,indices] - pos[:,:-1,indices]).square().sum(-1).mean(-1).sum() * time_consistency_coef
        return loss_total
    return point_objective

def get_eval_fn(pos_target, last_n_step, gamma=2):
    """pos: [B, steps, F], pos_target: [F]"""
    assert len(pos_target.shape) == 1
    def point_eval_objective(pos):
        n_bodies = pos.shape[-1] // 4
        loss = torch.stack([(((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)).mean() for jj in range(n_bodies)]).mean()
        return loss.item()
    return point_eval_objective


def get_eval_fn_std(pos_target, last_n_step, gamma=2):
    """pos: [B, steps, F], pos_target: [F]"""
    assert len(pos_target.shape) == 1
    def point_eval_objective_std(pos):
        n_bodies = pos.shape[-1] // 4
        loss = torch.cat([(((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)) for jj in range(n_bodies)], -1).mean(-1)
        loss_std = loss.std()
        return loss_std.item()
    return point_eval_objective_std

def get_eval_fn_loss_each(pos_target, last_n_step=1, gamma=2):
    """pos: [B, steps, F], pos_target: [F]"""
    assert len(pos_target.shape) == 1
    def point_eval_objective_std(pos):
        n_bodies = pos.shape[-1] // 4
        loss = torch.cat([(((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)) for jj in range(n_bodies)], -1).mean(-1)
        return loss
    return point_eval_objective_std


# In[ ]:
# args.val_batch_size=args.num_batchs*sum(ast.literal_eval(args.batch_size_list))
p.print(f"test_start inverse_design ", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
batch_size_list=ast.literal_eval(args.batch_size_list)
sample_steps_list=ast.literal_eval(args.sample_steps_list)
design_obj_list=[]
MAE_list=[]
for sample_steps in sample_steps_list:
    Rt_list=[]
    diffusion.sampling_timesteps=sample_steps
    for batch_size_val in batch_size_list:
        args.val_batch_size=batch_size_val
        best_loss_sum=0
        for _ in range(args.num_batchs):
            # With composition:
            print(f"n_composed: {args.n_composed}")
            print(f"compose_n_bodies: {args.compose_n_bodies}")
            print("model name: ", args.model_name)
            pp.pprint(args.__dict__)
            start_eval_t = 0
            pos_target = torch.tensor([0.5,0.5], device=device, dtype=float)
            loss_each_fn=get_eval_fn_loss_each(pos_target, last_n_step=1)
            for design_guidance in args.design_guidance.split(","):
                for design_coef in args.design_coef.split(","):
                    design_coef = eval(design_coef)
                    for consistency_coef in args.consistency_coef.split(","):
                        data_record = {}
                        consistency_coef = eval(consistency_coef)
                        design_fn = get_design_fn(
                            pos_target,
                            last_n_step=1,
                            coef=design_coef,
                            time_consistency_coef=consistency_coef,
                            design_fn_mode=args.design_fn_mode,
                        )
                        eval_fn = get_eval_fn(pos_target, last_n_step=1)
                        eval_fn_std = get_eval_fn_std(pos_target, last_n_step=1)
                        p.print(f"Design guidance: {design_guidance}, design_coef: {design_coef}, consistency_coef: {consistency_coef}", banner_size=100)
                        data_record.update(args.__dict__)
                        data_record["design_coef"] = design_coef
                        data_record["consistency_coef"] = consistency_coef
                        data_record["design_guidance"] = design_guidance
                        
                        pred = diffusion.sample(
                            batch_size=args.val_batch_size,
                            cond=cond,
                            is_composing_time=args.n_composed>0,
                            n_composed=args.n_composed,
                            compose_start_step=args.compose_start_step,
                            compose_n_bodies=args.compose_n_bodies,
                            compose_mode=args.compose_mode,
                            design_fn=design_fn,
                            design_guidance=design_guidance,
                            initialization_mode=args.initialization_mode,
                            initialization_img=y_gt.to(device),
                        )
                        pred_simu, design_obj_simu = eval_simu(
                            cond_design=pred[:args.val_batch_size,start_eval_t:start_eval_t+1],
                            design_fn=eval_fn,
                            n_bodies=args.compose_n_bodies,
                            rollout_steps=output_steps - 1,
                        )
                        data_record["pred"] = to_np_array(pred)
                        data_record["pred_simu"] = to_np_array(pred_simu)
                        data_record["design_obj_simu"] = design_obj_simu
                        design_obj_simu_CI = eval_fn_std(pred_simu) * 1.96 / np.sqrt(args.val_batch_size)
                        data_record["design_obj_simu_CI"] = design_obj_simu_CI
                        pred_simu = torch.cat([pred[:args.val_batch_size,:start_eval_t+1].to(device), pred_simu], 1)
                        diff = pred_simu - pred
                        RMSE = diff.square().mean((1,2)).sqrt().mean()
                        # 95% confidence interval:
                        RMSE_CI = diff.square().mean((1,2)).sqrt().std() * 1.96 / np.sqrt(args.val_batch_size)
                        MAE = torch.nn.L1Loss()(pred_simu, pred).item()
                        MAE_CI = diff.abs().mean((1,2)).std().item() * 1.96 / np.sqrt(args.val_batch_size)
                        data_record["RMSE"] = RMSE
                        data_record["RMSE_CI"] = RMSE_CI
                        data_record["MAE"] = MAE
                        data_record["MAE_CI"] = MAE_CI
                        print(f"design_obj_simu: {design_obj_simu:.6f} ± {design_obj_simu_CI:.6f}", )
                        print(f"RMSE: {RMSE} ± {RMSE_CI}", )
                        print(f"MAE: {MAE} ± {MAE_CI}", )
                        if np.isnan(design_obj_simu):
                            pred_simu_mask = ~torch.isnan(pred_simu.mean((1,2)))
                            design_obj_simu_nonan = eval_fn(pred_simu[pred_simu_mask])
                            print(f"{torch.sum(~pred_simu_mask).item()} elements are NaN. After excluding, design_obj_simu = {design_obj_simu_nonan}")
                            data_record["design_obj_simu_nonan"] = design_obj_simu_nonan

                        fontsize = 16
                        T = pred.shape[1]

                        dirname = f"results/inverse_design_diffusion/{args.exp_id}_{args.date_time}/"
                        filename = f"comp_{args.compose_n_bodies}_nt_{args.n_composed}_guid_{design_guidance}_descoef_{design_coef}_conscoef_{consistency_coef}_desmode_{args.design_fn_mode}_compmode_{args.compose_mode}_val_{args.val_batch_size}_initialization_mode-{args.initialization_mode}"
                        make_dir(dirname + filename)
                        pdump(data_record, dirname + "record_" + filename + ".p")
                        pdf = matplotlib.backends.backend_pdf.PdfPages(dirname + filename + ".pdf")

                        for ball_id in range(5):
                            fig = plt.figure(figsize=(20,8))
                            plt.subplot(1,2,1)
                            # diffused traj:
                            for ii in range(args.compose_n_bodies):
                                plt.plot(pred.cpu()[ball_id, :, ii*4], pred.cpu()[ball_id, :, ii*4+1])
                                plt.scatter(pred.cpu()[ball_id, :, ii*4], pred.cpu()[ball_id, :, ii*4+1], s=np.arange(1, T+1)*5, marker="v")
                            # evolved traj with initial design:
                            for ii in range(args.compose_n_bodies):
                                plt.plot(pred_simu.cpu()[ball_id, :, ii*4], pred_simu.cpu()[ball_id, :, ii*4+1])
                                plt.scatter(pred_simu.cpu()[ball_id, :, ii*4], pred_simu.cpu()[ball_id, :, ii*4+1], s=np.arange(1, T+1)*5, marker="+")
                            plt.xlim([0,1])
                            plt.ylim([0,1])
                            plt.title(f"design_obj_eval = {design_obj_simu:.9f} ± {design_obj_simu_CI:.6f}", fontsize=fontsize)
                            plt.subplot(1,2,2)
                            for ii in range(args.compose_n_bodies):
                                plt.plot(pred_simu.cpu()[ball_id, :, ii*4], pred_simu.cpu()[ball_id, :, ii*4+1])
                                plt.scatter(pred_simu.cpu()[ball_id, :, ii*4], pred_simu.cpu()[ball_id, :, ii*4+1], s=np.arange(1, T+1)*5, marker="+")
                            plt.xlim([0,1])
                            plt.ylim([0,1])
                            plt.title(f"RMSE = {RMSE:.9f}    MAE = {MAE:.9f} ± {MAE_CI:.6f}")
                            pdf.savefig(fig)
                            if is_jupyter:
                                plt.show()
                        pdf.close()
            # pdb.set_trace()
            loss_batch_size=loss_each_fn(pred_simu)
            _,__,___,best_loss=caculate_confidence_interval(loss_batch_size)
            best_loss_sum=best_loss_sum+best_loss
            #caculate Rt from
        Rt_list.append(best_loss_sum.to("cpu")/args.num_batchs)
    design_obj_list.append(design_obj_simu)
    MAE_list.append(MAE)
    Rt_array=np.array(Rt_list)
    file_path=dirname+f"num_batchs-{args.num_batchs}_batchsize_list-{args.batch_size_list}"
    np.save(file_path+"_CinDM.npy",Rt_array)
    import matplotlib.pyplot as plt
    plt.plot(batch_size_list, Rt_list, marker='o', linestyle='-')

    plt.xlabel('T')
    plt.ylabel('R_T')
    plt.title('Curve of R_T')
    plt.savefig(file_path+".png")
    plt.show()
    p.print(f"test_end inverse_design ", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)

#draw curve for design obj or MAE of different sample steps
design_obj_array=np.array(design_obj_list)
MAE_array=np.array(MAE_list)
plt.figure()
fontsize=18
plt.plot( sample_steps_list,design_obj_list, marker='o', linestyle='-',color="green",label="design obj")
plt.legend()
# plt.title(r'design obj of different $\lambda$')
plt.xlabel(r'sample_steps',fontsize=fontsize)
plt.ylabel(r'design obj',fontsize=fontsize)
plt.tick_params(labelsize=12)
plt.savefig(dirname+f"design_obj_of_different_sample_steps_{args.sample_steps_list}.pdf")
print("save design obj of different sample steps at "+dirname+"design_obj_of_different_sample_steps.pdf")
np.save(dirname+f"design_obj_of_different_sample_steps_{args.sample_steps_list}.npy",design_obj_array)
plt.figure()
fontsize=18
plt.plot( sample_steps_list,MAE_list, marker='^', linestyle='-',color="purple",label="MAE")
plt.legend()
# plt.title(r'design obj of different $\lambda$')
plt.xlabel(r'sample_steps',fontsize=fontsize)
plt.ylabel(r'MAE',fontsize=fontsize)
plt.tick_params(labelsize=12)
plt.savefig(dirname+f'MAE_of_different_sample_steps_{args.sample_steps_list}.pdf')
np.save(dirname+f"MAE_of_different_sample_steps_{args.sample_steps_list}.npy",MAE_array)
print("save design obj of different sample steps at "+dirname+"MAE_of_different_sample_steps.pdf")

