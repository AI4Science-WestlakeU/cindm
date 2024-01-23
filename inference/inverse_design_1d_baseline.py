#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import argparse
from CinDM_anonymous.model.diffusion_1d import Unet1D, GaussianDiffusion1D, Trainer1D, num_to_groups,TemporalUnet1D,Unet1D_forward_model,linear_beta_schedule
from CinDM_anonymous.filepath import EXP_PATH
import matplotlib.pylab as plt
import matplotlib.backends.backend_pdf
from CinDM_anonymous.data.nbody_dataset import NBodyDataset
import numpy as np
import pdb
import torch
from torch_geometric.data.dataloader import DataLoader
from CinDM_anonymous.utils import p, get_item_1d, COLOR_LIST,CustomSampler,simulation,eval_simu,caculate_confidence_interval,cosine_beta_schedule
from torch.autograd import grad

# In[ ]:
import argparse
import CinDM_anonymous.GNS_model
import torch.nn as nn
import os
import math
import CinDM_anonymous.filepath as filepath
parser = argparse.ArgumentParser(description='Analyze the trained model')

parser.add_argument('--exp_id', default='inv_design', type=str, help='experiment folder id')
parser.add_argument('--date_time', default='2023-09-21_1d_baseline', type=str, help='date for the experiment folder')
parser.add_argument('--dataset', default='nbody-2', type=str, help='dataset to evaluate')

parser.add_argument('--model_type', default='temporal-unet1d', type=str, help='model type.')
parser.add_argument('--conditioned_steps', default=1, type=int, help='conditioned steps')
parser.add_argument('--rollout_steps', default=23, type=int, help='rollout steps')
parser.add_argument('--time_interval', default=4, type=int, help='time interval')
parser.add_argument('--attention', default=True, type=bool, help='whether to use attention block')
parser.add_argument('--loss_weight_discount', default=1, type=float,
                    help='multiplies t^th timestep of trajectory loss by discount**t')

parser.add_argument('--milestone', default=1, type=int, help='in which milestone model was saved')
parser.add_argument('--val_batch_size', default=50, type=int, help='batch size for validation')
parser.add_argument('--is_test', default=True, type=bool,help='flag for testing')
parser.add_argument('--sample_steps', default=250, type=int, help='sample steps')
parser.add_argument('--num_features', default=4, type=int,
                    help='in original datset,every data have 4 features,and processed datset just have 11 features ')

parser.add_argument('--noncollision_hold_probability', default=0.0, type=float,
                    help='probability of preserving non-collision trajectory data  ')

parser.add_argument('--distance_threshold', default=40.5, type=float,
                    help=' the distance threshold of two bodies collision')
parser.add_argument('--is_unconditioned', default=False, type=bool,
                    help=' is unconditioned or not')
parser.add_argument('--is_diffusion_condition', default=False, type=bool,
                    help=' whther do diffusion on conditioned steps or not')
parser.add_argument('--method_type', default="Unet_single_step", type=str,
                    help='the method to predict trajectory : 1. GNS_direct 2.Unet 3. Diffusion 4. GNS_autoregress 5. Unet_single_step ')
parser.add_argument('--checkpoint_path', default=filepath.current_wp+"/checkpoint_path/Unet_cond-1_rollout-1_bodies-2.pt", type=str,
                    help='the path to load checkpoint')

parser.add_argument('--dataset_path', default=filepath.current_wp+"/dataset/nbody_dataset", type=str,
                    help='the path to load dataset')

parser.add_argument('--n_composed', default=1, type=int,
                    help='rollout n_composed*rolloutss_steps steps trajectory')
parser.add_argument('--coef', default=1, type=float,
                    help='design_fn hyparameters')

parser.add_argument('--gamma', default=2, type=int,
                    help='design_fn hyparameters')
parser.add_argument('--max_design_steps', default=1, type=int,
                    help='the max design iteration steps')

parser.add_argument('--GNS_output_size', default=46, type=int,
                    help='the putput size of last layer of GNS')

parser.add_argument('--coef_max_noise', default=1, type=float,
                    help='the max value of noise coeffient')
parser.add_argument('--coef_grad', default=0.001, type=float,
                    help='the max value of grad coeffient')
parser.add_argument('--design_method', default="CEM", type=str,
                    help='the design method: 1. CEM 2. backprop')
parser.add_argument('--N', default=1000, type=int,
                    help='the initial number of samples of CEM')
parser.add_argument('--Ne', default=50, type=int,
                    help='the selected sample number  of CEM')

parser.add_argument('--is_batch_for_GNS', default=False, type=bool,
                    help='whether to batch different graph to a batch to speed up inference speed of GNS')
parser.add_argument('--L_bnd', default=False, type=bool,
                    help='whether to use L_bnd loss in backprop')
parser.add_argument('--initialization_mode', default=0, type=int,
                    help='in wgich mode to iniatialize cond: 0. random noise;1. data;2. data+random noise ')
parser.add_argument('--num_batchs', default=1, type=int,
                    help='number of batchs ')
parser.add_argument('--batch_size_list', default="[50]", type=str,
                    help='the list of different batch_size ')

args = parser.parse_args()

# def L_bnd(x):
    
#     l_bnd=torch.nn.ReLU()
mu_x=torch.tensor([0.,0.,0.,0.],requires_grad=True)
# def L_bnd(x):
#     L_bnd=torch.nn.ReLU(x-R_x)
class MyAbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.abs(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone().sign()
# Define objective:
# def get_eval_fn(pos_target, last_n_step, gamma=2):
#     def point_eval_objective(pos):
#         if pos.shape[-1]==8:
#             loss = (((pos[...,-last_n_step:,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) + \
#                     ((pos[...,-last_n_step:,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)).mean()/2
#         elif pos.shape[-1]==16:
#             loss = (((pos[...,-last_n_step:,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) + \
#                     ((pos[...,-last_n_step:,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)+\
#                     ((pos[...,-last_n_step:,8:10] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)+\
#                     ((pos[...,-last_n_step:,12:14] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) 
#                     ).mean()/4
#         elif pos.shape[-1]==32:
#             loss = (((pos[...,-last_n_step:,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) + \
#                     ((pos[...,-last_n_step:,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)+\
#                     ((pos[...,-last_n_step:,8:10] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)+\
#                     ((pos[...,-last_n_step:,12:14] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) +\
#                     ((pos[...,-last_n_step:,16:18] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) + \
#                     ((pos[...,-last_n_step:,20:22] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)+\
#                     ((pos[...,-last_n_step:,24:26] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)+\
#                     ((pos[...,-last_n_step:,28:30] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)
#                     ).mean()/8
#         return loss.item()
#     return point_eval_objective

def get_eval_fn(pos_target, last_n_step, gamma=2):
    """pos: [B, steps, F], pos_target: [F]"""
    assert len(pos_target.shape) == 1
    def point_eval_objective(pos):
        n_bodies = pos.shape[-1] // 4
        loss = torch.stack([(((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)).mean() for jj in range(n_bodies)]).mean()
        return loss.item()
    return point_eval_objective

def lastpoint_eval_objective(pos,pos_target,gamma=2):
        # loss = torch.mean((((pos[:,-1,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) + \
        #         ((pos[:,-1,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma))/2,dim=1)
        if pos.shape[-1]==8:
            loss = torch.mean(torch.cat([((pos[:,-1:,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) ,
                    ((pos[:,-1:,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)],dim=1),dim=1)
        elif pos.shape[-1]==16:
            loss = torch.mean(torch.cat([((pos[:,-1:,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) ,
                    ((pos[:,-1:,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,8:10] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,12:14] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)
                    ],dim=1),dim=1)
        elif pos.shape[-1]==32:
            loss = torch.mean(torch.cat([((pos[:,-1:,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) ,
                    ((pos[:,-1:,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,8:10] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,12:14] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,16:18] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,20:22] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,24:26] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma),
                    ((pos[:,-1:,28:30] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)
                    ],dim=1),dim=1)
        return loss
def get_eval_fn_loss_each(pos_target, last_n_step=1, gamma=2):
    """pos: [B, steps, F], pos_target: [F]"""
    assert len(pos_target.shape) == 1
    def point_eval_objective_std(pos):
        n_bodies = pos.shape[-1] // 4
        loss = torch.cat([(((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)) for jj in range(n_bodies)], -1).mean(-1)
        return loss
    return point_eval_objective_std
# def lastpoint_eval_objective(pos,pos_target,gamma=2):
#     last_n_step=1
#     n_bodies = pos.shape[-1] // 4
#     loss = torch.stack([(((pos[...,-last_n_step:,jj*4:jj*4+2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)).mean() for jj in range(n_bodies)]).mean()
#     return loss.item()
def get_obj(cond,model,model_method,design_fn):
    '''
    cond:[batch_size,num_steps,n_bodies*n_features] 
    model: function to output pred given condition
    model_method:str
    '''
    if model_method=="Unet":
        pred=model(cond)[:,1:,:]
        return design_fn(pred)
def get_cond(cond=None,initialization_mode=0):
    if initialization_mode==0:
        cond=torch.randn_like(cond,device=cond.device)
    elif initialization_mode==1:
        cond=cond
    elif initialization_mode==2:
        cond=cond+torch.randn_like(cond,device=cond.device)
    return cond        
     
def CEM_1d(cond,model,model_method,design_fn,max_design_steps,args,metadata=None):
    with torch.no_grad():
        is_GNS_data=False
        if model_method=="GNS_direct" or model_method=="GNS_autoregress":
            data_GNS=cond
            is_GNS_data=True
            poss_cond,vel, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
            cond=torch.cat([poss_cond,vel],dim=3)
        batch_size=cond.shape[0]
        device=cond.device
        # pdb.set_trace()
        mean=torch.randn_like(cond)#cond [batch_size,1,n_bodies*n_features]/[batch_size,n_bodies,1,n_features]
        mean=cond_clamp(mean,is_GNS_data=is_GNS_data)
        std=torch.clamp(torch.randn_like(cond),min=0)
        design_obj=0
        design_obj_list=[]
        for i in range(max_design_steps):
            p.print(f"test_start  {i}  design_obj {design_obj}", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
            # c_list=[]
            for j in range(args.N):
                #generate N samples
                c=torch.normal(mean=mean,std=std)
                c=cond_clamp(c,is_GNS_data=is_GNS_data)
                # c_list.append(c) #c_list [N,cond.shape]
                if j==0:
                    c_tensor=c
                else:
                    c_tensor=torch.cat([c_tensor,c],dim=0)
            # c_tensor=torch.stack(c_list) #[N,cond.shape]
            if model_method=="Unet":
                if args.n_composed==1:
                    pred=model(c_tensor)
                elif args.n_composed>1:
                    pred_step=torch.cat([c_tensor.clone()]*(args.rollout_steps*args.n_composed),dim=1)
                    pred=torch.cat([c_tensor.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                    # pdb.set_trace()
                    for i in range(args.n_composed):
                        if i==0:
                            pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=model(c_tensor)[:,1:]
                            pred[:,:args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                        else:
                            # pdb.set_trace()
                            pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=model(pred[:,(10*i-1):10*i])[:,1:]
                            # pred[:,23:33]=pred_step[]
                            pred[:,i*10:i*10+args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
            elif args.method_type=="Unet_single_step":
                pred=torch.cat([c_tensor]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                for i in range(args.rollout_steps+(args.n_composed-1)*10):
                    if i==0:
                        pred[:,i,:]=model(c_tensor)[:,-1,:]
                    else:
                        pred[:,i,:]=model(pred[:,i-1:i,:])[:,-1,:]
            elif args.method_type=="GNS_direct":
                poss_cond,vel, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
                # cond=torch.cat([poss_cond,vel],dim=3)
                y_gt=torch.cat([tgt_poss,tgt_vels],dim=3).reshape(tgt_poss.shape[0],tgt_poss.shape[2],-1)
                #c_tensor [batch_size*N,n_bodies,n_steps,n_features]
                pred_step=torch.cat([c_tensor.clone()]*(args.rollout_steps*args.n_composed),dim=2)
                pred_step=pred_step.permute(0,2,1,3)
                pred_step=pred_step.reshape(pred_step.shape[0],pred_step.shape[1],-1)
                pred=torch.cat([c_tensor.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=2)
                for i in range(args.n_composed):
                    if i==0:
                        # pdb.set_trace()
                        _,pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps],c_tensor=GNS_model.dyn_model.GNS_inference(data_GNS,c_tensor.to(device),model,metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        pred[:,:,i*10:i*10+args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                    else:
                        # pdb.set_trace()
                        _,pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps],pred[:,:,(10*i-1):10*i]=GNS_model.dyn_model.GNS_inference(data_GNS,pred[:,:,(10*i-1):10*i],model,metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                pred[:,:,:,2:4]=pred[:,:,:,2:4]*(60./4.) #[batch_size,n_bodies,n_steps,n_features]
                pred=pred.permute(0,2,1,3)
                pred=pred.reshape(pred.shape[0],pred.shape[1],-1)
            elif args.method_type=="GNS_autoregress":
                __,pred,_=GNS_model.dyn_model.GNS_inference(data_GNS,c_tensor,model,metadata,device,rollout_steps=args.rollout_steps+(args.n_composed-1)*10,is_batch_for_GNS=args.is_batch_for_GNS)
            loss_list=[]
            # pdb.set_trace()
            for k in range(args.N):
                loss=design_fn(pred[k*batch_size:(k+1)*batch_size])
                loss_list.append(loss)
            #sort loss
            # pdb.set_trace()
            indexed_list = list(enumerate(loss_list))
            sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
            original_indices = [x[0] for x in sorted_indexed_list]

            #get first Ne samples to update mean,std
            if is_GNS_data:
                c_tensor=c_tensor.reshape(-1,batch_size,c_tensor.shape[1],c_tensor.shape[2],c_tensor.shape[3])
            else:
                c_tensor=c_tensor.reshape(-1,batch_size,c_tensor.shape[1],c_tensor.shape[2])
            c_tensor=c_tensor[original_indices[:args.Ne]]
            mean=torch.mean(c_tensor,dim=0)
            mean=cond_clamp(mean,is_GNS_data=is_GNS_data)
            std=torch.std(c_tensor,dim=0)

            cond_design=torch.normal(mean=mean,std=std)
            cond_design=cond_clamp(cond_design,is_GNS_data=is_GNS_data)
            if model_method=="Unet":
                if args.n_composed==1:
                    pred_design=model(cond_design)[:,1:,:]
                elif args.n_composed>1:
                    pred_step_design=torch.cat([cond_design.clone()]*(args.rollout_steps*args.n_composed),dim=1)
                    pred_design=torch.cat([cond_design.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                    # pdb.set_trace()
                    for i in range(args.n_composed):
                        if i==0:
                            pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=model(cond_design)[:,1:]
                            pred_design[:,:args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                        else:
                            # pdb.set_trace()
                            pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=model(pred_design[:,(10*i-1):10*i])[:,1:]
                            # pred[:,23:33]=pred_step[]
                            pred_design[:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
            elif args.method_type=="Unet_single_step":
                pred_design=torch.cat([cond_design]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                for i in range(args.rollout_steps+(args.n_composed-1)*10):
                    if i==0:
                        pred_design[:,i,:]=model(cond_design)[:,-1,:]
                    else:
                        pred_design[:,i,:]=model(pred_design[:,i-1:i,:])[:,-1,:]
            elif args.method_type=="GNS_direct":##input cond_design,output pred_design
                pred_step_design=torch.cat([cond_design.clone()]*(args.rollout_steps*args.n_composed),dim=2)
                pred_step_design=pred_step_design.reshape(pred_step_design.shape[0],pred_step_design.shape[2],-1)
                pred_design=torch.cat([cond_design.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=2)
                for i in range(args.n_composed):
                    if i==0:
                        _,pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps],cond_design=GNS_model.dyn_model.GNS_inference(data_GNS,cond_design,model,metadata,device)
                        pred_design[:,:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                        # pred_design[:,:,:args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_step_design.shape[0],pred_design[:,:,:args.rollout_steps].shape[1],args.rollout_steps,-1)
                    else:
                        # pdb.set_trace()
                        _,pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps],pred_design[:,:,(10*i-1):10*i]=GNS_model.dyn_model.GNS_inference(data_GNS,pred_design[:,:,(10*i-1):10*i],model,metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        pred_design[:,:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                pred_design[:,:,:,2:4]=pred_design[:,:,:,2:4]*(60./4.) #[batch_size,n_bodies,n_steps,n_features]
                pred_design=pred_design.permute(0,2,1,3)
                pred_design=pred_design.reshape(pred_design.shape[0],pred_design.shape[1],-1)
            elif args.method_type=="GNS_autoregress":
                _,pred_design,_=GNS_model.dyn_model.GNS_inference(data_GNS,cond_design,model,metadata,device,rollout_steps=args.rollout_steps+(args.n_composed-1)*10,is_batch_for_GNS=args.is_batch_for_GNS)
                pred_design=pred_design.reshape(pred_design.shape[0],pred_design.shape[1],-1,4)
                pred_design[:,:,:,2:4]=pred_design[:,:,:,2:4]/(4./60.)
                pred_design=pred_design.reshape(pred_design.shape[0],pred_design.shape[1],-1)
            design_obj=design_fn(pred_design)
            design_obj_list.append(design_obj.to("cpu").detach().numpy())
        return cond_design,pred_design,design_obj_list
def cond_clamp(cond,is_GNS_data=False):
    c=cond.reshape(cond.shape[0],cond.shape[1],-1,4)
    if is_GNS_data:
        c[:,:,:,:2]=torch.clamp(c[:,:,:,:2],min=0.1,max=0.9)
        c[:,:,:,2:]=torch.clamp(c[:,:,:,2:],min=-0.5*(4./60.),max=0.5*(4./60.))
        return c
    else:
        c[:,:,:,:2]=torch.clamp(c[:,:,:,:2],min=0.1,max=0.9)
        c[:,:,:,2:]=torch.clamp(c[:,:,:,2:],min=-0.5,max=0.5)

        return c.reshape(cond.shape[0],cond.shape[1],-1)
# Define objective:
# def get_design_fn(pos_target, last_n_step, coef=1,gamma=2):
#     def point_objective(pos):
#         loss=0
#         for i in range(int(pos.shape[-1]/4)):
#             loss =loss+ ((pos[...,-last_n_step:,i*2:(2+i*2)] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)
#         loss=loss.mean()/(pos.shape[-1]/4)
#         # pdb.set_trace() 
#         return loss*coef
#     return point_objective

def get_design_fn(pos_target, last_n_step, coef=1,gamma=2):
    def point_objective(pos):
        loss = (((pos[...,-last_n_step:,0:2] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma) + \
                ((pos[...,-last_n_step:,4:6] - pos_target).abs() ** gamma).sum(-1) ** (1/gamma)).mean()/2
        
        return loss*coef
    return point_objective
def analyse(val_batch_size,loss_list):
    args.val_batch_size=val_batch_size
    print(args.__dict__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")
    R_x=torch.tensor([0.45,0.45,0.5,0.5,0.45,0.45,0.5,0.5],requires_grad=True,device=device)
    id = f"sample-{args.sample_steps}-1"
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
        is_jupyter = True
    except:
        is_jupyter = False

    def get_str_item(string, key):
        string_split = string.split("_")
        item = string_split[string_split.index(key)+1]
        return item
    if True:
        n_bodies = eval(args.dataset.split("-")[1])
        conditioned_steps =args.conditioned_steps
        rollout_steps = args.rollout_steps
        time_interval = args.time_interval
        save_step_load = args.milestone
        is_valid = True
        if save_step_load >= 0:
            # pdb.set_trace()
            ##load model
            if args.method_type=="Unet":
                model_results=torch.load(args.checkpoint_path)
                Unet=Unet1D_forward_model(
                    horizon=args.rollout_steps+args.conditioned_steps,### horizon Maybe match the time_steps
                    transition_dim=n_bodies*args.num_features, #this matches num_bodies*nun_feactures
                    cond_dim=False,
                    dim=64,
                    dim_mults=(1, 2, 4, 8),
                    attention=True,
                )
                Unet.load_state_dict(model_results["model"])
                Unet.to(device)
            if args.method_type=="Unet_single_step":
                model_results=torch.load(args.checkpoint_path)
                Unet=Unet1D_forward_model(
                    horizon=2,### horizon Maybe match the time_steps
                    transition_dim=n_bodies*args.num_features, #this matches num_bodies*nun_feactures
                    cond_dim=False,
                    dim=64,
                    dim_mults=(1, 2, 4, 8),
                    attention=True,
                )
                Unet.load_state_dict(model_results["model"])
                Unet.to(device)
            elif args.method_type=="GNS_direct":
                # model setting
                model_results=torch.load(args.checkpoint_path)
                gns_model=GNS_model.dyn_model.Net_cond_one(
                    output_size=args.GNS_output_size,
                    n_bodies=n_bodies
                )
                    #dataset
                test_dataset=GNS_model.Nbody_gns_dataset.nbody_gns_dataset_cond_one(
                    data_dir="/user/project/inverse_design/dataset/nbody_dataset/",
                    phase='test',
                    time_interval=4,
                    verbose=0,
                    output_steps=args.rollout_steps+(args.n_composed-1)*10,
                    n_bodies=n_bodies,
                    is_train=False,
                    device=device
                    )
                gns_model.load_state_dict(model_results["model"])
                gns_model=gns_model.to(device)
                gns_model.eval()
            elif args.method_type=="GNS_autoregress":
                # model setting
                model_results=torch.load(args.checkpoint_path)
                gns_model=GNS_model.dyn_model.Net_cond_one(
                    output_size=args.GNS_output_size,
                    n_bodies=n_bodies
                )
                #dataset
                # pdb.set_trace()
                test_dataset=GNS_model.Nbody_gns_dataset.nbody_gns_dataset_cond_one(
                    data_dir="/user/project/inverse_design/dataset/nbody_dataset/",
                    phase='test',
                    time_interval=4,
                    verbose=0,
                    output_steps=args.rollout_steps+(args.n_composed-1)*10,
                    n_bodies=n_bodies,
                    is_train=False,
                    device=device
                    )
                gns_model.load_state_dict(model_results["model"])
                gns_model=gns_model.to(device)
                gns_model.eval()
        ##load Datset
        if args.is_unconditioned:
            dataset = NBodyDataset(
                dataset=f"nbody-{n_bodies}",
                input_steps=4,
                output_steps=20,
                time_interval=time_interval,
                is_y_diff=False,
                is_train=not args.is_test,
                is_testdata=False,
                dataset_path=args.dataset_path
            )
        else:
            dataset = NBodyDataset(
                dataset=f"nbody-{n_bodies}",
                input_steps=args.conditioned_steps,
                output_steps=args.rollout_steps+(args.n_composed-1)*10,
                time_interval=time_interval,
                is_y_diff=False,
                is_train=not args.is_test,
                is_testdata=True,
                dataset_path=args.dataset_path
            )
        # s=CustomSampler(data=dataset,batch_size=args.val_batch_size,noncollision_hold_probability=args.noncollision_hold_probability,distance_threshold=args.distance_threshold)
        # dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6,sampler=s)
        dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6)
        # pdb.set_trace()


        for data in dataloader:
            break
        if args.conditioned_steps!=0:
            cond = get_item_1d(data, "x").to(device)
        else:
            cond=None
        y_gt = get_item_1d(data, "y")

        #inference
        coef=args.coef
        gamma=args.gamma
        if args.design_method=="backprop":
            coef_grad_schedule=cosine_beta_schedule(timesteps=args.max_design_steps+200)
            if args.method_type=="Unet":
                if args.n_composed==1:
                    cond=torch.tensor(cond,requires_grad=True)
                    cond=get_cond(cond,initialization_mode=args.initialization_mode)
                    cond=torch.tensor(cond,requires_grad=True)
                    # pdb.set_trace()
                    pred=Unet(cond)
                    
                    target = torch.tensor([0.5,0.5], device=device, dtype=float)
                    eval_fn = get_eval_fn(
                        torch.tensor([0.5,0.5], device=device, dtype=float),
                        last_n_step=1,
                        gamma=2,
                    )
                    design_fn = get_design_fn(target, last_n_step=1, coef=coef,gamma=gamma)
                    design_obj_list=[]
                    design_obj_simu_list=[]
                    pred_design=pred
                    if design_fn is not None:
                        with torch.enable_grad():
                            design_obj= design_fn(pred)
                            design_obj_list.append(design_obj.to("cpu").detach().numpy())
                            grad_design = grad(design_obj, cond)[0]
                            # pdb.set_trace()
                            if args.L_bnd:
                                print("Use L_bnd loss")
                                cond_abs=MyAbs.apply(cond).to(device)
                                L_bnd=torch.sum(torch.relu(cond_abs-R_x))
                                grad_L_bnd=grad(L_bnd,cond)[0]
                                cond_design = cond - (grad_design+grad_L_bnd)
                            else:
                                cond_design = cond - grad_design
                            num_design=1
                            coefficient_noise=linear_beta_schedule(args.max_design_steps)*args.coef_max_noise
                            # coef_grad_schedule=linear_beta_schedule(args.max_design_steps)
                            # pdb.set_trace()
                            for i in range(args.max_design_steps):
                                pred_design=pred_design.clone().detach()
                                pred_design=Unet(cond_design)
                                design_obj = design_fn(pred_design)
                                design_obj_list.append(design_obj.to("cpu").detach().numpy())
                                if design_obj<0.01:
                                    break
                                grad_design = grad(design_obj, cond_design)[0]
                                if args.L_bnd:
                                    cond_design_abs=MyAbs.apply(cond_design)
                                    L_bnd_design=torch.sum(torch.relu(cond_design_abs-R_x))
                                    grad_design_L_bnd=grad(L_bnd_design,cond_design)[0]
                                    if i>900:
                                        args.coef_grad=0
                                    cond_design = (cond_design - grad_design-grad_design_L_bnd*args.coef_grad*coef_grad_schedule[i-args.max_design_steps] + coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design))
                                else:
                                    cond_design = (cond_design - grad_design+ \
                                        coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design))
                                cond_design[:,:,:2]=torch.clamp(cond_design[:,:,:2],max=0.9,min=0.1)
                                cond_design[:,:,2:4]=torch.clamp(cond_design[:,:,2:4],max=0.5,min=-0.5)
                                cond_design[:,:,4:6]=torch.clamp(cond_design[:,:,4:6],max=0.9,min=0.1)
                                cond_design[:,:,6:8]=torch.clamp(cond_design[:,:,6:8],max=0.5,min=-0.5)
                                
                                # pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=design_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps)
                                # design_obj_simu_list.append(design_obj_simu.to("cpu").detach().numpy())
                                num_design=num_design+1
                            
                            pred_design=Unet(cond_design)
                            design_obj = design_fn(pred_design)
                            design_obj_list.append(design_obj.to("cpu").detach().numpy())
                            pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
                            design_obj_simu_list.append(design_obj_simu)
                            # pdb.set_trace()
                    pred=pred[:,1:,:]
                    pred_design=pred_design[:,1:,:]
                elif args.n_composed>=2:
                    cond=torch.tensor(cond,requires_grad=True)
                    cond=get_cond(cond,initialization_mode=args.initialization_mode)
                    pred_step=torch.cat([cond.clone()]*(args.rollout_steps*args.n_composed),dim=1)
                    pred=torch.cat([cond.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                    # pdb.set_trace()
                    for i in range(args.n_composed):
                        if i==0:
                            pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=Unet(cond)[:,1:]
                            pred[:,:args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                        else:
                            # pdb.set_trace()
                            pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=Unet(pred[:,(10*i-1):10*i])[:,1:]
                            # pred[:,23:33]=pred_step[]
                            pred[:,i*10:i*10+args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                    
                    # pred_first_init=Unet(cond)
                    # pred_second_init=Unet(pred_first_init[:,-1:,:])
                    
                    target = torch.tensor([0.5,0.5], device=device, dtype=float)
                    eval_fn = get_eval_fn(
                        torch.tensor([0.5,0.5], device=device, dtype=float),
                        last_n_step=1,
                        gamma=2,
                    )
                    design_fn = get_design_fn(target, last_n_step=1, coef=coef,gamma=gamma)

                    design_obj_list=[]
                    design_obj_simu_list=[]

                    if design_fn is not None:
                        with torch.enable_grad():
                            design_obj= design_fn(pred)
                            design_obj_list.append(design_obj.to("cpu").detach().numpy())
                            grad_design = grad(design_obj, cond)[0]

                            # cond_design = cond - grad_design
                            if args.L_bnd:
                                print("Use L_bnd loss")
                                cond_abs=MyAbs.apply(cond).to(device)
                                L_bnd=torch.sum(torch.relu(cond_abs-R_x))
                                grad_L_bnd=grad(L_bnd,cond)[0]
                                cond_design = cond - (grad_design+grad_L_bnd)
                            else:
                                cond_design = cond - grad_design

                            num_design=1
                            coefficient_noise=linear_beta_schedule(args.max_design_steps)*args.coef_max_noise
                            pred_step_design=torch.cat([cond.clone()]*(args.rollout_steps*args.n_composed),dim=1)
                            pred_design=torch.cat([cond.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                            for j in range(args.max_design_steps):
                                pred_design=pred_design.clone().detach()
                                pred_step_design=pred_step_design.clone().detach()
                                cond_design=torch.tensor(cond_design.clone().detach(),requires_grad=True)
                                for i in range(args.n_composed):
                                    if i==0:
                                        pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=Unet(cond_design)[:,1:]
                                        pred_design[:,:args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                                    else:
                                        pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=Unet(pred_design[:,(10*i-1):10*i])[:,1:]
                                        pred_design[:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                                pred_design[:,-13:]=pred_step_design[:,-13:]
                                design_obj = design_fn(pred_design)
                                design_obj_list.append(design_obj.to("cpu").detach().numpy())
                                if design_obj<0.01:
                                    break
                                grad_design = grad(design_obj, cond_design)[0]
                                # cond_design = cond_design - grad_design+coefficient_noise[j-args.max_design_steps]*torch.randn_like(grad_design)
                                if args.L_bnd:
                                    cond_design_abs=MyAbs.apply(cond_design)
                                    L_bnd_design=torch.sum(torch.relu(cond_design_abs-R_x))
                                    grad_design_L_bnd=grad(L_bnd_design,cond_design)[0]
                                    if i>900:
                                        args.coef_grad=0
                                    cond_design = (cond_design - grad_design-grad_design_L_bnd*args.coef_grad*coef_grad_schedule[i-args.max_design_steps] + coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design))
                                else:
                                    cond_design = cond_design - grad_design+coefficient_noise[j-args.max_design_steps]*torch.randn_like(grad_design)
                                cond_design[:,:,:2]=torch.clamp(cond_design[:,:,:2],max=0.9,min=0.1)
                                cond_design[:,:,2:4]=torch.clamp(cond_design[:,:,2:4],max=0.5,min=-0.5)
                                cond_design[:,:,4:6]=torch.clamp(cond_design[:,:,4:6],max=0.9,min=0.1)
                                cond_design[:,:,6:8]=torch.clamp(cond_design[:,:,6:8],max=0.5,min=-0.5)
                                
                                # pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=design_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps)
                                # design_obj_simu_list.append(design_obj_simu.to("cpu").detach().numpy())
                                num_design=num_design+1
                            # pdb.set_trace()
                            for i in range(args.n_composed):
                                if i==0:
                                    pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=Unet(cond_design)[:,1:]
                                    pred_design[:,:args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                                else:
                                    pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]=Unet(pred_design[:,(10*i-1):(10*i)])[:,1:]
                                    pred_design[:,i*10:(i*10+args.rollout_steps)]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps]
                            
                            design_obj = design_fn(pred_design)
                            design_obj_list.append(design_obj.to("cpu").detach().numpy())
                            pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
                            design_obj_simu_list.append(design_obj_simu)
                    # pdb.set_trace()
            elif args.method_type=="Unet_single_step":
                cond=torch.tensor(cond,requires_grad=True)
                cond=get_cond(cond,initialization_mode=args.initialization_mode)
                pred=torch.cat([cond]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                for i in range(args.rollout_steps+(args.n_composed-1)*10):
                    if i==0:
                        pred_step=Unet(cond)[:,-1:,:]
                        pred[:,i,:]=pred_step[:,0,:]
                    else:
                        pred_step=Unet(pred[:,i-1:i,:])[:,-1:,:]
                        pred[:,i,:]=pred_step[:,0,:]
                # pdb.set_trace()
                target = torch.tensor([0.5,0.5], device=device, dtype=float)
                eval_fn = get_eval_fn(
                    torch.tensor([0.5,0.5], device=device, dtype=float),
                    last_n_step=1,
                    gamma=2,
                )
                design_fn = get_design_fn(target, last_n_step=1, coef=coef,gamma=gamma)
                design_obj_list=[]
                design_obj_simu_list=[]
                # pdb.set_trace()
                if design_fn is not None:
                    with torch.enable_grad():
                        design_obj= design_fn(pred)
                        design_obj_list.append(design_obj.to("cpu").detach().numpy())
                        grad_design = grad(design_obj, cond)[0]
                        if args.L_bnd:
                            cond_abs=MyAbs.apply(cond).to(device)
                            L_bnd=torch.sum(torch.relu(cond_abs-R_x))
                            grad_L_bnd=grad(L_bnd,cond)[0]
                            cond_design = cond - (grad_design+grad_L_bnd)
                        else:
                            cond_design = cond - grad_design
                        num_design=1
                        # pdb.set_trace()
                        coefficient_noise=linear_beta_schedule(args.max_design_steps)*args.coef_max_noise
                        pred_design=torch.cat([cond]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                        for i in range(args.max_design_steps):
                            # pdb.set_trace()
                            pred_design=pred_design.clone().detach()
                            cond_design=torch.tensor(cond_design.clone().detach(),requires_grad=True)
                            for j in range(args.rollout_steps+(args.n_composed-1)*10):
                                if j==0:
                                    pred_design[:,j:j+1,:]=Unet(cond_design)[:,-1:,:]
                                else:
                                    pred_design[:,j:j+1,:]=Unet(pred_design[:,j-1:j,:])[:,-1:,:]
                            design_obj = design_fn(pred_design)
                            design_obj_list.append(design_obj.to("cpu").detach().numpy())
                            if design_obj<0.01:
                                break
                            # p.print(f"test_start  {i}  design_obj {design_obj}", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
                            
                            # print("pred_design store address",pred_design.data_ptr())
                            # pdb.set_trace()
                            # design_obj.backward()
                            # grad_design=cond_design.grad
                            grad_design = grad(design_obj, cond_design)[0]
                            if args.L_bnd:
                                cond_design_abs=MyAbs.apply(cond_design)
                                L_bnd_design=torch.sum(torch.relu(cond_design_abs-R_x))
                                grad_design_L_bnd=grad(L_bnd_design,cond_design)[0]
                                if i>900:
                                    args.coef_grad=0
                                cond_design = (cond_design - grad_design-grad_design_L_bnd*args.coef_grad*coef_grad_schedule[i-args.max_design_steps] + \
                                    coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design))
                            else:
                                cond_design =(cond_design- ( grad_design+coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design)))
                            # print("cond_design store address",cond_design.data_ptr())
                            # p.print(f"test_end!!!!!!!!!!!!!!!  pred_design  {pred_design.data_ptr()}  cond_design {cond_design.data_ptr()}", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
                            # cond_design =(cond_design- ( grad_design+coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design)))
                            
                            # print("cond_design store address",cond_design.data_ptr())
                            # torch.cuda.empty_cache() 

                            cond_design[:,:,:2]=torch.clamp(cond_design[:,:,:2],max=0.9,min=0.1)
                            cond_design[:,:,2:4]=torch.clamp(cond_design[:,:,2:4],max=0.5,min=-0.5)
                            cond_design[:,:,4:6]=torch.clamp(cond_design[:,:,4:6],max=0.9,min=0.1)
                            cond_design[:,:,6:8]=torch.clamp(cond_design[:,:,6:8],max=0.5,min=-0.5)
                            
                            # pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=design_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps)
                            # design_obj_simu_list.append(design_obj_simu.to("cpu").detach().numpy())
                            num_design=num_design+1
                        
                        for j in range(args.rollout_steps+(args.n_composed-1)*10):
                            if j==0:
                                pred_design[:,j:j+1,:]=Unet(cond_design)[:,-1:,:]
                            else:
                                pred_design[:,j:j+1,:]=Unet(pred_design[:,j-1:j,:])[:,-1:,:]
                        design_obj = design_fn(pred_design)
                        design_obj_list.append(design_obj.to("cpu").detach().numpy())
                        pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
                        design_obj_simu_list.append(design_obj_simu)
                # pdb.set_trace()
                # pred_design=pred_design[:,1:,:]
            elif args.method_type=="GNS_direct":##To do 注意GNS——direct和Unet compose的时候索引问题
                dataloader_GNS=DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6)
                for data_GNS in dataloader_GNS:
                    break
                poss_cond,vel, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
                cond=torch.cat([poss_cond,vel],dim=3)
                y_gt=torch.cat([tgt_poss,tgt_vels],dim=3).reshape(tgt_poss.shape[0],tgt_poss.shape[2],-1)
                cond=torch.tensor(cond,requires_grad=True).to(device)
                cond=get_cond(cond,initialization_mode=args.initialization_mode)
                
                pred_step=torch.cat([cond.clone()]*(args.rollout_steps*args.n_composed),dim=2)
                pred_step=pred_step.reshape(pred_step.shape[0],pred_step.shape[2],-1)
                pred=torch.cat([cond.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=2)
                for i in range(args.n_composed):
                    if i==0:
                        _,pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps],_=GNS_model.dyn_model.GNS_inference(data_GNS,cond,gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        pred[:,:,i*10:i*10+args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                    else:
                        # pdb.set_trace()
                        _,pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps],pred[:,:,(10*i-1):10*i]=GNS_model.dyn_model.GNS_inference(data_GNS,pred[:,:,(10*i-1):10*i],gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        # pred[:,23:33]=pred_step[]
                        #pred_design_step [batch_size,num_steps,n_bodies*n_bn_features]
                        # temp=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred.shape[0],args.rollout_steps,-1,4)
                        # temp=temp.permute(0,2,1,3)
                        # pred[:,:,i*10:i*10+args.rollout_steps]=temp
                        # del temp
                        pred[:,:,i*10:i*10+args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)

                pred[:,:,:,2:4]=pred[:,:,:,2:4]*(60./4.)
                target = torch.tensor([0.5,0.5], device=device, dtype=float)
                eval_fn = get_eval_fn(
                        torch.tensor([0.5,0.5], device=device, dtype=float),
                        last_n_step=1,
                        gamma=2,
                    )
                design_fn = get_design_fn(target, last_n_step=1, coef=args.coef,gamma=gamma)
                design_obj_list=[]
                if design_fn is not None:
                    with torch.enable_grad():
                        design_obj = design_fn(pred.reshape(pred.shape[0],pred.shape[2],-1))
                        design_obj_list.append(design_obj.to("cpu").detach().numpy())
                        grad_design = grad(design_obj, cond)[0]
                    if args.L_bnd:
                        cond_abs=MyAbs.apply(cond.reshape(50,8)).to(device)
                        L_bnd=torch.sum(torch.relu(cond_abs-R_x))
                        grad_L_bnd=grad(L_bnd,cond)[0]
                        cond_design = cond - (grad_design+grad_L_bnd)
                    else:
                        cond_design = cond - grad_design*args.coef_grad
                    # pdb.set_trace()
                    num_design=1
                    coefficient_noise=linear_beta_schedule(args.max_design_steps)*args.coef_max_noise
                    pred_step_design=torch.cat([cond.clone()]*(args.rollout_steps*args.n_composed),dim=2)
                    pred_step_design=pred_step_design.reshape(pred_step_design.shape[0],pred_step_design.shape[2],-1)
                    pred_design=torch.cat([cond.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=2)
                    for j in range(args.max_design_steps):
                        # coef_grad=args.coef_grad*(0.95**(j/10))
                        coef_grad=args.coef_grad
                        # y_gt,pred_design,cond_design=GNS_model.dyn_model.GNS_inference(data_GNS,cond_design,gns_model,test_dataset.metadata,device)
                        pred_design=pred_design.clone().detach()
                        pred_step_design=pred_step_design.clone().detach()
                        cond_design=torch.tensor(cond_design.clone().detach(),requires_grad=True)
                        for i in range(args.n_composed):
                            if i==0:
                                _,pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps],_=GNS_model.dyn_model.GNS_inference(data_GNS,cond_design,gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                                pred_design[:,:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                                # pred_design[:,:,:args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_step_design.shape[0],pred_design[:,:,:args.rollout_steps].shape[1],args.rollout_steps,-1)
                            else:
                                # pdb.set_trace()
                                _,pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps],pred_design[:,:,(10*i-1):10*i]=GNS_model.dyn_model.GNS_inference(data_GNS,pred_design[:,:,(10*i-1):10*i],gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                                pred_design[:,:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                                # temp=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4)
                                # temp=temp.permute(0,2,1,3)
                                # pred_design[:,:,i*10:i*10+args.rollout_steps]=temp
                        design_obj= design_fn(pred_design.reshape(pred_design.shape[0],pred_design.shape[2],-1))
                        design_obj_list.append(design_obj.to("cpu").detach().numpy())
                        if design_obj<0.05:
                            break
                        # p.print(f" design_step {j} test_start  design_obj {design_obj} pred_design  {pred_design.data_ptr()}  cond_design {cond_design.data_ptr()} ", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
                        grad_design = grad(design_obj, cond_design)[0]
                        if args.L_bnd:
                            cond_design_abs=MyAbs.apply(cond_design.reshape(50,8)).to(device)
                            L_bnd_design=torch.sum(torch.relu(cond_design_abs-R_x))
                            grad_design_L_bnd=grad(L_bnd_design,cond_design)[0]
                            if i>900:
                                args.coef_grad=0
                            cond_design = (cond_design - grad_design*0.001-grad_design_L_bnd*args.coef_grad*coef_grad_schedule[i-args.max_design_steps] + \
                                coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design))
                        else:
                            cond_design= cond_design - grad_design*args.coef_grad+coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design)
                        
                        torch.cuda.empty_cache()
                        for k in range(cond_design.shape[1]):
                            cond_design[:,k,-1,:2]=torch.clamp(cond_design[:,k,-1,:2],max=0.9,min=0.1)
                            cond_design[:,k,-1,2:4]=torch.clamp(cond_design[:,k,-1,2:4],max=0.5*(4./60.),min=-0.5*(4./60.))
                        num_design=num_design+1
                # pdb.set_trace()
                for i in range(args.n_composed):
                    if i==0:
                        _,pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps],cond_design=GNS_model.dyn_model.GNS_inference(data_GNS,cond_design,gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        pred_design[:,:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                        # pred_design[:,:,:args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_step_design.shape[0],pred_design[:,:,:args.rollout_steps].shape[1],args.rollout_steps,-1)
                    else:
                        # pdb.set_trace()
                        _,pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps],pred_design[:,:,(10*i-1):10*i]=GNS_model.dyn_model.GNS_inference(data_GNS,pred_design[:,:,(10*i-1):10*i],gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        pred_design[:,:,i*10:i*10+args.rollout_steps]=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                        # temp=pred_step_design[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred_design.shape[0],args.rollout_steps,-1,4)
                        # temp=temp.permute(0,2,1,3)
                        # pred_design[:,:,i*10:i*10+args.rollout_steps]=temp
                
                pred_design[:,:,:,2:4]=pred_design[:,:,:,2:4]*(60./4.) #[batch_size,n_bodies,n_steps,n_features]

                pred_design=pred_design.permute(0,2,1,3)
                pred_design=pred_design.reshape(pred_design.shape[0],pred_design.shape[1],-1)
                design_obj = design_fn(pred_design)
                design_obj_list.append(design_obj.to("cpu").detach().numpy())

                pred=pred.reshape(pred.shape[0],pred.shape[2],-1)
                cond=cond.reshape(cond.shape[0],1,-1)
                # pdb.set_trace()



                cond_design[:,:,:,2:4]=cond_design[:,:,:,2:4]*(60./4.)
                cond_design=cond_design.permute(0,2,1,3).reshape(cond_design.shape[0],1,-1)
                pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
            elif args.method_type=="GNS_autoregress":
                # pdb.set_trace()
                dataloader_GNS=DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6)
                for data_GNS in dataloader_GNS:
                    break
                poss_cond,vel, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
                #tgt_poss [batch_size,n_bodies,n_steps,n_features]
                cond=torch.cat([poss_cond,vel],dim=3)
                cond=torch.tensor(cond,requires_grad=True).to(device)
                cond=get_cond(cond,initialization_mode=args.initialization_mode)
                # y_gt=torch.cat([tgt_poss,tgt_vels*(60./4.)],dim=3).permute(0,2,1,3).reshape(tgt_poss.shape[0],tgt_poss.shape[2],-1)
                y_gt,pred,_=GNS_model.dyn_model.GNS_inference(data_GNS,cond,gns_model,test_dataset.metadata,device,rollout_steps=args.rollout_steps+(args.n_composed-1)*10,is_batch_for_GNS=args.is_batch_for_GNS)
                target = torch.tensor([0.5,0.5], device=device, dtype=float)
                eval_fn = get_eval_fn(
                        torch.tensor([0.5,0.5], device=device, dtype=float),
                        last_n_step=1,
                        gamma=2,
                    )
                design_fn = get_design_fn(target, last_n_step=1, coef=args.coef,gamma=gamma)
                design_obj_list=[]
                if design_fn is not None:
                    with torch.enable_grad():
                        design_obj = design_fn(pred)
                        design_obj_list.append(design_obj.to("cpu").detach().numpy())
                        grad_design = grad(design_obj, cond)[0]
                    if args.L_bnd:
                        cond_abs=MyAbs.apply(cond.reshape(50,8)).to(device)
                        L_bnd=torch.sum(torch.relu(cond_abs-R_x))
                        grad_L_bnd=grad(L_bnd,cond)[0]
                        cond_design = cond - (grad_design+grad_L_bnd)
                    else:
                        cond_design = cond - grad_design*args.coef_grad
                    
                    for k in range(cond_design.shape[1]):
                        cond_design[:,k,-1,:2]=torch.clamp(cond_design[:,k,-1,:2],max=0.9,min=0.1)
                        cond_design[:,k,-1,2:4]=torch.clamp(cond_design[:,k,-1,2:4],max=0.5*(4./60.),min=-0.5*(4./60.))
                    # pdb.set_trace()
                    num_design=1
                    coefficient_noise=linear_beta_schedule(args.max_design_steps)*args.coef_max_noise
                    # pdb.set_trace()
                    pred_design=pred
                    for i in range(args.max_design_steps):
                        coef_grad=args.coef_grad*(0.95**(i/10.))
                        # coef_grad=args.coef_grad
                        pred_design=pred_design.clone().detach()
                        cond_design=torch.tensor(cond_design.clone().detach(),requires_grad=True)
                        _,pred_design,_=GNS_model.dyn_model.GNS_inference(data_GNS,cond_design,gns_model,test_dataset.metadata,device,rollout_steps=args.rollout_steps+(args.n_composed-1)*10,is_batch_for_GNS=args.is_batch_for_GNS)
                        design_obj= design_fn(pred_design)
                        design_obj_list.append(design_obj.to("cpu").detach().numpy())
                        if design_obj<0.05:
                            break
                        # p.print(f" design_step {i} test_start  {i}  design_obj {design_obj} pred_design  {pred_design.data_ptr()}  cond_design {cond_design.data_ptr()} ", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
                        grad_design = grad(design_obj, cond_design)[0]
                        if args.L_bnd:
                            cond_design_abs=MyAbs.apply(cond_design.reshape(50,8)).to(device)
                            L_bnd_design=torch.sum(torch.relu(cond_design_abs-R_x))
                            grad_design_L_bnd=grad(L_bnd_design,cond_design)[0]
                            if i>900:
                                args.coef_grad=0
                            cond_design = (cond_design - grad_design*0.001-grad_design_L_bnd*args.coef_grad*coef_grad_schedule[i-args.max_design_steps] + \
                                coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design))
                        else:
                            cond_design= cond_design - grad_design*coef_grad+coefficient_noise[i-args.max_design_steps]*torch.randn_like(grad_design)
                        
                        torch.cuda.empty_cache()
                        
                        for k in range(cond_design.shape[1]):
                            cond_design[:,k,-1,:2]=torch.clamp(cond_design[:,k,-1,:2],max=0.9,min=0.1)
                            cond_design[:,k,-1,2:4]=torch.clamp(cond_design[:,k,-1,2:4],max=0.5*(4./60.),min=-0.5*(4./60.))
                        num_design=num_design+1
                _,pred_design,_=GNS_model.dyn_model.GNS_inference(data_GNS,cond_design,gns_model,test_dataset.metadata,device,rollout_steps=args.rollout_steps+(args.n_composed-1)*10,is_batch_for_GNS=args.is_batch_for_GNS)
                design_obj = design_fn(pred_design)
                design_obj_list.append(design_obj.to("cpu").detach().numpy())

                pred_design=pred_design.reshape(pred_design.shape[0],pred_design.shape[1],n_bodies,-1)
                pred_design[:,:,:,2:4]=pred_design[:,:,:,2:4]/(4./60.)
                pred_design=pred_design.reshape(pred_design.shape[0],pred_design.shape[1],-1)

                cond_design[:,:,:,2:4]=cond_design[:,:,:,2:4]/(4./60.)
                cond[:,:,:,2:4]=cond[:,:,:,2:4]/(4./60.)
                cond=cond.permute(0,2,1,3).reshape(cond.shape[0],1,-1)
                cond_design=cond_design.permute(0,2,1,3).reshape(cond_design.shape[0],1,-1)
                pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
        elif args.design_method=="CEM":
            num_design=args.max_design_steps
            target = torch.tensor([0.5,0.5], device=device, dtype=float)
            eval_fn = get_eval_fn(
                    torch.tensor([0.5,0.5], device=device, dtype=float),
                    last_n_step=1,
                    gamma=2,
                )
            design_fn = get_design_fn(target, last_n_step=1, coef=coef,gamma=gamma)
                # design_obj_list=[]
            design_obj_simu_list=[]
            cond=get_cond(cond,initialization_mode=args.initialization_mode)
            if args.method_type=="Unet":
                pred=Unet(cond)[:,1:]
                target = torch.tensor([0.5,0.5], device=device, dtype=float)
                eval_fn = get_eval_fn(
                    torch.tensor([0.5,0.5], device=device, dtype=float),
                    last_n_step=1,
                    gamma=2,
                )
                design_fn = get_design_fn(target, last_n_step=1, coef=coef,gamma=gamma)
                # design_obj_list=[]
                design_obj_simu_list=[]
                cond_design,pred_design,design_obj_list=CEM_1d(cond,Unet,args.method_type,design_fn,args.max_design_steps,args=args)
                # cond_design,pred_design=CEM_1d(cond,Unet,args.method_type,design_fn,1,args.N,args.Ne)
                design_obj=design_fn(pred_design)
                design_obj_list.append(design_obj.to("cpu").detach().numpy())
                pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
                design_obj_simu_list.append(design_obj_simu)
            elif args.method_type=="Unet_single_step":
                pred=torch.cat([cond]*(args.rollout_steps+(args.n_composed-1)*10),dim=1)
                for i in range(args.rollout_steps+(args.n_composed-1)*10):
                    if i==0:
                        pred_step=Unet(cond)[:,-1:,:]
                        pred[:,i,:]=pred_step[:,0,:]
                    else:
                        pred_step=Unet(pred[:,i-1:i,:])[:,-1:,:]
                        pred[:,i,:]=pred_step[:,0,:]
                cond_design,pred_design,design_obj_list=CEM_1d(cond,Unet,args.method_type,design_fn,args.max_design_steps,args=args)
                # cond_design,pred_design=CEM_1d(cond,Unet,args.method_type,design_fn,1,args.N,args.Ne)
                design_obj=design_fn(pred_design)
                design_obj_list.append(design_obj.to("cpu").detach().numpy())
                pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
                design_obj_simu_list.append(design_obj_simu)
            elif args.method_type=="GNS_direct":
                dataloader_GNS=DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6)
                for data_GNS in dataloader_GNS:
                    break
                poss_cond,vel, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
                cond=torch.cat([poss_cond,vel],dim=3)
                y_gt=torch.cat([tgt_poss,tgt_vels],dim=3).reshape(tgt_poss.shape[0],tgt_poss.shape[2],-1)
                cond=torch.tensor(cond,requires_grad=True).to(device)
                cond=get_cond(cond,initialization_mode=args.initialization_mode)
                
                pred_step=torch.cat([cond.clone()]*(args.rollout_steps*args.n_composed),dim=2)
                pred_step=pred_step.reshape(pred_step.shape[0],pred_step.shape[2],-1)
                pred=torch.cat([cond.clone()]*(args.rollout_steps+(args.n_composed-1)*10),dim=2)
                for i in range(args.n_composed):
                    if i==0:
                        _,pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps],cond=GNS_model.dyn_model.GNS_inference(data_GNS,cond,gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        pred[:,:,i*10:i*10+args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)
                    else:
                        # pdb.set_trace()
                        _,pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps],pred[:,:,(10*i-1):10*i]=GNS_model.dyn_model.GNS_inference(data_GNS,pred[:,:,(10*i-1):10*i],gns_model,test_dataset.metadata,device,is_batch_for_GNS=args.is_batch_for_GNS)
                        # pred[:,23:33]=pred_step[]
                        #pred_design_step [batch_size,num_steps,n_bodies*n_bn_features]
                        # temp=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred.shape[0],args.rollout_steps,-1,4)
                        # temp=temp.permute(0,2,1,3)
                        # pred[:,:,i*10:i*10+args.rollout_steps]=temp
                        # del temp
                        pred[:,:,i*10:i*10+args.rollout_steps]=pred_step[:,i*args.rollout_steps:(i+1)*args.rollout_steps].reshape(pred.shape[0],args.rollout_steps,-1,4).permute(0,2,1,3)

                pred[:,:,:,2:4]=pred[:,:,:,2:4]*(60./4.) #[batch_size,n_bodies,n_steps,n_features]
                pred=pred.permute(0,2,1,3)
                pred=pred.reshape(pred.shape[0],pred.shape[1],-1)
                # pdb.set_trace()
                data_GNS[0]=data_GNS[0].to(device)
                data_GNS[1]=data_GNS[1].to(device)
                cond_design,pred_design,design_obj_list=CEM_1d(data_GNS,gns_model,args.method_type,design_fn,args.max_design_steps,args=args,metadata=test_dataset.metadata)
                # cond_design,pred_design=CEM_1d(cond,Unet,args.method_type,design_fn,1,args.N,args.Ne)
                design_obj=design_fn(pred_design)
                cond_design[:,:,:,2:4]=cond_design[:,:,:,2:4]/(4./60.) #[n_batch_size,n_bodies,n_step,n_features]
                cond_design=cond_design.permute(0,2,1,3).reshape(cond_design.shape[0],1,-1)
                design_obj_list.append(design_obj.to("cpu").detach().numpy())
                pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
                design_obj_simu_list.append(design_obj_simu)
            elif args.method_type=="GNS_autoregress":
                dataloader_GNS=DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6)
                for data_GNS in dataloader_GNS:
                    break
                poss_cond,vel, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
                #tgt_poss [batch_size,n_bodies,n_steps,n_features]
                cond=torch.cat([poss_cond,vel],dim=3)
                cond=torch.tensor(cond,requires_grad=True).to(device)
                cond=get_cond(cond,initialization_mode=args.initialization_mode)
                # y_gt=torch.cat([tgt_poss,tgt_vels*(60./4.)],dim=3).permute(0,2,1,3).reshape(tgt_poss.shape[0],tgt_poss.shape[2],-1)
                y_gt,pred,_=GNS_model.dyn_model.GNS_inference(data_GNS,cond,gns_model,test_dataset.metadata,device,rollout_steps=args.rollout_steps+(args.n_composed-1)*10,is_batch_for_GNS=args.is_batch_for_GNS)
                data_GNS[0]=data_GNS[0].to(device)
                data_GNS[1]=data_GNS[1].to(device)
                cond_design,pred_design,design_obj_list=CEM_1d(data_GNS,gns_model,args.method_type,design_fn,args.max_design_steps,args=args,metadata=test_dataset.metadata)


                cond_design[:,:,:,2:4]=cond_design[:,:,:,2:4]/(4./60.)
                cond[:,:,:,2:4]=cond[:,:,:,2:4]/(4./60.)
                cond=cond.permute(0,2,1,3).reshape(cond.shape[0],1,-1)
                cond_design=cond_design.permute(0,2,1,3).reshape(cond_design.shape[0],1,-1)
                pred_simu,design_obj_simu=eval_simu(cond_design=cond_design,design_fn=eval_fn,n_bodies=n_bodies,rollout_steps=args.rollout_steps+(args.n_composed-1)*10)
                design_obj=design_fn(pred_design)


        print("last design_obj",design_obj)
        print("simu_eval_obj",design_obj_simu)
        plt.figure()
        plt.plot(design_obj_list)

        plt.xlabel('design_iter_steps')
        plt.ylabel('design_obj')
        plt.title('design_obj_list')
        plt.grid(True)  
        plt.legend()
        Result_path=EXP_PATH + f"{args.exp_id}_{args.date_time}/" + f"{args.design_method}_cond-{args.conditioned_steps}_rollout-{args.rollout_steps+(args.n_composed-1)*10}_{args.method_type}_design_steps-{num_design-1}_coef-{args.coef}_gamma-{args.gamma}_coef_grad-{args.coef_grad}max-coef-noise-{args.coef_max_noise}_N-{args.N}_Ne-{args.Ne}_initialization_mode-{args.initialization_mode}"
        os.makedirs(Result_path, exist_ok=True)     
        plt.savefig(Result_path+f'/design_obj_list.png')
        
        cond_design=cond_design.to("cpu")
        pred=pred.to("cpu")
        pred_designed=pred_design.to("cpu")
        pred_simu=pred_simu.to("cpu")
        # pdb.set_trace()
        # y_gt_split = y_gt.view(y_gt.shape[0],y_gt.shape[1],n_bodies,args.num_features)
        # pred_split = pred.view(pred.shape[0],pred.shape[1],n_bodies,args.num_features)
        y_gt_split = pred_simu.view(pred_simu.shape[0],pred_simu.shape[1],n_bodies,args.num_features)
        pred_split = pred_designed.view(pred_designed.shape[0],pred_designed.shape[1],n_bodies,args.num_features)
        loss_mean = (pred_split[:,:,:,0:4].to("cpu") - y_gt_split[:,:,:,0:4].to("cpu") ).abs().mean().item() #when testing,we just analyze the accuracy of x,y,vx,y
        loss_list.append(loss_mean)

        if args.is_unconditioned: # reset these parameters for better results visulization
            args.conditioned_steps=4
            args.rollout_steps=20
        

        #save result as np

        np.save(Result_path+"/cond.npy",cond.to('cpu').detach().numpy())
        np.save(Result_path+"/GD_rollout.npy",y_gt.to('cpu').detach().numpy())
        np.save(Result_path+"/cond_design.npy",cond_design.to('cpu').detach().numpy())
        np.save(Result_path+"/pred_design.npy",pred_designed.to('cpu').detach().numpy())
        np.save(Result_path+"/pred_simu.npy",pred_simu.to('cpu').detach().numpy())
        
        #save results as txt
        # pdb.set_trace()
        print("Results analysis")
        MAE_mean,MAE_std,margin_of_error_MAEMAE,_=caculate_confidence_interval(torch.abs(pred_designed-pred_simu))
        obj_pred_simu=lastpoint_eval_objective(pred_simu.to(device),target,gamma=2)
        design_obj_mean,design_obj_std,margin_of_error_design_obj,_=caculate_confidence_interval(obj_pred_simu)
        txt_name =Result_path+f"/Results.txt"
        variables={"MAE_mean":MAE_mean,"MAE_std":MAE_std,"margin_of_error_MAE":margin_of_error_MAEMAE,"design_obj_mean":design_obj_mean,"design_obj_std":design_obj_std,"margin_of_error_design_obj":margin_of_error_design_obj,"design_pbj":design_obj_simu}
        with open(txt_name, "w") as file:
            for var_name, var_value in variables.items():
                file.write(f"{var_name}: {var_value}\n")
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(Result_path + f"/results_visulization.pdf")
        fontsize = 14
        # pdb.set_trace()
        for i in range(20):
            i=i*2
            fig = plt.figure(figsize=(18,15))
            if args.conditioned_steps!=0:
                cond_reshape = cond.reshape(cond.shape[0], args.conditioned_steps, n_bodies,args.num_features).to('cpu').detach().numpy()
                cond_design_reshape = cond_design.reshape(cond_design.shape[0], args.conditioned_steps, n_bodies,args.num_features).to('cpu').detach().numpy()
            pred_reshape = pred.reshape(pred.shape[0], pred.shape[1], n_bodies,args.num_features).to("cpu").detach().numpy()
            pred_designed_reshape = pred_designed.reshape(pred_designed.shape[0], pred_designed.shape[1], n_bodies,args.num_features).to("cpu").detach().numpy()
            pred_simu_reshape = pred_simu.reshape(pred_simu.shape[0], pred_simu.shape[1], n_bodies,args.num_features).to("cpu").detach().numpy()
            y_gt_reshape = y_gt.reshape(cond.shape[0], y_gt.shape[1], n_bodies,args.num_features).to("cpu").detach().numpy()
            for j in range(n_bodies):
                # cond:
                marker_size_cond = np.linspace(1, 2, args.conditioned_steps) * 100
                marker_size_cond = np.linspace(1, 2, args.conditioned_steps) * 100
                marker_size_y_gt = np.linspace(2, 3,  y_gt.shape[1]) * 100
                marker_size_pred = np.linspace(2, 3, pred.shape[1]) * 100
                marker_size_pred_designed = np.linspace(2, 3,  pred_designed.shape[1]) * 100

                plt.scatter(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color =COLOR_LIST[j], marker="+", linestyle="--", s=marker_size_cond)
                plt.scatter(cond_design_reshape[i,:,j,0], cond_design_reshape[i,:,j,1], color=COLOR_LIST[j], marker="*", linestyle="--", s=marker_size_cond)
                plt.scatter(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], marker=".", linestyle="-.", s=marker_size_y_gt)
                # plt.scatter(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], marker="v", linestyle="-", s=marker_size_pred)
                plt.scatter(pred_designed_reshape[i,:,j,0], pred_designed_reshape[i,:,j,1], color=COLOR_LIST[j], marker="o", linestyle="-", s=marker_size_y_gt)
                plt.scatter(pred_simu_reshape[i,:,j,0], pred_simu_reshape[i,:,j,1], color=COLOR_LIST[j], marker=",", linestyle="-", s=marker_size_y_gt)

                plt.legend(["condition","condition_designed","rollout_steps_groundtruth","prediction","prediction_designed","prediction of simuSolver on cond_design"],fontsize=16)

                plt.plot(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                

                # cond_design:
                
                plt.plot(cond_design_reshape[i,:,j,0], cond_design_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                
                # y_gt:
                
                plt.plot(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-.")
                
                # pred:

                # plt.plot(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
                
                
                #pred_designed
                
                plt.plot(pred_designed_reshape[i,:,j,0], pred_designed_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")

                #pred_simu
                plt.plot(pred_simu_reshape[i,:,j,0], pred_simu_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
                
                
                
                plt.xlim([0,1])
                plt.ylim([0,1])
            loss_item = (pred_simu[i].to("cpu") - pred_designed[i].to("cpu")).abs().mean().item()
            # loss_item = torch.tensor(pred_reshape[i,:,:,0:4] - y_gt_reshape[i,:,:,0:4]).abs().mean().item()
            plt.title(f"design_obj_mean:{obj_pred_simu[i]:.4f}  simu_eval_obj_mean:{design_obj_simu:.4f}  loss_mean:{loss_mean:.6f}  loss_item:{loss_item:.6f}", fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)
            pdf.savefig(fig)
            if is_jupyter:
                plt.show()
        pdf.close()
        print(f"save results at "+Result_path)
if __name__ == "__main__":
    loss_list=[]
    analyse(args.val_batch_size,loss_list=loss_list)
    # for i in range(10):
    #     analyse((i+1)*10,1000,loss_list=loss_list)
    # np.save("/user/project/inverse_design/results/inv_design_2023-08-14_test_for_old_dataset_2000_simulations/loss_list_old_dataset",loss_list)


