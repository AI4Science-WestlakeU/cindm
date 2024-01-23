#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from diffusion_1d import Unet1D, GaussianDiffusion1D, Trainer1D, num_to_groups,TemporalUnet1D,Unet1D_forward_model
from filepath import EXP_PATH
import matplotlib.pylab as plt
import matplotlib.backends.backend_pdf
from nbody_dataset import NBodyDataset
import numpy as np
import pdb
import torch
from torch_geometric.data.dataloader import DataLoader
from utils import p, get_item_1d, COLOR_LIST,CustomSampler,caculate_num_parameters
import filepath

# In[ ]:
import argparse
import GNS_model
parser = argparse.ArgumentParser(description='Analyze the trained model')

parser.add_argument('--exp_id', default='inv_design', type=str, help='experiment folder id')
parser.add_argument('--date_time', default='2023-09-22_1d_baseline_GNS_autoregress', type=str, help='date for the experiment folder')
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
# parser.add_argument('--num_features', default=4, type=int,
#                     help='in original datset,every data have 4 features,and processed datset just have 11 features ')
parser.add_argument('--is_diffusion_condition', default=False, type=bool,
                    help=' whther do diffusion on conditioned steps or not')
parser.add_argument('--method_type', default="GNS_cond_one", type=str,
                    help='the method to predict trajectory : 1. GNS 2.forward_model 3. Diffusion 4. GNS_cond_one 5. Unet_rollout_one')
parser.add_argument('--checkpoint_path', default=None, type=str,
                    help='the path to load checkpoint')

parser.add_argument('--dataset_path', default=filepath.current_wp+"/dataset/nbody_dataset", type=str,
                    help='the path to load dataset')

parser.add_argument('--GNS_output_size', default=2, type=int,
                    help='the putput size of last layer of GNS')
parser.add_argument('--Unet_dim', default=64, type=int,
                    help='dim of Unet')
args = parser.parse_args()


def analyse(milestone,val_batch_size,loss_list):
    args.milestone=milestone
    args.val_batch_size=val_batch_size
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")
    id = f"sample-{args.sample_steps}-1"
    dirname_list = [
        f"1D_dataset_{args.dataset}_cond_{args.conditioned_steps}_roll_{args.rollout_steps}_itv_4"
    ]
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
    for dirname in dirname_list:
        n_bodies = eval(args.dataset.split("-")[1])
        conditioned_steps = eval(get_str_item(dirname, "cond"))
        rollout_steps = eval(get_str_item(dirname, "roll"))
        time_interval = eval(get_str_item(dirname, "itv"))
        save_step_load = args.milestone
        is_valid = True
        while save_step_load >= 0:
            try:
                if args.checkpoint_path!=None:
                    model_results = torch.load(args.checkpoint_path)
                else:
                    model_results = torch.load(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + dirname + f"/model-{save_step_load}.pt")
            except:
                print(f"model-{save_step_load} does not exist!")
                is_valid = False
                break

            if args.model_type == "temporal-unet1d":
                model = TemporalUnet1D(
                horizon=args.rollout_steps+args.conditioned_steps,
                transition_dim=n_bodies*args.num_features,
                cond_dim=False,
                dim=args.Unet_dim,
                dim_mults=(1, 2, 4, 8),
                attention=args.attention,
                ).to(device)
            else:
                model = Unet1D(
                    dim = 64,
                    dim_mults = (1, 2, 4, 8),
                    channels =n_bodies*args.num_features,
                ).to(device)
            diffusion = GaussianDiffusion1D(
                model,
                image_size = args.rollout_steps,
                timesteps = 1000,           # number of steps
                sampling_timesteps = args.sample_steps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                loss_type = 'l1', 
                conditioned_steps=args.conditioned_steps,        # L1 or L2
                loss_weight_discount=args.loss_weight_discount,
                is_diffusion_condition=args.is_diffusion_condition
            ).to(device)
            if args.method_type=="forward_model":
                forward_model=Unet1D_forward_model(
                    horizon=args.rollout_steps+args.conditioned_steps,### horizon Maybe match the time_steps
                    transition_dim=n_bodies*args.num_features, #this matches num_bodies*nun_feactures
                    cond_dim=False,
                    dim=64,
                    dim_mults=(1, 2, 4, 8),
                    attention=True,
                )
                caculate_num_parameters(forward_model)
                forward_model.load_state_dict(model_results["model"])
                forward_model.to(device)
            elif args.method_type=="Unet_rollout_one":
                forward_model=Unet1D_forward_model(
                    horizon=1+args.conditioned_steps,### horizon Maybe match the time_steps
                    transition_dim=n_bodies*args.num_features, #this matches num_bodies*nun_feactures
                    cond_dim=False,
                    dim=64,
                    dim_mults=(1, 2, 4, 8),
                    attention=True,
                )
                caculate_num_parameters(forward_model)
                forward_model.load_state_dict(model_results["model"])
                forward_model.to(device)
            elif args.method_type=="GNS":
                # model setting
                gns_model=GNS_model.dyn_model.Net()
                    #dataset
                test_dataset=GNS_model.Nbody_gns_dataset.nbody_gns_dataset(
                    data_dir="/user/project/inverse_design/dataset/nbody_dataset/",
                    phase='test',
                    rollout_steps=240,
                    n_his=4,
                    time_interval=4,
                    verbose=1,
                    input_steps=4,
                    output_steps=args.rollout_steps,
                    n_bodies=n_bodies,
                    is_train=False,
                    device=device
                    )
                caculate_num_parameters(gns_model)
                gns_model.load_state_dict(model_results["model"])
                gns_model=gns_model.to(device)
                gns_model.eval()
            elif args.method_type=="GNS_cond_one":
                # model setting
                gns_model=GNS_model.dyn_model.Net_cond_one(
                    output_size=args.GNS_output_size
                )
                    #dataset
                test_dataset=GNS_model.Nbody_gns_dataset.nbody_gns_dataset_cond_one(
                    data_dir="/user/project/inverse_design/dataset/nbody_dataset/",
                    phase='test',
                    rollout_steps=240,
                    time_interval=4,
                    verbose=1,
                    output_steps=args.rollout_steps,
                    n_bodies=n_bodies,
                    is_train=True,
                    device=device
                    )
                caculate_num_parameters(gns_model)
                gns_model.load_state_dict(model_results["model"])
                gns_model=gns_model.to(device)
                gns_model.eval()
            else:
                caculate_num_parameters(diffusion.model)
                diffusion.load_state_dict(model_results["model"])
            if torch.isnan(next(iter(model.parameters()))).any():
                print(f"There are NaNs in the model parameters for {dirname} at save_step {save_step_load}! Reduce save_step by 5.")
                save_step_load -= 5
                continue
            else:
                break

        if torch.isnan(next(iter(model.parameters()))).any():
            print(f"all models have NaNs for {dirname}.")
            is_valid = False
        if not is_valid:
            continue
        p.print(f"Loading {dirname}, at save_step = {args.milestone}:", banner_size=50)
        _ = diffusion.eval()
        if args.is_unconditioned:
            dataset = NBodyDataset(
                dataset=f"nbody-{n_bodies}",
                input_steps=4,
                output_steps=args.rollout_steps-4,
                time_interval=time_interval,
                is_y_diff=False,
                is_train=True,
                is_testdata=False,
                dataset_path=args.dataset_path
            )
        else:
            dataset = NBodyDataset(
                dataset=f"nbody-{n_bodies}",
                input_steps=args.conditioned_steps,
                output_steps=args.rollout_steps,
                time_interval=time_interval,
                is_y_diff=False,
                is_train=not args.is_test,
                is_testdata=True,
                dataset_path=args.dataset_path
            )
        # s=CustomSampler(data=dataset,batch_size=args.val_batch_size,noncollision_hold_probability=args.noncollision_hold_probability,distance_threshold=args.distance_threshold)
        # dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=6,sampler=s)
        dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=6)
        # pdb.set_trace()


        for data in dataloader:
            break
        if args.conditioned_steps!=0:
            cond = get_item_1d(data, "x").to(device)
        else:
            cond=None
        # for i in range(cond.shape[0]):
        #     cond[i,:,2]=cond[i,-1,2]*2
        #     cond[i,:,3]=cond[i,-1,3]*2

        y_gt = get_item_1d(data, "y")
        # pdb.set_trace()
        if args.is_unconditioned:
            cond=get_item_1d(data, "x").to(device=device)
            # y_gt=y_gt[:,4:,:]
        # cond=cond[:,:,0:4]
        # y_gt=y_gt[:,:,0:4]
        if args.method_type=="forward_model":
            pred=forward_model(cond).to('cpu')
            pred=pred[:,args.conditioned_steps:,:]
        elif args.method_type=="Unet_rollout_one":
            pred=torch.cat([cond.clone()]*args.rollout_steps,dim=1)
            for i in range(pred.shape[1]):
                if i==0:
                    pred[:,i:i+1]=forward_model(cond)[:,-1:]
                else:
                    pred[:,i:i+1]=forward_model(pred[:,i-1:i])[:,-1:]
        elif args.method_type=="GNS":
            # pdb.set_trace()
            dataloader_GNS=DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=6)
            for data_GNS in dataloader_GNS:
                break

            for i in range(args.val_batch_size):
                # if i>0:
                # pdb.set_trace()
                poss_cond, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
                ##now the model just support batch_size=1
                poss_cond=poss_cond[i]
                tgt_accs=tgt_accs[i]
                tgt_vels=tgt_vels[i]
                particle_type=particle_type[i]
                nonk_mask=nonk_mask[i]
                tgt_poss=tgt_poss[i]

                cond_i=torch.cat([poss_cond,torch.zeros_like(poss_cond)],dim=2).permute(1,0,2)
                cond_i=cond_i.reshape(-1,cond_i.shape[0],cond_i.shape[1]*cond_i.shape[2]) #[batch_size, num_steps,n_bodies*n_features] just for convenient visulization
                num_rollouts=args.rollout_steps
                test_dataset.metadata={key: value.to(device) for key, value in test_dataset.metadata.items()}
                outputs = gns_model(poss_cond.to(device), particle_type.to(device), test_dataset.metadata, nonk_mask.to(device), tgt_poss.to(device), num_rollouts=num_rollouts, phase='test')
                torch.cuda.empty_cache()

                y_gt_i=tgt_poss
                pred_i=outputs["pred_poss"]
                y_gt_i=torch.cat([y_gt_i,torch.zeros_like(y_gt_i)],dim=2).permute(1,0,2)
                y_gt_i=y_gt_i.reshape(-1,y_gt_i.shape[0],y_gt_i.shape[1]*y_gt_i.shape[2])
                pred_i=torch.cat([pred_i,torch.zeros_like(pred_i)],dim=2).permute(1,0,2)
                pred_i=pred_i.reshape(-1,pred_i.shape[0],pred_i.shape[1]*pred_i.shape[2])
                if i==0:
                    y_gt=y_gt_i
                    pred=pred_i
                    cond=cond_i
                else:
                    y_gt=torch.cat([y_gt,y_gt_i],dim=0)
                    pred=torch.cat([pred,pred_i],dim=0)
                    cond=torch.cat([cond,cond_i],dim=0)
        elif args.method_type=="GNS_cond_one":

            dataloader_GNS=DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=6)
            for data_GNS in dataloader_GNS:
                break

            for i in range(args.val_batch_size):

                poss_cond, vel,tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
                ##now the model just support batch_size=1
                poss_cond=poss_cond[i]
                vel=vel[i]
                tgt_accs=tgt_accs[i]
                tgt_vels=tgt_vels[i]
                particle_type=particle_type[i]
                nonk_mask=nonk_mask[i]
                tgt_poss=tgt_poss[i]

                cond_i=torch.cat([poss_cond,torch.zeros_like(poss_cond)],dim=2).permute(1,0,2)
                cond_i=cond_i.reshape(-1,cond_i.shape[0],cond_i.shape[1]*cond_i.shape[2]) #[batch_size, num_steps,n_bodies*n_features] just for convenient visulization
                num_rollouts=args.rollout_steps
                test_dataset.metadata={key: value.to(device) for key, value in test_dataset.metadata.items()}
                outputs = gns_model(poss_cond.to(device),vel.to(device) ,particle_type.to(device), test_dataset.metadata, nonk_mask.to(device), tgt_poss.to(device), num_rollouts=num_rollouts, phase='test')
                torch.cuda.empty_cache()

                y_gt_i=tgt_poss
                pred_i=outputs["pred_poss"]
                y_gt_i=torch.cat([y_gt_i,torch.zeros_like(y_gt_i)],dim=2).permute(1,0,2)
                y_gt_i=y_gt_i.reshape(-1,y_gt_i.shape[0],y_gt_i.shape[1]*y_gt_i.shape[2])
                pred_i=torch.cat([pred_i,torch.zeros_like(pred_i)],dim=2).permute(1,0,2)
                pred_i=pred_i.reshape(-1,pred_i.shape[0],pred_i.shape[1]*pred_i.shape[2])
                if i==0:
                    y_gt=y_gt_i
                    pred=pred_i
                    cond=cond_i
                else:
                    y_gt=torch.cat([y_gt,y_gt_i],dim=0)
                    pred=torch.cat([pred,pred_i],dim=0)
                    cond=torch.cat([cond,cond_i],dim=0)
            # pdb.set_trace()
        else:
            pred = diffusion.sample(
                batch_size=args.val_batch_size,
                cond=cond,
                is_composing_time=False,
                n_composed=0  # [B, conditioned_steps, n_bodies*feature_size]
            ).to('cpu')# [B, rollout_steps, n_bodies*feature_size]
        
        if args.is_unconditioned:
            cond=pred[:,:4,:]
            pred=pred[:,4:,:]
            
        y_gt_split = y_gt.view(y_gt.shape[0],y_gt.shape[1],n_bodies,args.num_features)
        pred_split = pred.view(pred.shape[0],pred.shape[1],n_bodies,args.num_features)
        loss_mean = (pred_split[:,:,:,0:2].to("cpu") - y_gt_split[:,:,:,0:2].to("cpu") ).abs().mean().item() #when testing,we just analyze the accuracy of x,y,vx,y
        loss_list.append(loss_mean)

        if args.is_unconditioned: # reset these parameters for better results visulization
            args.conditioned_steps=4
            args.rollout_steps=args.rollout_steps-args.conditioned_steps
        pdf = matplotlib.backends.backend_pdf.PdfPages(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + f"figure_{dirname}_model-{args.milestone}_id_{id}.pdf")
        fontsize = 16

        for i in range(20):
            i=i*1
            fig = plt.figure(figsize=(18,15))
            if args.conditioned_steps!=0:
                cond_reshape = cond.reshape(cond.shape[0], args.conditioned_steps, n_bodies,args.num_features).to('cpu')
            pred_reshape = pred.reshape(pred.shape[0], args.rollout_steps, n_bodies,args.num_features).to("cpu").detach().numpy()
            y_gt_reshape = y_gt.reshape(cond.shape[0], args.rollout_steps, n_bodies,args.num_features)
            for j in range(n_bodies):
                # cond:
                if args.conditioned_steps!=0:
                    marker_size_cond = np.linspace(1, 2, args.conditioned_steps) * 100
                    plt.plot(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                    plt.scatter(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], marker="+", linestyle="--", s=marker_size_cond)
                # y_gt:
                marker_size_y_gt = np.linspace(2, 3, args.rollout_steps) * 100
                plt.plot(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-.")
                plt.scatter(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], marker=".", linestyle="-.", s=marker_size_y_gt)
                # pred:
                marker_size_pred = np.linspace(2, 3, args.rollout_steps) * 100
                plt.plot(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
                plt.scatter(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], marker="v", linestyle="-", s=marker_size_pred)
                plt.xlim([0,1])
                plt.ylim([0,1])
            loss_item = (pred[i].to("cpu") - y_gt[i].to("cpu")).abs().mean().item()
            plt.title(f"loss_mean: {loss_mean:.6f}   loss_item: {loss_item:.6f}", fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)
            pdf.savefig(fig)
            if is_jupyter:
                plt.show()
        pdf.close()
if __name__ == "__main__":
    loss_list=[]
    analyse(args.milestone,args.val_batch_size,loss_list=loss_list)

