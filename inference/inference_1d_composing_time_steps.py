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
from utils import p, get_item_1d, COLOR_LIST,get_item_1d_for_solver,simulation
import GNS_model

# In[ ]:
import argparse

parser = argparse.ArgumentParser(description='Analyze the trained model')

parser.add_argument('--exp_id', default='inv_design', type=str, help='experiment folder id')
parser.add_argument('--date_time', default='2023-09-14', type=str, help='date for the experiment folder')
parser.add_argument('--dataset', default='nbody-2', type=str, help='dataset to evaluate')

parser.add_argument('--model_type', default='temporal-unet1d', type=str, help='model type.')
parser.add_argument('--conditioned_steps', default=4, type=int, help='conditioned steps')
parser.add_argument('--rollout_steps', default=20, type=int, help='rollout steps')
parser.add_argument('--time_interval', default=4, type=int, help='time interval')
parser.add_argument('--attention', default=True, type=bool, help='whether to use attention block')

parser.add_argument('--milestone', default=100, type=int, help='in which milestone model was saved')
parser.add_argument('--val_batch_size', default=1000, type=int, help='batch size for validation')
parser.add_argument('--is_test', default=True, type=bool,help='flag for testing')
parser.add_argument('--sample_steps', default=1000, type=int, help='sample steps')
parser.add_argument('--num_features', default=4, type=int,
                    help='in original datset,every data have 4 features,and processed datset just have 11 features ')

parser.add_argument('--time_compose_method', default="autoregress", type=str,
                    help='the method to compose more time steps: 1. autoregress 2. direct 3. EBMs_compose 4. GNS 5. SimuSolver 6. Forward_model ')
parser.add_argument('--is_single_step_prediction', default=False, type=bool,
                    help='whether to use single step prediction model')
parser.add_argument('--dataset_path', default="/user/project/inverse_design/dataset/nbody_dataset", type=str,
                    help='the path to load dataset')

parser.add_argument('--n_composed', default=1, type=int,
                    help='how many prediction to be composed')
parser.add_argument('--checkpoint_path_basic_model', default=None, type=str,
                    help='the path to load checkpoint of basic model')

parser.add_argument('--checkpoint_path_unconditioned', default=None, type=str,
                    help='the path to load checkpoint of unconditioned model')

parser.add_argument('--checkpoint_path_single_step', default=None, type=str,
                    help='the path to load checkpoint of single step model')

parser.add_argument('--checkpoint_path_direct', default=None, type=str,
                    help='the path to load checkpoint of direct model')

parser.add_argument('--checkpoint_path_GNS', default=None, type=str,
                    help='the path to load checkpoint of GNS model')

parser.add_argument('--checkpoint_path_forward_model', default=None, type=str,
                    help='the path to load checkpoint of forward model')

args = parser.parse_args()



def analyse(milestone,val_batch_size,n_composed):
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
        n_bodies = eval(get_str_item(dirname, "dataset").split("-")[1])
        conditioned_steps = eval(get_str_item(dirname, "cond"))
        rollout_steps = eval(get_str_item(dirname, "roll"))
        time_interval = eval(get_str_item(dirname, "itv"))
        save_step_load = args.milestone
        is_valid = True
        while save_step_load >= 0:
            try:
                # model_results = torch.load("/user/project/inverse_design/results/inv_design_2023-08-29_test_continue_training_with_more_collisions_with_shuffle/1D_dataset_nbody-2_cond_4_roll_20_itv_4/model-10.pt")
                if args.checkpoint_path_basic_model!=None:
                    model_results= torch.load(args.checkpoint_path_basic_model)
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
                dim=64,
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
                conditioned_steps=args.conditioned_steps,
                timesteps = 1000,           # number of steps
                sampling_timesteps = args.sample_steps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                loss_type = 'l1',           # L1 or L2
            ).to(device)
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
        dataset = NBodyDataset(
            dataset=f"nbody-{n_bodies}",
            input_steps=conditioned_steps,
            output_steps=rollout_steps*(n_composed+1),
            time_interval=time_interval,
            is_y_diff=False,
            is_train=not args.is_test,
            is_testdata=False,
            dataset_path=args.dataset_path
        )
        dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=6)


        for data in dataloader:
            break
        cond = get_item_1d(data, "x").to(device)
        y_gt = get_item_1d(data, "y")
        if args.time_compose_method=="EBMs_compose":
            pred,pred_infered = diffusion.sample(
                batch_size=args.val_batch_size,
                cond=cond,
                is_composing_time=True,
                n_composed=n_composed  # [B, conditioned_steps, n_bodies*feature_size]
            )# [B, rollout_steps, n_bodies*feature_size]
            y=torch.cat([pred,pred_infered],dim=1).to("cpu")
        elif args.time_compose_method=="autoregress":
            if args.is_single_step_prediction:
                model_single_step = TemporalUnet1D(
                    horizon=args.conditioned_steps+args.conditioned_steps,
                    transition_dim=n_bodies*args.num_features,
                    cond_dim=False,
                    dim=64,
                    dim_mults=(1, 2, 4, 8),
                    attention=args.attention,
                ).to(device)
                diffusion_single_step = GaussianDiffusion1D(
                    model_single_step,
                    image_size = args.conditioned_steps,
                    conditioned_steps=args.conditioned_steps,
                    timesteps = 1000,           # number of steps
                    sampling_timesteps = args.sample_steps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                    loss_type = 'l1',           # L1 or L2
                ).to(device)
                model_results_single_steps=torch.load(args.checkpoint_path_single_step)
                # model_results_single_steps = torch.load("/user/project/inverse_design/results/inv_design_2023-09-07_test_for_2_bodies_cond-4_rollout-4/1D_dataset_nbody-2_cond_4_roll_4_itv_4/model-100.pt")
                diffusion_single_step.load_state_dict(model_results_single_steps["model"])
                pred = diffusion_single_step.autoregress_time_compose_sample(
                    batch_size=args.val_batch_size,
                    cond=cond,
                    n_composed=n_composed,  # [B, conditioned_steps, n_bodies*feature_size]
                    is_single_step_prediction=args.is_single_step_prediction,
                    prediction_steps=args.rollout_steps*(1+n_composed)
            )# [B, rollout_steps, n_bodies*feature_size]
            else:
                pred = diffusion.autoregress_time_compose_sample(
                    batch_size=args.val_batch_size,
                    cond=cond,
                    n_composed=n_composed,  # [B, conditioned_steps, n_bodies*feature_size]
                    is_single_step_prediction=args.is_single_step_prediction,
                    prediction_steps=args.rollout_steps*(1+n_composed)
                )# [B, rollout_steps, n_bodies*feature_size]
            y=pred.to("cpu")
            pred=y[:,:args.rollout_steps,:]
            pred_infered=y[:,args.rollout_steps:,:]
        elif args.time_compose_method=="direct":
            model_results=torch.load(args.checkpoint_path_direct)
            # model_results=torch.load("/user/project/inverse_design/results/inv_design_2023-09-07_test_for_2_bodies_cond-4_rollout-40/1D_dataset_nbody-2_cond_4_roll_40_itv_4/model-100.pt")
            if args.model_type == "temporal-unet1d":
                model = TemporalUnet1D(
                horizon=(n_composed+1)*args.rollout_steps+args.conditioned_steps,
                transition_dim=n_bodies*args.num_features,
                cond_dim=False,
                dim=64,
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
                image_size = args.rollout_steps*(n_composed+1),
                conditioned_steps=args.conditioned_steps,
                timesteps = 1000,           # number of steps
                sampling_timesteps = args.sample_steps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                loss_type = 'l1',           # L1 or L2
            ).to(device)
            diffusion.load_state_dict(model_results["model"])
            # pdb.set_trace()
            pred = diffusion.sample(
                batch_size=args.val_batch_size,
                cond=cond,
                is_composing_time=False,
                n_composed=0  # [B, conditioned_steps, n_bodies*feature_size]
            )# [B, rollout_steps, n_bodies*feature_size]
            y=pred.to('cpu')
            pred=y[:,:args.rollout_steps,:]
            pred_infered=y[:,args.rollout_steps:,:]
        elif args.time_compose_method=="GNS":
            #define model
            gns_model=GNS_model.dyn_model.Net()
            #dataset
            test_dataset=GNS_model.Nbody_gns_dataset.nbody_gns_dataset(
                data_dir=args.dataset_path,
                phase='test',
                rollout_steps=240,
                n_his=4,
                time_interval=4,
                verbose=1,
                input_steps=4,
                output_steps=args.rollout_steps*(n_composed+1),
                n_bodies=n_bodies,
                is_train=False,
                device=device
                )
            
            #load check_point
            model_results=torch.load(args.checkpoint_path_GNS)
            gns_model.load_state_dict(model_results["model"])
            gns_model=gns_model.to(device)
            gns_model.eval()

            #inference
            dataloader_GNS=DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=6)
            for data_GNS in dataloader_GNS:
                break

            for i in range(args.val_batch_size):
                # if i>0:
                #     pdb.set_trace()
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
                num_rollouts=args.rollout_steps*(1+n_composed)
                test_dataset.metadata={key: value.to(device) for key, value in test_dataset.metadata.items()}
                outputs = gns_model(poss_cond.to(device), particle_type.to(device), test_dataset.metadata, nonk_mask.to(device), tgt_poss.to(device), num_rollouts=num_rollouts, phase='train')
                # pdb.set_trace()
                torch.cuda.empty_cache()
                # pdb.set_trace()
                # for i in range(args.rollout_steps):
                #     data=train_dataset[i]
                #     poss, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss = data
                    # if i==0:
                    #     y_gt=tgt_poss 
                    # else:
                    #     y_gt=torch.cat([y_gt,tgt_poss],dim=1)
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
                    y=pred
                else:
                    y_gt=torch.cat([y_gt,y_gt_i],dim=0).to("cpu")
                    pred=torch.cat([pred,pred_i],dim=0)
                    cond=torch.cat([cond,cond_i],dim=0)
                    y=pred.to("cpu")
            pred=y[:,:1,:]
            pred_infered=y[:,1:,:]

        elif args.time_compose_method=="SimuSolver":
            cond = get_item_1d_for_solver(data, "x").to('cpu')
            # y_gt = get_item_1d_for_solver(data, "y").to('cpu')

            cond_reshape=cond.reshape(cond.shape[0],cond.shape[1],n_bodies,int(cond.shape[2]/n_bodies))

            # pdb.set_trace()
            simulation(features=cond_reshape[:,-1,:,:],n_steps=args.rollout_steps*(n_composed+1)*args.time_interval,filename=EXP_PATH + f"{args.exp_id}_{args.date_time}/")
            feature_simulation=np.load(EXP_PATH + f"{args.exp_id}_{args.date_time}/"+f"trajectory_balls_{n_bodies}_simu_{args.val_batch_size}_steps_{args.rollout_steps*(n_composed+1)*args.time_interval}.npy")
            feature_simulation_reshape=torch.tensor(feature_simulation.reshape((feature_simulation.shape[0],feature_simulation.shape[1],feature_simulation.shape[2]*feature_simulation.shape[3])))
            
            #normalization and sample each 4 time_interval
            cond=cond/200.
            y=feature_simulation_reshape[:,::4,:]/200.
            y=torch.cat([y[:,1:,:],feature_simulation_reshape[:,-1:,:]/200.],dim=1)
            
            pred=y[:,:1,:]
            pred_infered=y[:,1:,:]
            # pdb.set_trace()
        elif args.time_compose_method=="Forward_model":
            forward_model=Unet1D_forward_model(
                    horizon=args.rollout_steps*(1+n_composed),### horizon Maybe match the time_steps
                    transition_dim=n_bodies*args.num_features, #this matches num_bodies*nun_feactures
                    cond_dim=False,
                    dim=64,
                    dim_mults=(1, 2, 4, 8),
                    attention=True,
                )
            model_results_forward=torch.load(args.checkpoint_path_forward_model)
            forward_model.load_state_dict(model_results_forward["model"])
            forward_model.to(device)

            y=forward_model(cond).to('cpu')
            pred=y[:,:1,:]
            pred_infered=y[:,1:,:]
        y_gt_split = y_gt.view(y_gt.shape[0],y_gt.shape[1],n_bodies,y_gt.shape[2]//2)
        y_split = y.view(y.shape[0],y.shape[1],n_bodies,y.shape[2]//2)
        # pdb.set_trace()
        loss_mean = (y_split[:,:,:,0:2] - y_gt_split[:,:,:,0:2]).abs().mean().item() #when testing,we just analyze the accuracy of x,y,vx,y
        if args.is_single_step_prediction:
            pdf = matplotlib.backends.backend_pdf.PdfPages(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + f"figure_{dirname}_model-{args.milestone}_composed_{n_composed}_{args.time_compose_method}_steps-{args.sample_steps}_single_step.pdf")
        else:
            pdf = matplotlib.backends.backend_pdf.PdfPages(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + f"figure_{dirname}_model-{args.milestone}_composed_{n_composed}_{args.time_compose_method}_steps-{args.sample_steps}.pdf")
        fontsize = 16
        for i in range(20):
            i=i*1
            fig = plt.figure(figsize=(18,15))
            cond_reshape = cond.reshape(args.val_batch_size, conditioned_steps, n_bodies, cond.shape[2]//n_bodies).to('cpu')
            pred_reshape = pred.reshape(args.val_batch_size, pred.shape[1], n_bodies,pred.shape[2]//n_bodies).to('cpu').detach().numpy()
            pred_infered_reshape = pred_infered.reshape(args.val_batch_size, pred_infered.shape[1], n_bodies,pred_infered.shape[2]//n_bodies).to('cpu').detach().numpy()
            y_gt_reshape = y_gt.reshape(args.val_batch_size,y_gt.shape[1], n_bodies,y_gt.shape[2]//n_bodies)
            for j in range(n_bodies):
                # cond:
                marker_size_cond = np.linspace(1, 2, conditioned_steps) * 100
                plt.plot(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                plt.scatter(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], marker="+", linestyle="--", s=marker_size_cond)
                # y_gt:
                marker_size_y_gt = np.linspace(2, 3, y_gt.shape[1]) * 100
                plt.plot(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-.")
                plt.scatter(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], marker=".", linestyle="-.", s=marker_size_y_gt)
                # pred:
                marker_size_pred = np.linspace(2, 3, pred_reshape.shape[1]) * 100
                plt.plot(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
                plt.scatter(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], marker="v", linestyle="-", s=marker_size_pred)
                #pred_infered
                marker_size_pred_infered = np.linspace(2, 3, pred_infered_reshape.shape[1]) * 100
                plt.plot(pred_infered_reshape[i,:,j,0], pred_infered_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
                plt.scatter(pred_infered_reshape[i,:,j,0], pred_infered_reshape[i,:,j,1], color=COLOR_LIST[j], marker="o", linestyle="-", s=marker_size_pred_infered)

                plt.xlim([0,1])
                plt.ylim([0,1])
            loss_item = (y[i] - y_gt[i]).abs().mean().item()
            plt.title(f"loss_mean: {loss_mean:.6f}   loss_item: {loss_item:.6f}", fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)
            pdf.savefig(fig)
            if is_jupyter:
                plt.show()
            i=i/1
        pdf.close()
if __name__ == "__main__":

    analyse(args.milestone,args.val_batch_size,n_composed=args.n_composed)
    # for i in range(10):
    #     analyse((i+1),200)


