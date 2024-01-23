#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from diffusion_1d import Unet1D, GaussianDiffusion1D, Trainer1D, num_to_groups,TemporalUnet1D,linear_beta_schedule,Unet1D_forward_model
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
parser.add_argument('--date_time', default='2023-09-07_test_for_2_bodies', type=str, help='date for the experiment folder')
parser.add_argument('--dataset', default='nbody-2', type=str, help='dataset to evaluate')

parser.add_argument('--model_type', default='temporal-unet1d', type=str, help='model type.')
parser.add_argument('--conditioned_steps', default=4, type=int, help='conditioned steps')
parser.add_argument('--rollout_steps', default=20, type=int, help='rollout steps')
parser.add_argument('--time_interval', default=4, type=int, help='time interval')
parser.add_argument('--attention', default=True, type=bool, help='whether to use attention block')

parser.add_argument('--milestone', default=100, type=int, help='in which milestone model was saved')
parser.add_argument('--val_batch_size', default=1, type=int, help='batch size for validation')
parser.add_argument('--is_test', default=True, type=bool,help='flag for testing')
parser.add_argument('--sample_steps', default=250, type=int, help='sample steps')
parser.add_argument('--num_features', default=4, type=int,
                    help='in original datset,every data have 4 features,and processed datset just have 11 features ')

parser.add_argument('--dataset_path', default="/user/project/inverse_design/dataset/nbody_dataset", type=str,
                    help='the path to load dataset')

parser.add_argument('--n_composed', default=2, type=int,
                    help='how many prediction to be composed')

parser.add_argument('--multi_bodies_method', default="EBMs_compose", type=str,
                    help='The method to predict more bodies : 1. EBMs_compose 2. GNS 3. Forward_model 4. Direct_diffusion 5. SimuSolver')

parser.add_argument('--checkpoint_path_basic_model', default=None, type=str,
                    help='the path to load checkpoint of basic model')

parser.add_argument('--checkpoint_path_unconditioned', default=None, type=str,
                    help='the path to load checkpoint of unconditioned model')

parser.add_argument('--checkpoint_path_single_step', default=None, type=str,
                    help='the path to load checkpoint of single step model')

parser.add_argument('--checkpoint_path_direct_diffusion', default=None, type=str,
                    help='the path to load checkpoint of direct diffusion model')
parser.add_argument('--checkpoint_path_GNS', default=None, type=str,
                    help='the path to load checkpoint of GNS model')

parser.add_argument('--checkpoint_path_forward_model', default=None, type=str,
                    help='the path to load checkpoint of forward model')

args = parser.parse_args()




def analyse(milestone,val_batch_size,n_composed,N,L):
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
        "1D_dataset_nbody-2_cond_4_roll_20_itv_4"
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
                print(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + dirname + f"/model-{save_step_load}.pt")
                # model_results = torch.load("/user/project/inverse_design/results/inv_design_2023-08-25_test_for_old_dataset_316800_datas/1D_dataset_nbody-2_cond_4_roll_20_itv_4_StepLR/model-19.pt")
                if args.checkpoint_path_basic_model!=None:
                    model_results = torch.load(args.checkpoint_path_basic_model)
                else:
                    model_results = torch.load(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + dirname + f"/model-{save_step_load}.pt")
                if args.checkpoint_path_unconditioned!=None:
                    model_results_unconditional=torch.load(args.checkpoint_path_unconditioned)
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
                model_unconditioned=TemporalUnet1D(
                horizon=args.conditioned_steps + args.rollout_steps,### horizon Maybe match the time_steps
                transition_dim=1*args.num_features, #this matches num_bodies*nun_feactures
                cond_dim=False,
                dim=64,
                dim_mults=(1, 2, 4, 8),
                attention=True,
        ).to(device)
            else:
                model = Unet1D(
                    dim = 64,
                    dim_mults = (1, 2, 4, 8),
                    channels =n_bodies*args.num_features,
                ).to(device)
            
                        #init assistant
                        
            diffusion_assistant = GaussianDiffusion1D(
                model=model_unconditioned,
                image_size = args.rollout_steps,
                conditioned_steps=args.conditioned_steps,
                timesteps = 1000,           # number of steps
                sampling_timesteps = args.sample_steps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                loss_type = 'l1',           # L1 or L2
            ).to(device)
            diffusion_assistant.load_state_dict(model_results_unconditional['model'])

            diffusion = GaussianDiffusion1D(
                model,
                image_size = args.rollout_steps,
                conditioned_steps=args.conditioned_steps,
                timesteps = 1000,           # number of steps
                sampling_timesteps = args.sample_steps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                loss_type = 'l1',           # L1 or L2
            ).to(device)


            diffusion.load_state_dict(model_results["model"])
            if args.multi_bodies_method=="EBMs_compose":
                diffusion.model_unconditioned=diffusion_assistant.model
            betas_inference=linear_beta_schedule(N)
            diffusion.betas_inference=betas_inference
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
            dataset=f"nbody-{int(n_bodies*n_composed)}",
            input_steps=conditioned_steps,
            output_steps=rollout_steps,
            time_interval=time_interval,
            is_y_diff=False,
            is_train=not args.is_test,
            is_testdata=False,
            dataset_path=args.dataset_path
        )
        dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=6)

        i=0
        # pdb.set_trace()
        # for data in dataloader:
        #     cond_i=get_item_1d(data, "x").to(device) # [batch_size ,4,16]
        #     y_gt_i = get_item_1d(data, "y").to(device) # [batch_size ,20,16]
        #     if i==0:
        #         cond=cond_i
        #         y_gt=y_gt_i
        #     elif i%10==0 and i!=0:
        #         cond=torch.cat([cond,cond_i],dim=0)
        #         y_gt=torch.cat([y_gt,y_gt_i],dim=0)    
        #         if i==(args.val_batch_size-1)*10:
        #             break
        #     i=i+1

        # cond1 = get_item_1d(data1, "x").to(device)
        # cond100 = get_item_1d(data100, "x").to(device)
        # cond=torch.cat([cond1,cond100],dim=2)
        # y_gt1 = get_item_1d(data1, "y").to(device)
        # y_gt100 = get_item_1d(data100, "y").to(device)
        # y_gt=torch.cat([y_gt1,y_gt100],dim=2)
        # cond=get_item_1d(data1, "x").to(device) # [batch_size ,4,16]
        # y_gt = get_item_1d(data1, "y").to(device) # [batch_size ,20,16]
        for data in dataloader:
            break
        cond=get_item_1d(data, "x").to(device)
        y_gt = get_item_1d(data, "y").to(device)

        if args.multi_bodies_method=="EBMs_compose":
        # pdb.set_trace()
            pred=diffusion.sample_compose_multibodies(cond=cond,N=N,L=L,n_bodies=int(n_bodies*n_composed))
        elif args.multi_bodies_method=="GNS":
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
                output_steps=args.rollout_steps,
                n_bodies=n_bodies*n_composed,
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
                num_rollouts=args.rollout_steps
                test_dataset.metadata={key: value.to(device) for key, value in test_dataset.metadata.items()}
                outputs = gns_model(poss_cond.to(device), particle_type.to(device), test_dataset.metadata, nonk_mask.to(device), tgt_poss.to(device), num_rollouts=num_rollouts, phase='train')
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
                    y=pred
                else:
                    y_gt=torch.cat([y_gt,y_gt_i],dim=0).to("cpu")
                    pred=torch.cat([pred,pred_i],dim=0)
                    cond=torch.cat([cond,cond_i],dim=0)
                    y=pred.to("cpu")
            pred=y[:,:,:]
        elif args.multi_bodies_method=="Forward_model":
            forward_model=Unet1D_forward_model(
                horizon=args.rollout_steps,### horizon Maybe match the time_steps
                transition_dim=n_composed*n_bodies*args.num_features, #this matches num_bodies*nun_feactures
                cond_dim=False,
                dim=64,
                dim_mults=(1, 2, 4, 8),
                attention=True,
            )
            model_result_forward=torch.load(args.checkpoint_path_forward_model)
            forward_model.load_state_dict(model_result_forward["model"])
            forward_model.to(device)

            pred=forward_model(cond).to('cpu')
        elif args.multi_bodies_method=="Direct_diffusion":
            model_results_direct_diffusion=torch.load(args.checkpoint_path_direct_diffusion)
            if args.model_type == "temporal-unet1d":
                model = TemporalUnet1D(
                horizon=args.rollout_steps+args.conditioned_steps,
                transition_dim=n_bodies*n_composed*args.num_features,
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
            diffusion.load_state_dict(model_results_direct_diffusion["model"])
            # pdb.set_trace()
            pred = diffusion.sample(
                batch_size=args.val_batch_size,
                cond=cond,
                is_composing_time=False,
                n_composed=0  # [B, conditioned_steps, n_bodies*feature_size]
            )# [B, rollout_steps, n_bodies*feature_size]
            y=pred.to('cpu')
        elif args.multi_bodies_method=="SimuSolver":
            cond = get_item_1d_for_solver(data, "x").to('cpu')
            # y_gt = get_item_1d_for_solver(data, "y").to('cpu')

            cond_reshape=cond.reshape(cond.shape[0],cond.shape[1],n_bodies*n_composed,int(cond.shape[2]/(n_bodies*n_composed)))

            # pdb.set_trace()
            simulation(features=cond_reshape[:,-1,:,:],n_steps=args.rollout_steps*args.time_interval,filename=EXP_PATH + f"{args.exp_id}_{args.date_time}/")
            feature_simulation=np.load(EXP_PATH + f"{args.exp_id}_{args.date_time}/"+f"trajectory_balls_{n_bodies*n_composed}_simu_{args.val_batch_size}_steps_{args.rollout_steps*args.time_interval}.npy")
            feature_simulation_reshape=torch.tensor(feature_simulation.reshape((feature_simulation.shape[0],feature_simulation.shape[1],feature_simulation.shape[2]*feature_simulation.shape[3])))
            
            #normalization and sample each 4 time_interval
            cond=cond/200.
            y=feature_simulation_reshape[:,::4,:]/200.
            y=torch.cat([y[:,1:,:],feature_simulation_reshape[:,-1:,:]/200.],dim=1)
            
            pred=y

        loss_mean = (pred[:,:,0:2].to("cpu") - y_gt[:,:,0:2].to("cpu")).abs().mean().item()
        if args.multi_bodies_method=="EBMs_compose":
            pdf = matplotlib.backends.backend_pdf.PdfPages(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + f"figure_{dirname}_model-{args.milestone}_composed_{n_composed}_N_{N}_L_{L}.pdf")
        else:
            pdf = matplotlib.backends.backend_pdf.PdfPages(EXP_PATH + f"{args.exp_id}_{args.date_time}/" + f"figure_{dirname}_model-{args.milestone}_composed_{n_composed}_{args.multi_bodies_method}.pdf")
        fontsize = 18

        for i in range(20):
            i=i*20
            fig = plt.figure(figsize=(18,15))
            cond_reshape = cond.reshape(args.val_batch_size, conditioned_steps, int(n_bodies*n_composed), int(cond.shape[2]/(int(n_bodies*n_composed)))).to('cpu').detach().numpy()
            pred_reshape = pred.reshape(args.val_batch_size, rollout_steps, int(n_bodies*n_composed),int(pred.shape[2]/(int(n_bodies*n_composed)))).to('cpu').detach().numpy()
            y_gt_reshape = y_gt.reshape(args.val_batch_size, rollout_steps, int(n_bodies*n_composed),int(y_gt.shape[2]/(int(n_bodies*n_composed)))).to('cpu').detach().numpy()
            for j in range(int(int(n_bodies*n_composed))):
                # cond:
                marker_size_cond = np.linspace(1, 2, conditioned_steps) * 100
                marker_size_y_gt = np.linspace(2, 3, rollout_steps) * 100
                marker_size_pred = np.linspace(2, 3, rollout_steps) * 100

                plt.scatter(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], marker="+", linestyle="--", s=marker_size_cond,label="condition steps")
                plt.scatter(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], marker=".", linestyle="-.", s=marker_size_y_gt,label="rollout_steps_groundtruth")
                plt.scatter(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], marker="v", linestyle="-", s=marker_size_pred,label="rollout_steps_prediction")

                plt.legend(["condition steps","rollout_steps_groundtruth","rollout_steps_prediction"])

                plt.plot(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                
                # y_gt:
                
                plt.plot(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-.")
                
                # pred:
                
                plt.plot(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
                

                plt.xlim([0,1])
                plt.ylim([0,1])
                
            loss_item = (pred[i].to("cpu") - y_gt[i].to("cpu")).abs().mean().item()
            plt.title(f"trajectories_of_bodies-{int(n_bodies*n_composed)}_conditioned_steps-{args.conditioned_steps}_rollout_steps-{args.rollout_steps}_loss_mean-{loss_mean}_loss_item-{loss_item}", fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)
            if is_jupyter:
                plt.show()
            pdf.savefig(fig)
        pdf.close()
if __name__ == "__main__":

    analyse(milestone=args.milestone,val_batch_size=args.val_batch_size,n_composed=args.n_composed,N=400,L=0)
    # for i in range(10):
    #     analyse((i+1),200)


