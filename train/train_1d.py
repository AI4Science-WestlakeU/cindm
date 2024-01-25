#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import sys
import os
import pdb
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from cindm.model.diffusion_1d import Unet1D, TemporalUnet1D, GaussianDiffusion1D, Trainer1D,Unet1D_forward_model
from cindm.filepath import EXP_PATH
import pprint as pp
import torch
from cindm.utils import Printer, make_dir,caculate_num_parameters
p = Printer(n_digits=6)
import logging
import datetime
import cindm.GNS_model
import cindm.filepath as filepath
parser = argparse.ArgumentParser(description='Train EBM model')
parser.add_argument('--exp_id', default='inv_design', type=str,
                    help='experiment folder id')
parser.add_argument('--date_time', default='2023-09-22_test', type=str,
                    help='date for the experiment folder')
parser.add_argument('--dataset', default='nbody-2', type=str,
                    help='dataset to evaluate')
parser.add_argument('--model_type', default='temporal-unet1d', type=str,
                    help='model type.')
parser.add_argument('--conditioned_steps', default=1, type=int,
                    help='conditioned steps')
parser.add_argument('--rollout_steps', default=23, type=int,
                    help='rollout steps,to fit the arctecture of the network,this parameter should be set to a multiple of 8 ')
parser.add_argument('--time_interval', default=4, type=int,
                    help='time interval')
parser.add_argument('--batch_size', default=32, type=int,
                    help='size of batch of input to use')
parser.add_argument('--train_lr', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--loss_weight_discount', default=1, type=float,
                    help='multiplies t^th timestep of trajectory loss by discount**t')
parser.add_argument('--train_num_steps', default=6, type=int,
                    help='total training steps')
parser.add_argument('--save_and_sample_every', default=2, type=int,
                    help='save model every such steps')
parser.add_argument('--gradient_accumulate_every', default=2, type=int,
                    help='gradient accumulation steps')
parser.add_argument('--amp', default=True, type=bool,
                    help='turn on mixed precision')
parser.add_argument('--ema_decay', default=0.995, type=float,
                    help='exponential moving average decay')
parser.add_argument('--calculate_fid', default=False, type=bool,
                    help='whether to caculate FidScore during training')
parser.add_argument('--num_features', default=4, type=int,
                    help='in original datset,every data have 4 features,and processed datset just have 11 features ')

parser.add_argument('--noncollision_hold_probability', default=0.0, type=float,
                    help='probability of preserving non-collision trajectory data  ')

parser.add_argument('--distance_threshold', default=40.5, type=float,
                    help=' the distance threshold of two bodies collision')

parser.add_argument('--loss_type', default="l1", type=str,
                    help=' the type of lossfunction')

parser.add_argument('--is_diffusion_condition', default=False, type=bool,
                    help=' whther do diffusion on conditioned steps or not')

parser.add_argument('--method_type', default="GNS_cond_one", type=str,
                    help='the method to predict trajectory : 1. GNS 2.forward_model 3. Diffusion 4. GNS_cond_one 5. Unet_rollout_one' )

parser.add_argument('--dataset_path', default=filepath.current_wp+"/dataset/nbody_dataset", type=str,
                    help='the path to load dataset')

parser.add_argument('--GNS_output_size', default=2, type=int,
                    help='the putput size of last layer of GNS')
parser.add_argument('--Unet_dim', default=64, type=int,
                    help='dim of Unet')
args= parser.parse_args()
# In[ ]:
if __name__ == "__main__":
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
        is_jupyter = True
        FLAGS = parser.parse_args([])
    except:
        FLAGS = parser.parse_args()
    pp.pprint(FLAGS.__dict__)
    n_bodies = eval(FLAGS.dataset.split("-")[1])
    print("test:",args.model_type)
    if args.model_type == "unet1d":
        model = Unet1D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels=n_bodies*FLAGS.num_features
        )
    elif FLAGS.model_type == "temporal-unet1d":
        model = TemporalUnet1D(
        horizon=FLAGS.conditioned_steps + FLAGS.rollout_steps,### horizon Maybe match the time_steps
        transition_dim=n_bodies*FLAGS.num_features, #this matches num_bodies*nun_feactures
        cond_dim=False,
        dim=args.Unet_dim,
        dim_mults=(1, 2, 4, 8),
        attention=True,
        )
        model_unconditioned=TemporalUnet1D(
        horizon=FLAGS.conditioned_steps + FLAGS.rollout_steps,### horizon Maybe match the time_steps
        transition_dim=n_bodies*FLAGS.num_features, #this matches num_bodies*nun_feactures
        cond_dim=False,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        attention=True,
        )
        caculate_num_parameters(model)
    else:
        raise RuntimeError("Some error message")

    diffusion = GaussianDiffusion1D(
        model,
        image_size = FLAGS.rollout_steps,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type =args.loss_type,
        conditioned_steps=FLAGS.conditioned_steps,
        loss_weight_discount=FLAGS.loss_weight_discount,                    # L1 or L2
        is_diffusion_condition=FLAGS.is_diffusion_condition
    )

    #reload model and continue training
    # model_results = torch.load("/user/project/inverse_design/results/inv_design_2023-09-04_test_for_4_bodies/1D_dataset_nbody-4_cond_4_roll_20_itv_4/model-20.pt")
    # model_results = torch.load("/user/project/inverse_design/results/inv_design_2023-08-29_test_continue_training_with_more_collisions_with_shuffle/1D_dataset_nbody-2_cond_4_roll_20_itv_4/model-10.pt")
    # diffusion.load_state_dict(model_results["model"])
    if FLAGS.method_type=="Unet_rollout_one":
        forward_model=Unet1D_forward_model(
            horizon=1+FLAGS.conditioned_steps,### horizon Maybe match the time_steps
            transition_dim=n_bodies*FLAGS.num_features, #this matches num_bodies*nun_feactures
            cond_dim=False,
            dim=64,
            dim_mults=(1, 2, 4, 8),
            attention=True,
        )
    else:
        forward_model=Unet1D_forward_model(
                horizon=FLAGS.rollout_steps+FLAGS.conditioned_steps,### horizon Maybe match the time_steps
                transition_dim=n_bodies*FLAGS.num_features, #this matches num_bodies*nun_feactures
                cond_dim=False,
                dim=64,
                dim_mults=(1, 2, 4, 8),
                attention=True,
            )
    train_dataset=None
    if FLAGS.method_type=="GNS":
        # model setting
        forward_model=GNS_model.dyn_model.Net()

        #dataset
        train_dataset=GNS_model.Nbody_gns_dataset.nbody_gns_dataset(
            data_dir="/user/project/inverse_design/dataset/nbody_dataset/",
            output_steps=FLAGS.rollout_steps,
            
        )
        # pdb.set_trace()
    if FLAGS.method_type=="GNS_cond_one":
        # model setting
        forward_model=GNS_model.dyn_model.Net_cond_one(
            output_size=args.GNS_output_size,
            n_bodies=n_bodies
        )

        #dataset
        # caculate_num_parameters(forward_model)
        train_dataset=GNS_model.Nbody_gns_dataset.nbody_gns_dataset_cond_one(
            data_dir="/user/project/inverse_design/dataset/nbody_dataset/",
            output_steps=FLAGS.rollout_steps,
            
        )
        # pdb.set_trace()
    exp_dirname = f"{FLAGS.exp_id}_{FLAGS.date_time}/"
    results_folder = EXP_PATH + exp_dirname + f"1D_dataset_{FLAGS.dataset}_cond_{FLAGS.conditioned_steps}_roll_{FLAGS.rollout_steps}_itv_{FLAGS.time_interval}"
    # pdb.set_trace()
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"{results_folder} created")
    else:
        print(f"{results_folder} exists")
    make_dir(results_folder + "/test")
    trainer = Trainer1D(
        diffusion,
        FLAGS.dataset,
        train_batch_size = FLAGS.batch_size,
        train_lr = FLAGS.train_lr,
        train_num_steps = FLAGS.train_num_steps,         # total training steps
        save_and_sample_every = FLAGS.save_and_sample_every,     # save model every such steps
        gradient_accumulate_every =FLAGS.gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = FLAGS.ema_decay,                # exponential moving average decay
        amp = FLAGS.amp,                       # turn on mixed precision
        calculate_fid = FLAGS.calculate_fid,            # whether to calculate fid during training
        conditioned_steps = FLAGS.conditioned_steps,
        rollout_steps = FLAGS.rollout_steps,
        time_interval = FLAGS.time_interval,
        results_folder = results_folder,
        method_type=FLAGS.method_type,
        forward_model=forward_model,
        train_dataset=train_dataset,
        dataset_path=FLAGS.dataset_path
    )

    # if FLAGS.method_type=="forward_model":
    #     trainer.train_forward_model()
    # else:
    trainer.train()


