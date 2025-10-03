# CinDM: Compositional Generative Inverse  Design (ICLR 2024 spotlight)

[Paper](https://openreview.net/forum?id=wmX0CqFSd7) | [arXiv](https://arxiv.org/abs/2401.13171) | [Poster](https://github.com/AI4Science-WestlakeU/cindm/blob/main/assets/CinDM_poster.pdf) | [Tweet](https://twitter.com/tailin_wu/status/1747259448635367756) 

Official repo for the paper [Compositional Generative Inverse Design](https://openreview.net/forum?id=wmX0CqFSd7).<br />
[Tailin Wu*](https://tailin.org/), [Takashi Maruyama*](https://sites.google.com/view/tmaruyama/home), [Long Wei*](), [Tao Zhang*](https://zhangtao167.github.io), [Yilun Du*](https://yilundu.github.io/), [Gianluca Iaccarino](https://profiles.stanford.edu/gianluca-iaccarino), [Jure Leskovec](https://cs.stanford.edu/people/jure/)<br />
ICLR 2024 **spotlight**. 

We propose a novel formulation for inverse design as an energy optimization problem and introduce Compositional Inverse Design with Diffusion Models method(CinDM) to enable us to generalize to out-of-distribution and more complex design inputs than seen in training, which outperforms the existing works in n-body and 2D airfoil design.

Framework of CinDM:
<a href="url"><img src="https://github.com/AI4Science-WestlakeU/cindm/blob/main/assets/fig1.png" align="center" width="600" ></a>

Visualization of generation process for airfoil design:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/cindm/blob/main/assets/generation_process.gif" align="center" width="600" ></a>

Example generated trajectories and the airfoil boundary:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/cindm/blob/main/assets/generated_examples.gif" align="center" width="600" ></a>

## Installation


1. Install dependencies.

First, create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html) (with python >= 3.7). Then install pytorch, torch-geometric and other dependencies as follows (the repository is run with the following dependencies. Other version of torch-geometric or deepsnap may work but there is no guarentee.)

Install pytorch (replace "cu113" with appropriate cuda version. For example, cuda11.1 will use "cu111"):
```code
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install torch-geometric. Run the following commands:
```code
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-geometric==1.7.2
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
```

Install other dependencies:
```code
pip install -r requirements.txt
```

If wanting to use wandb (--wandb=True), need to set up wandb, following [this link](https://docs.wandb.ai/quickstart).

If wanting to run 2d mesh-based simulation, FEniCS needs to be installed:

```code
conda install -c conda-forge fenics
```


## Dataset and checkpoint

All the dataset can be downloaded in this [this link](https://drive.google.com/file/d/1qolmtsYF4zAQwsUt6OM0nrwJUfWpHqA3/view?usp=drive_link). Checkpoints are in [this link](https://drive.google.com/file/d/1ZQi7iyudhezO7I6lx7-L7HEpLGBHcgrI/view?usp=drive_link). Both dataset.zip and checkpoint_path.zip should be decompressed to the root directory of this project.


## Training

Below we provide example commands for training the diffusion model/forward model.

### training model for N-body inverse design

An example command for training 2-body diffusion model conditioned on 0 steps to rollout 24 steps is as follows, more training setting details are in train_1d.py.
```code
python train/train_1d.py  --date_time '2023-11-20' --dataset 'nbody-2' --model_type "temporal-unet1d" --conditioned_steps 0 --rollout_steps 24 --train_num_steps 1000000 --save_and_sample_every 10000 --method_type "Diffusion"  --Unet_dim 64
```

### training model for 2D airfoils inverse design

```code
python traing/train_2d.py --results_folder "./checkpoint_path/diffusion_2d/"
```

### training baselines

To run experiment with FNO baseline, run:

```code
python3 train/train_baseline.py --exp_id=naca_ellipse --date_time=2023-11-14 --dataset=naca_ellipse_lepde --n_train=-1 --time_interval=4 --save_interval=5 --algo=fno-m20-w32 --no_latent_evo=False --encoder_type=cnn-s --input_steps=4 --evolution_type=mlp-3-silu-3-silu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-4 --normalization_type=gn --latent_size=32 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=silu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --loss_type_consistency=mse --batch_size=64 --val_batch_size=64 --epochs=100 --opt=adam --weight_decay=0 --seed=0 --id=0 --verbose=1 --save_iterations=-1 --latent_loss_normalize_mode=targetindi --n_workers=20 --is_unittest=False --output_padding_str=0-1-1-0 --static_latent_size=32 --gpuid=3 --n_workers=24 --is_timing=0 --test_interval=1
```
```code
python3 train/train_baseline.py --exp_id=naca_ellipse --date_time=2023-11-19 --dataset=naca_ellipse_lepde --n_train=-1 --time_interval=4 --save_interval=5 --algo=contrast --no_latent_evo=False --encoder_type=cnn-s --input_steps=4 --evolution_type=mlp-5-silu-5-silu-3 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-4 --normalization_type=gn --latent_size=160 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=silu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --loss_type_consistency=mse --batch_size=64 --val_batch_size=64 --epochs=100 --opt=adam --weight_decay=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=16 --is_unittest=False --output_padding_str=0-1-1-0 --static_encoder_type="cnn-s" --static_latent_size=32 --gpuid=2 --n_workers=30 --is_timing=0 --test_interval=1 --load_dirname "naca_ellipse_2023-11-14/" --load_filename "naca_ellipse_lepde_train_-1_algo_contrast_enc_cnn-s_evo_cnn_act_silu_hid_160_lo_mse_recef_1.0_conef_1.0_nconv_4_nlat_1_clat_3_nl_False_lf_True_reg_None_gpu:4_id_0_Hash_yAlVxifp_whdeng.p"
```

## Inverse design

Here we provide commands for inverse design using the trained diffusion/forward model:

### N-body inverse design

Inverse design N-body CinDM:

Trained with 2-body 24 steps, at inference, we can generalize diffusion to 8 bodies and 44 steps. More design setteing are in inverse_design_diffusion_1d.py
```code
python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=11-20 --n_composed=0 --compose_n_bodies=2 --design_coef="0.4" --consistency_coef="0.1" --design_guidance="standard-recurrence-10" --val_batch_size=50 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0
```
Inverse design N-body baseline:

backprop with U-Net
```code 
python inference/inverse_design_1d_baseline.py --date_time "2023-11-20_1d_baseline_Unet_NA" --method_type "Unet"  --design_method "backprop"  --max_design_steps 1000 --coef 1 --gamma 2 --coef_max_noise 0  --coef_grad 1 --n_composed 1 --L_bnd False --checkpoint_path "/inverse_design/checkpoint_path/Unet_cond-1_rollout-23_bodies-2.pt"
```

CEM with U-Net
```code 
python inference/inverse_design_1d_baseline.py --date_time "2023-11-20_CEM_Unet" --method_type "Unet"  --design_method "CEM" --max_design_steps 1000 --coef 1 --gamma 2 --coef_max_noise 0 --n_composed 1 --N 1000 --Ne 100 --checkpoint_path "/project/inverse_design/checkpoint_path/Unet_cond-1_rollout-23_bodies-2.pt"
```

More inverse design setting details for N-body baseline in inverse_design_1d_baseline.py. 
### 2D airfoil inverse design

Inverse design with CinDM

```code
python inference/inverse_design_2d.py --ForceNet_path "checkpoint_path/force_surrogate_model.pth" --diffusion_model_path "checkpoint_path/diffusion_2d/"
```

Inverse design with baselines

To perform inverse design with diffusion models, use python filename.py
Example for two airfoils design with CEM using FNO:
```code
python inference/baseline/inverse_design_CEM_discrete_fno_twobds.py
```



## Related Projects

* [DiffPhyCon](https://github.com/AI4Science-WestlakeU/diffphycon) (NeurIPS 2024): We introduce DiffPhyCon which uses diffusion generative models to jointly model control and simulation of complex physical systems as a single task. 

* [WDNO](https://github.com/AI4Science-WestlakeU/wdno) (ICLR 2025): We propose Wavelet Diffusion Neural Operator (WDNO), a novel method for generative PDE simulation and control, to address diffusion models' challenges of modeling system states with abrupt changes and generalizing to higher resolutions, via performing diffusion in the wavelet space.

* [BENO](https://github.com/AI4Science-WestlakeU/beno) (ICLR 2024): A boundary-embedded neural operator that incorporates complex boundary shape and inhomogeneous boundary values into the solving of Elliptic PDEs.
  
* [LAMP](https://github.com/snap-stanford/lamp) (ICLR 2023 spotlight): First fully DL-based surrogate model that jointly optimizes spatial resolutions to reduce computational cost and learns the evolution model, learned via reinforcement learning.


## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wu2024compositional,
  title={Compositional Generative Inverse  Design},
  author={Tailin Wu and Takashi Maruyama and Long Wei and Tao Zhang and Yilun Du and Gianluca Iaccarino and Jure Leskovec},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=wmX0CqFSd7}
}
```
