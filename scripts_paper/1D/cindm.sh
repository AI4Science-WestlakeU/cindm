#! /bin/bash

# cd /
# source /opt/conda/bin/activate

conda activate invDes_env 
cd /zhangtao/project/inverse_design

#2024010204
#Table 1

python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=0 --compose_n_bodies=2 --design_coef="0.2" --consistency_coef="0.2" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 7
# python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=1 --compose_n_bodies=2 --design_coef="0.4" --consistency_coef="0.1" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 0
# python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=2 --compose_n_bodies=2 --design_coef="0.4" --consistency_coef="0.1" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 0
# python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=3 --compose_n_bodies=2 --design_coef="0.4" --consistency_coef="0.1" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 0

#Table 2
# python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=0 --compose_n_bodies=4 --design_coef="0.2" --consistency_coef="0.2" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 7
python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=2 --compose_n_bodies=4 --design_coef="0.2" --consistency_coef="0.2" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 7
python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=0 --compose_n_bodies=8 --design_coef="0.2" --consistency_coef="0.2" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 7
python inference/inverse_design_diffusion_1d.py --exp_id=new-standard-noise_sum --date_time=02-04 --n_composed=2 --compose_n_bodies=8 --design_coef="0.2" --consistency_coef="0.2" --design_guidance="standard-recurrence-10" --val_batch_size=500 --model_name="Diffusion_cond-0_rollout-24_bodies-2_more_collision" --sample_steps=1000 --compose_mode=mean-inside --design_fn_mode=L2 --initialization_mode 0 --gpuid 7