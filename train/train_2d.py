import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from CinDM_anonymous.model.diffusion_2d import Unet, GaussianDiffusion, Trainer
import argparse

parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--dataset', default='naca_ellipse', type=str,
                    help='dataset to evaluate')
parser.add_argument('--batch_size', default=48, type=int,
                    help='size of batch of input to use')
parser.add_argument('--cond_frames', default=2, type=int,
                    help='number of frames to condition')
parser.add_argument('--pred_frames', default=4, type=int,
                    help='number of frames to predict')
parser.add_argument('--ts', default=4, type=int,
                    help='timeskip between frames')
parser.add_argument('--is_train', default=True, type=bool,
                    help='where to train or test the model')
parser.add_argument('--is_testdata', default=True, type=bool,
                    help='where use mini test data or not')
parser.add_argument('--results_folder', default="./checkpoint_path/diffusion_2d/", type=str,
                    help='fold to save the model results')

FLAGS = parser.parse_args()
if __name__ == "__main__":
    print("FLAGS: ", FLAGS)

    inp_dim = (FLAGS.pred_frames + FLAGS.cond_frames) * 3 + 3 # use boundary mask and offset

    model = Unet(
        dim = 64,
        dim_mults = (1, 2),
        channels=inp_dim
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        frames=FLAGS.pred_frames + FLAGS.cond_frames,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2',            # L1 or L2
        diffuse_cond = True # diffuse on both cond states, pred states and boundary
    )

    trainer = Trainer(
        diffusion,
        FLAGS.dataset,
        FLAGS.cond_frames,
        FLAGS.pred_frames,
        FLAGS.ts,
        train_batch_size = FLAGS.batch_size,
        train_lr = 1e-4,
        train_num_steps = 500000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 1000,
        results_folder = FLAGS.results_folder, # diffuse on both cond states, pred states and boundary
        amp = False,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
        is_train = FLAGS.is_train,
        is_testdata=FLAGS.is_testdata
    )

    # trainer.load(110)

    trainer.train()
