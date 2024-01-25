import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from cindm.le_pde.pytorch_net.util import str2bool

def arg_parse():
    parser = argparse.ArgumentParser(description='PDE argparse.')
    # Experiment management:
    parser.add_argument('--exp_id', type=str,
                        help='Experiment id')
    parser.add_argument('--date_time', type=str,
                        help='date and time')
    parser.add_argument('--decoder_act_name', type=str,
                        help='decoder_act_name')
    parser.add_argument('--is_prioritized_dropout', type=str2bool, nargs='?', const=True, default=False,
                        help="is_prioritized_dropout")
    parser.add_argument('--save_interval', type=int,
                        help='Interval for saving the model_dict.')
    parser.add_argument('--test_interval', type=int,
                        help='Interval for val and test.')
    parser.add_argument('--save_iterations', type=int,
                        help='Iterations at which model is saved, -1 means not saved in the middle of an epoch')
    parser.add_argument('--verbose', type=int,
                        help='How much to print. Default "1". "2" for printing over minibatches')
    parser.add_argument('--is_test_only', type=str2bool, nargs='?', const=True, default=False,
                        help="If True, will only load the test dataset and test_loader")
    parser.add_argument('--is_tensorboard', type=str2bool, nargs='?', const=True, default=False,
                        help="If True, use Tensorboard to record.")
    parser.add_argument('--wandb', type=eval, 
                        help='boolean on whether to use wandb')
    parser.add_argument('--wandb_project_name', type=str, 
                        help='wandb project name.')
    parser.add_argument('--wandb_step', type=int, 
                        help='wandb project log num freq.')
    parser.add_argument('--wandb_step_plot', type=int, 
                        help='wandb project plot freq.')
    
    parser.add_argument('--is_timing', type=int,
                        help="If True, will print out timing. Use level of 0,1,2,...")

    parser.add_argument('--is_unittest', type=str2bool, nargs='?', const=True, default=True,
                        help="If True, perform unittest")
    parser.add_argument('--seed', type=int,
                        help="Seed for the experiment. Default None which do not set a fixed seed.")
    parser.add_argument('--id', type=str,
                        help='ID, additional information for distinguishing the experiment.')
    # Loading previous experiments:
    parser.add_argument('--load_dirname', type=str,
                        help='Directory to load previous experiment from.')
    parser.add_argument('--load_filename', type=str,
                        help='Filename to load previous experiment from.')
    parser.add_argument('--load_exp_renew', type=str2bool, nargs='?', const=True, default=False,
                        help="If True, reset the data_record and epoch.")
    # Dataset:
    parser.add_argument('--dataset', type=str,
                        help='Dataset. Choose from "burgers[...]", "ks[...]", "karman-2d", "advection", "VL-small2", "VL-large", "PL-1Dsmall" and "PL-1Dlarge".')
    parser.add_argument('--dataset_split_type', type=str,
                        help='Split type for the dataset. Choose from "standard", "random".')
    parser.add_argument('--data_noise_amp', type=float,
                        help='Gaussian noise amplitude for dataset. Default 0')
    parser.add_argument('--train_fraction', type=float,
                        help='fraction of training inside train_val.')
    parser.add_argument('--time_interval', type=int,
                        help='Time interval of sampling the data. Default 1.')
    parser.add_argument('--n_train', type=str,
                        help='The first n_train examples will be used for the dataset. If -1, will use the full dataset.')
    parser.add_argument('--data_dropout', type=str,
                        help='Dropout mode for PyG graph. Choose from "None", e.g. "node:0.4"')
    parser.add_argument('--exclude_bdd', type=eval,
                        help='If True, when doing the data_dropout=node:..., will not dropout the boundary nodes.')
    parser.add_argument('--sector_size', type=str,
                        help='Size of the sector. Default "-1".')
    parser.add_argument('--sector_stride', type=str,
                        help='Stride for the sectors. Default "-1".')
    parser.add_argument('--is_y_variable_length', type=str2bool, nargs='?', const=False, default=False,
                        help='Whether to use difference as y.')
    parser.add_argument('--is_offmask', type=str2bool, nargs='?', const=False, default=False,
                        help='Whether to use mask in computing loss')
    # Model:
    ## Global:
    parser.add_argument('--algo', type=str,
                        help='Algorithm to use. Choose from "contrast" and "gns".')
    parser.add_argument('--latent_size', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--act_name', type=str,
                        help='activation name.')
    parser.add_argument('--decoder_last_act_name', type=str,
                        help="Decoder's last layer's activation name.")
    ## Regularization:
    parser.add_argument('--reg_type', type=str,
                        help='Regularization. Default "None". Has the format of f"{reg-type}[-{model-target}]^..." as splited by "^" for different types of regularizations.'
                         'where {reg-type} chooses from "srank", "sn", "snn", "Jsim" (Jacobian simplicity), "l2", "l1", "fro".'
                         'The optional {model-target} chooses from "all" or "evo" (only effective for Contrastive). If not appearing, default "all".'
                         'The "Jsim" only targets "evo".')
    parser.add_argument('--reg_coef', type=float,
                        help='Coefficient for regularization.')
    parser.add_argument('--is_reg_anneal', type=str2bool, nargs='?', const=True, default=False,
                        help="If True, will anneal up the regularization from 0 quadratically to reg_coef at epoch.")
    ## For contrastive:
    parser.add_argument('--no_latent_evo', type=str2bool, nargs='?', const=True, default=False,
                        help="If True, will turn of latent evolution, and instead predict the data at each time step.")
    parser.add_argument('--encoder_type', type=str,
                        help='Encoder type for Contrastive. Choose from "cnn", "gnn".')
    parser.add_argument('--encoder_n_linear_layers', type=int,
                        help='Number of linear layers following the last layer of the encoder. Default 0.')
    parser.add_argument('--n_conv_blocks', type=int,
                        help='Number of conv blocks.')
    parser.add_argument('--n_latent_levs', type=int,
                        help='Number of latent levels. 1 for only the innermost latent.')
    parser.add_argument('--n_conv_layers_latent', type=int,
                        help='Number of convolutional layers for the latent levels>=2.')
    parser.add_argument('--evo_conv_type', type=str,
                        help='Convolution layer type for the evolution_op.')
    parser.add_argument('--evo_pos_dims', type=int,
                        help='pos_dim for evolution_op.')
    parser.add_argument('--evo_inte_dims', type=int,
                        help='Dimensions for integration for CNN_Integral')
    parser.add_argument('--is_latent_flatten', type=str2bool, nargs='?', const=True, default=True,
                        help="If True, the last layer's latent will be obtained from MLP on flattened feature maps.")
    parser.add_argument('--encoder_mode', type=str,
                        help='Encoder mode for Contrastive. Choose from e.g. "dense", "sector-8-4".')
    parser.add_argument('--recons_coef', type=float,
                        help='Coefficient for reconstruction loss. If 0, will turn off.')
    parser.add_argument('--consistency_coef', type=float,
                        help='Coefficient for consistency of prediction on latent space. If 0, will turn off.')
    parser.add_argument('--contrastive_rel_coef', type=float,
                        help='Coefficient for contrastive loss on latent space, relative to consistency_coef. If 0, will turn off.')
    parser.add_argument('--hinge', type=float,
                        help='Hinge for the contrastive loss.')
    parser.add_argument('--latent_noise_amp', type=float,
                        help='Amplitude of noise added to the latent variable.')
    parser.add_argument('--density_coef', type=float,
                        help='Coefficient for the loss for the maximum likelihood for particles for PIC.')
    parser.add_argument('--is_pos_transform', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will first pass x_pos into an MLP before evaluate on the distribution.')
    parser.add_argument('--normalization_type', type=str,
                        help='Normalization type for CNN encoder and decoder. Choose from "bn2d", "gn".')
    # For ComponentDGL:
    parser.add_argument('--gnn_mlp_n_neurons', type=int,
                        help='Number of neurons for MLP inside ComponentDGL')

    # For GNNDGL:
    parser.add_argument('--gnn_mlp_n_layers', type=int,
                        help='Number of layers for MLP inside GNNDGL')
    parser.add_argument('--synch', type=str2bool, nargs='?', const=True, default=False,
                        help="Whether to use synchronous update.")
    parser.add_argument('--agg_type', type=str,
                        help='Aggregation type for GNNDGL.')


    ## For CNN and hybrid:
    parser.add_argument('--channel_mode', type=str,
                        help='channel_mode. Choose from "exp-{NUM}", "c-{NUM}", "{NUM}-{NUM}-...".')
    parser.add_argument('--kernel_size', type=int,
                        help='Kernel size.')
    parser.add_argument('--stride', type=int,
                        help='stride.')
    parser.add_argument('--padding', type=int,
                        help='padding.')
    parser.add_argument('--padding_mode', type=str,
                        help="padding_mode. Choose from 'zeros', 'reflect', 'replicate' or 'circular'")
    parser.add_argument('--output_padding_str', type=str,
                        help="output_padding_str. Choose from 'None', or e.g. '1-0-0-0', '1-1-0-0'.")
    parser.add_argument('--evo_groups', type=int,
                        help="Number of groups for evolution_op.")
    ## For evolution operator:
    parser.add_argument('--evolution_type', type=str,
                        help="evolution_type for Contrastive."
                             "Format: (1) mlp-{n_layers}[-{act_name}][-{n_linear_layers}], or"
                             "        (2) SINDy-{poly_order}[-{additional_nonlinearities}][-{n_linear_layers}]"
                             "E.g. 'mlp-3' where the number is the number of layers.")
    parser.add_argument('--forward_type', type=str,
                        help="forward_type for evolution_op. Choose from 'Euler', 'RK4'.")
    ## For mixture distributions:
    parser.add_argument('--decoder_type', type=str,
                        help='decoder_type. Format: "MixGau-full-10" where the number is the number of components.')
    ## For gns:
    parser.add_argument('--layer_type', type=str,
                        help='layer_type for GNN. Choose from "graphsage".')
    parser.add_argument('--n_layers', type=int,
                        help='Number of graph convolution layers.')
    parser.add_argument('--dropout', type=float,
                        help='The dropout ratio.')
    ## For Unet:
    parser.add_argument('--unet_fmaps', type=int,
                        help='Number of feature maps for UNetnD.')

    # Training:
    parser.add_argument('--is_pretrain_autoencode', type=str2bool, nargs='?', const=True, default=False,
                        help="If True, reset the data_record and epoch.")
    parser.add_argument('--vae_mode', type=str,
                        help="Choose from 'None', 'recons', 'recons+sample'.")
    parser.add_argument('--vae_beta', type=float,
                        help="beta value for the VAE's KL term.")
    parser.add_argument('--epochs_pretrain', type=int,
                        help='Pretrain epochs')
    parser.add_argument('--dp_mode', type=str,
                        help='Choose from "None", "dp", "ddp"')
    parser.add_argument('--zero_weight', type=float,
                        help='weight for the ground-truth elements that has value 0.')
    parser.add_argument('--input_steps', type=int,
                        help='Number of input steps')
    parser.add_argument('--temporal_bundle_steps', type=int,
                        help='Number of temporal bundle steps. Default 1 (no bundle)')
    parser.add_argument('--is_multistep_detach', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will detach when doing multistep.')
    parser.add_argument('--input_steps_lazy', type=bool,
                        help='Whether to materialize input_steps dimension data after preprocessing, avoiding out-of-memory errors')                
    parser.add_argument('--multi_step', type=str,
                        help='Multi-step prediction mode. Default "1", meaning only 1 step MSE. "1^2:1e-2^4:1e-3" means loss has 1, 2, 4 steps, with the number after ":" being the scale.')
    parser.add_argument('--multi_step_start_epoch', type=int,
                        help='Starting epoch for multi_step. Before that, use single step.')
    parser.add_argument('--latent_multi_step', type=str,
                        help='Multi-step prediction mode for latent variable. Default "1", meaning only 1 step MSE. "1^2:1e-2^4:1e-3" means loss has 1, 2, 4 steps, with the number after ":" being the scale.')
    parser.add_argument('--use_grads', type=str2bool, nargs='?', const=True,
                        help='whether to use data gradient as feature.')
    parser.add_argument('--use_pos', type=str2bool, nargs='?', const=True,
                        help='whether to use normalized position data to augment the features.')
    parser.add_argument('--is_y_diff', type=str2bool, nargs='?', const=False,
                        help='Whether to use difference as y.')
    parser.add_argument('--epsilon_latent_loss', type=float,
                        help='epsilon added to the denominator of latent loss with "target" or "targetindi".')
    parser.add_argument('--loss_type',
                        help='loss type. Choose from "mse", "huber", "l1", "dl".')
    parser.add_argument('--loss_type_consistency',
                        help='loss type on the latent space. Choose from "mse", "huber", "l1", "dl".')
    parser.add_argument('--latent_loss_normalize_mode',
                        help='Choose from "None", "target", "targetindi".')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    parser.add_argument('--n_workers', type=int,
                        help='Number of workers')
    parser.add_argument('--val_batch_size', type=int,
                        help='Batch size for validation and test.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--early_stopping_patience', type=int,
                        help='Patience for early_stopping.')
    parser.add_argument('--opt', type=str,
                        help='Optimizer such as adam, sgd, rmsprop or adagrad.')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay.')
    parser.add_argument('--is_clip_grad', type=eval,
                        help="If True, will clip gradient according to exp_avg_sq.")
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--lr_min_cos', type=float,
                        help='Minimal learning rate for cosine scheduler.')
    parser.add_argument('--lr_scheduler_type', type=str,
                        help='type of the lr-scheduler. Choose from "rop", "cos", "cos-re", "steplr-s100-g0.5" (e.g.) and "None".')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1,
                        help='Multiplication factor for ReduceOnPlateau lr-scheduler.')
    parser.add_argument('--lr_scheduler_T0', type=int, default=50,
                        help='T0 for CosineAnnealingWarmRestarts (cos-re) scheduler')
    parser.add_argument('--lr_scheduler_T_mult', type=int, default=1,
                        help='Multiplication factor for increasing T_i after a restart, for CosineAnnealingWarmRestarts (cos-re) scheduler.')
    parser.add_argument('--gpuid', type=str,
                        help='gpu id or comma-delimited gpu ids.')
    parser.add_argument('--static_latent_size', type=int, default=0,
                        help='dimension of latent vector of static featurs.')
    parser.add_argument('--static_encoder_type', type=str, default="None",
                        help='encoder type of converting static featurs to latent vectors.')
    parser.add_argument('--max_grad_norm', type=float, default=-1,
                        help='maximul gradient L2 norm. Default -1 of not using the grad norm.')
    parser.add_argument('--gnn_output_dim', type=int, default=32,
                        help='dimension of node feature of output.')
    parser.add_argument('--gnn_latent_dim', type=int, default=32,
                        help='dimension of latent vector in gnn.')
    parser.add_argument('--gnn_num_steps', type=int, default=3,
                        help='number of message passing steps.')
    parser.add_argument('--gnn_layer_norm', type=str2bool, nargs='?', const=True, default=False,
                        help='whether or not use layer norm for gnn')
    parser.add_argument('--gnn_activation', type=str,
                        help='actibation function for gnn')
    parser.add_argument('--gnn_diffMLP', type=str2bool, nargs='?', const=True, default=False,
                        help='whether or not use different message passing layer for gnn')
    parser.add_argument('--gnn_global_pooling', type=str,
                        help='function name for global pooling')
    parser.add_argument('--gnn_is_virtual', type=str2bool, nargs='?', const=True, default=True,
                        help='whether or not use of virtual node')
    
    
    parser.set_defaults(
        exp_id="contrast",
        date_time="1-1",
        save_interval=10,
        test_interval=1,
        save_iterations=-1,
        verbose=1,
        seed=-1,
        id="0",
        is_tensorboard=False,
        wandb=False,
        wandb_step_plot=100,
        wandb_step=20,
        wandb_project_name="test",
        is_unittest=True,
        is_timing=0,

        # Loading previous experiments:
        load_dirname="None",
        load_filename="None",
        load_exp_renew=False,

        # Dataset:
        dataset='burgers',  # 'burgers', 'karman-2d', 'advection', 'PL-1Dsmall', 'PL-1Dlarge'
        dataset_split_type="standard",
        train_fraction=float(8/9),
        n_train="-1",
        time_interval=1,
        data_noise_amp=0.,
        data_dropout="None",
        exclude_bdd=False,
        sector_size="-1",
        sector_stride="-1",
        is_test_only=False,
        is_y_variable_length=False,
        is_offmask=False,

        ## Reg:
        reg_type="None",
        reg_coef=0,
        is_reg_anneal=True,

        ## Model:
        algo='contrast',  # 'contrast', 'gns'
        no_latent_evo=False,
        encoder_type="cnn-s",  # "hybrid", "cnn-s", "cnn"
        evolution_type="mlp-3-rational-2",
        decoder_type="cnn-tr",  # "cnn-tr", "mixGau-diag-16"
        encoder_n_linear_layers=0,
        n_conv_blocks=4,
        n_latent_levs=2,
        n_conv_layers_latent=1,
        forward_type="Euler",
        evo_conv_type="cnn",
        evo_pos_dims=-1,
        evo_inte_dims=-1,
        is_latent_flatten=True,
        encoder_mode="dense",
        recons_coef=1,
        consistency_coef=1,
        contrastive_rel_coef=0,
        hinge=1.,
        density_coef=0.001,
        latent_noise_amp=0,
        is_pos_transform=True,
        normalization_type="gn",
        layer_type='graphsage',
        latent_size=16,
        channel_mode="exp-16",
        kernel_size=4,
        stride=2,
        padding=1,
        padding_mode="zeros",  # TODO: Plasma particles only have "zeros" padding_mode
        output_padding_str="None",
        evo_groups=1,
        n_layers=3,
        dropout=0.0,
        act_name="rational",
        decoder_last_act_name="linear",
        is_prioritized_dropout=False,
        decoder_act_name="None",

        ## For ComponentDGL:
        gnn_mlp_n_neurons=64,


        ## For GNNDGL:
        gnn_mlp_n_layers=2,
        synch=False,
        agg_type="mean",

        ## Discriminator:
        disc_coef=0,
        disc_mode="Siamese-2",
        disc_lr=-1,
        disc_reg_type="snn",
        disc_iters=5,
        disc_loss_type="hinge",
        disc_t="None",

        ## Other baselines:
        unet_fmaps=64,

        ## Training:
        is_pretrain_autoencode=False,
        vae_mode="None",
        vae_beta=1,
        epochs_pretrain=0,
        dp_mode="None",
        is_mesh=False,
        zero_weight=1,
        input_steps=1,
        input_steps_lazy=False,
        temporal_bundle_steps=1,
        is_multistep_detach=False,
        multi_step="1^2:0.1^3:0.1^4:0.1", #"1^2:0.1^3:0.1^4:0.1", "1"
        multi_step_start_epoch=0,
        latent_multi_step=None,
        latent_loss_normalize_mode="None",
        use_grads=True,
        use_pos=False,
        is_y_diff=False,
        epsilon_latent_loss=0,
        loss_type="mse",
        loss_type_consistency="mse",
        batch_size=16,
        val_batch_size=64,
        n_workers=4,
        epochs=100,
        opt='adam',
        weight_decay=0,
        early_stopping_patience=-1,
        is_clip_grad=False,
        lr=1e-3,
        lr_min_cos=0,
        lr_scheduler_type="cos",  # "rop", "cos", "cos-re"
        lr_scheduler_factor=0.1,
        lr_scheduler_T0=50,
        lr_scheduler_T_mult=1,
        max_grad_norm=-1,
        gpuid="0",

        ## Static features:
        static_latent_size=0,
        static_encoder_type="None",
        gnn_output_dim=64,
        gnn_latent_dim=32,
        gnn_num_steps=3,
        gnn_layer_norm=False,
        gnn_activation="relu",
        gnn_diffMLP=False,
        gnn_global_pooling="mean",
        gnn_is_virtual=True,
    )
    add_rl_args(parser)
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    args.lr_scheduler_T0 = args.epochs // 8
    if args.algo == "contrast":
        args.is_y_diff = False
    if args.disc_lr == -1:
        args.disc_lr = args.lr
    if args.evo_conv_type.endswith("vaa") or args.evo_conv_type.endswith("vab") or args.evo_conv_type.endswith("vba") or args.evo_conv_type.endswith("vbb") or args.evo_conv_type.endswith("v00") or args.evo_conv_type.endswith("vc"):
        # These modes requires that the evolution operator is "direct" mode:
        args.forward_type = "direct"
    if args.static_encoder_type == "None":
        args.static_latent_size = 0
    if args.static_latent_size == 0:
        args.static_encoder_type="None"

    return args


def add_rl_args(parser):
    parser.add_argument('--rl_coefs', type=str,
                        help='Coefficients for rl. Choose from "None", "reward:1+value:0.1" e.g..')
    parser.add_argument('--rl_horizon', type=int,
                        help='Horizon for RL.')
    parser.add_argument('--reward_mode', type=str,
                        help='Reward mode. Choose from "None" (loss+time), "loss+state".')
    parser.add_argument('--reward_beta', type=str,
                        help='beta on the reward=loss + beta * compute. E.g. "1", "0.5-2".')
    parser.add_argument('--reward_loss_coef', type=float,
                        help='reward_loss_coef on the reward=loss * reward_loss_coef + beta * compute. E.g. "1", "0.5-2".')
    parser.add_argument('--reward_src', type=str,
                        help='src of reward. Choose from "pred" or "env".')
    parser.add_argument('--rl_gamma', type=float,
                        help='gamma as the discount factor.')
    parser.add_argument('--rl_lambda', type=float,
                        help='lambda for the coefficient of value function.')
    parser.add_argument('--rl_rho', type=float,
                        help='rho for the coefficient of the Reinforce loss, and (1-reward_rho) for the dynamics loss.')
    parser.add_argument('--rl_eta', type=float,
                        help='eta for the coefficient of the entropy for the actor loss.')
    parser.add_argument('--rl_critic_update_iterations', type=int,
                        help="How many iteration steps to update the critic_target.")
    parser.add_argument('--rl_data_dropout', type=str,
                        help='Dropout mode for PyG graph for RL. Choose from "None", e.g. "node:0.4", "node:0-0.1:0.1" (prob of 0.1 to perform dropout. If dropout, drop 0-0.1 fraction of nodes)')
    parser.add_argument('--rl_is_finetune_evolution', type=eval,
                        help="Whether to train the evolution model at the same time.")
    parser.add_argument('--rl_is_alt_remeshing', type=eval,
                        help="Whether to use the remeshed result for alt before doing next step.")
    parser.add_argument('--top_k_action', type=int,
                        help="top_k_action selected to sample")
    parser.add_argument('--opt_evl_horizon', type=int,
                        help="opt_evl_horizon")
    parser.add_argument('--evl_stop_gradient', type=eval,
                        help="evl_stop_gradient")
    
    
    parser.add_argument('--actor_lr', type=float,
                        help="Learning rate for value model.")
    parser.add_argument('--actor_batch_norm', type=eval,
                        help="batch_norm for Value_Model.")
    parser.add_argument('--skip_coarse', type=eval,
                        help="batch_norm for Value_Model.")
    parser.add_argument('--skip_split', type=eval,
                        help="batch_norm for Value_Model.")
    parser.add_argument('--skip_flip', type=eval,
                        help="skip_flip.")

    parser.add_argument('--value_latent_size', type=int,
                        help="latent_size for Value_Model.")
    parser.add_argument('--value_num_pool', type=int,
                        help="num_pool for Value_Model.")
    parser.add_argument('--value_act_name', type=str,
                        help="act_name for Value_Model.")
    parser.add_argument('--value_act_name_final', type=str,
                        help="act_name_final for Value_Model.")
    parser.add_argument('--value_layer_norm', type=eval,
                        help="layer_norm for Value_Model.")
    parser.add_argument('--value_batch_norm', type=eval,
                        help="batch_norm for Value_Model.")
    parser.add_argument('--value_num_steps', type=int,
                        help="num_steps for Value_Model.")
    parser.add_argument('--value_pooling_type', type=str,
                        help="pooling_type for Value_Model.")
    parser.add_argument('--value_lr', type=float,
                        help="Learning rate for value model.")
    parser.add_argument('--value_loss_type', type=str,
                        help="Loss_type for value model.")
    parser.add_argument('--value_loss_coef', type=float,
                        help="Coefficient for the loss for value model.")
    parser.add_argument('--test_value_model', type=eval,
                        help="test Value_Model.")
    parser.add_argument('--value_target_mode', type=str,
                        help="Mode for value target. Choose from 'value-lambda', 'vanilla', 'value-n-step'.")
    
    parser.add_argument('--use_reward_vanilla', type=eval,
                        help="use_reward_vanilla - no recursion")
    
    parser.add_argument('--reward_condition', type=eval,
                        help="reward_condition")
    
    parser.add_argument('--is_alternating_train', type=eval,
                        help="is_alternating_train")
    parser.add_argument('--is_single_action', type=eval,
                        help="only one action per state")
    
    parser.add_argument('--value_steps', type=int,
                        help="value_steps")
    parser.add_argument('--actor_steps', type=int,
                        help="actor_steps")
    parser.add_argument('--test_data_interp', type=eval,
                        help="if plot interp forward result")
    
        
    parser.add_argument('--actor_critic_step', type=int,
                        help="actor_critic_step")
    parser.add_argument('--evolution_steps', type=int,
                        help="evolution_steps")
    parser.add_argument('--offset_coarse', type=float,
                        help="policy offset_coarse as prior")
    parser.add_argument('--offset_split', type=float,
                        help="policy offset_split as prior")
    parser.add_argument('--rl_finetune_evalution_mode', type=str,
                        help="rl_finetune_evalution_mode")
    
    parser.add_argument('--max_action', type=int,
                        help="max_action")
    parser.add_argument('--kaction_pooling_type', type=str,
                        help="kaction_pooling_type")
    parser.add_argument('--connect_bdd', type=eval,
                        help="if connect the bdd nodes for periodic boundary condition")
    parser.add_argument('--stop_all_gradient', type=eval,
                        help="if stop_all_gradient")
    parser.add_argument('--is_eval_sample', type=eval,
                        help="if is_eval_sample ")
    parser.add_argument('--debug', type=eval,
                        help="if debug mode ")
    parser.add_argument('--is_1d_periodic', type=eval,
                        help="if is_1d_periodic")
    parser.add_argument('--is_normalize_pos', type=eval,
                        help="if is_normalize_pos")
    parser.add_argument('--fine_tune_gt_input', type=eval,
                        help="if fine_tune_gt_input")
    parser.add_argument('--soft_update', type=eval,
                        help="if soft_update")
    parser.add_argument('--share_processor', type=eval,
                        help="if share_processor")

    parser.add_argument('--policy_input_feature',type=str,
                        help="input feature for policy network")
    parser.add_argument('--load_hash',type=str,
                        help="load hash")
    parser.add_argument('--test_reward_random_sample', type=eval,
                        help="if test_reward_random_sample")
    parser.add_argument('--processor_aggr', type=str,
                        help="if max")
    parser.add_argument('--fix_alt_evolution_model', type=eval,
                        help="if fix_alt_evolution_model")

    parser.set_defaults(
        rl_coefs="None",
        rl_horizon=4,
        reward_mode="lossdiff+statediff",
        reward_beta="0.5",
        reward_loss_coef=5,
        reward_src="env",
        rl_lambda=0.95,
        rl_gamma=0.99,
        rl_rho=1.,
        rl_eta=1e-4,
        rl_critic_update_iterations=100,
        rl_data_dropout="None",
        rl_is_finetune_evolution=False,
        rl_finetune_evalution_mode=None,
        actor_critic_step=None,
        evolution_steps=None,
        rl_is_alt_remeshing=False,
        
        is_single_action=False,
        top_k_action=1,

        actor_lr=5e-4,
        actor_batch_norm=False,
        skip_coarse=False,
        skip_split=False,
        skip_flip=False,
        evl_stop_gradient=False,

        value_latent_size=32,
        value_num_pool=1,
        value_act_name="elu",
        value_act_name_final="linear",
        value_layer_norm=False,
        value_batch_norm=False,
        value_num_steps=3,
        value_pooling_type="global_mean_pool",
        value_lr=1e-4,
        value_loss_type="mse",
        value_loss_coef=0.5,
        value_target_mode="value-lambda",
        test_value_model=False,
        use_reward_vanilla=False,
        
        is_alternating_train=False,
        is_eval_sample=True,
        is_1d_periodic=False,
        is_normalize_pos=True,
        value_steps=None,
        actor_steps=None,
        test_data_interp=False,
        
        offset_coarse=0.,
        offset_split=0.,
        reward_condition=False,
        kaction_pooling_type="global_mean_pool",
        max_action=5,
        opt_evl_horizon=-1,
        connect_bdd=False,
        stop_all_gradient=True,

        debug=False,
        fine_tune_gt_input=False,
        soft_update=False,
        policy_input_feature="velocity",

        test_reward_random_sample=False,
        share_processor=True,
        processor_aggr="max",
        fix_alt_evolution_model=False,
        load_hash="None",
    )