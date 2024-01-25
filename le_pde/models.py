import datetime
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import pdb
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch_geometric.nn.inits import reset
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from tqdm import tqdm
import matplotlib
import math

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append("GraphPDE")

from cindm.le_pde.datasets.load_dataset import load_data
from cindm.le_pde.pytorch_net.util import get_repeat_interleave, forward_Runge_Kutta, tuple_add, tuple_mul, to_np_array, record_data, ddeepcopy as deepcopy, Attr_Dict, set_seed, pdump, pload, get_time, check_same_model_dict, print_banner, to_string
from cindm.le_pde.utils import PyG_to_Attr_Dict, SpectralNorm, SpectralNormReg, requires_grad, process_data_for_CNN, get_regularization, get_batch_size, get_Hessian_penalty
from cindm.le_pde.utils import detach_data, get_model_dict, loss_op_core, MLP, get_keys_values, flatten, get_elements, get_activation, to_cpu, to_tuple_shape, parse_multi_step, parse_act_name, parse_reg_type, loss_op, get_normalization, get_edge_index_kernel, loss_hybrid, stack_tuple_elements, add_noise, get_neg_loss, get_pos_dims_dict
from cindm.le_pde.utils import p, seed_everything, is_diagnose, get_precision_floor, parse_string_idx_to_list, parse_loss_type, get_loss_ar, get_max_pool, get_data_next_step, get_LCM_input_shape, expand_same_shape, Sum, Mean, Channel_Gen, Flatten, Permute, Reshape, add_data_noise 
from cindm.GraphPDE.gnn_module import mesh_PDE
from torch_geometric.data import Data

def get_conv_func(pos_dim, *args, **kwargs):
    if "reg_type_list" in kwargs:
        reg_type_list = kwargs.pop("reg_type_list")
    else:
        reg_type_list = None
    if pos_dim == 1:
        conv = nn.Conv1d(*args, **kwargs)
    elif pos_dim == 2:
        conv = nn.Conv2d(*args, **kwargs)
    elif pos_dim == 3:
        conv = nn.Conv3d(*args, **kwargs)
    else:
        raise Exception("The pos_dim can only be 1, 2 or 3!")
    if reg_type_list is not None:
        if "snn" in reg_type_list:
            conv = SpectralNorm(conv)
        elif "snr" in reg_type_list:
            conv = SpectralNormReg(conv)
    return conv


def get_conv_trans_func(pos_dim, *args, **kwargs):
    if "reg_type_list" in kwargs:
        reg_type_list = kwargs.pop("reg_type_list")
    else:
        reg_type_list = None
    if pos_dim == 1:
        conv_trans = nn.ConvTranspose1d(*args, **kwargs)
    elif pos_dim == 2:
        conv_trans = nn.ConvTranspose2d(*args, **kwargs)
    elif pos_dim == 3:
        conv_trans = nn.ConvTranspose3d(*args, **kwargs)
    else:
        raise Exception("The pos_dim can only be 1, 2 or 3!")
     # The weight's output dim=1 for ConvTranspose
    if reg_type_list is not None:
        if "snn" in reg_type_list:
            conv_trans = SpectralNorm(conv_trans, dim=1)
        elif "snr" in reg_type_list:
            conv_trans = SpectralNormReg(conv_trans, dim=1)
    return conv_trans



class Contrastive(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        latent_size,
        encoder_type,
        evolution_type,
        decoder_type,
        input_shape,
        grid_keys,
        part_keys,
        no_latent_evo=False,
        temporal_bundle_steps=1,
        forward_type="Euler",
        channel_mode="exp-16",
        kernel_size=4,
        stride=2,
        padding=1,
        padding_mode="zeros",
        output_padding_str="None",
        encoder_mode="dense",
        encoder_n_linear_layers=0,
        act_name="rational",
        decoder_last_act_name="linear",
        is_pos_transform=False,
        normalization_type="bn2d",
        cnn_n_conv_layers=2,
        is_latent_flatten=True,
        reg_type="None",
        n_conv_blocks=4,
        n_latent_levs=1,
        # Evolution_op specific:
        n_conv_layers_latent=1,
        evo_conv_type="cnn",
        evo_pos_dims=-1,
        evo_inte_dims=-1,
        evo_groups=1,
        loss_type=None,
        static_latent_size=0,
        static_encoder_type="None",
        #static_axis=0,
        static_input_size={"n0": 0},
        decoder_act_name="None",
        is_prioritized_dropout=False,
        vae_mode="None",
    ):
        super(Contrastive, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.encoder_type = encoder_type
        self.evolution_type = evolution_type
        self.decoder_type = decoder_type
        #pdb.set_trace()
        self.static_latent_size = static_latent_size
        self.static_encoder_type = static_encoder_type
        self.encoder_n_linear_layers = encoder_n_linear_layers
        self.is_latent_flatten = is_latent_flatten
        self.encoder_mode = encoder_mode
        self.grid_keys = grid_keys
        self.part_keys = part_keys
        self.no_latent_evo = no_latent_evo
        self.temporal_bundle_steps = temporal_bundle_steps
        self.forward_type = forward_type
        self.channel_mode = channel_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding_str = output_padding_str
        self.act_name = act_name
        self.decoder_last_act_name = decoder_last_act_name
        self.is_pos_transform = is_pos_transform
        self.normalization_type = normalization_type
        self.normalization_n_groups = 2
        self.cnn_n_conv_layers = cnn_n_conv_layers
        self.input_shape = input_shape
        self.n_conv_blocks = n_conv_blocks
        self.n_latent_levs = n_latent_levs
        # Evolution_op specific:
        self.n_conv_layers_latent = n_conv_layers_latent
        self.evo_conv_type = evo_conv_type
        self.evo_pos_dims = evo_pos_dims
        self.evo_inte_dims = evo_inte_dims
        self.evo_groups = evo_groups
        self.loss_type = loss_type
        self.static_input_size = static_input_size
        self.decoder_act_name = decoder_act_name
        self.is_prioritized_dropout = is_prioritized_dropout
        self.vae_mode = vae_mode
        if vae_mode != "None":
            assert is_latent_flatten is True
        if decoder_act_name is None or decoder_act_name == "None":
            decoder_act_name = act_name

        self.reg_type = reg_type
        reg_type_list = parse_reg_type(self.reg_type)
        encoder_list = []
        if self.encoder_type == "cnn":
            pdb.set_trace()
            self.encoder = CNNEncoder(
                in_channels=input_size,
                out_channels=latent_size,
                n_conv_layers=cnn_n_conv_layers,
                encoder_mode=encoder_mode,
                init_channel_number=32,
                input_shape=input_shape,
                act_name=act_name,
                kernel_size=kernel_size,
                padding_size=1,
                padding_mode=padding_mode,
                dilation_type="None",
                dilation_base=2,
            )
        elif self.encoder_type == "cnn-s":
            self.encoder = CNN_Encoder(
                in_channels=input_size,
                output_size=latent_size,
                input_shape=input_shape,
                grid_keys=grid_keys,
                part_keys=part_keys,
                channel_mode=channel_mode,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                last_n_linear_layers=self.encoder_n_linear_layers,
                act_name=act_name,
                normalization_type=normalization_type,
                n_conv_blocks=self.n_conv_blocks,
                n_latent_levs=self.n_latent_levs,
                is_latent_flatten=self.is_latent_flatten,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evoenc"]],
                vae_mode=self.vae_mode,
            )
        elif self.encoder_type == "hybrid":
            self.encoder = Hybrid(
                input_size=input_size,
                output_size=latent_size,
                input_shape=input_shape,
                grid_keys=grid_keys,
                part_keys=part_keys,
                channel_mode=self.channel_mode,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                act_name=act_name,
                normalization_type=normalization_type,
                last_n_linear_layers=self.encoder_n_linear_layers,
                n_conv_blocks=self.n_conv_blocks,
                n_latent_levs=self.n_latent_levs,
                is_latent_flatten=self.is_latent_flatten,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evoenc"]],
            )
        elif self.encoder_type == "cnn-VL":
            self.encoder = Vlasov_Encoder(
                input_size=input_size,
                output_size=latent_size,
                input_shape=input_shape,
                n_conv_blocks=self.n_conv_blocks,
                act_name=act_name,
                normalization_type=self.normalization_type,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evoenc"]],
            )
        elif self.encoder_type.startswith("VL-u"):
            self.encoder = Vlasov_U_Encoder(
                model_type=encoder_type,
                input_size=input_size,
                output_size=latent_size,
                input_shape=input_shape,
                n_conv_blocks=self.n_conv_blocks,
                padding_mode=padding_mode,
                act_name=act_name,
                normalization_type=self.normalization_type,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evoenc"]],
            )
        else:
            raise Exception("encoder_type {} is not valid!".format(self.encoder_type))

        if self.static_encoder_type == "cnn-s":
            
            # pdb.set_trace()
            assert not (self.static_latent_size == 0 or self.static_input_size["n0"] == 0)
            self.static_encoder = CNN_Encoder(
                in_channels=static_input_size,
                output_size=static_latent_size,
                input_shape=input_shape,
                grid_keys=grid_keys,
                part_keys=part_keys,
                channel_mode=channel_mode,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                last_n_linear_layers=self.encoder_n_linear_layers,
                act_name=act_name,
                normalization_type=normalization_type,
                n_conv_blocks=self.n_conv_blocks,
                n_latent_levs=self.n_latent_levs,
                is_latent_flatten=self.is_latent_flatten,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evoenc"]],
            )
        elif self.static_encoder_type.startswith("param"):
            assert not (self.static_latent_size == 0 or self.static_input_size["n0"] == 0)
            if len(static_encoder_type.split("-")) == 3:
                string, static_encoder_n_layers, static_encoder_act_name = static_encoder_type.split("-")
            else:
                string, static_encoder_n_layers = static_encoder_type.split("-")
                static_encoder_act_name = act_name
            if static_encoder_n_layers == "expand":
                self.static_encoder = None
            else:
                static_encoder_n_layers = int(static_encoder_n_layers)
                if static_encoder_n_layers == 0:
                    if static_latent_size == static_input_size["n0"]:
                        self.static_encoder = nn.Identity()
                    else:
                        self.static_encoder = get_repeat_interleave(
                            input_size=static_input_size["n0"],
                            output_size=static_latent_size,
                            dim=-1,
                        )
                else:
                    self.static_encoder = MLP(
                        input_size=static_input_size["n0"],
                        n_layers=static_encoder_n_layers,
                        n_neurons=static_latent_size,
                        output_size=static_latent_size,
                        act_name=static_encoder_act_name,
                    )

        # Evolution operator:
        self.evolution_op = Evolution_Op(
            evolution_type=self.evolution_type,
            latent_size=self.latent_size,
            pos_dims=get_pos_dims_dict(self.input_shape),
            normalization_type=self.normalization_type,
            normalization_n_groups=self.normalization_n_groups,
            n_latent_levs=self.n_latent_levs,
            n_conv_layers_latent=self.n_conv_layers_latent,
            evo_conv_type=self.evo_conv_type,
            evo_pos_dims=self.evo_pos_dims,
            evo_inte_dims=self.evo_inte_dims,
            evo_groups=evo_groups,
            channel_size_dict=self.encoder.channel_dict,
            padding_mode=padding_mode,
            act_name=self.act_name,
            is_latent_flatten=is_latent_flatten,
            reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evoenc", "evo"]],
            static_latent_size=self.static_latent_size,
            is_prioritized_dropout=self.is_prioritized_dropout,
        )
        if self.evo_conv_type.startswith("VL-u"):
            pass
#             assert self.evolution_op.evolution_op1.model_version == self.encoder.model_version

        self.is_single_decoder = True
        if self.decoder_type.startswith("mixGau"):
            Gaussian_mode = self.decoder_type.split("-")[1]
            n_components = eval(self.decoder_type.split("-")[2])
            self.decoder = Mixture_Gaussian_model(
                latent_size=latent_size,
                output_size=output_size,
                n_components=n_components,
                Gaussian_mode=Gaussian_mode,
                MLP_n_neurons=32,
                MLP_n_layers=2,
                act_name=act_name,
                is_pos_transform=is_pos_transform,
            )
        elif self.decoder_type == "cnn-tr-hybrid":
            self.decoder = CNN_Decoder_Hybrid(
                latent_size=latent_size,
                latent_shape=self.encoder.latent_shape,
                output_size=output_size,
                output_shape=dict(input_shape),
                fc_output_dim=self.encoder.flat_fts,
                channel_mode=self.channel_mode,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="zeros",
                act_name=act_name,
                last_act_name=self.decoder_last_act_name,
                normalization_type=normalization_type,
                n_conv_blocks=self.n_conv_blocks,
                n_latent_levs=self.n_latent_levs,
                is_latent_flatten=self.is_latent_flatten,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evodec"]],
            )
        elif self.decoder_type == "cnn-tr-VL":
            self.decoder = Vlasov_Decoder_Hybrid(
                latent_size=latent_size,
                latent_shape=self.encoder.latent_shape,
                output_size=output_size,
                output_shape=dict(input_shape),
                flat_sizes=self.encoder.flat_sizes,
                conv_lat_sizes=self.encoder.conv_lat_sizes,
                act_name=act_name,
                normalization_type=self.normalization_type,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evodec"]],
            )
        elif self.decoder_type.startswith("VL-u"):
            self.decoder = Vlasov_U_Decoder(
                model_type=decoder_type,
                latent_size=latent_size,
                latent_shape=self.encoder.latent_shape,
                output_size=output_size,
                output_shape=dict(input_shape),
                fc_output_dim=self.encoder.flat_fts,
                act_name=act_name,
                normalization_type=self.normalization_type,
                reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evodec"]],
            )
        elif self.decoder_type == "cnn-tr":
            self.is_single_decoder = False
            for key in output_size:
                setattr(self, f"decoder_{key}", CNN_Decoder(
                    latent_size=latent_size,
                    latent_shape=self.encoder.latent_shape,
                    output_size=output_size[key],
                    output_shape=dict(input_shape)[key if key in self.grid_keys else self.grid_keys[0]],
                    fc_output_dim=self.encoder.flat_fts,
                    temporal_bundle_steps=self.temporal_bundle_steps,
                    channel_mode=self.channel_mode,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_mode="zeros",
                    output_padding_str=output_padding_str,
                    act_name=act_name,
                    normalization_type=normalization_type,
                    n_conv_blocks=self.n_conv_blocks,
                    n_latent_levs=self.n_latent_levs,
                    is_latent_flatten=self.is_latent_flatten,
                    reg_type_list=[reg_type_core for reg_type_core, reg_target in reg_type_list if reg_target in ["all", "evodec"]],
                    decoder_act_name=decoder_act_name,
                ))
        elif self.decoder_type.startswith("neural-basis"):
            decoder_type_split = self.decoder_type.split("-")
            coupling_mode = "concat" if len(decoder_type_split) == 2 else decoder_type_split[2]
            freq_order = 6 if len(decoder_type_split) <= 3 else int(decoder_type_split[3])
            n_layers = 4 if len(decoder_type_split) <= 4 else int(decoder_type_split[4])
            assert self.is_latent_flatten == True
            self.is_single_decoder = False
            for key in output_size:
                if decoder_act_name == "siren":
                    is_pos_encoding = False
                else:
                    is_pos_encoding = True
                setattr(self, f"decoder_{key}", NeuralBasis(
                    x_size=len(dict(input_shape)[key]),
                    z_size=latent_size,
                    n_neurons=64,
                    n_layers=n_layers,
                    output_size=len(input_size),
                    act_name=decoder_act_name,
                    is_z_x=False,
                    is_pos_encoding=is_pos_encoding,
                    freq_order=freq_order,
                    is_freeze_basis=False,
                    coupling_mode=coupling_mode,
                ))
        else:
            raise Exception("decoder_type {} is not valid!".format(self.decoder_type))

    def requires_grad(self, is_requires_grad, targets):
        """
        Args:
            is_requires_grad: True/False
            target: list subset of ["encoder", "static-encoder", "evolution", "decoder"]
        """
        if not isinstance(targets, list):
            targets = [targets]
        for target in targets:
            if target == "encoder":
                requires_grad(self.encoder.parameters(), is_requires_grad)
            elif target == "static-encoder":
                if hasattr(self, "static_encoder"):
                    requires_grad(self.static_encoder.parameters(), is_requires_grad)
            elif target == "evolution":
                requires_grad(self.evolution_op.parameters(), is_requires_grad)
            elif target == "decoder":
                if hasattr(self, "decoder"):
                    requires_grad(self.decoder.parameters(), is_requires_grad)
                else:
                    for key in self.output_size:
                        requires_grad(getattr(self, f"decoder_{key}").parameters(), is_requires_grad)
            else:
                raise

    def set_input_shape(self, input_shape):
        """Update the input_shape."""
        self.input_shape = input_shape
        self.encoder.input_shape = input_shape
        if self.is_single_decoder:
            self.output_shape = dict(input_shape)
        else:
            for key in output_size:
                getattr(self, f"decoder_{key}").output_shape = dict(input_shape)[key if key in self.grid_keys else self.grid_keys[0]]

    def evolve_latent(self, latent):
        """Evolve latent using residual connection."""
        if self.forward_type == "direct":
            return self.evolution_op(latent)
        elif self.forward_type == "Euler":
            if self.static_encoder_type != "None":
                out_latent = self.evolution_op(latent)
                if isinstance(latent, tuple):
                    latent_dynamic = tuple(latent_ele[:,:out_latent[jj].shape[1]] if latent_ele is not None else None for jj, latent_ele in enumerate(latent))
                    out = tuple_add(latent_dynamic, out_latent)
                else:
                    out = tuple_add(latent[:,:out_latent.shape[1]], out_latent)
            else:
                out = tuple_add(latent, self.evolution_op(latent))
            return out
        elif self.forward_type.startswith("RK"):
            return forward_Runge_Kutta(self.evolution_op, latent, mode=self.forward_type)
        else:
            raise Exception("forward_type '{}' is not valid!".format(self.forward_type))

    def get_latent_targets(self, data, latent_pred_steps, temporal_bundle_steps, use_grads=True, use_pos=False):
        """Get the latent representation of future targets.
        Args:
            data: data.
            latent_pred_steps: a list of latent pred steps. E.g. at current time t, an element of 1 means that will
                get the latent target for next-step's input (t+1)
            temporal_bundle_steps: temporal bundling, default 1.
            use_grads: if True, will augment the data with derivative w.r.t. spatial directions.
            use_pos: if True, will augment the data with normalized position on the grid.

        Returns:
            latent_targets: has shape of [B, max_pred_steps, latent_size]
        """
        def get_future_data(data, k, temporal_bundle_steps):
            """Get the input data for the k'th step in the future.

            If temporal_bundle_steps > 1, then each k will include {temporal_bundle_steps} number of steps
            """
            dyn_dims_dict = dict(to_tuple_shape(data.dyn_dims))
            compute_func_dict = dict(to_tuple_shape(data.compute_func))
            static_dims_dict = {key: data.node_feature[key].shape[-1] - dyn_dims_dict[key] - compute_func_dict[key][0] for key in data.node_feature}
            data_k = deepcopy(data)
            for key in data.node_feature:
                dynamic_input_list = []
                input_steps_full = data.node_feature[key].shape[-2]
                assert input_steps_full % temporal_bundle_steps == 0
                input_steps_effective = input_steps_full // temporal_bundle_steps
                y_idx_list = np.arange((k-1)*temporal_bundle_steps, k*temporal_bundle_steps).tolist()
                dynamic_features = data.node_label[key][:, y_idx_list]  # [n_nodes, temporal_bundle_steps, dyn_dims]
                static_features = data.node_feature[key][:, -1:, -static_dims_dict[key]-dyn_dims_dict[key]:-dyn_dims_dict[key]]  # [n_nodes, 1, static_dims]
                if input_steps_full > 1:
                    static_features = static_features.expand(static_features.shape[0], input_steps_full, static_features.shape[-1])  # [n_nodes, args.input_steps*temporal_bundle_steps, static_dims]
                dynamic_input_list.append(dynamic_features)

                start_effective = k - input_steps_effective  # k = 1, input_steps_effective = 2
                start_effective_nonneg = max(0, k - input_steps_effective)
                if k - 1 > 0:
                    start_effective_idx_list = np.arange(start_effective_nonneg * temporal_bundle_steps, (k-1)*temporal_bundle_steps).tolist()
                    prev_label_dynamic = data.node_label[key][:, start_effective_idx_list]
                    dynamic_input_list.insert(0, prev_label_dynamic)
                if start_effective < 0:
                    prev_node_feature_idx = np.arange(start_effective * temporal_bundle_steps, 0).tolist()
                    prev_node_feature_dynamic = data.node_feature[key][:, prev_node_feature_idx, -dyn_dims_dict[key]:]
                    dynamic_input_list.insert(0, prev_node_feature_dynamic)
                dynamic_input_list = torch.cat(dynamic_input_list, 1)  # [n_nodes, input_steps*temporal_bundle_steps, dyn_dims]

                compute_dims = compute_func_dict[key][0]
                if compute_dims > 0:
                    compute_features = compute_func_dict[key][1](dynamic_input_list)
                    node_features = torch.cat([compute_features, static_features, dynamic_input_list], -1)
                else:
                    node_features = torch.cat([static_features, dynamic_input_list], -1)  # [n_nodes, temporal_bundle_steps, static_dims+dyn_dims]
                
                data_k.node_feature[key] = node_features
            return data_k

        latent_targets = []
        for k in range(1, max(latent_pred_steps + [0]) + 1):
            data_k = get_future_data(data, k, temporal_bundle_steps=temporal_bundle_steps)
            latent_target_k = self.encoder(data_k, use_grads=use_grads, use_pos=use_pos)  # [B, latnet_size]
            if self.vae_mode != "None":
                latent_target_k = latent_target_k[0]
            if k in latent_pred_steps:
                latent_targets.append(latent_target_k)  # [(z11, z12, ...), (z21, z22, ...)]
        if len(latent_targets) > 0:
            if not isinstance(latent_targets[0], tuple):
                latent_targets = torch.stack(latent_targets, 1)
            else:
                latent_targets = stack_tuple_elements(latent_targets, 1)  # [(z11, z12, ...), (z21, z22, ...)] -> (torch.stack([z11, z21, ...], 1), torch.stack([z12, z22, ...], 1))
        return latent_targets


    def get_reg(self, reg_type):
        """Get regularization."""
        reg_type_list = parse_reg_type(reg_type)
        reg_sum = 0
        for reg_type_core, reg_target in reg_type_list:
            if reg_type_core == "None":
                reg = 0
            else:
                # Collect models:
                model_list = []
                if reg_target == "evo":
                    model_list.append(self.evolution_op)
                elif reg_target == "all":
                    model_list += [self.evolution_op, self.encoder]
                    if self.is_single_decoder:
                        model_list.append(self.decoder)
                    else:
                        for key in self.output_size:
                            model_list.append(getattr(self, f"decoder_{key}"))
                elif reg_target == "evoenc":
                    model_list += [self.evolution_op, self.encoder]
                elif reg_target == "evodec":
                    model_list += [self.evolution_op]
                    if self.is_single_decoder:
                        model_list.append(self.decoder)
                    else:
                        for key in self.output_size:
                            model_list.append(getattr(self, f"decoder_{key}"))
                else:
                    raise Exception("reg_target {} is not supported! Choose from 'evo' or 'all'.".format(reg_target))
                # Get regularization:
                reg = get_regularization(model_list, reg_type_core)
            reg_sum = reg_sum + reg
        return reg_sum


    def forward_nolatent(
        self,
        data,
        use_grads=True,
        use_pos=False,
    ):
        """Make a forward step without latent evolution."""
        # Encode:
        latent = self.encoder(data, use_grads=use_grads, use_pos=use_pos)  # [B, latent_size]
        # Decode:
        if self.is_single_decoder:
            pred = self.decoder(latent)
        else:
            pred = {key: getattr(self, f"decoder_{key}")(latent) for key in self.output_size}
        return pred, {}

    def forward(
        self,
        data,
        pred_steps=1,
        latent_pred_steps=None,
        is_recons=False,
        use_grads=True,
        is_y_diff=False,
        reg_type="None",
        use_pos=False,
        latent_noise_amp=0,
        is_rollout=False,
        static_data=None,
    ):
        """Predict one or multiple steps into the future using latent evolution.
            If self.no_latent_evo is True, then will not perform latent evolution. Instead,
            will use the decoder to directly predict the output at the next time step.

        Args:
            data: Deepsnap Data instance
            pred_steps: a list of predicting steps.
            latent_pred_steps: a list of latent predicting steps. E.g. at current time t, an element of 1 means
                that will predict the latent at time t+1.
            is_recons: if True, will also return the reconstructed input
            use_grads: if True, will augment the data with gradient w.r.t. rows and columns.
            use_pos: if True, will augment the data.x with normalized position in the grid.
            is_rollout: if True, will transform the output to the original representation of the input space.
                E.g. if self.loss_type contains mselog, will do exp(pred) - precision_floor.

        Returns:
            preds: having format of {key: [B, len(pred_steps), dyn_dims]}}.
            If is_recons is True, will also return recons that has format of {key: [B, 1, dyn_dims]}}.
        """
        def expand_static_latent(static_latent, latent):
            if isinstance(latent, tuple):
                return tuple(expand_static_latent(static_latent, latent_ele) if latent_ele is not None else None for latent_ele in latent)
            for i in range(len(latent.shape) - len(static_latent.shape)):
                static_latent = static_latent[...,None]
            static_latent = static_latent.expand(*static_latent.shape[:2], *latent.shape[2:])  # [B, C, (H, W, ...)]
            return static_latent

        # Reshape x_pos:
        info = {}
        if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
            pred_steps = [pred_steps]
        if latent_pred_steps is None:
            latent_pred_steps = pred_steps
        if not isinstance(latent_pred_steps, list) and not isinstance(latent_pred_steps, np.ndarray):
            latent_pred_steps = [latent_pred_steps]

        max_pred_step = max(pred_steps + [0])
        max_latent_pred_step = max(latent_pred_steps + [0])
        original_shape = dict(to_tuple_shape(data.original_shape))
        n_pos = np.array(original_shape[self.grid_keys[0]]).prod()
        if hasattr(data, "node_pos") and use_pos:
            batch_size = data.node_feature[self.grid_keys[0]].shape[0] // n_pos
            node_pos_item = data.node_pos[0][0] if isinstance(data.node_pos[0], list) or isinstance(data.node_pos[0], tuple) else data.node_pos[0]
            x_pos = {key: node_pos_item[key].reshape(1, -1, len(original_shape[key if key in self.grid_keys else self.grid_keys[0]])).repeat_interleave(repeats=batch_size, dim=0).to(data.node_feature[key].device) for key in self.output_size}  #  [B, n_grid: prod(input_shape), pos_dim: len(input_shape)]
        else:
            x_pos = None

        # Compute regularization:
        info["reg"] = self.get_reg(reg_type)

        # Compute loss:
        if self.no_latent_evo:
            if len(pred_steps) == 1 and max_pred_step == 1:
                # Single-step prediction:
                preds, _ = self.forward_nolatent(data, use_grads=use_grads)
            else:
                # Multi-step prediction:
                dyn_dims = dict(to_tuple_shape(data.dyn_dims))
                preds = {}
                for k in range(1, max_pred_step + 1):
                    if k != max_pred_step:
                        data, pred = get_data_next_step(self, data, forward_func_name="forward_nolatent",
                                                        use_grads=use_grads, is_y_diff=is_y_diff, return_data=True, is_rollout=is_rollout)
                    else:
                        _, pred = get_data_next_step(self, data, forward_func_name="forward_nolatent",
                                                     use_grads=use_grads, is_y_diff=is_y_diff, return_data=False, is_rollout=is_rollout)
                    if k in pred_steps:
                        record_data(preds, list(pred.values()), list(pred.keys()))
                if len(preds) > 0:
                    for key in self.output_size:
                        preds[key] = torch.cat(preds[key], 1)
        else:
            # Encode:
            latent = self.encoder(data, use_grads=use_grads, use_pos=use_pos)  # [B, latent_size]
            if self.vae_mode != "None":
                assert len(latent) == 2
                info["latent_loc"] = latent[0]
                info["latent_logscale"] = latent[1]
                if self.training:
                    latent_recons = latent[0] + torch.exp(latent[1]) * torch.randn_like(latent[1])
                else:
                    latent_recons = latent[0]
                if self.vae_mode == "recons":
                    latent_forward = latent[0]
                elif self.vae_mode == "recons+sample":
                    if self.training:
                        latent_forward = latent_recons
                    else:
                        latent_forward = latent[0]
                else:
                    raise
            else:
                latent_recons = latent_forward = latent
            if self.static_encoder_type != "None":
                # pdb.set_trace()
                if self.static_encoder_type.startswith("param"):
                    if self.static_encoder_type.startswith("param-expand"):
                        static_latent = data.param["n0"]
                        static_latent = expand_static_latent(static_latent, latent_forward)
                    else:
                        if static_data is None:
                            static_data = data.param["n0"]
                            static_latent = self.static_encoder(static_data)
                        else:
                            static_latent = self.static_encoder(static_data)
                else:
                    if static_data is None:
                        # pdb.set_trace()
                        static_data = deepcopy(data)
                        static_dims = data.node_feature["n0"].shape[-1] - dict(to_tuple_shape(data.dyn_dims))["n0"] - dict(to_tuple_shape(data.compute_func))["n0"][0]
                        static_feature = data.node_feature["n0"][:,:,:static_dims]
                        static_data.node_feature["n0"] = static_feature
                        static_latent = self.static_encoder(static_data, use_grads=use_grads, use_pos=use_pos)
                    else:
                        static_latent = self.static_encoder(static_data, use_grads=use_grads, use_pos=use_pos)

            info["latent"] = latent_forward
            # Reconstruct:
            if is_recons:
                if hasattr(self, "decoder"):
                    recons = self.decoder(latent_recons)
                else:
                    recons = {key: getattr(self, f"decoder_{key}")(latent_recons, x_pos=x_pos[key] if x_pos is not None else None) for key in self.output_size}
            # Prediction:
            info["latent_preds"] = []
            preds = {key: [] for key in self.output_size}
            for k in range(1, max(max_pred_step, max_latent_pred_step) + 1):
                if self.training and latent_noise_amp > 0:
                    latent_forward = add_noise(latent_forward, latent_noise_amp)
                # latent: [B, latent_size]
                if self.static_encoder_type != "None":
                    if self.n_latent_levs == 1:
                        # pdb.set_trace()
                        latent_forward = torch.cat([latent_forward, static_latent], -1)
                    else:
                        latent_forward = tuple(torch.cat([latent_ele, static_latent_ele], 1) if latent_ele is not None else None for latent_ele, static_latent_ele in zip(latent_forward, static_latent))
                        # raise Exception("Boundary concatenation is not implemented for n_latent_levs > 1")
                latent_forward = self.evolve_latent(latent_forward)
                if k in latent_pred_steps:
                    info["latent_preds"].append(latent_forward)
                if k in pred_steps:
                    if self.is_single_decoder:
                        pred = self.decoder(latent_forward, x_pos=x_pos)
                        for key in self.output_size:
                            preds[key].append(pred[key])
                    else:
                        for key in self.output_size:
                            preds[key].append(getattr(self, f"decoder_{key}")(latent_forward, x_pos=x_pos[key] if x_pos is not None else None))
            for key in self.output_size:
                if len(preds[key]) > 0:
                    preds[key] = torch.cat(preds[key], 1)
            if len(info["latent_preds"]) > 0:
                if not isinstance(info["latent_preds"][0], tuple):
                    info["latent_preds"] = torch.stack(info["latent_preds"], 1)  # [B, max_pred_steps, latent_size]
                else:
                    info["latent_preds"] = stack_tuple_elements(info["latent_preds"], dim=1)
            # Returns:
            if is_recons:
                info["recons"] = recons
        if is_rollout:
            """Go to original representation."""
            info["input"] = deepcopy(data.node_feature)
            precision_floor = get_precision_floor(self.loss_type)
            if self.loss_type is not None and precision_floor is not None:
                preds_core = {}
                if is_recons and "recons" in info:
                    recons_core = {}
                for loss_type_key in self.loss_type.split("^"):
                    key = loss_type_key.split(":")[0]
                    if "mselog" in loss_type_key or "huberlog" in loss_type_key or "l1log" in loss_type_key:
                        if len(preds) > 0 and len(preds[key]) > 0:
                            preds_core[key] = torch.exp(preds[key]) - precision_floor
                        if is_recons and "recons" in info:
                            recons_core[key] = torch.exp(info["recons"][key]) - precision_floor
                    else:
                        if len(preds) > 0:
                            preds_core[key] = preds[key]
                        if is_recons and "recons" in info:
                            recons_core[key] = info["recons"][key]
                preds = preds_core
                if is_recons and "recons" in info:
                    info["recons"] = recons_core
        return preds, info

    def get_loss(self, data, args, is_rollout=False, **kwargs):
        """Get loss."""
        # Make prediction:
        if is_diagnose(loc="loss:0", filename=args.filename):
            pdb.set_trace()
        multi_step_dict = parse_multi_step(args.multi_step)
        latent_multi_step_dict = parse_multi_step(args.latent_multi_step) if args.latent_multi_step is not None else multi_step_dict
        if args.consistency_coef > 0 or args.contrastive_rel_coef > 0:
            data_copy_cons = deepcopy(data)
        self.info = {}
        if args.loss_type == "lp" or args.is_y_variable_length:
            original_shape = dict(to_tuple_shape(data.original_shape))
            n_pos = np.array(original_shape[self.grid_keys[0]]).prod()
            batch_size = data.node_feature[self.grid_keys[0]].shape[0] // n_pos
        else:
            batch_size = args.batch_size

        # Compute prediction:
        """Only set is_rollout=True if self.loss_type contains e.g. 'mselog' but the args.loss_type does not contain:"""
        precision_floor_self = get_precision_floor(self.loss_type)
        precision_floor_args = get_precision_floor(args.loss_type)
        if precision_floor_self is not None and precision_floor_args is None:
            is_rollout_core = is_rollout
        else:
            is_rollout_core = False
        # Perform prediction:
        preds, info = self(
            data,
            pred_steps=list(multi_step_dict.keys()),
            latent_pred_steps=list(latent_multi_step_dict.keys()),
            is_recons=True if args.recons_coef > 0 else False,
            use_grads=args.use_grads,
            is_y_diff=args.is_y_diff,
            use_pos=args.use_pos,
            latent_noise_amp=args.latent_noise_amp,
            reg_type=args.reg_type if args.reg_coef > 0 else "None",
            is_rollout=is_rollout_core,
        )
        if is_diagnose(loc="loss:1", filename=args.filename):
            pdb.set_trace()

        # Compute main losses:
        if self.no_latent_evo:
            # Prediction loss:
            loss = 0
            for pred_idx, k in enumerate(multi_step_dict):
                loss_k = loss_op(
                    preds, data.node_label, data.mask,
                    pred_idx=pred_idx,
                    y_idx=k-1,
                    loss_type=args.loss_type,
                    keys=self.grid_keys,
                    batch_size=batch_size,
                    is_y_variable_length=args.is_y_variable_length,
                    **kwargs
                )
                loss = loss + loss_k
            self.info["loss_pred"] = to_np_array(loss)
        else:
            # Prediction loss:
            if args.loss_type == "lfm":
                """
                Latent Field Model, from "Learning latent field dynamics of PDEs", Kochkov et al. 2020.
                """
                # (1) Reconstruction loss:
                y_idx_recons = np.arange(
                    data.node_feature[list(data.node_feature)[0]].shape[-2] - args.temporal_bundle_steps, 
                    data.node_feature[list(data.node_feature)[0]].shape[-2]).tolist()
                loss_recons = loss_op(
                    info["recons"], data.node_feature, data.mask,
                    y_idx=y_idx_recons,
                    dyn_dims=dict(to_tuple_shape(data.dyn_dims)),
                    loss_type="mse",
                    keys=self.grid_keys,
                    batch_size=batch_size,
                    is_y_variable_length=args.is_y_variable_length,
                    **kwargs
                )
                self.info["loss_recons"] = loss_recons.item()

                # Compute difference on latent and input:
                latent_preds = info["latent_preds"]  # [B, pred_steps, C, ...]
                if isinstance(latent_preds, tuple):
                    assert latent_preds[-1].shape[1] == 1  # pred_steps==1
                    latent_diff = tuple(latent_pred_ele[:,0] - latent_ele if latent_ele is not None else None for latent_pred_ele, latent_ele in zip(latent_preds, info["latent"]))
                else:
                    assert latent_preds.shape[1] == 1, "For lfm loss, the pred_steps must be 1!"
                    latent_diff = latent_preds[:,0] - info["latent"]  # [B, C]
                node_feature = deepcopy(data.node_feature["n0"])
                input_steps = node_feature.shape[-2]
                node_feature_new = torch.cat([node_feature, data.node_label["n0"]], -2)[...,-input_steps:,:]
                node_feature_diff = node_feature_new - node_feature

                # (2) Loss for latent:
                data_copy = deepcopy(data)
                latent_has_none = False
                if isinstance(info["latent"], tuple):
                    if info["latent"][0] is None:
                        latent_has_none = True
                def get_latent_from_input(node_feature):
                    data_copy.node_feature["n0"] = node_feature
                    latent = self.encoder(data_copy, use_grads=args.use_grads, use_pos=args.use_pos)
                    if latent_has_none:
                        latent = latent[1:]
                    return latent

                _, latent_diff_target = jvp(get_latent_from_input, node_feature, v=node_feature_diff)
                if isinstance(info["latent"], tuple):
                    if latent_has_none:
                        latent_diff_core = latent_diff[1:]
                        assert len(info["latent"]) == 2, "currently can only work for up to one latent element."
                        latent_core = info["latent"][1]
                    else:
                        latent_diff_core = latent_diff
                        latent_core = info["latent"]
                    loss_latent = torch.stack([nn.MSELoss()(latent_diff_ele, latent_diff_target_ele) for latent_diff_ele, latent_diff_target_ele in zip(latent_diff_core, latent_diff_target)]).sum()
                else:
                    latent_diff_core = latent_diff
                    latent_core = info["latent"]
                    loss_latent = nn.MSELoss()(latent_diff, latent_diff_target)
                self.info["loss_latent"] = loss_latent.item()

                # (3) Loss for input:
                def get_output_from_latent(latent):
                    pred = self.decoder_n0((None, latent) if latent_has_none else latent)
                    return pred
                _, pred_diff = jvp(get_output_from_latent, latent_core, v=latent_diff_core[0] if latent_has_none else latent_diff_core)
                pred_steps = pred_diff.shape[-2]
                loss_pred = nn.MSELoss()(pred_diff, node_feature_diff[...,-pred_steps:,:])
                self.info["loss_pred"] = loss_pred.item()
                loss = loss_recons + loss_latent + loss_pred
                return loss

            # Not LFM loss:
            loss = 0
            for pred_idx, k in enumerate(multi_step_dict):
                pred_idx_list = np.arange(pred_idx*args.temporal_bundle_steps, (pred_idx+1)*args.temporal_bundle_steps).tolist()
                y_idx_list = np.arange((k-1)*args.temporal_bundle_steps, k*args.temporal_bundle_steps).tolist()
                loss_k = loss_op(
                    preds, data.node_label, data.mask,
                    pred_idx=pred_idx_list,
                    y_idx=y_idx_list,
                    loss_type=args.loss_type,
                    keys=self.grid_keys,
                    batch_size=batch_size,
                    is_y_variable_length=args.is_y_variable_length,
                    **kwargs
                )
                if len(self.part_keys) > 0:
                    input_shape_grid = dict(self.input_shape)
                    input_shape = input_shape_grid[list(input_shape_grid.keys())[0]]
                    loss_dict = loss_hybrid(
                        preds, data.node_label, data.mask,
                        node_pos_label=data.node_pos_label,
                        input_shape=input_shape,
                        pred_idx=pred_idx_list,
                        y_idx=y_idx_list,
                        loss_type=args.loss_type,
                        part_keys=self.part_keys,
                        batch_size=batch_size,
                        **kwargs
                       )
                    loss_k = loss_k + loss_dict["feature"]
                    if args.density_coef > 0:
                        loss_k = loss_k + loss_dict["density"] * args.density_coef
                loss = loss + loss_k * multi_step_dict[k]
            self.info["loss_pred"] = to_np_array(loss)

            # Reconstruction loss:
            if args.recons_coef > 0:
                y_idx_recons = np.arange(
                    data.node_feature[list(data.node_feature)[0]].shape[-2] - args.temporal_bundle_steps, 
                    data.node_feature[list(data.node_feature)[0]].shape[-2]).tolist()
                # pdb.set_trace()
                loss_recons = loss_op(
                    info["recons"], data.node_feature, data.mask,
                    y_idx=y_idx_recons,
                    dyn_dims=dict(to_tuple_shape(data.dyn_dims)),
                    loss_type=args.loss_type,
                    keys=self.grid_keys,
                    batch_size=batch_size,
                    is_y_variable_length=args.is_y_variable_length,
                    **kwargs
                )
                if len(self.part_keys) > 0:
                    if len(self.part_keys) > 0:
                        input_shape_grid = dict(self.input_shape)
                        input_shape = input_shape_grid[list(input_shape_grid.keys())[0]]
                        # TODO: fix pred_idx and y_idx:
                        loss_dict_recons = loss_hybrid(
                            info["recons"], data.node_feature, data.mask,
                            node_pos_label=data.node_pos,
                            dyn_dims=dict(to_tuple_shape(data.dyn_dims)),
                            input_shape=input_shape,
                            pred_idx=0,
                            y_idx=-1,
                            loss_type=args.loss_type,
                            part_keys=self.part_keys,
                            batch_size=args.batch_size,
                            **kwargs
                           )
                        loss_recons = loss_recons + loss_dict_recons["feature"]
                        if args.density_coef > 0:
                            loss_recons = loss_recons + loss_dict_recons["density"] * args.density_coef
                loss = loss + loss_recons * args.recons_coef
                self.info["loss_recons"] = to_np_array(loss_recons) * args.recons_coef
                if self.vae_mode != "None":
                    latent_loc = info["latent_loc"]  # [B, C]
                    latent_logscale = info["latent_logscale"]  # [B, C]
                    loss_vae_kl = -(1 + latent_logscale * 2 - latent_loc ** 2 - torch.exp(latent_logscale * 2)).mean() / 2
                    loss = loss + loss_vae_kl * args.vae_beta
                    self.info["loss_vae_kl"] = to_np_array(loss_vae_kl) * args.vae_beta

            if args.consistency_coef > 0 or args.contrastive_rel_coef > 0:
                latent_time_step_weights = torch.FloatTensor(list(latent_multi_step_dict.values())).to(args.device)
                latent_targets = self.get_latent_targets(data_copy_cons, list(latent_multi_step_dict.keys()), temporal_bundle_steps=args.temporal_bundle_steps, use_grads=args.use_grads, use_pos=args.use_pos)  # [B, pred_steps, [latent_size]] or tuple(...)

                # Compute consistency loss:
                if args.consistency_coef > 0:
                    loss_type_consistency = args.loss_type_consistency if args.loss_type_consistency != "None" else args.loss_type
                    if not isinstance(latent_targets, tuple):
                        loss_consistency = loss_op(
                            info["latent_preds"],
                            latent_targets,
                            loss_type=args.loss_type_consistency,
                            time_step_weights=latent_time_step_weights,
                            normalize_mode=args.latent_loss_normalize_mode,
                            epsilon_latent_loss=args.epsilon_latent_loss,
                            batch_size=batch_size,
                            is_y_variable_length=args.is_y_variable_length,
                        )
                    else:
                        loss_consistency = torch.stack([loss_op(
                            info["latent_preds"][i],
                            latent_targets[i],
                            loss_type=args.loss_type_consistency,
                            time_step_weights=latent_time_step_weights,
                            normalize_mode=args.latent_loss_normalize_mode,
                            epsilon_latent_loss=args.epsilon_latent_loss,
                            batch_size=batch_size,
                            is_y_variable_length=args.is_y_variable_length,
                           ) for i in range(len(latent_targets)) if latent_targets[i] is not None]
                        ).mean()
                    loss = loss + loss_consistency * args.consistency_coef
                    self.info["loss_cons"] = to_np_array(loss_consistency) * args.consistency_coef

                # Compute contrastive loss:
                if args.contrastive_rel_coef > 0:
                    """Inspired by https://github.com/tkipf/c-swm/blob/e944b24bcaa42d9ee847f30163437a50f0237aa0/modules.py#L94"""
                    loss_neg = get_neg_loss(
                        info["latent_preds"],
                        latent_targets,
                        loss_type=args.loss_type_consistency,
                        time_step_weights=latent_time_step_weights,
                    )
                    loss_contrastive = torch.max(torch.zeros_like(loss_neg), args.hinge - loss_neg)
                    loss = loss + loss_contrastive * args.consistency_coef * args.contrastive_rel_coef
                    self.info["loss_contr"] = to_np_array(loss_contrastive) * args.consistency_coef * args.contrastive_rel_coef

        # Add regularization:
        if args.reg_coef > 0:
            reg_coef = args.reg_coef
            if args.is_reg_anneal:
                reg_coef *= (kwargs["current_epoch"] / args.epochs) ** 2
            loss = loss + info["reg"] * reg_coef
            self.info["reg"] = to_np_array(info["reg"]) * reg_coef
        if is_diagnose(loc="loss:end", filename=args.filename):
            pdb.set_trace()
        return loss

    @property
    def model_dict(self):
        model_dict = {"type": "Contrastive"}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["latent_size"] = self.latent_size
        model_dict["encoder_type"] = self.encoder_type
        model_dict["evolution_type"] = self.evolution_type
        model_dict["decoder_type"] = self.decoder_type
        model_dict["encoder_n_linear_layers"] = self.encoder_n_linear_layers
        model_dict["encoder_mode"] = self.encoder_mode
        model_dict["input_shape"] = self.input_shape
        model_dict["grid_keys"] = self.grid_keys
        model_dict["part_keys"] = self.part_keys
        model_dict["no_latent_evo"] = self.no_latent_evo
        model_dict["temporal_bundle_steps"] = self.temporal_bundle_steps
        model_dict["forward_type"] = self.forward_type
        model_dict["channel_mode"] = self.channel_mode
        model_dict["kernel_size"] = self.kernel_size
        model_dict["stride"] = self.stride
        model_dict["padding"] = self.padding
        model_dict["padding_mode"] = self.padding_mode
        model_dict["output_padding_str"] = self.output_padding_str
        model_dict["evo_groups"] = self.evo_groups
        model_dict["act_name"] = self.act_name
        model_dict["decoder_last_act_name"] = self.decoder_last_act_name
        model_dict["is_pos_transform"] = self.is_pos_transform
        model_dict["normalization_type"] = self.normalization_type
        model_dict["cnn_n_conv_layers"] = self.cnn_n_conv_layers
        model_dict["n_conv_blocks"] = self.n_conv_blocks
        model_dict["n_latent_levs"] = self.n_latent_levs
        model_dict["n_conv_layers_latent"] = self.n_conv_layers_latent
        model_dict["evo_conv_type"] = self.evo_conv_type
        model_dict["evo_pos_dims"] = self.evo_pos_dims
        model_dict["evo_inte_dims"] = self.evo_inte_dims
        model_dict["is_latent_flatten"] = self.is_latent_flatten
        model_dict["reg_type"] = self.reg_type
        model_dict["loss_type"] = self.loss_type
        model_dict["state_dict"] = to_cpu(self.state_dict())
        model_dict["static_latent_size"] = self.static_latent_size
        model_dict["static_encoder_type"] = self.static_encoder_type
        model_dict["static_input_size"] = self.static_input_size
        model_dict["decoder_act_name"] = self.decoder_act_name
        model_dict["is_prioritized_dropout"] = self.is_prioritized_dropout
        model_dict["vae_mode"] = self.vae_mode
        return model_dict


def get_loss(model, data, args, is_rollout=False, **kwargs):
    if args.algo == "contrast":
        return get_loss_contrast(model, data, args, is_rollout=False, **kwargs)
    else:
        raise


# ### Load model:

# In[ ]:


def load_model(model_dict, device, multi_gpu=False, **kwargs):
    """Load saved model using model_dict."""
    def process_state_dict(state_dict):
        """Deal with SpectralNorm:"""
        keys_to_delete = []
        for key in state_dict:
            if key.endswith("weight_bar"):
                keys_to_delete.append(key[:-4])
        for key in keys_to_delete:
            if key in state_dict:
                state_dict.pop(key)
        return state_dict

    model_type = model_dict["type"]

    
    if model_type == "FNOModel":
        model = FNOModel(
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            input_shape=model_dict["input_shape"],
            modes=model_dict["modes"],
            width=model_dict["width"],
            loss_type=model_dict["loss_type"],
            temporal_bundle_steps=model_dict["temporal_bundle_steps"] if "temporal_bundle_steps" in model_dict else 1,
            static_encoder_type=model_dict["static_encoder_type"] if "static_encoder_type" in model_dict else "None",
            static_latent_size=model_dict["static_latent_size"] if "static_latent_size" in model_dict else 0,
            static_param_size=model_dict["static_param_size"] if "static_param_size" in model_dict else 0,
            gnn_output_dim=model_dict["gnn_output_dim"] if "gnn_output_dim" in model_dict else 32,
            gnn_latent_dim=model_dict["gnn_latent_dim"] if "gnn_latent_dim" in model_dict else 32,
            gnn_num_steps=model_dict["gnn_num_steps"] if "gnn_num_steps" in model_dict else 3,
            gnn_layer_norm=model_dict["gnn_layer_norm"] if "gnn_layer_norm" in model_dict else False,
            gnn_activation=model_dict["gnn_activation"] if "gnn_activation" in model_dict else "relu",
            gnn_diffMLP=model_dict["gnn_diffMLP"] if "gnn_diffMLP" in model_dict else False,
            gnn_global_pooling=model_dict["gnn_global_pooling"] if "gnn_global_pooling" in model_dict else "mean",
            gnn_is_virtual=model_dict["gnn_is_virtual"] if "gnn_is_virtual" in model_dict else True,
        )
    elif model_type == "Contrastive":
        #pdb.set_trace()
        model = Contrastive(
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            latent_size=model_dict["latent_size"],
            encoder_type=model_dict["encoder_type"],
            evolution_type=model_dict["evolution_type"],
            decoder_type=model_dict["decoder_type"],
            encoder_n_linear_layers=model_dict["encoder_n_linear_layers"] if "encoder_n_linear_layers" in model_dict else 0,
            encoder_mode=model_dict["encoder_mode"],
            input_shape=kwargs["input_shape"] if "input_shape" in kwargs else model_dict["input_shape"],
            grid_keys=model_dict["grid_keys"] if "grid_keys" in model_dict else ("n0",),
            part_keys=model_dict["part_keys"] if "part_keys" in model_dict else (),
            temporal_bundle_steps=model_dict["temporal_bundle_steps"] if "temporal_bundle_steps" in model_dict else 1,
            no_latent_evo=model_dict["no_latent_evo"] if "no_latent_evo" in model_dict else False,
            forward_type=model_dict["forward_type"] if "forward_type" in model_dict else "Euler",
            channel_mode=model_dict["channel_mode"] if "channel_mode" in model_dict else "exp-16",
            kernel_size=model_dict["kernel_size"] if "kernel_size" in model_dict else 4,
            stride=model_dict["stride"] if "stride" in model_dict else 2,
            padding=model_dict["padding"] if "padding" in model_dict else 1,
            padding_mode=kwargs["padding_mode"] if "padding_mode" in kwargs else model_dict["padding_mode"] if "padding_mode" in model_dict else "zeros",
            output_padding_str=model_dict["output_padding_str"] if "output_padding_str" in model_dict else "None",
            evo_groups=model_dict["evo_groups"] if "evo_groups" in model_dict else 1,
            act_name=model_dict["act_name"],
            decoder_last_act_name=model_dict["decoder_last_act_name"] if "decoder_last_act_name" in model_dict else "linear",
            is_pos_transform=model_dict["is_pos_transform"],
            normalization_type=model_dict["normalization_type"] if "normalization_type" in model_dict else "bn2d",
            cnn_n_conv_layers=model_dict["cnn_n_conv_layers"],
            n_conv_blocks=model_dict["n_conv_blocks"] if "n_conv_blocks" in model_dict else 4,
            n_latent_levs=model_dict["n_latent_levs"] if "n_latent_levs" in model_dict else 1,
            n_conv_layers_latent=model_dict["n_conv_layers_latent"] if "n_conv_layers_latent" in model_dict else 1,
            evo_conv_type=model_dict["evo_conv_type"] if "evo_conv_type" in model_dict else "cnn",
            evo_pos_dims=model_dict["evo_pos_dims"] if "evo_pos_dims" in model_dict else -1,
            evo_inte_dims=model_dict["evo_inte_dims"] if "evo_inte_dims" in model_dict else -1,
            is_latent_flatten=model_dict["is_latent_flatten"] if "is_latent_flatten" in model_dict else True,
            reg_type=model_dict["reg_type"] if "reg_type" in model_dict else "None",
            loss_type=model_dict["loss_type"] if "loss_type" in model_dict else None,
            static_latent_size=model_dict["static_latent_size"] if "static_latent_size" in model_dict else 0,
            static_encoder_type=model_dict["static_encoder_type"] if "static_encoder_type" in model_dict else "None",
            static_input_size=model_dict["static_input_size"] if "static_input_size" in model_dict else {"n0": 0},
            decoder_act_name=model_dict["decoder_act_name"] if "decoder_act_name" in model_dict else None,
            is_prioritized_dropout=model_dict["is_prioritized_dropout"] if "is_prioritized_dropout" in model_dict else False,
            vae_mode=model_dict["vae_mode"] if "vae_mode" in model_dict else "None",
        )
    else:
        raise Exception("model type {} is not supported!".format(model_type)) 

    if model_type not in ["Actor_Critic"]:
        model.load_state_dict(model_dict["state_dict"])
    model.to(device)
    return model


# ### Get model:

# In[ ]:


def get_model(
    args,
    data_eg,
    device,
):
    """Get model as specified by args."""
    # breakpoint()
    if isinstance(data_eg, Data):
        data_eg = PyG_to_Attr_Dict(data_eg)
    if len(dict(data_eg.original_shape)["n0"]) != 0:
        original_shape = to_tuple_shape(data_eg.original_shape)
        pos_dims = get_pos_dims_dict(original_shape)
    grid_keys = to_tuple_shape(data_eg.grid_keys)
    part_keys = to_tuple_shape(data_eg.part_keys)
    dyn_dims = dict(to_tuple_shape(data_eg.dyn_dims))
    static_input_size = {"n0": 0}
    # pdb.set_trace()
    if "node_feature" in data_eg:
        # The data object contains the actual data:
        output_size = {key: data_eg.node_label[key].shape[-1] + 1 if key in part_keys else data_eg.node_label[key].shape[-1] for key in data_eg.node_label}
        # pdb.set_trace()
        input_size = {key: np.prod(data_eg.node_feature[key].shape[-2:]) for key in data_eg.node_feature}
        if args.static_encoder_type != "None":
            if args.static_encoder_type.startswith("param"):
                static_input_size = {key: data_eg.param[key].shape[-1] for key in input_size}
            else:
                #input_steps = 
                static_input_size = {key: input_size[key]-dyn_dims[key]*data_eg.node_feature[key].shape[-2] for key in input_size}
    else:
        # The data object contains necessary information for JIT loading:
        static_dims = dict(to_tuple_shape(data_eg.static_dims))
        compute_func = dict(to_tuple_shape(data_eg.compute_func))
        input_size = {key: compute_func[key][0] + static_dims[key] + dyn_dims[key] for key in dyn_dims}
        output_size = deepcopy(dyn_dims)
        if args.static_encoder_type != "None":
            static_input_size = static_dims
    if not args.is_latent_flatten:
        assert args.n_latent_levs >= 2
    for key in data_eg.grid_keys:
        if args.use_grads:
            input_size[key] += dyn_dims[key] * pos_dims[key]
        # if args.use_pos:
        #     input_size[key] += pos_dims[key]
    if args.algo == "contrast":
        model = Contrastive(
            input_size=input_size,
            output_size=output_size,
            latent_size=args.latent_size,
            encoder_type=args.encoder_type,
            evolution_type=args.evolution_type,
            decoder_type=args.decoder_type,
            encoder_n_linear_layers=args.encoder_n_linear_layers,
            temporal_bundle_steps=args.temporal_bundle_steps,
            n_conv_blocks=args.n_conv_blocks,
            n_latent_levs=args.n_latent_levs,
            n_conv_layers_latent=args.n_conv_layers_latent,
            evo_conv_type=args.evo_conv_type,
            evo_pos_dims=args.evo_pos_dims,
            evo_inte_dims=args.evo_inte_dims,
            is_latent_flatten=args.is_latent_flatten,
            encoder_mode="dense",
            grid_keys=grid_keys,
            part_keys=part_keys,
            no_latent_evo=args.no_latent_evo,
            forward_type=args.forward_type,
            channel_mode=args.channel_mode,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            padding_mode=args.padding_mode,
            output_padding_str=args.output_padding_str,
            evo_groups=args.evo_groups,
            act_name=args.act_name,
            decoder_last_act_name=args.decoder_last_act_name,
            is_pos_transform=args.is_pos_transform,
            normalization_type=args.normalization_type,
            reg_type=args.reg_type,
            loss_type=args.loss_type,
            input_shape=original_shape,
            static_latent_size=args.static_latent_size,
            static_encoder_type=args.static_encoder_type,
            static_input_size=static_input_size,
            decoder_act_name=args.decoder_act_name,
            is_prioritized_dropout=args.is_prioritized_dropout,
            vae_mode=args.vae_mode,
        ).to(device)
    elif args.algo.startswith("fno"):
        if args.static_encoder_type.startswith("param"):
            static_param_size = data_eg.param["n0"].reshape(1, -1).shape[-1]
        else:
            static_param_size = 0
        if (args.static_encoder_type.startswith("param")) and ("gnn" in args.static_encoder_type):
            # pdb.set_trace()
            input_size["n0"] = np.prod([data_eg.node_feature["n0"].shape[-2], dyn_dims["n0"]])
        """fno-w20-m12: fno with width=20 and modes=12. Default"""
        algo_dict = {}
        for ele in args.algo.split("-")[1:]:
            algo_dict[ele[0]] = int(ele[1:])
        if "w" not in algo_dict:
            algo_dict["w"] = None
        if "m" not in algo_dict:
            algo_dict["m"] = None
        model = FNOModel(
            input_size=input_size["n0"],
            output_size=output_size["n0"] * args.temporal_bundle_steps,
            input_shape=dict(original_shape)["n0"],
            modes=algo_dict["m"],
            width=algo_dict["w"],
            loss_type=args.loss_type,
            temporal_bundle_steps=args.temporal_bundle_steps,
            static_encoder_type=args.static_encoder_type,
            static_latent_size=args.static_latent_size,
            static_param_size = static_param_size,
            gnn_output_dim = args.gnn_output_dim,
            gnn_latent_dim = args.gnn_latent_dim,
            gnn_num_steps = args.gnn_num_steps,
            gnn_layer_norm = args.gnn_layer_norm,
            gnn_activation = args.gnn_activation,
            gnn_diffMLP = args.gnn_diffMLP,
            gnn_global_pooling = args.gnn_global_pooling,
            gnn_is_virtual = args.gnn_is_virtual,
        ).to(device)
    else:
        raise Exception("Algo {} is not supported!".format(args.algo))
    return model


# # CNN-based:



# ### CNN_Encoder:

# In[ ]:


class CNN_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        output_size,
        grid_keys,
        part_keys,
        input_shape=None,
        channel_mode="exp-16",
        kernel_size=4,
        stride=2,
        padding=1,
        padding_mode="zeros",
        last_n_linear_layers=0,
        act_name="rational",
        normalization_type="gn",
        n_conv_blocks=4,
        n_latent_levs=1,
        is_latent_flatten=True,
        reg_type_list=None,
        vae_mode="None",
    ):
        super(CNN_Encoder, self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.pos_dims = get_pos_dims_dict(input_shape)
        self.output_size = output_size
        self.grid_keys = grid_keys
        self.part_keys = part_keys
        self.channel_mode = channel_mode
        self.channel_gen = Channel_Gen(channel_mode=self.channel_mode)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.last_n_linear_layers = last_n_linear_layers
        self.act_name = act_name
        self.normalization_type = normalization_type
        self.n_conv_blocks = n_conv_blocks
        self.n_latent_levs = n_latent_levs
        self.is_latent_flatten = is_latent_flatten
        self.normalization_n_groups = 2
        self.reg_type_list = reg_type_list
        self.vae_mode = vae_mode

        # Convolutions:
        for key, pos_dim in self.pos_dims.items():
            setattr(self, "conv_grid_{}".format(key),
                    nn.Sequential(get_conv_func(
                        pos_dim=pos_dim,
                        in_channels=self.in_channels[key],
                        out_channels=self.channel_gen(0),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode=padding_mode,
                        reg_type_list=self.reg_type_list,
                    ),
                    get_activation(self.act_name)))

        self.pos_dim_max = max(list(self.pos_dims.values()))
        self.conv_list = nn.ModuleList()
        assert self.n_conv_blocks >=0 and self.n_conv_blocks <= 6
        if self.n_conv_blocks >= 1:
            self.conv_list.append(nn.Sequential(
                get_conv_func(self.pos_dim_max, self.channel_gen(0)*len(self.pos_dims), self.channel_gen(1), self.kernel_size, self.stride, self.padding, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(1), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 2:
            self.conv_list.append(nn.Sequential(
                get_conv_func(self.pos_dim_max, self.channel_gen(1), self.channel_gen(2), self.kernel_size, self.stride, self.padding, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(2), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 3:
            self.conv_list.append(nn.Sequential(
                get_conv_func(self.pos_dim_max, self.channel_gen(2), self.channel_gen(3), self.kernel_size, self.stride, self.padding, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(3), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 4:
            self.conv_list.append(nn.Sequential(
                get_conv_func(self.pos_dim_max, self.channel_gen(3), self.channel_gen(4), self.kernel_size-1 if self.n_conv_blocks==4 else self.kernel_size, self.stride, self.padding, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(4), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 5:
            self.conv_list.append(nn.Sequential(
                get_conv_func(self.pos_dim_max, self.channel_gen(4), self.channel_gen(5), self.kernel_size-1 if self.n_conv_blocks==5 else self.kernel_size, self.stride, self.padding, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(5), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 6:
            self.conv_list.append(nn.Sequential(
                get_conv_func(self.pos_dim_max, self.channel_gen(5), self.channel_gen(6), self.kernel_size-1 if self.n_conv_blocks==6 else self.kernel_size, self.stride, self.padding, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(6), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))

        self.flat_fts = self.get_flat_fts(self.conv_list)

        if self.is_latent_flatten:
            if vae_mode == "None":
                last_layers = [
                    nn.Linear(self.flat_fts, self.output_size),
                    get_normalization('bn1d' if self.normalization_type != "gn" else "gn", output_size, n_groups=self.normalization_n_groups),
                    get_activation(self.act_name),
                ]
                if self.last_n_linear_layers > 0:
                    last_layers.append(
                        MLP(input_size=self.output_size,
                            n_neurons=self.output_size,
                            n_layers=self.last_n_linear_layers,
                            act_name="linear",
                            output_size=self.output_size,
                        )
                    )
                self.linear = nn.Sequential(*last_layers)
            else:
                self.linear_base = nn.Sequential(*[
                    nn.Linear(self.flat_fts, self.output_size),
                    get_normalization('bn1d' if self.normalization_type != "gn" else "gn", output_size, n_groups=self.normalization_n_groups),
                    get_activation(self.act_name),
                ])
                self.linear_loc = MLP(
                    input_size=self.output_size,
                    n_neurons=self.output_size,
                    n_layers=self.last_n_linear_layers,
                    act_name="linear",
                    output_size=self.output_size,
                )
                self.linear_logscale = MLP(
                    input_size=self.output_size,
                    n_neurons=self.output_size,
                    n_layers=self.last_n_linear_layers,
                    act_name="linear",
                    output_size=self.output_size,
                )

    def get_flat_fts(self, conv_list):
        self.input_shape_LCM = get_LCM_input_shape(self.input_shape)
        x = torch.ones(1, self.channel_gen(0) * len(self.pos_dims), *self.input_shape_LCM)
        self.channel_dict = {}
        if self.n_latent_levs == self.n_conv_blocks + 2:
            self.channel_dict[self.n_latent_levs-1] = x.shape[1]
        for i, conv in enumerate(conv_list):
            x = conv(x)
            if self.n_latent_levs + i >= self.n_conv_blocks + 1:
                self.channel_dict[self.n_conv_blocks-i] = x.shape[1]
        f_shape = tuple(x.shape)  # [B, C, [latent_shape]]
        self.latent_shape = f_shape[2:]   # [[latent_shape]]
        return int(np.prod(f_shape[1:]))

#     def get_flat_fts(self, fts):
#         f_shape = tuple(fts(torch.ones(1, self.in_channels_sum, *self.input_shape)).size())  # [B, C, [latent_shape]]
#         self.latent_shape = f_shape[2:]
#         return int(np.prod(f_shape[1:]))

#     def forward(self, data, use_grads=True, use_pos=False):
#         data_node_feature = process_data_for_CNN(data, use_grads=use_grads, use_pos=use_pos)
#         x_list = [x for key, x in data_node_feature.items()]
#         x = torch.cat(x_list, 1)
#         x = self.conv(x)
#         x = x.reshape(-1, self.flat_fts)
#         return self.linear(x)


    def forward(
        self,
        data,
        pred_steps=1,
        is_recons=False,
        use_grads=True,
        use_pos=False,
    ):
        data_node_feature = process_data_for_CNN(data, use_grads=use_grads, use_pos=use_pos)
        x = []
        for key in self.grid_keys:
            x_ele = getattr(self, "conv_grid_{}".format(key))(data_node_feature[key])
            x.append(x_ele)
        x = expand_same_shape(x, self.input_shape_LCM)
        x = torch.cat(x, 1)

        x_list = []
        if self.n_latent_levs + (-1) == self.n_conv_blocks + 1:
            x_list.insert(0, x)
        for i, conv in enumerate(self.conv_list):
            x = conv(x)
            if self.n_latent_levs + i >= self.n_conv_blocks + 1:
                x_list.insert(0, x)
        if self.is_latent_flatten:
            if self.vae_mode != "None":
                x = self.linear_base(flatten(x))
                x = self.linear_loc(x), self.linear_logscale(x)
            else:
                x = self.linear(flatten(x))
            x_list.insert(0, x)
        else:
            x_list.insert(0, None)
        if self.n_latent_levs > 1:
            return tuple(x_list)
        else:
            return x


# ### CNN_Decoder:

# In[ ]:


class CNN_Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        latent_shape,
        output_size,
        output_shape,
        fc_output_dim,
        temporal_bundle_steps=1,
        channel_mode="exp-16",
        kernel_size=4,
        stride=2,
        padding=1,
        padding_mode="zeros",
        output_padding_str="None",
        act_name="rational",
        normalization_type="gn",
        n_conv_blocks=4,
        n_latent_levs=1,
        is_latent_flatten=True,
        reg_type_list=None,
        decoder_act_name="None",
    ):
        super(CNN_Decoder, self).__init__()
        self.latent_size = latent_size
        self.latent_shape = latent_shape
        self.output_size = output_size
        self.output_shape = output_shape
        self.channel_mode = channel_mode
        self.channel_gen = Channel_Gen(self.channel_mode)
        self.temporal_bundle_steps = temporal_bundle_steps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding_str = output_padding_str
        if output_padding_str != "None":
            output_padding_list = [int(ele) for ele in output_padding_str.split("-")]
            assert len(output_padding_list) == n_conv_blocks
        else:
            output_padding_list = [1] + [0] * (n_conv_blocks - 1)
            self.output_padding_list = output_padding_list
        self.act_name = act_name
        self.normalization_type = normalization_type
        self.normalization_n_groups = 2
        self.n_conv_blocks=n_conv_blocks
        self.n_latent_levs=n_latent_levs
        self.is_latent_flatten = is_latent_flatten
        self.reg_type_list = reg_type_list
        if decoder_act_name == "None":
            decoder_act_name = act_name
        self.decoder_act_name = decoder_act_name

        self.fc_output_dim = fc_output_dim

        self.pos_dims = len(output_shape)
        if self.is_latent_flatten:
            self.fc = nn.Sequential(
                nn.Linear(self.latent_size, self.fc_output_dim),
                get_normalization('bn1d' if self.normalization_type != "gn" else "gn", self.fc_output_dim, n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            )

        self.deconv_list = nn.ModuleList()
        if self.n_conv_blocks >= 6:
            channels = self.fc_output_dim // np.prod(self.latent_shape) if self.n_conv_blocks == 6 else self.channel_gen(6)
            if self.n_latent_levs >= self.n_conv_blocks - 4 and not (self.is_latent_flatten is False and self.n_conv_blocks==6):
                channels *= 2   # Append 256
            self.deconv_list.append(nn.Sequential(
                get_conv_trans_func(self.pos_dims, channels, self.channel_gen(5),  # [256, 128]
                                    self.kernel_size, self.stride, self.padding, output_padding=0, bias=False, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(5), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 5:
            channels = self.fc_output_dim // np.prod(self.latent_shape) if self.n_conv_blocks == 5 else self.channel_gen(5)
            if self.n_latent_levs >= self.n_conv_blocks - 3 and not (self.is_latent_flatten is False and self.n_conv_blocks==5):
                channels *= 2   # Append 256
            self.deconv_list.append(nn.Sequential(
                get_conv_trans_func(self.pos_dims, channels, self.channel_gen(4),  # [256, 128]
                                    self.kernel_size, self.stride, self.padding, output_padding=output_padding_list[-5], bias=False, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(4), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 4:
            channels = self.fc_output_dim // np.prod(self.latent_shape) if self.n_conv_blocks == 4 else self.channel_gen(4)
            if self.n_latent_levs >= self.n_conv_blocks - 2 and not (self.is_latent_flatten is False and self.n_conv_blocks==4):
                channels *= 2   # Append 256
            self.deconv_list.append(nn.Sequential(
                get_conv_trans_func(self.pos_dims, channels, self.channel_gen(3),  # [256, 128]
                                    self.kernel_size-1 if self.n_conv_blocks == 4 else self.kernel_size, self.stride, self.padding, output_padding=output_padding_list[-4], bias=False, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(3), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 3:
            channels = self.channel_gen(3)
            if self.n_latent_levs >= self.n_conv_blocks - 1 and not (self.is_latent_flatten is False and self.n_conv_blocks==3):
                channels *= 2   # Append 128
            self.deconv_list.append(nn.Sequential(
                get_conv_trans_func(self.pos_dims, channels, self.channel_gen(2),  # [128, 64]
                                    self.kernel_size, self.stride, self.padding, bias=False, padding_mode=padding_mode, output_padding=output_padding_list[-3], reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(2), n_groups=self.normalization_n_groups),
                get_activation(self.act_name),
            ))
        if self.n_conv_blocks >= 2:
            channels = self.channel_gen(2)
            if self.n_latent_levs >= self.n_conv_blocks and not (self.is_latent_flatten is False and self.n_conv_blocks==2):
                channels *= 2     # Append 64
            self.deconv_list.append(nn.Sequential(
                get_conv_trans_func(self.pos_dims, channels, self.channel_gen(1),  # [64, 32]
                                    self.kernel_size, self.stride, self.padding, bias=False, padding_mode=padding_mode, output_padding=output_padding_list[-2], reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(1), n_groups=self.normalization_n_groups),
                get_activation(self.decoder_act_name),
            ))
        if self.n_conv_blocks >= 1:
            channels = self.channel_gen(1)
            if self.n_latent_levs >= self.n_conv_blocks + 1:
                channels += self.channel_gen(0) * 3     # Append 48
            self.deconv_list.append(nn.Sequential(
                get_conv_trans_func(self.pos_dims, channels, self.channel_gen(1),  # [32, 16]
                                    self.kernel_size, self.stride, self.padding, bias=False, padding_mode=padding_mode, output_padding=output_padding_list[-1], reg_type_list=self.reg_type_list),
                get_normalization(self.normalization_type, self.channel_gen(1), n_groups=self.normalization_n_groups),
                get_activation(self.decoder_act_name),
            ))

        channels = self.channel_gen(1)
        if self.n_latent_levs >= self.n_conv_blocks + 2:
            channels += self.channel_gen(0) * 3     # Append 48

        self.deconv_list.append(nn.Sequential(
            get_conv_trans_func(self.pos_dims, channels, self.output_size * self.temporal_bundle_steps, 3, 1, 1, bias=False, padding_mode=padding_mode, reg_type_list=self.reg_type_list),
        ))

#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(self.fc_output_dim // (self.latent_shape[0] * self.latent_shape[1]), self.channel_mult*4,
#                                 4, 1, 0, bias=False),
#             nn.BatchNorm2d(self.channel_mult*4),
#             get_activation(self.act_name),
#             # state size. self.channel_mult*32 x 4 x 4
#             nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
#                                 3, 2, 1, bias=False),
#             nn.BatchNorm2d(self.channel_mult*2),
#             get_activation(self.act_name),
#             # state size. self.channel_mult*16 x 7 x 7
#             nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1,
#                                 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.channel_mult*1),
#             get_activation(self.act_name),
#             # state size. self.channel_mult*8 x 14 x 14
#             nn.ConvTranspose2d(self.channel_mult*1, self.output_size, 4, 2, 1, bias=False),
#             # state size. self.output_channels x 28 x 28


    def forward(self, latent, **kwargs):
        if not isinstance(latent, tuple):
            x = self.fc(latent)
            x = x.view(x.shape[0], -1, *self.latent_shape)
            for i, deconv in enumerate(self.deconv_list):
                x = deconv(x)                 # [B, output_channels, [pos_dims]]
            permute_order = (0,) + tuple(range(2, 2+self.pos_dims)) + (1,)
            x = x.permute(*permute_order)      # [B, [pos_dims], output_channels]
            x = x.reshape(-1, self.temporal_bundle_steps, self.output_size)  # [B * prod([pos_dims]), self.temporal_bundle_steps, self.output_size]
        else:
            if self.is_latent_flatten:
                x = self.fc(latent[0])
                x = x.view(x.shape[0], -1, *self.latent_shape)  # [B, 256, 64]
            else:
                assert latent[0] is None
            for i, deconv in enumerate(self.deconv_list):
                if self.n_latent_levs >= i + 2:
                    if (not self.is_latent_flatten) and i == 0:
                        # Only if not flatten latent and at the first deconv layer:
                        x = latent[1+i]
                    else:
                        x = torch.cat([latent[1+i], x], 1)
                x = deconv(x)                 # [B, output_channels, [pos_dims]]
            permute_order = (0,) + tuple(range(2, 2+self.pos_dims)) + (1,)
            x = x.permute(*permute_order)      # [B, [pos_dims], output_channels]
            x = x.reshape(-1, self.temporal_bundle_steps, self.output_size)  # [B * prod([pos_dims]), self.temporal_bundle_steps, self.output_size]
        return x








class Evolution_Op(nn.Module):
    def __init__(
        self,
        evolution_type,
        latent_size,
        pos_dims,
        normalization_type="gn",
        normalization_n_groups=2,
        n_latent_levs=1,
        n_conv_layers_latent=1,
        evo_conv_type="cnn",
        evo_pos_dims=-1,
        evo_inte_dims=-1,
        is_latent_flatten=True,
        channel_size_dict=None,
        act_name="rational",
        padding_mode="zeros",
        evo_groups=1,
        reg_type_list=None,
        static_latent_size=0,
        is_prioritized_dropout=False,
    ):
        super(Evolution_Op, self).__init__()
        self.evolution_type = evolution_type
        self.latent_size = latent_size
        self.pos_dims = pos_dims
        self.normalization_type = normalization_type
        self.normalization_n_groups = normalization_n_groups
        self.n_latent_levs = n_latent_levs
        self.n_conv_layers_latent = n_conv_layers_latent
        self.evo_conv_type = evo_conv_type
        self.evo_pos_dims = evo_pos_dims
        self.evo_inte_dims = evo_inte_dims
        self.is_latent_flatten = is_latent_flatten
        self.act_name = act_name
        self.padding_mode = padding_mode
        self.evo_groups = evo_groups
        self.reg_type_list = reg_type_list
        self.static_latent_size=static_latent_size
        self.is_prioritized_dropout = is_prioritized_dropout

        if self.is_latent_flatten:
            evolution_op_list = []
            evolution_type_split = self.evolution_type.split("-")
            evolution_type_core = evolution_type_split[0]
            if len(evolution_type_split) >= 4:
                evolution_n_linear_layers = eval(evolution_type_split[3])
            else:
                evolution_n_linear_layers = 0
            if evolution_type_core == "mlp":
                """
                mlp-{n_layers}: mlp, multi-layers with the same activation as the global activation.
                mlp-{n_layers}-{act_name}: mlp, multi-layers of specified activation
                mlp-{n_layers}-{act_name}-{n_linear_layers}, multi-layers of specified activation, followed by multi-layers of linear activation.
                """
                evolution_n_layers = eval(evolution_type_split[1])
                if len(evolution_type_split) >= 3:
                    evolution_act_name = evolution_type_split[2]
                else:
                    evolution_act_name = self.act_name
                #pdb.set_trace()
                evolution_op_list.append(
                    MLP(input_size=latent_size+static_latent_size,
                        n_neurons=latent_size,
                        n_layers=evolution_n_layers,
                        act_name=evolution_act_name,
                        output_size=latent_size,
                        last_layer_linear=False,
                        is_prioritized_dropout=is_prioritized_dropout,
                ))
            elif evolution_type_core == "SINDy":
                """
                SINDy-{poly_order}[-{additional_nonlinearities}][-{n_linear_layers}]
                """
                poly_order = eval(evolution_type_split[1])
                additional_nonlinearities = evolution_type_split[2] if len(evolution_type_split) >= 3 else "None"
                evolution_op_list.append(
                    SINDy(
                        input_size=latent_size,
                        poly_order=poly_order,
                        additional_nonlinearities=additional_nonlinearities,
                    )
                )
            else:
                raise Exception("evolution_type_core '{}' is not supported!".format(evolution_type_core))

            # Appending additional linear layers for implicit rank regularization (Implicit Rank-Minimizing Autoencoder, Li Jing et al. 2020):
            if evolution_n_linear_layers > 0:
                evolution_op_list.append(
                    MLP(input_size=latent_size,
                        n_neurons=latent_size,
                        n_layers=evolution_n_linear_layers,
                        act_name="linear",
                        output_size=latent_size,
                    )
                )
            self.evolution_op0 = nn.Sequential(*evolution_op_list)

        # Convolution operator in lower latent layers:
        if self.n_latent_levs > 1:
            pos_dim_max = max(list(self.pos_dims.values())) if self.evo_pos_dims == -1 else evo_pos_dims
            for i in range(1, self.n_latent_levs):
                if self.evo_conv_type == "cnn":
                    setattr(self, "evolution_op{}".format(i),
                            nn.Sequential(
                                *([get_conv_func(pos_dim_max, channel_size_dict[i]+self.static_latent_size, channel_size_dict[i], 3, 1, 1, padding_mode=padding_mode, groups=evo_groups, reg_type_list=self.reg_type_list),
                                  get_normalization(self.normalization_type, channel_size_dict[i], n_groups=self.normalization_n_groups),
                                  get_activation(self.act_name)] + \
                                  [get_conv_func(pos_dim_max, channel_size_dict[i], channel_size_dict[i], 3, 1, 1, padding_mode=padding_mode, groups=evo_groups, reg_type_list=self.reg_type_list),
                                  get_normalization(self.normalization_type, channel_size_dict[i], n_groups=self.normalization_n_groups),
                                  get_activation(self.act_name),
                                  ] * (self.n_conv_layers_latent - 1)
                                 )
                            ))
                elif self.evo_conv_type == "cnn-inte":
                    assert evo_inte_dims != -1
                    setattr(self, "evolution_op{}".format(i),
                            CNN_Integral(
                                pos_dim=pos_dim_max,
                                in_channels=channel_size_dict[i],
                                n_conv_layers_latent=self.n_conv_layers_latent,
                                inte_dims=self.evo_inte_dims,
                                padding_mode=self.padding_mode,
                                groups=self.evo_groups,
                                normalization_type=self.normalization_type,
                                act_name=self.act_name,
                                reg_type_list=self.reg_type_list,
                            ))
                elif self.evo_conv_type.startswith("VL-u"):
                    assert pos_dim_max == 1
                    setattr(self, "evolution_op{}".format(i),
                            Vlasov_U_Evo(
                                model_type=self.evo_conv_type,
                                in_channels=channel_size_dict[i],
                                n_conv_layers_latent=self.n_conv_layers_latent,
                                padding_mode=self.padding_mode,
                                groups=self.evo_groups,
                                normalization_type=self.normalization_type,
                                act_name=self.act_name,
                                reg_type_list=self.reg_type_list,
                           ))
                else:
                    raise

    def forward_core(self, latent):
        def get_Hessian_penalty_latent(G, z, reg_mode, G_z):
            return get_Hessian_penalty(
                G=G,
                z=z,
                mode=reg_mode,
                k=2,
                epsilon=0.1,
                reduction=torch.max,
                return_separately=False,
                G_z=G_z,
                is_nondimensionalize=True,
            )
        Hreg_mode = None
        Hreg = 0
        if self.training and self.reg_type_list is not None and ("Hall" in self.reg_type_list or "Hdiag" in self.reg_type_list or "Hoff" in self.reg_type_list):
            for ele in self.reg_type_list:
                if ele == "Hall" or ele == "Hdiag" or ele == "Hoff":
                    Hreg_mode = ele
                    break

        if not isinstance(latent, tuple):
            latent_out = self.evolution_op0(latent)
            if Hreg_mode is not None:
                self.Hreg = get_Hessian_penalty_latent(self.evolution_op0, latent, Hreg_mode, G_z=latent_out)
            return latent_out
        else:
            latent_list = []
            if self.is_latent_flatten:
                latent_out = self.evolution_op0(latent[0])
                latent_list.append(latent_out)
                if Hreg_mode is not None:
                    # Hessian regularization:
                    Hreg = Hreg + get_Hessian_penalty_latent(
                        G=self.evolution_op0,
                        z=latent[0],
                        reg_mode=Hreg_mode,
                        G_z=latent_out,
                    )
            else:
                assert latent[0] is None
                latent_list.append(None)
            for i in range(1, self.n_latent_levs):
                evolution_op_i = getattr(self, "evolution_op{}".format(i))
                latent_out = evolution_op_i(latent[i])
                latent_list.append(latent_out)
                if Hreg_mode is not None:
                    # Hessian regularization:
                    Hreg = Hreg + get_Hessian_penalty_latent(
                        G=evolution_op_i,
                        z=latent[i],
                        reg_mode=Hreg_mode,
                        G_z=latent_out,
                    )
            if Hreg_mode is not None:
                self.Hreg = Hreg
            return tuple(latent_list)

    def forward(self, latent):
        return self.forward_core(latent)



################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, input_size, output_size):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_size + 1, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
# 2d fourier layers
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, input_size, output_size):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_size + 2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, input_size, output_size):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_size + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # After: [B, C, H, W, D]
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)  # [B, C, H, W, D]
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # After: [B, H, W, D, C]. pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class FNOModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        input_shape,
        modes=None,
        width=None,
        loss_type="lp",
        temporal_bundle_steps=1,
        static_encoder_type="None",
        static_latent_size=0,
        static_param_size=0,
        gnn_output_dim=32,
        gnn_latent_dim=32,
        gnn_num_steps=3,
        gnn_layer_norm=False,
        gnn_activation="relu",
        gnn_diffMLP=False,
        gnn_global_pooling="mean",
        gnn_is_virtual=True,
    ):
        super().__init__()
        self.input_size = input_size  # steps*feature_size
        self.output_size = output_size
        self.input_shape = input_shape
        self.loss_type = loss_type
        self.temporal_bundle_steps = temporal_bundle_steps
        self.static_encoder_type = static_encoder_type
        self.static_latent_size = static_latent_size
        self.static_param_size = static_param_size
        self.pos_dims = len(input_shape)
        static_latent_size_core = self.static_latent_size if self.static_encoder_type.startswith("param") else 0
        if (self.static_encoder_type.startswith("param")) and ("gnn" in self.static_encoder_type):
            self.gnn_output_dim = gnn_output_dim  ### 3 has to be variable
            self.gnn_latent_dim = gnn_latent_dim
            self.gnn_num_steps = gnn_num_steps
            self.gnn_layer_norm = gnn_layer_norm
            self.gnn_activation = gnn_activation
            self.gnn_diffMLP = gnn_diffMLP
            self.gnn_global_pooling = gnn_global_pooling
            self.gnn_is_virtual=gnn_is_virtual
            self.mpnn = mesh_PDE(2, 2, self.gnn_output_dim, latent_dim=self.gnn_latent_dim, 
                                 num_steps=self.gnn_num_steps, layer_norm=self.gnn_layer_norm, 
                                 nl = self.gnn_activation, diffMLP = self.gnn_diffMLP) 
        if self.pos_dims == 1:
            self.modes = 16 if modes is None else modes
            self.width = 64 if width is None else width
            self.model = FNO1d(self.modes, width=self.width, input_size=self.input_size+static_latent_size_core, output_size=self.output_size)
        elif self.pos_dims == 2:
            self.modes = 12 if modes is None else modes
            self.width = 20 if width is None else width
            # self.model = FNO2d(self.modes, self.modes, width=self.width, input_size=self.input_size+static_latent_size_core, output_size=self.output_size)
            if (self.static_encoder_type.startswith("param")) and ("gnn" in self.static_encoder_type):
                # pdb.set_trace()
                ### self.input_size has to be changed, and 
                self.model = FNO2d(self.modes, self.modes, width=self.width, input_size=self.input_size+self.gnn_output_dim, output_size=self.output_size)
            else:
                self.model = FNO2d(self.modes, self.modes, width=self.width, input_size=self.input_size+self.static_param_size, output_size=self.output_size)
        elif self.pos_dims == 3:
            self.modes = 8 if modes is None else modes
            self.width = 20 if width is None else width
            self.model = FNO3d(self.modes, self.modes, self.modes, width=self.width, input_size=self.input_size+static_latent_size_core, output_size=self.output_size)
        else:
            raise Exception("self.pos_dims can only be 1, 2 or 3!")


    def forward(
        self,
        data,
        pred_steps=1,
        **kwargs
    ):
        if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
            pred_steps = [pred_steps]
        max_pred_steps = max(pred_steps)
        original_shape = dict(to_tuple_shape(data.original_shape))
        n_pos = np.array(original_shape["n0"]).prod()
        batch_size = data.node_feature["n0"].shape[0] // n_pos

        info = {}
        x = data.node_feature["n0"]
        x = x.reshape(batch_size, *self.input_shape, *x.shape[-2:])  # [B, (H, W, D), steps, feature_size]
        time_steps = x.shape[-2]
        x_feature_size = x.shape[-1]
        dyn_dims = dict(to_tuple_shape(data.dyn_dims))["n0"]
        static_dims = x_feature_size - dyn_dims
        if static_dims > 0:
            static_features = x[...,-self.temporal_bundle_steps:,:-dyn_dims]  # [B, (H, W, D), 1, static_dims]
        pos_dims = len(self.input_shape)
        assert pos_dims in [1,2,3]
        if (self.static_encoder_type.startswith("param")) and ("gnn" in self.static_encoder_type):
            # pdb.set_trace()
            x = x[..., static_dims:]
            x_feature_size = x.shape[-1]
        x = x.flatten(start_dim=1+pos_dims) # x: [B, (H, W, D), steps*feature_size]  , [20,64,64,10]
        if self.static_encoder_type.startswith("param"):
            if "gnn" in self.static_encoder_type:
                # pdb.set_trace()
                ### node_feature ###
                x_pyg = data.param["n0"].reshape(batch_size, -1)[0,:].reshape(40,2)   # next time, do 80 instead of 40
                
                ### edge index and edge feature ###
                node_index = torch.tensor([i for i in range(40)], device=x_pyg.device)
                roll_node_index = torch.roll(node_index, 1)
                edge_index = torch.cat([torch.stack([node_index, roll_node_index]), torch.stack([roll_node_index, node_index])], -1)

                if self.gnn_is_virtual:
                    ### Add virtual node ###
                    virtual_node = torch.tensor([[1.,1.]], device=x_pyg.device)
                    x_pyg = torch.cat((x_pyg, virtual_node), 0)

                    ### Create virtual edges ###
                    virtual_node_index = torch.tensor([40], device=x_pyg.device).repeat(40)
                    virtual_edge_index = torch.cat([torch.stack([node_index, virtual_node_index]), torch.stack([virtual_node_index, node_index])], -1)

                    ### Add virtual edges ###
                    edge_index = torch.cat((edge_index, virtual_edge_index), 1)
                                
                edge_attr = x_pyg[edge_index[0]] - x_pyg[edge_index[1]]
                
                ### Define graph ###
                graph = Data(x=x_pyg, edge_index=edge_index, edge_attr=edge_attr) 
                # pdb.set_trace()
                if self.gnn_global_pooling == "mean":
                    single_static_param = torch.mean(self.mpnn(graph).x, dim=0)
                elif self.gnn_global_pooling == "max":
                    single_static_param = torch.max(self.mpnn(graph).x, dim=0)[0]
                elif self.gnn_global_pooling == "sum":
                    single_static_param = torch.sum(self.mpnn(graph).x, dim=0)
                else:
                    raise
                static_param = single_static_param.repeat(batch_size, 1)
            else:
                static_param = data.param["n0"].reshape(batch_size, -1)  # [B, self.static_latent_size]
            static_param_expand = static_param
            for jj in range(pos_dims):
                static_param_expand = static_param_expand[...,None,:]
            static_param_expand = static_param_expand.expand(batch_size, *self.input_shape, static_param_expand.shape[-1])  # [B, (H, W, D), static_dims]
        preds = {"n0": []}

        is_multistep_detach = kwargs["is_multistep_detach"] if "is_multistep_detach" in kwargs else False

        for k in range(1, max_pred_steps + 1):
            # pdb.set_trace()
            # x: [B, (H, W, D), steps*feature_size]
            if self.static_encoder_type.startswith("param"):
                pred = self.model(torch.cat([static_param_expand, x], -1))
            else:
                pred = self.model(x)  # pred: [B, (H, W, D), temporal_bundle_steps*dyn_dims], x: [B, (H, W, D), steps*feature_size+static_latent_size]
            x_reshape = x.reshape(*x.shape[:1+pos_dims], time_steps, x_feature_size)  # [B, (H, W, D), steps, feature_size]
            pred_reshape = pred.reshape(*pred.shape[:1+pos_dims], self.temporal_bundle_steps, dyn_dims)  # [B, (H, W, D), self.temporal_bundle_steps, dyn_dims]
            if k in pred_steps:
                preds["n0"].append(pred_reshape)
            if (self.static_encoder_type.startswith("param")) and ("gnn" in self.static_encoder_type):
                # pdb.set_trace()
                pred_reshape = pred_reshape
            else:
                if static_dims > 0:
                    pred_reshape = torch.cat([static_features, pred_reshape], -1)  # [B, (H, W, D), 1, x_feature_size]
            new_x_reshape = torch.cat([x_reshape, pred_reshape], -2)[...,-time_steps:,:]   # [B, H, W, input_steps, x_feature_size]
            x = new_x_reshape.flatten(start_dim=1+pos_dims)  # x:   # [B, (H, W, D), input_steps*x_feature_size]
            if is_multistep_detach:
                x = x.detach()
        # Before concat, each element of preds["n0"] has shape of [B, (H, W, D), self.temporal_bundle_steps, dyn_dims]
        preds["n0"] = torch.cat(preds["n0"], -2)
        preds["n0"] = preds["n0"].reshape(-1, len(pred_steps) * self.temporal_bundle_steps, dyn_dims)  # [B*n_nodes_B, pred_steps * self.temporal_bundle_steps, dyn_dims]
        return preds, info


    def get_loss(self, data, args, is_rollout=False, **kwargs):
        multi_step_dict = parse_multi_step(args.multi_step)
        preds, info = self(
            data,
            pred_steps=list(multi_step_dict.keys()),
            is_multistep_detach=args.is_multistep_detach,
        )

        original_shape = dict(to_tuple_shape(data.original_shape))
        n_pos = np.array(original_shape["n0"]).prod()
        batch_size = data.node_feature["n0"].shape[0] // n_pos

        # Prediction loss:
        self.info = {}
        loss = 0
        for pred_idx, k in enumerate(multi_step_dict):
            pred_idx_list = np.arange(pred_idx*args.temporal_bundle_steps, (pred_idx+1)*args.temporal_bundle_steps).tolist()
            y_idx_list = np.arange((k-1)*args.temporal_bundle_steps, k*args.temporal_bundle_steps).tolist()
            loss_k = loss_op(
                preds, data.node_label, mask=None if args.is_offmask else data.mask,
                pred_idx=pred_idx_list,
                y_idx=y_idx_list,
                loss_type=args.loss_type,
                batch_size=batch_size,
                is_y_variable_length=args.is_y_variable_length,
                **kwargs
            )
            loss = loss + loss_k
        self.info["loss_pred"] = to_np_array(loss)
        return loss

    @property
    def model_dict(self):
        model_dict = {"type": "FNOModel"}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["input_shape"] = self.input_shape
        model_dict["width"] = self.width
        model_dict["modes"] = self.modes
        model_dict["loss_type"] = self.loss_type
        model_dict["temporal_bundle_steps"] = self.temporal_bundle_steps
        model_dict["static_encoder_type"] = self.static_encoder_type
        model_dict["static_latent_size"] = self.static_latent_size
        model_dict["static_param_size"] = self.static_param_size
        model_dict["state_dict"] = to_cpu(self.state_dict())
        if (self.static_encoder_type.startswith("param")) and ("gnn" in self.static_encoder_type):
            model_dict["gnn_output_dim"] = self.gnn_output_dim
            model_dict["gnn_latent_dim"] = self.gnn_latent_dim
            model_dict["gnn_num_steps"] = self.gnn_num_steps
            model_dict["gnn_layer_norm"] = self.gnn_layer_norm
            model_dict["gnn_activation"] = self.gnn_activation
            model_dict["gnn_diffMLP"] = self.gnn_diffMLP
            model_dict["gnn_global_pooling"] = self.gnn_global_pooling
            model_dict["gnn_is_virtual"] = self.gnn_is_virtual
        return model_dict



# # Training and evaluation:

# ### Test:

# In[ ]:


def build_optimizer(args, params, separate_params=None):
    weight_decay = args.weight_decay
    lr = args.lr
    if separate_params is not None:
        filter_fn = separate_params
    else:
        filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(filter_fn, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=lr, weight_decay=weight_decay)

    if args.lr_scheduler_type == "rop":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_scheduler_factor, verbose=True)
    elif args.lr_scheduler_type == "cos":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min_cos if hasattr(args, "lr_min_cos") else 0)
    elif args.lr_scheduler_type == "cos-re":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_scheduler_T0, T_mult=args.lr_scheduler_T_mult)
    elif args.lr_scheduler_type == "None":
        scheduler = None
    elif args.lr_scheduler_type.startswith("steplr"):
        """Example: "steplr-s100-g0.5" means step_size of 100 epochs and gamma decay of 0.5 (multiply 0.5 every 100 epochs)"""
        scheduler_dict = {}
        for item in args.lr_scheduler_type.split("-")[1:]:
            if item[0] == "s":
                scheduler_dict[item[0]] = int(item[1:])
            elif item[0] == "g":
                scheduler_dict[item[0]] = float(item[1:])
            else:
                raise
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_dict["s"], gamma=scheduler_dict["g"])
    else:
        raise
    return optimizer, scheduler


def test(data_loader, model, device, args, **kwargs):
    model.eval()
    count = 0
    total_loss = 0
    multi_step_dict = parse_multi_step(args.multi_step)
    args = deepcopy(args)
    info = {}
    keys_pop = []
    for key, value in kwargs.items():
        if (isinstance(value, Number) or isinstance(value, str)) and key != "current_epoch" and key != "current_minibatch":
            setattr(args, key, value)
            keys_pop.append(key)
    for key in keys_pop:
        kwargs.pop(key)
    # Compute loss:
    for jj, data in enumerate(data_loader):
        with torch.no_grad():
            data = data.to(device)
            if isinstance(data, Data):
                data = PyG_to_Attr_Dict(data)
            batch_size = get_batch_size(data)
            if args.normalization_type == "gn" and args.is_latent_flatten is True and args.latent_size == 2 and batch_size == 1:
                continue

            if is_diagnose(loc="test:1", filename=args.filename):
                pdb.set_trace()
            args = deepcopy(args)
            args.zero_weight = 1.0
            loss = model.get_loss(data, args, is_rollout=True, **kwargs).item()
            keys, values = get_keys_values(model.info, exclude=["pred"])
            record_data(info, values, keys)
            total_loss = total_loss + loss
            count += 1
    for key, item in info.items():
        info[key] = np.mean(item)
    if count == 0:
        return None, info
    else:
        return total_loss / count, info

    
def unittest_model(model, data, args, device, use_grads=True, use_pos=False, is_mesh=False, test_cases="all", algo="contrastive", **kwargs):
    """Test if the loaded model is exactly the same as the original model."""
    if test_cases == "all":
        test_cases = ["model_dict", "pred", "loss"]
    if not isinstance(test_cases, list):
        test_cases = [test_cases]

    def dataparallel_compat(model_dict):
        model_dict_copy = {}
        for (k, v) in model_dict.items():
            model_dict_copy[k.replace("module.", "")] = v
        return model_dict_copy

    with torch.no_grad():
        data = deepcopy(data).to(device)
        model.eval()
        multi_gpu=len(args.gpuid.split(",")) > 1
        model2 = load_model(get_model_dict(model), device, multi_gpu=multi_gpu)
        # model2.type(list(model.parameters())[0].dtype)
        model2.eval()

        if "model_dict" in test_cases:
            model_dict = get_model_dict(model)
            model_dict2 = get_model_dict(model2)
            diff_sum = 0
            # All other key: values must match:
            # pdb.set_trace()
            check_same_model_dict(model_dict, model_dict2)
            # The state_dict must match:
            if model_dict["type"] not in ["Actor_Critic"]:
                for k, v in model_dict["state_dict"].items():
                    v2 = model_dict2["state_dict"][k]
                    diff_sum += (v - v2).abs().max()
                assert diff_sum == 0, "The model_dict of the loaded model is not the same as the original model!"

        if "pred" in test_cases and model_dict["type"] not in ["Actor_Critic"]:
            # Evaluate the difference for three times:
            #pdb.set_trace()
            if is_mesh:
                data_c = deepcopy(data)
                pred, _ = model.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2, _ = model2.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                pred["n0"] = pred["n0"][0]
                pred2["n0"] = pred2["n0"][0]
            else:
                data_c = deepcopy(data)
                pred, _ = model(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2, _ = model2(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff = 0.0
            #pdb.set_trace()
            try:
                max_diff = max([(pred[key] - pred2[key]).abs().max().item() for key in pred])
            except:
                max_diff = max([(pred.node_feature[nt] - pred2.node_feature[nt]).abs().max().item() for nt in pred.node_feature])

            if is_mesh:
                data_c = deepcopy(data)
                pred_2, _ = model.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2_2, _ = model2.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                pred_2["n0"] = pred_2["n0"][0]
                pred2_2["n0"] = pred2_2["n0"][0]
            else:
                data_c = deepcopy(data)
                pred_2, _ = model(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2_2, _ = model2(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff_2 = 0.0
            try:
                max_diff_2 = max([(pred_2[key] - pred2_2[key]).abs().max().item() for key in pred])
            except:
                max_diff_2 = max([(pred_2.node_feature[nt] - pred2_2.node_feature[nt]).abs().max().item() for nt in pred.node_feature])

            if is_mesh:
                if max_diff < 9e-2 and max_diff_2 < 9e-2:
                    print("\nThe maximum difference between the predictions are {:.4e} and {:.4e}, within error tolerance.\n".format(max_diff, max_diff_2))
                else:
                    raise Exception("\nThe loaded model for 2d mesh is not exactly the same as the original model! The maximum difference between the predictions are {:.4e} and {:.4e}.\n".format(max_diff, max_diff_2))                
            else:
                if max_diff < 8e-5 and max_diff_2 < 8e-5:
                    print("\nThe maximum difference between the predictions for 2d meshes are {:.4e} and {:.4e}, within error tolerance.\n".format(max_diff, max_diff_2))
                else:
                    raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e} and {:.4e}.\n".format(max_diff, max_diff_2))

        if "one_step_interpolation" in test_cases:
            data_c = deepcopy(data)
            set_seed(42)
            pred, info, _ = model.one_step_interpolation_forward(data_c, 0)
            data_cc = deepcopy(data)
            set_seed(42)
            pred2, info2, _ = model.one_step_interpolation_forward(data_cc, 0)
            max_diff = 0.0
            max_diff = max([(pred["n0"][0] - pred2["n0"][0]).abs().max().item() for key in pred])
            if max_diff < 8e-5:
                print("\nThe output shape:{}".format(pred["n0"][0].shape))
                print("\nThe maximum difference between the predictions are {:.4e}, within error tolerance.\n".format(max_diff))
            else:
                raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e}.\n".format(max_diff))     

                    
        if "value_model" in test_cases:
            data_c = deepcopy(data)
            set_seed(42)
            pred = model(data_c, use_grads=use_grads, use_pos=use_pos)
            data_c = deepcopy(data)
            set_seed(42)
            pred2 = model2(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff = 0.0
            max_diff = max([(pred - pred2).abs().max().item() for key in pred])
            if max_diff < 8e-5:
                print("\nThe output shape:{}".format(pred.shape))
                print("\nThe maximum difference between the predictions are {:.4e}, within error tolerance.\n".format(max_diff))
            else:
                raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e}.\n".format(max_diff))     
            set_seed(args.seed)

        if "gnnpolicysizing" in test_cases:
            
            data_c = deepcopy(data)
            set_seed(42)
            pred,prob,entropy,dict1 = model.remeshing_forward_GNN(data_c, use_grads=use_grads, use_pos=use_pos)
            data_c = deepcopy(data)
            set_seed(42)
            pred2,prob2,entropy2,dict2 = model2.remeshing_forward_GNN(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff = 0.0
            max_diff = max([(prob[key] - prob2[key]).abs().max().item() for key in prob.keys()])
            print("max_diff, prob",max_diff)
            max_diff +=  max([(entropy[key] - entropy2[key]).abs().max().item() for key in prob.keys()])
            print("max_diff+entropy, prob",max_diff)
            if max_diff < 5e-5:
                print("\nThe maximum difference between the predictions are {:.4e}, within error tolerance.\n".format(max_diff))
            else:
                raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e}.\n".format(max_diff))  
            set_seed(args.seed)
                
        # Get difference in loss components:
        if "loss" in test_cases:
            # kwargs2 = deepcopy(kwargs)
            kwargs2 = kwargs
            set_seed(42)
            # model.type(torch.float64)
            # kwargs["evolution_model"].type(torch.float64)
            loss1 = model.get_loss(
                deepcopy(data), args,
                current_epoch=0,
                current_minibatch=0,
                **kwargs
            )
            info1 = deepcopy(model.info)
            set_seed(42)
            # model2.type(torch.float64)
            # kwargs2["evolution_model"].type(torch.float64)
            loss2 = model2.get_loss(
                deepcopy(data), args,
                current_epoch=0,
                current_minibatch=0,
                **kwargs2
            )
            info2 = deepcopy(model2.info)
            for key in info1:
                if key not in [
                    "loss_reward", "t_evolve", "t_evolve_alt", "t_evolve_interp_alt",
                    "loss_value", "loss_actor", "loss_reinforce",
                    "r_timediff", "r_timediff_alt", "r_timediff_interp_alt", "interp_r_timediff",
                    "v_timediff", "v_timediff_alt", "v_timediff_interp_alt", "interp_v_timediff",
                    "v_beta", "interp_v_beta", "v/timediff", "r/timediff",
                ]:
                    print("{} \t{:.4e}\t{:.4e}\tdiff: {:.4e}".format("{}:".format(key).ljust(16), info1[key], info2[key], abs(info1[key] - info2[key])))
                    if is_mesh:
                        if abs(info1[key] - info2[key]) > 1e-3:
                            raise Exception("{} for the loaded model differs by {:.4e}, which is more than 8e-5.".format(key, abs(info1[key] - info2[key])))
                    else:
                        if abs(info1[key] - info2[key]) > 8e-6:
                            pdb.set_trace()
                            raise Exception("{} for the loaded model differs by {:.4e}, which is more than 5e-6.".format(key, abs(info1[key] - info2[key])))
            set_seed(args.seed)
        print(get_time(), "Unittest passed for test cases {}!".format(test_cases))


def get_Hessian_penalty(
    G,
    z,
    mode,
    k=2,
    epsilon=0.1,
    reduction=torch.max,
    return_separately=False,
    G_z=None,
    is_nondimensionalize=False,
    **G_kwargs
):
    """
    Adapted from https://github.com/wpeebles/hessian_penalty/ (Peebles et al. 2020).
    Note: If you want to regularize multiple network activations simultaneously, you need to
    make sure the function G you pass to hessian_penalty returns a list of those activations when it's called with
    G(z, **G_kwargs). Otherwise, if G returns a tensor the Hessian Penalty will only be computed for the final
    output of G.
    
    Args:
        G: Function that maps input z to either a tensor or a list of tensors (activations)
        z: Input to G that the Hessian Penalty will be computed with respect to
        mode: choose from "Hdiag", "Hoff" or "Hall", specifying the scope of Hessian values to perform sum square on. 
                "Hall" will be the sum of "Hdiag" (for diagonal elements) and "Hoff" (for off-diagonal elements).
        k: Number of Hessian directions to sample (must be >= 2)
        epsilon: Amount to blur G before estimating Hessian (must be > 0)
        reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
        return_separately: If False, hessian penalties for each activation output by G are automatically summed into
                              a final loss. If True, the hessian penalties for each layer will be returned in a list
                              instead. If G outputs a single tensor, setting this to True will produce a length-1
                              list.
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>
    :return: A differentiable scalar (the hessian penalty), or a list of hessian penalties if return_separately is True
    """
    if G_z is None:
        G_z = G(z, **G_kwargs)
    rademacher_size = torch.Size((k, *z.size()))  # (k, N, z.size())
    if mode == "Hall":
        loss_diag = get_Hessian_penalty(G=G, z=z, mode="Hdiag", k=k, epsilon=epsilon, reduction=reduction, return_separately=return_separately, G_z=G_z, **G_kwargs)
        loss_offdiag = get_Hessian_penalty(G=G, z=z, mode="Hoff", k=k, epsilon=epsilon, reduction=reduction, return_separately=return_separately, G_z=G_z, **G_kwargs)
        if return_separately:
            loss = []
            for loss_i_diag, loss_i_offdiag in zip(loss_diag, loss_offdiag):
                loss.append(loss_i_diag + loss_i_offdiag)
        else:
            loss = loss_diag + loss_offdiag
        return loss
    elif mode == "Hdiag":
        xs = epsilon * complex_rademacher(rademacher_size, device=z.device)
    elif mode == "Hoff":
        xs = epsilon * rademacher(rademacher_size, device=z.device)
    else:
        raise
    second_orders = []

    if mode == "Hdiag" and isinstance(G, nn.Module):
        # Use the complex64 dtype:
        dtype_ori = next(iter(G.parameters())).dtype
        G.type(torch.complex64)
    if isinstance(G, nn.Module):
        G_wrapper = get_listified_fun(G)
        G_z = listity_tensor(G_z)
    else:
        G_wrapper = G

    for x in xs:  # Iterate over each (N, z.size()) tensor in xs
        central_second_order = multi_layer_second_directional_derivative(G_wrapper, z, x, G_z, epsilon, **G_kwargs)
        second_orders.append(central_second_order)  # Appends a tensor with shape equal to G(z).size()
    loss = multi_stack_metric_and_reduce(second_orders, mode, reduction, return_separately)  # (k, G(z).size()) --> scalar

    if mode == "Hdiag" and isinstance(G, nn.Module):
        # Revert back to original dtype:
        G.type(dtype_ori)

    if is_nondimensionalize:
        # Multiply a factor ||z||_2^2 so that the result is dimensionless:
        factor = z.square().mean()
        if return_separately:
            loss = [ele * factor for ele in loss]
        else:
            loss = loss * factor
    return loss

def rollout(
    dataset,
    init_step,
    model,
    device,
    algo,
    n_rollout_steps=100,
    interval=20,
    n_plots_row=6,
    use_grads=True,
    is_y_diff=False,
    loss_type="mse",
    isplot=2,
    dataset_name=None,
    **kwargs
):
    """Rollout for multiple time steps."""
    def plot_loss_list(loss_list, key):
        from sklearn.linear_model import LinearRegression
        fontsize=14
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        power_list = []
        for k in range(dyn_dims[key]):
            plt.plot(loss_list[:,k], label="{}th".format(k))
            linear_rg = LinearRegression().fit(np.log(np.arange(1, 1+len(loss_list[k])))[:, None], np.log(loss_list[k]))
            power_list.append(linear_rg.coef_[0])
        plt.xlabel("steps", fontsize=fontsize)
        plt.ylabel("mse", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.subplot(1,2,2)
        for k in range(dyn_dims[key]):
            plt.semilogy(loss_list[:,k], label="{}th".format(k))
        plt.xlabel("steps", fontsize=fontsize)
        plt.ylabel("mse", fontsize=fontsize)
        plt.title("power: {}".format(to_string(power_list, num_digits=3, connect=", ")))
        plt.tick_params(labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.show()

    def plot_array1D_time(pred, target, vmin, vmax):
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        plt.imshow(pred, vmin=vmin, vmax=vmax, origin="lower")
        plt.subplot(1,2,2)
        plt.imshow(target, vmin=vmin, vmax=vmax, origin="lower")
        plt.show()

    if isplot > 0:
        print_banner("Evaluate rollouts starting at t={}, for steps {} further:".format(init_step, np.concatenate([np.arange(0, n_rollout_steps, interval), np.array([n_rollout_steps-1])], 0)))
    model.eval()

    data = deepcopy(dataset[init_step]).to(device)  # node_feature: [n_nodes, input_steps, static_dims + dyn_dims]

    dyn_dims = dict(to_tuple_shape(data.dyn_dims))
    original_shape = dict(to_tuple_shape(data.original_shape))
    pos_dims = get_pos_dims_dict(original_shape)
    grid_keys = data.grid_keys
    loss_1step_list = []
    if algo in ["gns", "unet"]:
        preds = {}
        for i in range(n_rollout_steps):
            with torch.no_grad():
                # The returned data does not contain grads:
                data, pred = get_data_next_step(model, data, use_grads=use_grads, return_data=True, is_y_diff=is_y_diff)
                record_data(preds, list(data.node_feature.values()), list(data.node_feature.keys()))
                loss = to_np_array(loss_op(pred, dataset[init_step + i].to(device).node_label, dataset[init_step + i].mask, y_idx=0, loss_type="mse", reduction="mean-dyn", **kwargs), full_reduce=False)
                loss_1step_list.append(loss)
        for key in preds:
            preds[key] = torch.cat(preds[key], 1)[..., -dyn_dims[key]:]  # [n_nodes, pred_steps: n_rollout_steps, dyn_dims]
    elif algo in ["contrast", "contrast-ebm"]:
        preds, info = model(data, pred_steps=np.arange(1, n_rollout_steps+1), use_grads=use_grads, is_recons=True, is_y_diff=is_y_diff, is_rollout=True)  # [n_nodes, pred_steps: n_rollout_steps, dyn_dims]
        for i in range(n_rollout_steps):
            pred = {key: preds[key][...,i:i+1,:] for key in preds}
            loss = to_np_array(loss_op(pred, dataset[init_step+i].to(device).node_label, dataset[init_step+i].mask, y_idx=0, loss_type="mse", reduction="mean-dyn", keys=to_tuple_shape(grid_keys), **kwargs), full_reduce=False)
            loss_1step_list.append(loss)
        # Compute latent targets:
        latent_targets = []
        for i in range(n_rollout_steps):
            data = deepcopy(dataset[init_step+i+1])
            latent_targets.append(model.encoder(data, use_grads=use_grads))
        if not isinstance(latent_targets[0], tuple):
            latent_targets = torch.stack(latent_targets, 1)
        else:
            latent_targets = stack_tuple_elements(latent_targets, 1)
        info["latent_targets"] = latent_targets
    else:
        raise Exception("algo '{}' is not supported!".format(algo))
    loss_1step_list = np.stack(loss_1step_list)  # After stack: [n_rollout_steps, dyn_dims]
    permute_order = {key: (pos_dims[key],) + tuple(range(pos_dims[key])) + (pos_dims[key] + 1,) for key in pos_dims}
    # before: [n_grids, pred_steps, dyn_dims]; reshape: [[pos_dims], pred_steps, dyn_dims]; transpose: [pred_steps, [pos_dims], dyn_dims]:
    preds_list = {key: to_np_array(preds[key].reshape(*original_shape[key if key in grid_keys else grid_keys[0]], *preds[key].shape[1:])).transpose(*permute_order[key if key in grid_keys else grid_keys[0]]) for key in preds}

    targets = {}
    loss_list_dict = {}
    MAE_list_dict = {}
    for key in data.node_feature:
        # Get ground-truth:
        array_gt = []
        for k in range(n_rollout_steps):
            data_gt = dataset[init_step + 1 + k]
            array_gt.append(to_np_array(data_gt.node_feature[key][:, 0, -dyn_dims[key]:].reshape(1, *original_shape[key if key in grid_keys else grid_keys[0]], dyn_dims[key])))
        array_gt = np.vstack(array_gt)
        targets[key] = array_gt
        if pos_dims[key] == 2:
            loss_list_dict[key] = ((array_gt - preds_list[key]) ** 2).mean((1,2))
            MAE_list_dict[key] = np.abs(array_gt - preds_list[key]).mean((1,2))
        elif pos_dims[key] == 1:
            loss_list_dict[key] = ((array_gt - preds_list[key]) ** 2).mean(1)
            MAE_list_dict[key] = np.abs(array_gt - preds_list[key]).mean(1)
        else:
            raise

    if dataset_name is not None and dataset_name.startswith("VL-small"):
        if isplot >= 1:
            for i in range(0, n_rollout_steps, interval):
                print("Rollout at {}".format(i))
                plot_vlasov(spec=preds_list["2"][i, ..., 0], fld=preds_list["0"][i, ..., 0], 
                            spec2=targets["2"][i, ..., 0], fld2=targets["0"][i, ..., 0], 
                            mask=data.mask, mask_outer=dataset.mask_outer if hasattr(dataset, "mask_outer") else None,
                            title=("pred", "target"),
                            dataset_name=dataset_name,
                            vmin=0, vmax=7,
                           )
            print("pred:")
            plot_energy(preds_list, dataset_name=dataset_name)
            print("gt:")
            plot_energy(targets, dataset_name=dataset_name)
    else:
        for key in data.node_feature:
            array_gt = targets[key]
            # vmax = array_gt.max()
            # vmin = array_gt.min()
            # if abs(vmin) > abs(vmax) / 2:
            #     vmin = -vmax
            # else:
            #     vmin = 0
            if isplot >= 2:
                for k in range(dyn_dims[key]):
                    vmax = array_gt[...,k].max()
                    vmin = array_gt[...,k].min()
                    if abs(vmin) > abs(vmax) / 2:
                        vmin = -vmax
                    else:
                        vmin = 0
                    print("{}: the {}th dynamic feature in rollout, with maximum={:.4f}:".format(key, k, preds_list[key][:, k].max()))
                    if len(array_gt.shape) == 4:  # 2D grid
                        print("ground-truth:")
                        plot_array(array_gt[...,k], interval=interval, vmin=vmin, vmax=vmax, n_plots_row=n_plots_row)
                        print("prediction:")
                        plot_array(preds_list[key][...,k], interval=interval, loss_list=loss_list_dict[key][:,k], vmin=vmin, vmax=vmax, n_plots_row=n_plots_row)
                        print("diff:")
                        diff = preds_list[key][...,k] - array_gt[...,k]
                        plot_array(diff, interval=interval, vmin=-vmax, vmax=vmax, is_range=True, n_plots_row=n_plots_row, cmap="PiYG")
                    elif len(array_gt.shape) == 3:  # 1D grid
                        # Plot snapshots at intervals:
                        plot_array1D(preds_list[key][...,k], array_gt[...,k], loss_list=loss_list_dict[key][:, k], interval=interval, vmin=vmin, vmax=vmax, is_range=False, n_plots_row=n_plots_row)
                        # Plot full rollouts across time (x-axis: x, y-axis: time):
                        plot_array1D_time(preds_list[key][...,k], array_gt[...,k], vmin=vmin, vmax=vmax)
                    print()

            # Plot loss curve w.r.t. rollout steps:
            if isplot >= 1:
                print("MSE, cumulative:")
                plot_loss_list(loss_list_dict[key], key)
                print("MSE, cumulative input 1-step target:")
                plot_loss_list(loss_1step_list, key)
    losses_rollout_all = {"loss_list_dict": loss_list_dict,
                          "MAE_list_dict": MAE_list_dict,
                          "loss_1step_list": loss_1step_list,
                         }
    return (preds_list, targets), losses_rollout_all, info

def plot_array(array, interval=10, loss_list=None, is_range=False, vmin=0, vmax="start", n_plots_row=5, cmap="PiYG"):
    """
    Args:
        array: has shape of [steps, rows, cols, dyn_dims].
        interval: interval by which you want to plot the array.
        vmin, vmax: choose from 
            None: will use the min/max in the full array
            "start": use the min/max in the starting array
            "auto": use each respective array
            a number: user specified value.
        loss_list: losses corresponding to each array, for making subtitles.
        is_range: if True, will put the range at the subtitles.
        n_plots_row: number of subplots per row. Default 5.
        cmap: colormap for matplotlib.
    """
    nplots = int(np.ceil(len(array) / interval))
    nrows = int(np.ceil(nplots / n_plots_row))
    fig, axes = plt.subplots(nrows=nrows, ncols=n_plots_row, figsize=(22, 6*nrows))
    array_s = array[::interval]
    if vmin is None:
        vmin = array_s.min()
    elif vmin == "start":
        vmin = array_s[0].min()
    if vmax is None:
        vmax = array_s.max()
    elif vmax == "start":
        vmax = array_s[0].max()
    idx = 0
    for i, ax in enumerate(array):
        if i % interval == 0 or i == len(array) - 1: 
            if nrows > 1:
                row, col = divmod(idx, n_plots_row)
                ax = axes[row][col]
            else:
                ax = axes[idx]
            idx += 1
            im = ax.imshow(array[i], vmin=vmin if vmin!="auto" else None, vmax=vmax if vmax!="auto" else None, cmap=cmap)
            if is_range:
                ax.set_title("{}: range: [{:.3f}, {:.3f}]\nm: {:.2e}, std: {:.3f}".format(i, array[i].min(), array[i].max(), array[i].mean(), array[i].std()))
            else:
                if loss_list is not None:
                    ax.set_title("{}: mse: {:.6f}".format(i, loss_list[i]))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
