from collections import OrderedDict
from copy import deepcopy
from numbers import Number
import pdb
import torch
from torch.fft import fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import yaml

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from le_pde.pytorch_net.util import lcm, L2Loss, Attr_Dict, Printer

p = Printer(n_digits=6)

INVALID_VALUE = -200
PDE_PATH = "./dataset/"
EXP_PATH = "./results/"
DESIGN_PATH = ".."
MPPDE1D_PATH = "mppde1d_data/"
FNO_PATH = "fno_data/"
MOVINGGAS_PATH = "movinggas_data/"
KARMAN3D_PATH = "karman3d_data"


def flatten(tensor):
    """Flatten the tensor except the first dimension."""
    return tensor.reshape(tensor.shape[0], -1)


def get_activation(act_name, inplace=False):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "linear":
        return nn.Identity()
    elif act_name == "leakyrelu":
        return nn.LeakyReLU(inplace=inplace)
    elif act_name == "leakyrelu0.2":
        return nn.LeakyReLU(inplace=inplace, negative_slope=0.2)
    elif act_name == "elu":
        return nn.ELU(inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "softplus":
        return nn.Softplus()
    elif act_name == "exp":
        return Exp()
    elif act_name == "sine":
        from siren_pytorch import Sine
        return Sine()
    elif act_name == "rational":
        return Rational()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "celu":
        return nn.CELU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "prelu":
        return nn.PReLU()
    elif act_name == "rrelu":
        return nn.RReLU()
    elif act_name == "mish":
        return nn.Mish()
    else:
        raise Exception("act_name '{}' is not valid!".format(act_name))


class Apply_Activation(nn.Module):
    def __init__(self, apply_act_idx, act_name="relu", dim=1):
        super().__init__()
        if isinstance(apply_act_idx, str):
            apply_act_idx = [int(ele) for ele in apply_act_idx.split(",")]
        self.apply_act_idx = apply_act_idx
        self.act = get_activation(act_name)
        self.dim = dim

    def forward(self, input):
        assert len(input.shape) >= 2  # []
        out = []
        for i in range(input.shape[self.dim]):
            if i in self.apply_act_idx:
                if self.dim == 1:
                    out.append(self.act(input[:,i]))
                elif self.dim == 2:
                    out.append(self.act(input[:,:,i]))
                else:
                    raise
            else:
                if self.dim == 1:
                    out.append(input[:,i])
                elif self.dim == 2:
                    out.append(input[:,:,i])
                else:
                    raise
        return torch.stack(out, self.dim)


def get_LCM_input_shape(input_shape):
    input_shape_dict = dict(input_shape)
    key_max = None
    shape_len_max = -np.Inf
    for key, shape in input_shape_dict.items():
        if len(shape) > shape_len_max:
            key_max = key
            shape_len_max = len(shape)
    return input_shape_dict[key_max]


def expand_same_shape(x_list, input_shape_LCM):
    pos_dim_max = len(input_shape_LCM)
    x_expand_list = []
    for x in x_list:
        shape = x.shape
        if len(x.shape) < pos_dim_max + 2:
            for i in range(pos_dim_max + 2 - len(x.shape)):
                x = x.unsqueeze(2)  # Here the field [B, C, X] is expanded to the full distribution [B, C, U, X]
            x = x.expand(x.shape[:2] + input_shape_LCM)
            x_expand_list.append(x)
        else:
            x_expand_list.append(x)
    return x_expand_list


def get_data_next_step(
    model,
    data,
    use_grads=True,
    is_y_diff=False,
    return_data=True,
    forward_func_name=None,
    is_rollout=False,
):
    """Apply the model to data and obtain the data at the next time step without grads.

    Args:
        data:
            The returned data has features of
                [computed_features [not containing grad], static_features, dyn_features]

            The input data.node_feature does not contain the grads information.
            if return_data is False, will only return pred.
            if return_data is True, will also return the full data incorporating the prediction.

        forward_func_name: if None, will use the model's own forward function. If a string, will use model.forward_func_name as the forward function.
        is_rollout: if is_rollout=True, will stop gradient.

    Returns:
        pred: {key: [n_nodes, pred_steps, dyn_dims]}
    """
    dyn_dims = dict(to_tuple_shape(data.dyn_dims))  # data.node_feature: [n_nodes, input_steps, static_dims + dyn_dims]
    compute_func_dict = dict(to_tuple_shape(data.compute_func))
    static_dims = {key: data.node_feature[key].shape[-1] - dyn_dims[key] - compute_func_dict[key][0] for key in data.node_feature}

    # Compute pred:
    # After this application, the data.node_feature may append the grads information at the left:
    if is_rollout:
        with torch.no_grad():
            if forward_func_name is None:
                pred, _ = model(data, use_grads=use_grads)  # pred: [n_nodes, pred_steps, dyn_dims]
            else:
                pred, _ = getattr(model, forward_func_name)(data, use_grads=use_grads)  # pred: [n_nodes, pred_steps, dyn_dims]
    else:
        if forward_func_name is None:
            pred, _ = model(data, use_grads=use_grads)  # pred: [n_nodes, pred_steps, dyn_dims]
        else:
            pred, _ = getattr(model, forward_func_name)(data, use_grads=use_grads)  # pred: [n_nodes, pred_steps, dyn_dims]

    if not return_data:
        return None, pred
    # Update data:
    for key in pred:
        compute_dims = compute_func_dict[key][0]
        dynamic_features = pred[key]
        if is_y_diff:
            dynamic_features = dynamic_features + data.node_feature[key][..., -dyn_dims[key]:]
        # Append the computed node features:
        # [computed + static + dynamic]
        input_steps = data.node_feature[key].shape[-2]
        # pdb.set_trace()
        if input_steps > 1:
            dynamic_features = torch.cat([data.node_feature[key][...,-dyn_dims[key]:], dynamic_features], -2)[...,-input_steps:,:]
        static_features = data.node_feature[key][..., -static_dims[key]-dyn_dims[key]:-dyn_dims[key]]
        # The returned data will not contain grad information:
        if compute_dims > 0:
            compute_features = compute_func_dict[key][1](dynamic_features)
            node_features = torch.cat([compute_features, static_features, dynamic_features], -1)
        else:
            node_features = torch.cat([static_features, dynamic_features], -1)
        data.node_feature[key] = node_features
    return data, pred


def get_loss_ar(
    model,
    data,
    multi_step,
    use_grads=True,
    is_y_diff=False,
    loss_type="mse",
    **kwargs
):
    """Get auto-regressive loss for multiple steps."""
    multi_step_dict = parse_multi_step(multi_step)
    if len(multi_step_dict) == 1 and next(iter(multi_step_dict)) == 1:
        # Single-step prediction:
        pred, _ = model(data, use_grads=use_grads)
        loss = loss_op(pred, data.node_label, mask=data.mask, y_idx=0, loss_type=loss_type, **kwargs)
    else:
        # Multi-step prediction:
        max_step = max(list(multi_step_dict.keys()))
        loss = 0
        dyn_dims = dict(to_tuple_shape(data.dyn_dims))
        for i in range(1, max_step + 1):
            if i != max_step:
                data, _ = get_data_next_step(model, data, use_grads=use_grads, is_y_diff=is_y_diff, return_data=True)
                if i in multi_step_dict:
                    pred_new = {key: item[..., -dyn_dims[key]:] for key, item in data.node_feature.items()}
            else:
                _, pred_new = get_data_next_step(model, data, use_grads=use_grads, is_y_diff=is_y_diff, return_data=False)
            if i in multi_step_dict:
                loss_i = loss_op(pred_new, data.node_label, data.mask, y_idx=i-1, loss_type=loss_type, **kwargs)
                loss = loss + loss_i * multi_step_dict[i]
    return loss


def get_precision_floor(loss_type):
    """Get precision_floor from loss_type, if mselog, huberlog or l1 is inside loss_type. Otherwise return None"""
    precision_floor = None
    if loss_type is not None and ("mselog" in loss_type or "huberlog" in loss_type or "l1log" in loss_type):
        string_all = loss_type.split("+")
        for string in string_all:
            if "mselog" in string or "huberlog" in string or "l1log" in string:
                precision_floor = eval(string.split("#")[1])
                break
    return precision_floor


def loss_op(
    pred,
    y,
    mask=None,
    pred_idx=None,
    y_idx=None,
    dyn_dims=None,
    loss_type="mse",
    keys=None,
    reduction="mean",
    time_step_weights=None,
    normalize_mode="None",
    is_y_variable_length=False,
    **kwargs
):
    """Compute loss.

    Args:
        pred: shape [n_nodes, pred_steps, features]
        y: shape [n_nodes, out_steps, dyn_dims]
        mask: shape [n_nodes]
        pred_idx: range(0, pred_steps)
        y_idx: range(0, out_steps)
        dyn_dims: dictionary of {key: number of dynamic dimensions}. If not None, will get loss from pred[..., -dyn_dims:].
        loss_type: choose from "mse", "huber", "l1" and "dl", or use e.g. "0:huber^1:mse" for {"0": "huber", "1": "mse"}.
                   if "+" in loss_type, the loss will be the sum of multiple loss components added together.
                   E.g., if loss_type == '0:mse^2:mse+l1log#1e-3', then the loss is
                   {"0": mse, "2": mse_loss + l1log loss}
        keys: if not None, will only go through the keys provided. If None, will use the keys in "pred".
        time_step_weights: if not None but an array, will weight each time step by some coefficients.
        reduction: choose from "mean", "sum", "none" and "mean-dyn" (mean on the loss except on the last dyn_dims dimension).
        normalize_mode: choose from "target", "targetindi", "None". If "target", will divide the loss by the global norm of the target. 
                   if "targetindi", will divide the each individual loss by the individual norm of the target example. 
                   Default "None" will not normalize.
        **kwargs: additional kwargs for loss function.

    Returns:
        loss: loss.
    """
    # Make pred and y both dictionary:
    if (not isinstance(pred, dict)) and (not isinstance(y, dict)):
        pred = {"key": pred}
        y = {"key": y}
        if mask is not None:
            mask = {"key": mask}
    if keys is None:
        keys = list(pred.keys())

    # Individual loss components:
    loss = 0
    if time_step_weights is not None:
        assert len(time_step_weights.shape) == 1
        reduction_core = "none"
    else:
        if reduction == "mean-dyn":
            # Will perform mean on the loss except on the last dyn_dims dimension:
            reduction_core = "none"
        else:
            reduction_core = reduction
    if is_y_variable_length and loss_type != "lp":
        reduction_core = "none"
    if "^" in loss_type:
        # Different loss for different keys:
        loss_type_dict = parse_loss_type(loss_type)
    else:
        loss_type_dict = {key: loss_type for key in keys}

    # Compute loss
    for key in keys:
        # Specify which time step do we want to use from y:
        #   y has shape of [n_nodes, output_steps, dyn_dims]
        if pred[key] is None:
            # Due to latent level turning off:
            assert y[key] is None
            continue
        elif isinstance(pred[key], list) and len(pred[key]) == 0:
            # The multi_step="" or latent_multi_step="":
            continue
        if pred_idx is not None:
            if not isinstance(pred_idx, list):
                pred_idx = [pred_idx]
            pred_core = pred[key][..., pred_idx, :]
        else:
            pred_core = pred[key]
        if y_idx is not None:
            if not isinstance(y_idx, list):
                y_idx = [y_idx]
            y_core = y[key][..., y_idx, :]
        else:
            y_core = y[key]

        # y_core: [n_nodes, output_steps, dyn_dims]
        if is_y_variable_length:
            is_nan_full = (y_core == INVALID_VALUE).view(kwargs["batch_size"], -1, *y_core.shape[-2:])  # [batch_size, n_nodes_per_batch, output_steps, dyn_dims]
            n_nodes_per_batch = is_nan_full.shape[1]
            is_not_nan_batch = ~is_nan_full.any(1).any(-1)  # [batch_size, output_steps]
            if is_not_nan_batch.sum() == 0:
                continue
            is_not_nan_batch = is_not_nan_batch[:,None,:].expand(kwargs["batch_size"], n_nodes_per_batch, is_not_nan_batch.shape[-1])  # [batch_size, n_nodes_per_batch, output_steps]
            is_not_nan_batch = is_not_nan_batch.reshape(-1, is_not_nan_batch.shape[-1])  # [n_nodes, output_steps]
            if loss_type == "lp":
                kwargs["is_not_nan_batch"] = is_not_nan_batch[..., None]  # [n_nodes, output_steps, 1]
        else:
            is_not_nan_batch = None
        if dyn_dims is not None:
            y_core = y_core[..., -dyn_dims[key]:]
        if mask is not None:
            pred_core = pred_core[mask[key]]  # [n_nodes, pred_steps, dyn_dims]
            y_core = y_core[mask[key]]      # [n_nodes, output_steps, dyn_dims]
        # Compute loss:
        loss_i = loss_op_core(pred_core, y_core, reduction=reduction_core, loss_type=loss_type_dict[key], normalize_mode=normalize_mode, **kwargs)

        if time_step_weights is not None:
            shape = loss_i.shape
            assert len(shape) >= 3  # [:, time_steps, ...]
            time_step_weights = time_step_weights[None, :]
            for i in range(len(shape) - 2):
                time_step_weights = time_step_weights.unsqueeze(-1)  # [:, time_steps, [1, ...]]
            loss_i = loss_i * time_step_weights
            if is_y_variable_length and is_not_nan_batch is not None:
                loss_i = loss_i[is_not_nan_batch]
            if reduction == "mean-dyn":
                assert len(shape) == 3
                loss_i = loss_i.mean((0,1))
            else:
                loss_i = reduce_tensor(loss_i, reduction)
        elif is_y_variable_length and is_not_nan_batch is not None and loss_type != "lp":
            loss_i = loss_i[is_not_nan_batch]
            loss_i = reduce_tensor(loss_i, reduction)
        else:
            if reduction == "mean-dyn":
                assert len(loss_i.shape) == 3
                loss_i = loss_i.mean((0,1))  # [dyn_dims,]

        if loss_type == "rmse":
            loss_i = loss_i.sqrt()
        loss = loss + loss_i
    return loss


def reduce_tensor(tensor, reduction, dims_to_reduce=None, keepdims=False):
    """Reduce tensor using 'mean' or 'sum'."""
    if reduction == "mean":
        if dims_to_reduce is None:
            tensor = tensor.mean()
        else:
            tensor = tensor.mean(dims_to_reduce, keepdims=keepdims)
    elif reduction == "sum":
        if dims_to_reduce is None:
            tensor = tensor.sum()
        else:
            tensor = tensor.sum(dims_to_reduce, keepdims=keepdims)
    elif reduction == "none":
        pass
    else:
        raise
    return tensor


def loss_op_core(pred_core, y_core, reduction="mean", loss_type="mse", normalize_mode="None", zero_weight=1, **kwargs):
    """Compute the loss. Here pred_core and y_core must both be tensors and have the same shape. 
    Generically they have the shape of [n_nodes, pred_steps, dyn_dims].
    For hybrid loss_type, e.g. "mse+huberlog#1e-3", will recursively call itself.
    """
    if "+" in loss_type:
        loss_list = []
        precision_floor = get_precision_floor(loss_type)
        for loss_component in loss_type.split("+"):
            loss_component_coef = eval(loss_component.split(":")[1]) if len(loss_component.split(":")) > 1 else 1
            loss_component = loss_component.split(":")[0] if len(loss_component.split(":")) > 1 else loss_component
            if precision_floor is not None and not ("mselog" in loss_component or "huberlog" in loss_component or "l1log" in loss_component):
                pred_core_new = torch.exp(pred_core) - precision_floor
            else:
                pred_core_new = pred_core
            loss_ele = loss_op_core(
                pred_core=pred_core_new,
                y_core=y_core,
                reduction=reduction,
                loss_type=loss_component,
                normalize_mode=normalize_mode,
                zero_weight=zero_weight,
                **kwargs
            ) * loss_component_coef
            loss_list.append(loss_ele)
        loss = torch.stack(loss_list).sum(dim=0)
        return loss

    if normalize_mode != "None":
        assert normalize_mode in ["targetindi", "target"]
        dims_to_reduce = list(np.arange(2, len(y_core.shape)))  # [2, ...]
        epsilon_latent_loss = kwargs["epsilon_latent_loss"] if "epsilon_latent_loss" in kwargs else 0
        if normalize_mode == "target":
            dims_to_reduce.insert(0, 0)  # [0, 2, ...]

    if loss_type.lower() in ["mse", "rmse"]:
        if normalize_mode in ["target", "targetindi"]:
            loss = F.mse_loss(pred_core, y_core, reduction='none')
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.square(), "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                loss = F.mse_loss(pred_core, y_core, reduction=reduction)
            else:
                loss_inter = F.mse_loss(pred_core, y_core, reduction="none")
                zero_mask = y_core.abs() < 1e-8
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "huber":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.smooth_l1_loss(pred_core, y_core, reduction='none')
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                loss = F.smooth_l1_loss(pred_core, y_core, reduction=reduction)
            else:
                loss_inter = F.smooth_l1_loss(pred_core, y_core, reduction="none")
                zero_mask = y_core.abs() < 1e-6
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "l1":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.l1_loss(pred_core, y_core, reduction='none')
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                loss = F.l1_loss(pred_core, y_core, reduction=reduction)
            else:
                loss_inter = F.l1_loss(pred_core, y_core, reduction="none")
                zero_mask = y_core.abs() < 1e-6
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "l2":
        first_dim = kwargs["first_dim"] if "first_dim" in kwargs else 2
        if normalize_mode in ["target", "targetindi"]:
            loss = L2Loss(reduction='none', first_dim=first_dim)(pred_core, y_core)
            y_L2 = L2Loss(reduction='none', first_dim=first_dim)(torch.zeros_like(y_core), y_core)
            if normalize_mode == "target":
                y_L2 = y_L2.mean(0, keepdims=True)
            loss = loss / y_L2
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                loss = L2Loss(reduction=reduction, first_dim=first_dim)(pred_core, y_core)
            else:
                loss_inter = L2Loss(reduction="none", first_dim=first_dim)(pred_core, y_core)
                zero_mask = y_core.abs() < 1e-6
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "lp":
        assert normalize_mode not in ["target", "targetindi"]
        assert zero_weight == 1
        batch_size = kwargs["batch_size"]
        pred_core_reshape = pred_core.reshape(batch_size, -1)  # [B, -1]
        y_core_reshape = y_core.reshape(batch_size, -1)  # [B, -1]
        loss = LpLoss(reduction=True, size_average=True if reduction=="mean" else False)(pred_core_reshape, y_core_reshape, mask=kwargs["is_not_nan_batch"] if "is_not_nan_batch" in kwargs else None)
    elif loss_type.lower().startswith("mpe"):
        exponent = eval(loss_type.split("-")[1])
        if normalize_mode in ["target", "targetindi"]:
            loss = (pred_core - y_core).abs() ** exponent
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.abs() ** exponent, "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                loss = reduce_tensor((pred_core - y_core).abs() ** exponent, reduction=reduction)
            else:
                loss_inter = (pred_core - y_core).abs() ** exponent
                zero_mask = y_core.abs() < 1e-8
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "dl":
        if zero_weight == 1:
            loss = DLLoss(pred_core, y_core, reduction=reduction, **kwargs)
        else:
            loss_inter = DLLoss(pred_core, y_core, reduction="none", **kwargs)
            zero_mask = y_core.abs() < 1e-6
            nonzero_mask = ~zero_mask
            loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
            loss = reduce_tensor(loss, reduction)
    # loss where the target is taking the log scale:
    elif loss_type.lower().startswith("mselog"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.mse_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    elif loss_type.lower().startswith("huberlog"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.smooth_l1_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    elif loss_type.lower().startswith("l1log"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.l1_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    else:
        raise Exception("loss_type {} is not valid!".format(loss_type))
    return loss


def loss_hybrid(
    preds,
    node_label,
    mask,
    node_pos_label,
    input_shape,
    pred_idx=None,
    y_idx=None,
    dyn_dims=None,
    loss_type="mse",
    part_keys=None,
    reduction="mean",
    **kwargs
):
    """Compute the loss at particle locations using by interpolating the values at the field grid.

    Args:
        preds:      {key: [n_nodes, pred_steps, dyn_dims]}
        node_label: {key: [n_nodes, output_steps, [static_dims + compute_dims] + dyn_dims]}
        mask:       {key: [n_nodes]}
        node_pos_label: {key: [n_nodes, output_steps, pos_dims]}. Both used to obtain the pos_grid, and also compute the loss for the density.
        input_shape: a tuple of the actual grid shape.
        pred_idx: index for the pred_steps in preds
        y_idx:    index for the output_steps in node_label
        dyn_dims: the last dyn_dims to obtain from node_label. If None, use full node_label.
        loss_type: loss_type.
        part_keys: keys for particle node types.
        reduction: choose from "mean", "sum" and "none".
        **kwargs: additional kwargs for loss_core.

    Returns:
        if reduction is 'none':
            loss_dict = {"density": {key: loss_matrix with shape [B, n_grid, dyn_dims]}, "feature": {key: loss_matrix}}
        else:
            loss_dict = {"density": loss_density, "feature": loss_feature}
    """
    grid_key = None
    for key in node_pos_label:
        if key not in part_keys:
            grid_key = key
    assert grid_key is not None
    pos_dims = len(input_shape)
    pos_grid = node_pos_label[grid_key][:, y_idx].reshape(-1, *input_shape, pos_dims)  # [B, n_grid, pos_dims]
    batch_size = pos_grid.shape[0]
    n_grid = pos_grid.shape[1]
    pos_dict = {}
    for dim in range(pos_grid.shape[-1]):
        pos_dict[dim] = {"pos_min": pos_grid[..., dim].min(),
                         "pos_max": pos_grid[..., dim].max()}

    loss_dict = {"density": {}, "feature": {}}
    for key in part_keys:
        # Obtain the index information for each position dimension:
        pos_part = node_pos_label[key][:, pred_idx].reshape(batch_size, -1, pos_dims) # [B, n_part, pos_dims]
        n_part = pos_part.shape[1]
        idx_dict = {}
        for dim in range(pos_dims):
            # idx_dict records the left index and remainder for each pos_part[..., dim] (with shape of [B, n_part]):
            idx_left, idx_remainder = get_idx_rel(pos_part[..., dim], pos_dict[dim]["pos_min"], pos_dict[dim]["pos_max"], n_grid)
            idx_dict[dim] = {}
            idx_dict[dim]["idx_left"] = idx_left
            idx_dict[dim]["idx_remainder"] = idx_remainder

        # density_grid_logit, prection of the density logit at each location, shape [B, prod([pos_dims])]
        density_grid_logit = preds[key][:, pred_idx, 0].reshape(batch_size, -1)
        density_grid_logprob = F.log_softmax(density_grid_logit, dim=-1)  # [B, prod([pos_dims])]
        if pos_dims == 1:
            dim = 0
            # Density loss. density_part_logprob: [B, n_part]:
            density_part_logprob = torch.gather(density_grid_logprob, dim=1, index=idx_dict[dim]["idx_left"]) * (1 - idx_dict[dim]["idx_remainder"]) + \
                                   torch.gather(density_grid_logprob, dim=1, index=idx_dict[dim]["idx_left"]+1) * idx_dict[dim]["idx_remainder"]
            loss_dict["density"][key] = -density_part_logprob.mean()

            # Field value loss:
            if dyn_dims is not None:
                node_label_core = node_label[key][:, y_idx, -dyn_dims[key]:].reshape(batch_size, n_part, dyn_dims[key])  # [B, n_part, dyn_dims]
            else:
                node_label_core = node_label[key][:, y_idx].reshape(batch_size, n_part, -1)  # [B, n_part, dyn_dims]
            feature_size = node_label_core.shape[-1]
            node_feature_pred_grid = preds[key][:, pred_idx, 1:].reshape(batch_size, n_grid, feature_size)  # [B, n_grid, feature_size]
            idx_left = idx_dict[dim]["idx_left"][...,None].expand(batch_size, n_part, feature_size)
            # node_feature_pred_part: [B, n_part, feature_size]:
            node_feature_pred_part = torch.gather(node_feature_pred_grid, dim=1, index=idx_left) * (1 - idx_dict[dim]["idx_remainder"])[..., None] + \
                                     torch.gather(node_feature_pred_grid, dim=1, index=idx_left+1) * idx_dict[dim]["idx_remainder"][..., None]
            loss_dict["feature"][key] = loss_op_core(node_feature_pred_part, node_label_core, reduction=reduction, loss_type=loss_type, **kwargs)
        else:
            raise Exception("Currently only supports pos_dims=1!")

    if reduction != "none":
        for mode in ["density", "feature"]:
            if reduction == "mean":
                loss_dict[mode] = torch.stack(list(loss_dict[mode].values())).mean()
            elif reduction == "sum":
                loss_dict[mode] = torch.stack(list(loss_dict[mode].values())).sum()
            else:
                raise
    return loss_dict


def get_idx_rel(pos, pos_min, pos_max, n_grid):
    """
    Obtain the left index on the grid as well as the relative distance to the left index.

    Args:
        pos: any tensor
        pos_min, pos_max: position of the left and right end of the grid
        n_grid: number of grid vertices

    Returns:
        idx_left: the left index. The pos is within [left_index, left_index + 1). Same shape as pos.
        idx_remainder: distance to the left index (in index space). Same shape as pos.
    """
    idx_real = (pos - pos_min) / (pos_max - pos_min) * (n_grid - 1)
    idx_left = idx_real.long()
    idx_remainder = idx_real - idx_left
    return idx_left, idx_remainder


def DLLoss(pred, y, reduction="mean", quantile=0.5):
    """Compute the Description Length (DL) loss, according to AI Physicist (Wu and Tegmark, 2019)."""
    diff = pred - y
    if quantile == 0.5:
        precision_floor = diff.abs().median().item() + 1e-10
    else:
        precision_floor = diff.abs().quantile(quantile).item() + 1e-10
    loss = torch.log(1 + (diff / precision_floor) ** 2)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise Exception("Reduction can only choose from 'mean', 'sum' and 'none'.")
    return loss


def to_cpu(state_dict):
    state_dict_cpu = {}
    for k, v in state_dict.items():
        state_dict_cpu[k] = v.cpu()
    return state_dict_cpu


def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("plasma")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def to_tuple_shape(item):
    """Transform [tuple] or tuple into tuple."""
    if isinstance(item, list) or isinstance(item, torch.Tensor):
        item = item[0]
    if isinstance(item, list):
        item = tuple(item)
    assert isinstance(item, tuple) or isinstance(item, Number) or isinstance(item, torch.Tensor) or isinstance(item, str) or isinstance(item, bool)
    return item


def parse_multi_step(string):
    """
    Parse multi-step prediction setting from string to multi_step_dict.

    Args:
        string: default "1", meaning only 1 step MSE. "1^2:1e-2^4:1e-3" means loss has 1, 2, 4 steps, with the number after ":" being the scale.'

    Returns:
        multi_step_dict: E.g. {1: 1, 2: 1e-2, 4: 1e-3} for string="1^2:1e-2^4:1e-3".
    """
    if string == "":
        return {}
    multi_step_dict = {}
    if "^" in string:
        string_split = string.split("^")
    else:
        string_split = string.split("$")
    for item in string_split:
        item_split = item.split(":")
        time_step = eval(item_split[0])
        multi_step_dict[time_step] = eval(item_split[1]) if len(item_split) == 2 else 1
    multi_step_dict = OrderedDict(sorted(multi_step_dict.items()))
    return multi_step_dict


def parse_act_name(string):
    """
    Parse act_name_dict from string.

    Returns:
        act_name_dict: E.g. {"1": "softplus", "2": "elu"} for string="1:softplus^2:elu".
    """
    act_name_dict = {}
    string_split = string.split("^")
    for item in string_split:
        item_split = item.split(":")
        assert len(item_split) == 2
        time_step = item_split[0]
        act_name_dict[time_step] = item_split[1]
    return act_name_dict


def parse_loss_type(string):
    """
    Parse loss_type_dict from string.

    Returns:
        loss_type_dict: E.g.
            string == "1:mse^2:huber" => loss_type_dict = {"1": "mse", "2": "huber"} for .
            string == '0:mse^2:mse+l1log#1e-3' => loss_type_dict = {"0": mse, "2": "mse+l1log#1e-3"}
    """
    loss_type_dict = {}
    string_split = string.split("^")
    for item in string_split:
        item_split = item.split(":")
        assert len(item_split) == 2
        key = item_split[0]
        loss_type_dict[key] = item_split[1]
    return loss_type_dict


def parse_hybrid_targets(hybrid_targets, default_value=1.):
    """
    Example: M:0.1^xu
    """
    if hybrid_targets == "all":
        hybrid_target_dict = {key: 1 for key in ["M", "MNT", "xu", "J", "field", "full"]}
    elif isinstance(hybrid_targets, str):
        hybrid_target_dict = {}
        for item in hybrid_targets.split("^"):
            hybrid_target_dict[item.split(":")[0]] = eval(item.split(":")[1]) if len(item.split(":")) > 1 else default_value
    else:
        raise
    return hybrid_target_dict


def parse_reg_type(reg_type):
    """Parse reg_type and returns reg_type_core and reg_target.

    reg_type has the format of f"{reg-type}[-{model-target}]^..." as splited by "^"
        where {reg-type} chooses from "srank", "spectral", "Jsim" (Jacobian simplicity), "l2", "l1".
        The optional {model-target} chooses from "all" or "evo" (only effective for Contrastive).
        If not appearing, default "all". The "Jsim" only targets "evo".
    """
    reg_type_list = []
    for reg_type_ele in reg_type.split("^"):
        reg_type_split = reg_type_ele.split("-")
        reg_type_core = reg_type_split[0]
        if len(reg_type_split) == 1:
            reg_target = "all"
        else:
            assert len(reg_type_split) == 2
            reg_target = reg_type_split[1]
        if reg_type_core == "Jsim":
            assert len(reg_type_split) == 1 or reg_target == "evo"
            reg_target = "evo"
        elif reg_type_core == "None":
            reg_target = "None"
        reg_type_list.append((reg_type_core, reg_target))
    return reg_type_list


def get_cholesky_inverse(scale_tril_logit, size):
    """Get the cholesky-inverse from the lower triangular matrix.

    Args:
        scale_tril_logit: has shape of [B, n_components, size*(size+1)/2]. It should be a logit
            where the diagonal element will be passed into softplus.
        size: dimension of the matrix.

    Returns:
        cholesky_inverse: has shape of [B, n_components, size, size]
    """
    n_components = scale_tril_logit.shape[-2]
    scale_tril = fill_triangular(scale_tril_logit.view(-1, scale_tril_logit.shape[-1]), dim=size)
    scale_tril = matrix_diag_transform(scale_tril, F.softplus)
    cholesky_inverse = torch.stack([torch.cholesky_inverse(matrix) for matrix in scale_tril]).reshape(-1, n_components, size, size)
    return cholesky_inverse, scale_tril.reshape(-1, n_components, size, size)


class Rational(torch.nn.Module):
    """Rational Activation function.
    Implementation provided by Mario Casado (https://github.com/Lezcano)
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                         [1.5957, 2.383],
                                         [0.5, 0.0],
                                         [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


def get_device(args):
    """Initialize device."""
    if len(args.gpuid.split(",")) > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid # later retrieved to set gpuids
        # https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018/9
        cuda_str = args.gpuid.split(",")[0] # first device
    else:
        cuda_str = args.gpuid
    is_cuda = eval(cuda_str)
    if not isinstance(is_cuda, bool):
        is_cuda = "cuda:{}".format(is_cuda)
    device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")
    return device


def get_normalization(normalization_type, n_channels, n_groups=2):
    """Get normalization layer."""
    if normalization_type.lower() == "bn1d":
        layer = nn.BatchNorm1d(n_channels)
    elif normalization_type.lower() == "bn2d":
        layer = nn.BatchNorm2d(n_channels)
    elif normalization_type.lower() == "gn":
        layer = nn.GroupNorm(num_groups=n_groups, num_channels=n_channels)
    elif normalization_type.lower() == "ln":
        layer = nn.LayerNorm(n_channels)
    elif normalization_type.lower() == "none":
        layer = nn.Identity()
    else:
        raise Exception("normalization_type '{}' is not valid!".format(normalization_type))
    return layer


def get_max_pool(pos_dims, kernel_size):
    if pos_dims == 1:
        return nn.MaxPool1d(kernel_size=kernel_size)
    elif pos_dims == 2:
        return nn.MaxPool2d(kernel_size=kernel_size)
    elif pos_dims == 3:
        return nn.MaxPool3d(kernel_size=kernel_size)
    else:
        raise


def get_regularization(model_list, reg_type_core):
    """Get regularization.

    Args:
        reg_type_core, Choose from:
            "None": no regularization
            "l1": L1 regularization
            "l2": L2 regularization
            "nuc": nuclear regularization
            "fro": Frobenius norm
            "snr": spectral regularization
            "snn": spectral normalization
            "SINDy": L1 regularization on the coefficient of SINDy
            "Hall": regularize all the elements of Hessian
            "Hoff": regularize the off-diagonal elements of Hessian
            "Hdiag": regularize the diagonal elements of Hessian

    Returns:
        reg: computed regularization.
    """
    if reg_type_core in ["None", "snn"]:
        return 0
    else:
        List = []
        if reg_type_core in ["l1", "l2", "nuc", "fro"]:
            for model in model_list:
                for param_key, param in model.named_parameters():
                    if "weight" in param_key and param.requires_grad:
                        if reg_type_core in ["nuc", "fro"]:
                            norm = torch.norm(param, reg_type_core)
                        elif reg_type_core == "l1":
                            norm = param.abs().sum()
                        elif reg_type_core == "l2":
                            norm = param.square().sum()
                        else:
                            raise
                        List.append(norm)
        elif reg_type_core == "snr":
            for model in model_list:
                for module in model.modules():
                    if module.__class__.__name__ == "SpectralNormReg" and hasattr(module, "snreg"):
                        List.append(module.snreg)
        elif reg_type_core == "sindy":
            for model in model_list:
                for module in model.modules():
                    if module.__class__.__name__ == "SINDy":
                        List.append(module.weight.abs().sum())
        elif reg_type_core in ["Hall", "Hoff", "Hdiag"]:
            for model in model_list:
                if hasattr(model, "Hreg"):
                    List.append(model.Hreg)
        else:
            raise Exception("reg_type_core {} is not valid!".format(reg_type_core))
        if len(List) > 0:
            reg = torch.stack(List).sum()
        else:
            reg = 0
        return reg


def get_edge_index_kernel(pos_part, grid_pos, kernel_size, stride, padding, batch_size):
    """Get the edge index from particle to kernel indices.

    Args:
        pos_part: particle position [B, n_part]
        grid_pos: [B, n_grid, steps, pos_dims]
    """
    def get_index_pos(pos_part, pos_min, pos_max, n_grid):
        """Get the index position (index_pos) for each real position."""
        index_pos = (pos_part - pos_min) / (pos_max - pos_min) * (n_grid - 1)
        return index_pos

    def get_kernel_index_range(index_pos, kernel_size, stride, padding):
        """Get the kernel index range(index_left, index_right) for each index_pos."""
        index_left = torch.ceil((index_pos + padding - (kernel_size - 1)) / stride).long()
        index_right = (torch.floor((index_pos + padding) / stride) + 1).long()
        return index_left, index_right

    assert len(pos_part.shape) == 2
    device = pos_part.device
    pos_min = grid_pos.min()
    pos_max = grid_pos.max()
    n_grid = grid_pos.shape[1]
    n_part = pos_part.shape[1]
    n_kern = int((n_grid + 2 * padding - kernel_size) / stride + 1)
    index_pos = get_index_pos(pos_part, pos_min, pos_max, n_grid)  # [B, n_part]

    index_left, index_right = get_kernel_index_range(index_pos, kernel_size, stride, padding)  # [B, n_part]
    assert stride == 1, "Currently only supports stride=1."
    idx_kern_ori = torch.arange(kernel_size - 1).unsqueeze(0).unsqueeze(0).to(device) + index_left.unsqueeze(-1)  # [B, n_part, kernel_size-1], each denotes the index of the kernel that the particle will be mapped to.
    mask_valid = ((0 <= idx_kern_ori) & (idx_kern_ori < n_kern)).view(-1)
    # Compute edge_attr:
    idx_index = idx_kern_ori * stride - padding + (kernel_size - 1) / 2
    pos_kern = idx_index / (n_grid - 1) * (pos_max - pos_min) + pos_min
    pos_diff = pos_part.unsqueeze(-1) - pos_kern  # [B, n_part, kernel_size-1]
    pos_diff = pos_diff.view(-1)[:, None]
    edge_attr = torch.cat([pos_diff.abs(), pos_diff], -1)
    edge_attr = edge_attr[mask_valid]

    # Compute edge_index:
    idx_kern = idx_kern_ori + (torch.arange(batch_size) * n_kern).unsqueeze(-1).unsqueeze(-1).to(device)
    idx_part = (torch.ones(kernel_size - 1).unsqueeze(0).unsqueeze(0).long() * torch.arange(n_part).unsqueeze(0).unsqueeze(-1) + (torch.arange(batch_size) * n_part).unsqueeze(-1).unsqueeze(-1)).to(device)  # [B, n_part, kernel_size-1]
    edge_index = torch.stack([idx_part.view(-1), idx_kern.view(-1)]).to(device)
    edge_index = edge_index[:, mask_valid]
    return edge_index, edge_attr, n_kern


def stack_tuple_elements(list_of_tuples, dim=1):
    """
    Transform a list of tuples (with same format) into a tuple of stacked tensors, stacked over the list and along dimension dim.
        input: List_of_tuples: [(z11, z12, ...), (z21, z22, ...), ...]
        output: (torch.stack([z11, z21,...], dim),
                 torch.stack([z12, z22,...], dim), ...)
    """
    List = [[] for _ in range(len(list_of_tuples[0]))]
    for i in range(len(list_of_tuples)):  # info["latent_preds"]: [(z1,z2), (z1, z2)]
        for j in range(len(list_of_tuples[0])):
            List[j].append(list_of_tuples[i][j])
    for j in range(len(List)):
        if List[j][0] is not None:
            List[j] = torch.stack(List[j], dim)
        else:
            # In case one position is None, due to is_latent_flatten=False:
            for ele in List[j]:
                assert ele is None
            List[j] = None
    return tuple(List)    # (z1_aug, z2_aug, ...)


def add_noise(input, noise_amp):
    """Add independent Gaussian noise to each element of the tensor."""
    if not isinstance(input, tuple):
        input = input + torch.randn(input.shape).to(input.device) * noise_amp
        return input
    else:
        List = []
        for element in input:
            if element is not None:
                List.append(add_noise(element, noise_amp))
            else:
                List.append(None)
        return tuple(List)


def get_neg_loss(pred, target, loss_type="mse", time_step_weights=None):
    """Get the negative loss by permuting the target along batch dimension."""
    if not isinstance(pred, tuple):
        assert not isinstance(target, tuple)
        batch_size = pred.shape[0]
        perm_idx = np.random.permutation(batch_size)
        target_neg = target[perm_idx]
        loss_neg = loss_op(pred, target_neg, loss_type=loss_type, time_step_weights=time_step_weights)
    else:
        loss_neg = torch.stack([get_neg_loss(pred_ele, target_ele, loss_type=loss_type, time_step_weights=time_step_weights,
                                            ) for pred_ele, target_ele in zip(pred, target) if pred_ele is not None]).mean()
    return loss_neg


def get_pos_dims_dict(original_shape):
    """Obtain the position dimension based on original_shape.

    Args:
        original_shape: ((key1, shape_tuple1), (key2, shape_tuple2), ...) or the corresponding dict format dict(original_shape).

    Returns:
        pos_dims_dict: {key1: pos_dims1, key2: pos_dims2}
    """
    original_shape_dict = dict(original_shape)
    pos_dims_dict = {key: len(original_shape_dict[key]) for key in original_shape_dict}
    return pos_dims_dict


def process_data_for_CNN(data, use_grads=True, use_pos=False):
    """Process data for CNN, optionally adding gradient and normalized position."""
    data = endow_grads(data) if use_grads else data
    # if use_pos:
    #     for key in data.node_feature:
    #         data.node_feature[key] = torch.cat([data.node_pos[0][0][key].to(data.node_feature[key].device), data.node_feature[key]], -1)
    original_shape = dict(to_tuple_shape(data.original_shape))
    pos_dims = get_pos_dims_dict(original_shape)
    data_node_feature = {}
    for key in data.node_feature:
        if key in to_tuple_shape(data.grid_keys):  # Only for grid nodes
            x = data.node_feature[key]  # x: [n_nodes, input_steps, C]
            x = x.reshape(-1, *original_shape[key], *x.shape[-2:])  # [B, [pos_dims], T, C]
            # assert x.shape[-2] == 1
            """
            if mask_non_badpoints is present, then mask_non_badpoints denotes the points in the input that need to be set to zero.
                and mask denotes the nodes to compute the loss.
            if mask_non_badpoints is not present, then mask acts as both mask_non_badpoints and mask. If it is None, then all nodes are valid.
            """
            if hasattr(data, "mask_non_badpoints"):
                mask_non_badpoints = data.mask_non_badpoints
            else:
                mask_non_badpoints = data.mask
            if mask_non_badpoints is not None:
                mask_non_badpoints = mask_non_badpoints[key].reshape(-1, *original_shape[key])  # [B, [pos_dims]]
                mask_non_badpoints = mask_non_badpoints.unsqueeze(-1).unsqueeze(-1).to(x.device)  # [B, [pos_dims], T:1, C:1]
                x = x * mask_non_badpoints  # [B, [pos_dims], T, C]
            permute_order = (0,) + (1 + pos_dims[key], 2 + pos_dims[key]) + tuple(range(1, 1 + pos_dims[key]))
            x = x.permute(*permute_order)  # [B, T, C, [pos_dims]]
            data_node_feature[key] = x.reshape(x.shape[0], -1, *x.shape[3:])  # [B, T * C, [pos_dims]]
    return data_node_feature


def endow_grads_x(x, original_shape, dyn_dims):
    """
    Append data grad to the left of the x.

    The full x has feature semantics of [compute_dims, static_dims, dyn_dims]
    Let the original x has shape of [nodes, (input_steps), compute_dims + static_dims + dyn_dims]  (in order),
        then the x_new will have shape of [nodes, (input_steps), compute_dims ([dyn_dims * 2 + other_compute_dims]) + static_dims + dyn_dims]
    """
    dyn_dims = to_tuple_shape(dyn_dims)
    pos_dims = len(original_shape)
    x_reshape = x.reshape(-1, *original_shape, *x.shape[1:])  # [batch, [pos_dims], (input_steps), feature_size]
    x_core = x_reshape[..., -dyn_dims:]
    x_diff_list = []
    x_diff_list.append(torch.cat([x_core[:,1:2] - x_core[:,0:1], (x_core[:,2:] - x_core[:,:-2]) / 2, x_core[:,-1:] - x_core[:,-2:-1]], 1))
    if pos_dims >= 2:
        x_diff_list.append(torch.cat([x_core[:,:,1:2] - x_core[:,:,0:1], (x_core[:,:,2:] - x_core[:,:,:-2]) / 2, x_core[:,:,-1:] - x_core[:,:,-2:-1]], 2))
    if pos_dims >= 3:
        x_diff_list.append(torch.cat([x_core[:,:,:,1:2] - x_core[:,:,:,0:1], (x_core[:,:,:,2:] - x_core[:,:,:,:-2]) / 2, x_core[:,:,:,-1:] - x_core[:,:,:,-2:-1]], 3))
    x_new = torch.cat(x_diff_list + [x_reshape], -1)   # [batch, [pos_dims], (input_steps), feature_size]
    x_new = x_new.reshape(-1, *x_new.shape[1+pos_dims:])  # [-1, (input_steps), feature_size]
    return x_new


def endow_grads(data):
    grid_keys = to_tuple_shape(data.grid_keys)
    dyn_dims_dict = dict(to_tuple_shape(data.dyn_dims))
    original_shape = dict(to_tuple_shape(data.original_shape))
    for key in data.node_feature:
        if key in grid_keys:
            data.node_feature[key] = endow_grads_x(data.node_feature[key], original_shape[key], dyn_dims_dict[key])
    return data


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        n_neurons,
        n_layers,
        act_name="rational",
        output_size=None,
        last_layer_linear=True,
        is_res=False,
        normalization_type="None",
        is_prioritized_dropout=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        if not isinstance(n_neurons, Number):
            assert n_layers == len(n_neurons), "If n_neurons is not a number, then its length must be equal to n_layers={}".format(n_layers)
        self.n_layers = n_layers
        self.act_name = act_name
        self.output_size = output_size if output_size is not None else n_neurons
        self.last_layer_linear = last_layer_linear
        self.is_res = is_res
        self.normalization_type = normalization_type
        self.normalization_n_groups = 2
        self.is_prioritized_dropout = is_prioritized_dropout
        if act_name != "siren":
            last_out_neurons = self.input_size
            for i in range(1, self.n_layers + 1):
                out_neurons = self.n_neurons if isinstance(self.n_neurons, Number) else self.n_neurons[i-1]
                if i == self.n_layers and self.output_size is not None:
                    out_neurons = self.output_size

                if i == self.n_layers and self.last_layer_linear == "siren":
                    # Last layer is Siren:
                    from siren_pytorch import Siren
                    setattr(self, "layer_{}".format(i), Siren(
                        last_out_neurons,
                        out_neurons,
                    ))
                else:
                    setattr(self, "layer_{}".format(i), nn.Linear(
                        last_out_neurons,
                        out_neurons,
                    ))
                    last_out_neurons = out_neurons
                    torch.nn.init.xavier_normal_(getattr(self, "layer_{}".format(i)).weight)
                    torch.nn.init.constant_(getattr(self, "layer_{}".format(i)).bias, 0)

                # Normalization and activation:
                if i != self.n_layers:
                    # Intermediate layers:
                    if self.act_name != "linear":
                        if self.normalization_type != "None":
                            setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                        setattr(self, "activation_{}".format(i), get_activation(act_name))
                else:
                    # Last layer:
                    if self.last_layer_linear in [False, "False"]:
                        if self.act_name != "linear":
                            if self.normalization_type != "None":
                                setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                            setattr(self, "activation_{}".format(i), get_activation(act_name))
                    elif self.last_layer_linear in [True, "True", "siren"]:
                        pass
                    else:
                        if self.normalization_type != "None":
                            setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                        setattr(self, "activation_{}".format(i), get_activation(self.last_layer_linear))

        else:
            from siren_pytorch import SirenNet, Sine
            if self.last_layer_linear in [False, "False"]:
                if self.act_name == "siren":
                    last_layer = Sine()
                else:
                    last_layer = get_activation(act_name)
            elif self.last_layer_linear in [True, "True"]:
                last_layer = nn.Identity()
            elif self.last_layer_linear == "siren":
                last_layer = Sine()
            else:
                last_layer = get_activation(self.last_layer_linear)
            self.model = SirenNet(
                dim_in=input_size,               # input dimension, ex. 2d coor
                dim_hidden=n_neurons,            # hidden dimension
                dim_out=output_size,             # output dimension, ex. rgb value
                num_layers=n_layers,             # number of layers
                final_activation=last_layer,     # activation of final layer (nn.Identity() for direct output). If last_layer_linear is False, then last activation is Siren
                w0_initial=30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )

    def forward(self, x, n_dropout=None):
        if self.act_name != "siren":
            u = x
            if n_dropout is None:
                if self.training and self.is_prioritized_dropout:
                    max_dropout_size = min(self.input_size, self.output_size)  # 128
                    prioritized_dropout_size = max(1, max_dropout_size // 8)   # 16
                    max_dropout_chunks = (max_dropout_size - 1) // prioritized_dropout_size  # 7
                    n_dropout = np.random.randint(0, max_dropout_chunks+1) * prioritized_dropout_size
                    self.n_dropout = n_dropout
                else:
                    self.n_dropout = 0
            else:
                self.n_dropout = n_dropout

            for i in range(1, self.n_layers + 1):
                if self.n_dropout > 0:
                    u = get_prioritized_dropout(u, self.n_dropout)
                    u = getattr(self, "layer_{}".format(i))(u) * self.input_size / (self.input_size - self.n_dropout)
                else:
                    u = getattr(self, "layer_{}".format(i))(u)

                # Normalization and activation:
                if i != self.n_layers:
                    # Intermediate layers:
                    if self.act_name != "linear":
                        if self.normalization_type != "None":
                            u = getattr(self, "normalization_{}".format(i))(u)
                        u = getattr(self, "activation_{}".format(i))(u)
                else:
                    # Last layer:
                    if self.last_layer_linear in [True, "True", "siren"]:
                        pass
                    else:
                        if self.last_layer_linear in [False, "False"] and self.act_name == "linear":
                            pass
                        else:
                            if self.normalization_type != "None":
                                u = getattr(self, "normalization_{}".format(i))(u)
                            u = getattr(self, "activation_{}".format(i))(u)
            if self.is_res:
                x = x + u
            else:
                x = u
            return x
        else:
            return self.model(x)


class MLP_Coupling(nn.Module):
    def __init__(
        self,
        input_size,
        z_size,
        n_neurons,
        n_layers,
        act_name="rational",
        output_size=None,
        last_layer_linear=True,
        is_res=False,
        normalization_type="None",
        is_prioritized_dropout=False,
    ):
        super(MLP_Coupling, self).__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        if not isinstance(n_neurons, Number):
            assert n_layers == len(n_neurons), "If n_neurons is not a number, then its length must be equal to n_layers={}".format(n_layers)
        self.n_layers = n_layers
        self.act_name = act_name
        self.output_size = output_size if output_size is not None else n_neurons
        self.last_layer_linear = last_layer_linear
        self.is_res = is_res
        self.normalization_type = normalization_type
        self.normalization_n_groups = 2
        self.is_prioritized_dropout = is_prioritized_dropout
        assert act_name != "siren"
        last_out_neurons = self.input_size
        for i in range(1, self.n_layers + 1):
            out_neurons = self.n_neurons if isinstance(self.n_neurons, Number) else self.n_neurons[i-1]
            if i == self.n_layers and self.output_size is not None:
                out_neurons = self.output_size

            setattr(self, "layer_{}".format(i), nn.Linear(
                last_out_neurons,
                out_neurons,
            ))
            setattr(self, "z_layer_{}".format(i), nn.Linear(
                z_size,
                out_neurons*2,
            ))
            last_out_neurons = out_neurons
            torch.nn.init.xavier_normal_(getattr(self, "layer_{}".format(i)).weight)
            torch.nn.init.constant_(getattr(self, "layer_{}".format(i)).bias, 0)
            torch.nn.init.xavier_normal_(getattr(self, "z_layer_{}".format(i)).weight)
            torch.nn.init.constant_(getattr(self, "z_layer_{}".format(i)).bias, 0)

            # Normalization and activation:
            if i != self.n_layers:
                # Intermediate layers:
                if self.act_name != "linear":
                    if self.normalization_type != "None":
                        setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                    setattr(self, "activation_{}".format(i), get_activation(act_name))
            else:
                # Last layer:
                if self.last_layer_linear in [False, "False"]:
                    if self.act_name != "linear":
                        if self.normalization_type != "None":
                            setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                        setattr(self, "activation_{}".format(i), get_activation(act_name))
                elif self.last_layer_linear in [True, "True"]:
                    pass
                else:
                    if self.normalization_type != "None":
                        setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                    setattr(self, "activation_{}".format(i), get_activation(self.last_layer_linear))


    def forward(self, x, z, n_dropout=None):
        u = x

        for i in range(1, self.n_layers + 1):
            u = getattr(self, "layer_{}".format(i))(u)
            z_chunks = getattr(self, "z_layer_{}".format(i))(z)
            z_weight, z_bias = torch.chunk(z_chunks, 2, dim=-1)
            u = u * z_weight + z_bias

            # Normalization and activation:
            if i != self.n_layers:
                # Intermediate layers:
                if self.act_name != "linear":
                    if self.normalization_type != "None":
                        u = getattr(self, "normalization_{}".format(i))(u)
                    u = getattr(self, "activation_{}".format(i))(u)
            else:
                # Last layer:
                if self.last_layer_linear in [True, "True"]:
                    pass
                else:
                    if self.last_layer_linear in [False, "False"] and self.act_name == "linear":
                        pass
                    else:
                        if self.normalization_type != "None":
                            u = getattr(self, "normalization_{}".format(i))(u)
                        u = getattr(self, "activation_{}".format(i))(u)
        if self.is_res:
            x = x + u
        else:
            x = u
        return x


class MultiHeadAttnCoupling(nn.Module):
    def __init__(
        self,
        input_size,
        z_size,
        d_tensor,
        n_heads,
        seq_len,
    ):
        super().__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.d_tensor = d_tensor
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.w_q = nn.Linear(z_size, d_tensor*n_heads*seq_len)
        self.w_k = nn.Linear(input_size, d_tensor*n_heads*seq_len)
        self.w_v = nn.Linear(input_size, d_tensor*n_heads*seq_len)
        self.w_concat = nn.Linear(d_tensor*n_heads*seq_len, input_size)

    def forward(self, x, z):
        """
        Args:
            x: [*size, input_size]
            z: [*size, z_size]
        
        Q, K, V: [*size, heads, seq_len, d_tensor]

        Returns:
            out: [*size, input_size]
        """
        size = z.shape[:2]
        Q = self.w_q(z).view(*size, self.n_heads, self.seq_len, self.d_tensor)
        K = self.w_k(x).view(*size, self.n_heads, self.seq_len, self.d_tensor)
        V = self.w_v(x).view(*size, self.n_heads, self.seq_len, self.d_tensor)

        attention = F.softmax((Q @ K.transpose(-1,-2)) / np.sqrt(self.d_tensor), dim=-1)  # [*size, heads, seq_len, seq_len]
        out = attention @ V  # [*size, heads, seq_len, d_tensor]
        out = out.view(*size, -1)
        out = self.w_concat(out)
        return out


class EncoderLayerCoupling(nn.Module):
    def __init__(
        self,
        input_size,
        z_size,
        d_tensor,
        n_heads,
        seq_len,
        n_neurons,
        act_name,
        normalization_type="ln",
        is_res=True,
        drop_prob=0,
    ):
        super().__init__()
        self.attn_layer = MultiHeadAttnCoupling(
            input_size=input_size,
            z_size=z_size,
            d_tensor=d_tensor,
            n_heads=n_heads,
            seq_len=seq_len,
        )
        self.norm1 = get_normalization(normalization_type, input_size, n_groups=2)
        self.drop_prob = drop_prob
        self.act_name = act_name
        self.normalization_type = normalization_type
        self.is_res = is_res
        if self.drop_prob > 0:
            self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = MLP(
            input_size=input_size,
            n_neurons=n_neurons,
            output_size=input_size,
            n_layers=2,
            act_name=act_name,
        )
        self.norm2 = get_normalization(normalization_type, input_size, n_groups=2)
        if self.drop_prob > 0:
            self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, z):
        # 1. compute self attention
        _x = x
        x = self.attn_layer(x=x, z=z)
        
        # 2. add and norm
        if self.is_res:
            x = x + _x
        if self.normalization_type in ["ln", "gn"]:
            x = self.norm1(x)
        if self.drop_prob > 0:
            x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        if self.is_res:
            x = x + _x
        if self.normalization_type in ["ln", "gn"]:
            x = self.norm2(x)
        if self.drop_prob > 0:
            x = self.dropout2(x)
        return x


class MLP_Attn(nn.Module):
    def __init__(
        self,
        input_size,
        z_size,
        n_neurons,
        n_layers,
        output_size,
        d_tensor,
        n_heads,
        seq_len,
        act_name="rational",
        normalization_type="ln",
        is_res=True,
        last_layer_linear=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.act_name = act_name
        self.output_size = output_size
        self.last_layer_linear = last_layer_linear
        self.is_res = is_res
        if self.last_layer_linear is False:
            self.last_act = get_activation(self.act_name)

        for i in range(1, self.n_layers + 1):
            setattr(self, f"layer_{i}", EncoderLayerCoupling(
                input_size=input_size,
                z_size=z_size,
                d_tensor=d_tensor,
                n_heads=n_heads,
                seq_len=seq_len,
                n_neurons=n_neurons,
                act_name=act_name,
                normalization_type=normalization_type,
                is_res=is_res,
            ))
        self.last_layer = nn.Linear(input_size, output_size)
        torch.nn.init.xavier_normal_(self.last_layer.weight)
        torch.nn.init.constant_(self.last_layer.bias, 0)

    def forward(self, x, z):
        u = x
        for i in range(1, self.n_layers + 1):
            u = getattr(self, f"layer_{i}")(x=u, z=z)
        u = self.last_layer(u)
        if self.last_layer_linear is False:
            u = self.last_act(u)
        return u


class Sum(nn.Module):
    """Module to perform summation along one dimension."""
    def __init__(self, dim, keepshape=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keepshape = keepshape

    def forward(self, input):
        if self.keepshape:
            shape = input.shape
            x = input.sum(dim=self.dim, keepdims=True).expand(shape)
        else:
            x = input.sum(dim=self.dim)
        return x


class Mean(nn.Module):
    """Module to perform summation along one dimension."""
    def __init__(self, dim, keepshape=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepshape = keepshape

    def forward(self, input):
        if self.keepshape:
            shape = input.shape
            x = input.mean(dim=self.dim, keepdims=True).expand(shape)
        else:
            x = input.mean(dim=self.dim)
        return x


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, input):
        return input.exp()


class Flatten(nn.Module):
    def __init__(self, keepdims, is_reshape=False):
        super(Flatten, self).__init__()
        if not isinstance(keepdims, tuple):
            keepdims = (keepdims,)
        assert keepdims == tuple(range(len(keepdims)))
        self.keepdims = keepdims
        self.is_reshape = is_reshape

    def forward(self, input):
        if self.is_reshape:
            return input.reshape(*input.shape[:len(self.keepdims)], -1)
        else:
            return input.view(*input.shape[:len(self.keepdims)], -1)

    def __repr__(self):
        return "Flatten(keepdims={}, is_reshape={})".format(self.keepdims, self.is_reshape)


class Permute(nn.Module):
    def __init__(self, permute_idx):
        super(Permute, self).__init__()
        self.permute_idx = permute_idx

    def forward(self, input):
        return input.permute(*self.permute_idx).contiguous()

    def __repr__(self):
        return "Permute({})".format(self.permute_idx)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.reshape(self.shape)

    def __repr__(self):
        return "Reshape({})".format(self.shape)


class Channel_Gen(object):
    """Generator to generate number of channels depending on the block id (n)."""
    def __init__(self, channel_mode):
        self.channel_mode = channel_mode

    def __call__(self, n):
        if self.channel_mode.startswith("exp"):
            # Exponential growth:
            channel_mul = int(self.channel_mode.split("-")[1])
            n_channels = channel_mul * 2 ** n
        elif self.channel_mode.startswith("c"):
            # Constant:
            n_channels = int(self.channel_mode.split("-")[1])
        else:
            channels = [int(ele) for ele in self.channel_mode.split("-")]
            n_channels = channels[n]
        return n_channels


def get_batch_size(data):
    """Get batch_size"""
    if hasattr(data, "node_feature"):
        first_key = next(iter(data.node_feature))
        original_shape = dict(to_tuple_shape(data.original_shape))
        batch_size = data.node_feature[first_key].shape[0] // np.prod(original_shape[first_key])
    else:
        batch_size = len(data.t)
    return batch_size


def get_elements(src, string_idx):
    if ":" not in string_idx:
        idx = eval(string_idx)
        return src[idx: idx+1]
    else:
        if string_idx.startswith(":"):
            return src[:eval(string_idx[1:])]
        elif string_idx.endswith(":"):
            return src[eval(string_idx[:-1]):]
        else:
            string_idx_split = string_idx.split(":")
            assert len(string_idx_split) == 2
            return src[eval(string_idx_split[0]): eval(string_idx_split[1])]


def parse_string_idx_to_list(string_idx, max_t=None, is_inclusive=True):
    """Parse a index into actual list. E.g.
    E.g.:
        '4'  -> [4]
        ':3' -> [1, 2, 3]
        '2:' -> [2, 3, ... max_t]
        '2:4' -> [2, 3, 4]
    """
    if isinstance(string_idx, int):
        return [string_idx]
    elif isinstance(string_idx, str):
        if ":" not in string_idx:
            idx = eval(string_idx)
            return [idx]
        else:
            if string_idx.startswith(":"):
                return (np.arange(eval(string_idx[1:])) + (1 if is_inclusive else 0)).tolist()
            elif string_idx.endswith(":"):
                return (np.arange(eval(string_idx[:1]), max_t+1)).tolist()
            else:
                string_idx_split = string_idx.split(":")
                assert len(string_idx_split) == 2
                return (np.arange(eval(string_idx_split[0]), eval(string_idx_split[1])+(1 if is_inclusive else 0))).tolist()


def get_data_comb(dataset):
    """Get collated data for full dataset, collated along the batch dimension."""
    from deepsnap.batch import Batch as deepsnap_Batch
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, collate_fn=deepsnap_Batch.collate(), batch_size=len(dataset))
    for data in data_loader:
        break
    return data


def combine_node_label_time(dataset):
    """Combine the node_label across the time dimension, starting with the first data."""
    data = deepcopy(dataset[0])
    node_label_dict = {key: [] for key in data.node_label}
    for data in dataset:
        for key in data.node_label:
            node_label_dict[key].append(data.node_label[key])
    node_label_dict[key] = torch.cat(node_label_dict[key], 1)
    data.node_label = node_label_dict
    return data


def get_root_dir(level=0):
    """Obtain the root directory of the repo.
    Args:
        level: the relative level w.r.t. the repo.
    """
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("plasma")
    dirname = "/".join(dirname_split[:index + 1 + level])
    return dirname


def is_diagnose(loc, filename):
    """If the given loc and filename matches that of the diagose.yml, will return True and (later) call an pde.set_trace()."""
    try:
        with open(get_root_dir() + "/design/multiscale/diagnose.yml", "r") as f:
            Dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        return False
    Dict.pop(None, None)
    if not ("loc" in Dict and "dirname" in Dict and "filename" in Dict):
        return False
    if loc == Dict["loc"] and filename == op.path.join(Dict["dirname"], Dict["filename"]):
        return True
    else:
        return False


def get_keys_values(Dict, exclude=None):
    """Obtain the list of keys and values of the Dict, excluding certain keys."""
    if exclude is None:
        exclude = []
    if not isinstance(exclude, list):
        exclude = [exclude]
    keys = []
    values = []
    for key, value in Dict.items():
        if key not in exclude:
            keys.append(key)
            values.append(value)
    return keys, values


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x


def get_prioritized_dropout(x, n_dropout):
    if isinstance(n_dropout, Number):
        if n_dropout > 0:
            x = torch.cat([x[..., :-n_dropout], torch.zeros(*x.shape[:-1], n_dropout, device=x.device)], -1)
    else:
        if n_dropout.sum() == 0:
            return x
        assert x.shape[0] == len(n_dropout)
        x_list = []
        device = x.device
        for i in range(len(x)):
            if n_dropout[i] > 0:
                x_ele_dropout = torch.cat([x[i, ..., :-n_dropout[i]], torch.zeros(*x.shape[1:-1],n_dropout[i], device=device)], -1)
            else:
                x_ele_dropout = x[i]
            x_list.append(x_ele_dropout)
        x = torch.stack(x_list)
    return x


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y, mask=None):
        """
        Args:
            x, y: both have shape of [B, -1]
        """
        num_examples = x.size()[0]

        if mask is None:
            diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        else:
            mask = mask.view(num_examples, -1).float()
            diff_norms = torch.norm((x.reshape(num_examples,-1) - y.reshape(num_examples,-1)) * mask, self.p, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1) * mask + (1 - mask) * 1e-6, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y, mask=None):
        return self.rel(x, y, mask=mask)


def get_model_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.model_dict
    else:
        return model.model_dict


def add_data_noise(tensor, data_noise_amp):
    if data_noise_amp == 0:
        return tensor
    else:
        return tensor + torch.randn_like(tensor) * data_noise_amp


def deepsnap_to_pyg(data, is_flatten=False, use_pos=False, args_dataset=None):
    assert hasattr(data, "node_feature")
    from torch_geometric.data import Data
    data_pyg = Data(
        x=data.node_feature["n0"],
        y=data.node_label["n0"],
        edge_index=data.edge_index[("n0", "0", "n0")],
        x_pos=data.node_pos["n0"],
        xfaces=data.xfaces["n0"],
        x_bdd=data.x_bdd["n0"],
        original_shape=data.original_shape,
        dyn_dims=data.dyn_dims,
        compute_func=data.compute_func,
    )
    if hasattr(data, "edge_attr"):
        data_pyg.edge_attr = data.edge_attr[("n0", "0", "n0")]
    if hasattr(data, "is_1d_periodic"):
        data_pyg.is_1d_periodic = data.is_1d_periodic
    if hasattr(data, "is_normalize_pos"):
        data_pyg.is_normalize_pos = data.is_normalize_pos
    if hasattr(data, "dataset"):
        data_pyg.dataset = data.dataset
    if hasattr(data, "param"):
        data_pyg.param = data.param["n0"]
    if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
        data_pyg.yedge_index = data.yedge_index["n0"]
        data_pyg.y_tar = data.y_tar["n0"]
        data_pyg.y_back = data.y_back["n0"]
        data_pyg.yface_list = data.yface_list["n0"]
    if hasattr(data, "mask"):
        data_pyg.mask = data.mask["n0"]
    if is_flatten:
        is_1d_periodic = to_tuple_shape(data.is_1d_periodic)
        if is_1d_periodic:
            lst = [data_pyg.x.flatten(start_dim=1)]
        else:
            lst = [data_pyg.x.flatten(start_dim=1), data_pyg.x_bdd]
        if use_pos:
            lst.insert(1, data_pyg.x_pos)
        if hasattr(data_pyg, "dataset") and to_tuple_shape(data_pyg.dataset).startswith("mppde1dh"):
            lst.insert(1, data_pyg.param[:1].expand(data_pyg.x.shape[0], data_pyg.param.shape[-1]))
        data_pyg.x = torch.cat(lst, -1)
        if data_pyg.y is not None:
            data_pyg.y = data_pyg.y.flatten(start_dim=1)
    return data_pyg

def attrdict_to_pygdict(attrdict, is_flatten=False, use_pos=False):    
    pygdict = {
        "x": attrdict["node_feature"]["n0"],
        "y": attrdict["node_label"]["n0"],
        #"y_tar": attrdict["y_tar"]["n0"],
        #"y_back": attrdict["y_back"]["n0"],
        "x_pos": attrdict["x_pos"]["n0"],
        "edge_index": attrdict["edge_index"][("n0","0","n0")],
        "xfaces": attrdict["xfaces"]["n0"],
        #"yface_list": attrdict["yface_list"]["n0"], 
        #"yedge_index": attrdict["yedge_index"]["n0"],
        "x_bdd": attrdict["x_bdd"]["n0"], 
        "original_shape": attrdict["original_shape"],
        "dyn_dims": attrdict["dyn_dims"],
        "compute_func": attrdict["compute_func"],
        "grid_keys": attrdict["grid_keys"],
        "part_keys": attrdict["part_keys"], 
        "time_step": attrdict["time_step"], 
        "sim_id": attrdict["sim_id"],
        "time_interval": attrdict["time_interval"], 
        "cushion_input": attrdict["cushion_input"],
    }
    if "edge_attr" in attrdict:
        pygdict["edge_attr"] = attrdict["edge_attr"]["n0"][0]
    if len(dict(to_tuple_shape(attrdict["original_shape"]))["n0"]) == 0:
        pygdict["bary_weights"] = attrdict["bary_weights"]["n0"]
        pygdict["bary_indices"] = attrdict["bary_indices"]["n0"]
        pygdict["hist_weights"] = attrdict["hist_weights"]["n0"]
        pygdict["hist_indices"] = attrdict["hist_indices"]["n0"]
        pygdict["yedge_index"] = attrdict["yedge_index"]["n0"]
        pygdict["y_back"] = attrdict["y_back"]["n0"]
        pygdict["y_tar"] = attrdict["y_tar"]["n0"]
        pygdict["yface_list"] = attrdict["yface_list"]["n0"]
        pygdict["history"] = attrdict["history"]["n0"]
        pygdict["yfeatures"] = attrdict["yfeatures"]["n0"]
        pygdict["node_dim"] = attrdict["node_dim"]["n0"]
        pygdict["xface_list"] = attrdict["xface_list"]["n0"]
        pygdict["reind_yfeatures"] = attrdict["reind_yfeatures"]["n0"]
        pygdict["batch_history"] = attrdict["batch_history"]["n0"]
    if "onehot_list" in attrdict:
        pygdict["onehot_list"] = attrdict["onehot_list"]["n0"]
        pygdict["kinematics_list"] = attrdict["kinematics_list"]["n0"]   
    if is_flatten:
        if use_pos:
            pygdict["x"] = torch.cat([pygdict["x"].flatten(start_dim=1), pygdict["x_pos"], pygdict["x_bdd"]], -1)
        else:
            pygdict["x"] = torch.cat([pygdict["x"].flatten(start_dim=1), pygdict["x_bdd"]], -1)
    return Attr_Dict(pygdict)

def sample_reward_beta(reward_beta_str,batch_size=1):
    """
    Args:
        reward_beta_str: "0.5-2:linear" (sample from 0.5 to 2, with uniform sampling), "0.0001-1:log" (sample from 0.0001 to 1, with uniform sampling in log scale). Default "1".
    """
    if len(reward_beta_str.split(":")) == 1:
        value_str, mode = reward_beta_str, "linear"
    else:
        value_str, mode = reward_beta_str.split(":")
    if len(value_str.split("-")) == 1:
        min_value, max_value = eval(value_str), eval(value_str)
    else:
        min_value, max_value = value_str.split("-")
        min_value, max_value = eval(min_value), eval(max_value)
    assert min_value <= max_value
    if min_value == max_value:
        return np.ones(batch_size)*min_value
    if mode == "linear":
        reward_beta = np.random.rand(batch_size) * (max_value - min_value) + min_value
    elif mode == "log":
        reward_beta = np.exp(np.random.rand(batch_size) * (np.log(max_value) - np.log(min_value)) + np.log(min_value))
    else:
        raise
    return reward_beta


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0).item()
    return total_norm


def copy_data(data, detach=True):
    """Copy Data instance, and detach from source Data."""
    if isinstance(data, dict):
        dct = {key: copy_data(value) for key, value in data.items()}
        if data.__class__.__name__ == "Attr_Dict":
            dct = Attr_Dict(dct)
        return dct
    elif isinstance(data, list):
        return [copy_data(ele) for ele in data]
    elif isinstance(data, torch.Tensor):
        if detach:
            return data.detach().clone()
        else:
            return data.clone()
    elif isinstance(data, tuple):
        return tuple(copy_data(ele) for ele in data)
    elif data.__class__.__name__ in ['HeteroGraph', 'Data']:
        dct = Attr_Dict({key: copy_data(value) for key, value in vars(data).items()})
        assert len(dct) > 0, "Did not clone anything. Check that your PyG version is below 1.8, preferablly 1.7.1. Follow the the ./design/multiscale/README.md to install the correct version of PyG."
        return dct
    elif data is None:
        return data
    else:
        return deepcopy(data)


def detach_data(data):
    if hasattr(data, "detach"):
        return data.detach()
    elif data is None:
        return data
    else:
        for key, item in vars(data).items():
            if hasattr(item, "detach"):
                setattr(data, key, item.detach())
        return data


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, dim=0):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.dim = dim
        if not self._made_params():
            self._make_params()

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w_mat.data), u.data))
            u.data = l2normalize(torch.mv(w_mat.data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w_mat.mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        width = w_mat.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        if self.training:
            self._update_u_v()
        else:
            setattr(self.module, self.name, getattr(self.module, self.name + "_bar") / 1)
        return self.module.forward(*args)


# ### SpectralNormReg:

# In[ ]:


class SpectralNormReg(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, dim=0):
        super(SpectralNormReg, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.dim = dim
        if not self._made_params():
            self._make_params()

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_snreg(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w_mat.data), u.data))
            u.data = l2normalize(torch.mv(w_mat.data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w_mat.mv(v))
        self.snreg = sigma.square() / 2
        setattr(self.module, self.name, w / 1)  # Here the " / 1" is to prevent state_dict() to record self.module.weight

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        width = w_mat.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self.compute_snreg()
        return self.module.forward(*args)


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

def read_zipped_array(filename):
    file = np.load(filename)
    array = file[file.files[-1]]  # last entry in npz file has to be data array
    if array.shape[0] != 1 or len(array.shape) == 1:
        array = np.expand_dims(array, axis=0)
    # if not physics_config.is_x_first and array.shape[-1] != 1:
    if array.shape[-1] != 1:
        array = array[..., ::-1]  # component order in stored files is always XYZ
    return array


def write_zipped_array(filename, array):
    if array.shape[0] == 1 and len(array.shape) > 1:
        array = array[0, ...]
    #if not physics_config.is_x_first and array.shape[-1] != 1:
    if array.shape[-1] != 1:
        array = array[..., ::-1]  # component order in stored files is always XYZ
    np.savez_compressed(filename, array)


def update_legacy_default_hyperparam(Dict):
    """Default hyperparameters for legacy settings."""
    default_param = {
        # Dataset:
        "time_interval": 1,
        "sector_size": "-1",
        "sector_stride": "-1",
        "seed": -1,
        "dataset_split_type": "standard",
        "train_fraction": float(8/9),
        "temporal_bundle_steps": 1,
        "is_y_variable_length": False,
        "data_noise_amp": 0,
        "data_dropout": "None",

        # Model:
        "latent_multi_step": None,
        "padding_mode": "zeros",
        "latent_noise_amp": 0,
        "decoder_last_act_name": "linear",
        "hinge": 1,
        "contrastive_rel_coef": 0,
        "n_conv_layers_latent": 1,
        "is_latent_flatten": True,
        "channel_mode": "exp-16",
        "no_latent_evo": False,
        "reg_type": "None",
        "reg_coef": 0,
        "is_reg_anneal": True,
        "forward_type": "Euler",
        "evo_groups": 1,
        "evo_conv_type": "cnn",
        "evo_pos_dims": -1,
        "evo_inte_dims": -1,
        "decoder_act_name": "None",
        "vae_mode": "None",
        "is_1d_periodic": False,
        "is_normalize_pos": True,

        # Training:
        "is_pretrain_autoencode": False,
        "is_vae": False,
        "epochs_pretrain": 0,
        "dp_mode": "None",
        "latent_loss_normalize_mode": "None",
        "reinit_mode": "None",
        "is_clip_grad": False,
        "multi_step_start_epoch": 0,
        "epsilon_latent_loss": 0,
        "test_interval": 1,
        "lr_min_cos": 0,
        "is_prioritized_dropout": False,

        # Unet:
        "unet_fmaps": 64,
        
    }
    for key, item in default_param.items():
        if key not in Dict:
            Dict[key] = item
    if "seed" in Dict and Dict["seed"] is None:
        Dict["seed"] = -1
    return Dict


def PyG_to_Attr_Dict(data):
    attr_dict = Attr_Dict(
        node_feature={"n0": data.x},
        node_label={"n0": data.y},
        mask={"n0": data.mask},
        # ptr={"n0": data.ptr},
        param={"n0": data.param},
        sim_id={"n0": data.sim_id},
        time_id={"n0": data.time_id},
        x_bound={"n0": data.x_bound},
        x_pos={"n0": data.x_pos},
        y_bound={"n0": data.y_bound},
        edge_index={("n0","0","n0"): data.edge_index},
        original_shape=(("n0",to_tuple_shape(data.original_shape)),),
        dyn_dims=(("n0", to_tuple_shape(data.dyn_dims)),),
        compute_func=None,
        grid_keys=("n0",),
        part_keys=(),
    )
    return attr_dict