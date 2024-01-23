from __future__ import print_function
import argparse
from collections import Counter, OrderedDict, Iterable
import contextlib
from datetime import datetime
import os
import pdb
import math
from math import gcd
from numbers import Number
import numpy as np
from copy import deepcopy, copy
from functools import reduce
from IPython.display import Image, display
import itertools
import json
import operator
import pickle
import random
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
import scipy.linalg
import sys
from termcolor import colored
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.autograd import Function
from torch.utils.data import Dataset, Sampler
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union



PrecisionFloorLoss = 2 ** (-32)
CLASS_TYPES = ["MLP", "Multi_MLP", "Branching_Net", "Fan_in_MLP", "Model_Ensemble", "Model_with_uncertainty",
               "RNNCellBase", "LSTM", "Wide_ResNet", "Conv_Net", "Conv_Model", "Conv_Autoencoder", "VAE", "Net_reparam", "Mixture_Gaussian", "Triangular_dist"]
ACTIVATION_LIST = ["relu", "leakyRelu", "leakyReluFlat", "tanh", "softplus", "sigmoid", "selu", "elu", "sign", "heaviside", "softmax", "negLogSoftmax", "naturalLogSoftmax"]
COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine",
             "b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]
LINESTYLE_LIST = ["-", "--", ":", "-."]
MARKER_LIST = ["o", "+", "x", "v", ".", "D"]
T_co = TypeVar('T_co', covariant=True)



def plot_matrices(
    matrix_list, 
    shape = None, 
    images_per_row = 10, 
    scale_limit = None,
    figsize = None,
    x_axis_list = None,
    filename = None,
    title = None,
    subtitles = [],
    highlight_bad_values = True,
    plt = None,
    pdf = None,
    verbose = False,
    no_xlabel = False,
    cmap = None,
    is_balanced = False,
    ):
    """Plot the images for each matrix in the matrix_list.
    Adapted from https://github.com/useruser/pytorch_net/blob/c1cfda5e90fef9503c887f5061cb7b1262133ac0/util.py#L54
    
    Args:
        is_balanced: if True, the scale_min and scale_max will have the same absolute value but opposite sign.
        cmap: choose from None, "PiYG", "jet", etc.
    """
    import matplotlib
    from matplotlib import pyplot as plt
    n_rows = max(len(matrix_list) // images_per_row, 1)
    fig = plt.figure(figsize=(20, n_rows*7) if figsize is None else figsize)
    fig.set_canvas(plt.gcf().canvas)
    if title is not None:
        fig.suptitle(title, fontsize = 18, horizontalalignment = 'left', x=0.1)
    
    # To np array. If None, will transform to NaN:
    matrix_list_new = []
    for i, element in enumerate(matrix_list):
        if element is not None:
            matrix_list_new.append(to_np_array(element))
        else:
            matrix_list_new.append(np.array([[np.NaN]]))
    matrix_list = matrix_list_new
    
    num_matrixs = len(matrix_list)
    rows = int(np.ceil(num_matrixs / float(images_per_row)))
    try:
        matrix_list_reshaped = np.reshape(np.array(matrix_list), (-1, shape[0],shape[1])) \
            if shape is not None else np.array(matrix_list)
    except:
        matrix_list_reshaped = matrix_list
    if scale_limit == "auto":
        scale_min = np.Inf
        scale_max = -np.Inf
        for matrix in matrix_list:
            scale_min = min(scale_min, np.min(matrix))
            scale_max = max(scale_max, np.max(matrix))
        scale_limit = (scale_min, scale_max)
        if is_balanced:
            scale_min, scale_max = -max(abs(scale_min), abs(scale_max)), max(abs(scale_min), abs(scale_max))
    for i in range(len(matrix_list)):
        ax = fig.add_subplot(rows, images_per_row, i + 1)
        image = matrix_list_reshaped[i].astype(float)
        if len(image.shape) == 1:
            image = np.expand_dims(image, 1)
        if highlight_bad_values:
            cmap = copy(plt.cm.get_cmap("binary" if cmap is None else cmap))
            cmap.set_bad('red', alpha = 0.2)
            mask_key = []
            mask_key.append(np.isnan(image))
            mask_key.append(np.isinf(image))
            mask_key = np.any(np.array(mask_key), axis = 0)
            image = np.ma.array(image, mask = mask_key)
        else:
            cmap = matplotlib.cm.binary if cmap is None else cmap
        if scale_limit is None:
            ax.matshow(image, cmap = cmap)
        else:
            assert len(scale_limit) == 2, "scale_limit should be a 2-tuple!"
            if is_balanced:
                scale_min, scale_max = scale_limit
                scale_limit = -max(abs(scale_min), abs(scale_max)), max(abs(scale_min), abs(scale_max)) 
            ax.matshow(image, cmap = cmap, vmin = scale_limit[0], vmax = scale_limit[1])
        if len(subtitles) > 0:
            ax.set_title(subtitles[i])
        if not no_xlabel:
            try:
                xlabel = "({0:.4f},{1:.4f})\nshape: ({2}, {3})".format(np.min(image), np.max(image), image.shape[0], image.shape[1])
                if x_axis_list is not None:
                    xlabel += "\n{}".format(x_axis_list[i])
                plt.xlabel(xlabel)
            except:
                pass
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    # if cmap is not None:
    #     cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    #     plt.colorbar(cax=cax)
        # cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
        # plt.colorbar(cax=cbar_ax)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=400)
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

    if scale_limit is not None:
        if verbose:
            print("scale_limit: ({0:.6f}, {1:.6f})".format(scale_limit[0], scale_limit[1]))
    print()


def plot_simple(
    x=None,
    y=None,
    title=None,
    xlabel=None,
    ylabel=None,
    ylim=None,
    figsize=(7,5),
):
    plt.figure(figsize=figsize)
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x, y)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def plot_2_axis(
    x,
    y1,
    y2,
    xlabel=None,
    ylabel1=None,
    ylabel2=None,
    ylim1=None,
    ylim2=None,
    title=None,
    figsize=(7,5),
    fontsize=14,
):
    import matplotlib.pylab as plt
    fig, ax1 = plt.subplots(figsize=figsize)

    color = 'tab:blue'
    if xlabel is not None:
        ax1.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel1 is not None:
        ax1.set_ylabel(ylabel1, color=color, fontsize=fontsize)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize-1)
    if ylim1 is not None:
        ax1.set_ylim(ylim1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2, color=color, fontsize=fontsize)  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize-1)
    if ylim2 is not None:
        ax2.set_ylim(ylim2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.show()


def plot_vectors(
    Dict,
    x_range=None,
    xlabel=None,
    ylabel=None,
    title=None,
    ylim=None,
    fontsize=14,
    linestyle_dict=None,
    figsize=(15,5),
    is_logscale=True,
    is_standard_error=False,
    **kwargs
):
    """Plot learning curve (all losses)."""
    def get_setting(key, setting_dict):
        setting = "-"
        if setting_dict is not None:
            for setting_key in setting_dict:
                if setting_key in key:
                    setting = setting_dict[setting_key]
                    break
        return setting
    from matplotlib import pyplot as plt
    if not isinstance(Dict, dict):
        Dict = {"item": Dict}
    plt.figure(figsize=figsize)
    if is_logscale:
        plt.subplot(1,2,1)
    first_key = next(iter(Dict))
    if x_range is None:
        if isinstance(Dict[first_key][0], Number):
            x_range = np.arange(len(Dict[first_key]))
        else:
            x_range = np.arange(Dict[first_key].shape[1])
    for key in Dict:
        #pdb.set_trace()
        if type(Dict[key])!=list:
            continue
        if len(Dict[key])!=len(x_range):
            continue
        if isinstance(Dict[key][0], Number):
            plt.plot(x_range, to_np_array(Dict[key]), label=key, linestyle=get_setting(key, linestyle_dict), **kwargs)
        else:
            if is_standard_error:
                plt.errorbar(x_range, to_np_array(Dict[key]).mean(-1), to_np_array(Dict[key]).std(-1) / np.sqrt(to_np_array(Dict[key]).shape[0]), label=key, linestyle=get_setting(key, linestyle_dict), capsize=2, **kwargs)
            else:
                plt.errorbar(x_range, to_np_array(Dict[key]).mean(-1), to_np_array(Dict[key]).std(-1), label=key, linestyle=get_setting(key, linestyle_dict), capsize=2, **kwargs)
    #pdb.set_trace()
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tick_params(labelsize=fontsize)

    if is_logscale:
        plt.subplot(1,2,2)
        for key in Dict:
            #pdb.set_trace()
            if type(Dict[key])!=list:
                continue
            if len(Dict[key])!=len(x_range):
                continue
            if isinstance(Dict[key][0], Number):
                plt.semilogy(x_range, to_np_array(Dict[key]), label=key, linestyle=get_setting(key, linestyle_dict), **kwargs)
            else:
                if is_standard_error:
                    plt.errorbar(x_range, to_np_array(Dict[key]).mean(-1), to_np_array(Dict[key]).std(-1) / np.sqrt(to_np_array(Dict[key]).shape[0]), label=key, linestyle=get_setting(key, linestyle_dict), capsize=2, **kwargs)
                else:
                    plt.errorbar(x_range, to_np_array(Dict[key]).mean(-1), to_np_array(Dict[key]).std(-1), label=key, linestyle=get_setting(key, linestyle_dict), capsize=2, **kwargs)
                ax = plt.gca()
                ax.set_yscale("log")
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=fontsize)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=fontsize)
        if title is not None:
            plt.title("{} (log-scale)".format(title), fontsize=fontsize)
        if ylim is not None:
            plt.ylim(ylim)
        plt.tick_params(labelsize=fontsize)
    plt.legend(bbox_to_anchor=[1, 1], fontsize=fontsize-2)
    plt.show()


class Recursive_Loader(object):
    """A recursive loader, able to deal with any depth of X"""
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.length = int(len(self.y) / self.batch_size)
        self.idx_list = torch.randperm(len(self.y))

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < self.length:
            idx = self.idx_list[self.current * self.batch_size: (self.current + 1) * self.batch_size]
            self.current += 1
            return recursive_index((self.X, self.y), idx)
        else:
            self.idx_list = torch.randperm(len(self.y))
            raise StopIteration


def recursive_index(data, idx):
    """Recursively obtain the idx of data"""
    data_new = []
    for i, element in enumerate(data):
        if isinstance(element, tuple):
            data_new.append(recursive_index(element, idx))
        else:
            data_new.append(element[idx])
    return data_new


def record_data(data_record_dict, data_list, key_list, nolist=False, ignore_duplicate=False, recent_record=-1):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else: 
                if (not ignore_duplicate) or (data not in data_record_dict[key]):
                    data_record_dict[key].append(data)
            if recent_record != -1:
                # Only keep the most recent records
                data_record_dict[key] = data_record_dict[key][-recent_record:]


def transform_dict(Dict, mode="array"):
    if mode == "array":
        return {key: np.array(item) for key, item in Dict.items()}
    if mode == "concatenate":
        return {key: np.concatenate(item) for key, item in Dict.items()}
    elif mode == "torch":
        return {key: torch.FloatTensor(item) for key, item in Dict.items()}
    elif mode == "mean":
        return {key: np.mean(item) for key, item in Dict.items()}
    elif mode == "std":
        return {key: np.std(item) for key, item in Dict.items()}
    elif mode == "sum":
        return {key: np.sum(item) for key, item in Dict.items()}
    elif mode == "prod":
        return {key: np.prod(item) for key, item in Dict.items()}
    else:
        raise


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if array is None:
            array_list.append(array)
            continue
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        if not ("keep_list" in kwargs and kwargs["keep_list"]):
            array_list = array_list[0]
    return array_list


def to_Variable(*arrays, **kwargs):
    """Transform numpy arrays into torch tensors/Variables"""
    is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
    requires_grad = kwargs["requires_grad"] if "requires_grad" in kwargs else False
    array_list = []
    for array in arrays:
        is_int = False
        if isinstance(array, Number):
            is_int = True if isinstance(array, int) else False
            array = [array]
        if isinstance(array, np.ndarray) or isinstance(array, list) or isinstance(array, tuple):
            is_int = True if np.array(array).dtype.name == "int64" else False
            array = torch.tensor(array).float()
        if isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor):
            array = Variable(array, requires_grad=requires_grad)
        if "preserve_int" in kwargs and kwargs["preserve_int"] is True and is_int:
            array = array.long()
        array = set_cuda(array, is_cuda)
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def to_Variable_recur(item, type='float'):
    """Recursively transform numpy array into PyTorch tensor."""
    if isinstance(item, dict):
        return {key: to_Variable_recur(value, type=type) for key, value in item.items()}
    elif isinstance(item, tuple):
        return tuple(to_Variable_recur(element, type=type) for element in item)
    else:
        try:
            if type == "long":
                return torch.LongTensor(item)
            elif type == "float":
                return torch.FloatTensor(item)
            elif type == "bool":
                return torch.BoolTensor(item)
        except:
            return [to_Variable_recur(element, type=type) for element in item]


def to_Boolean(tensor):
    """Transform to Boolean tensor. For PyTorch version >= 1.2, use bool(). Otherwise use byte()"""
    version = torch.__version__
    version = eval(".".join(version.split(".")[:-1]))
    if version >= 1.2:
        return tensor.bool()
    else:
        return tensor.byte()


def init_module_weights(module_list, init_weights_mode = "glorot-normal"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_weights_mode == "glorot-uniform":
            glorot_uniform_limit = np.sqrt(6 / float(module.in_features + module.out_features))
            module.weight.data.uniform_(-glorot_uniform_limit, glorot_uniform_limit)
        elif init_weights_mode == "glorot-normal":
            glorot_normal_std = np.sqrt(2 / float(module.in_features + module.out_features))
            module.weight.data.normal_(mean = 0, std = glorot_normal_std)
        else:
            raise Exception("init_weights_mode '{0}' not recognized!".format(init_weights_mode))


def init_module_bias(module_list, init_bias_mode = "zeros"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_bias_mode == "zeros":
            module.bias.data.fill_(0)
        else:
            raise Exception("init_bias_mode '{0}' not recognized!".format(init_bias_mode))


def init_weight(weight_list, init):
    """Initialize the weights"""
    if not isinstance(weight_list, list):
        weight_list = [weight_list]
    for weight in weight_list:
        if len(weight.size()) == 2:
            rows = weight.size(0)
            columns = weight.size(1)
        elif len(weight.size()) == 1:
            rows = 1
            columns = weight.size(0)
        if init is None:
            init = "glorot-normal"
        if not isinstance(init, str):
            weight.data.copy_(torch.FloatTensor(init))
        else:
            if init == "glorot-normal":
                glorot_normal_std = np.sqrt(2 / float(rows + columns))
                weight.data.normal_(mean = 0, std = glorot_normal_std)
            else:
                raise Exception("init '{0}' not recognized!".format(init))


def init_bias(bias_list, init):
    """Initialize the bias"""
    if not isinstance(bias_list, list):
        bias_list = [bias_list]
    for bias in bias_list:
        if init is None:
            init = "zeros"
        if not isinstance(init, str):
            bias.data.copy_(torch.FloatTensor(init))
        else:
            if init == "zeros":
                bias.data.fill_(0)
            else:
                raise Exception("init '{0}' not recognized!".format(init))


def get_activation(activation):
    """Get activation"""
    assert activation in ACTIVATION_LIST + ["linear"]
    if activation == "linear":
        f = lambda x: x
    elif activation == "relu":
        f = F.relu
    elif activation == "leakyRelu":
        f = nn.LeakyReLU(negative_slope = 0.3)
    elif activation == "leakyReluFlat":
        f = nn.LeakyReLU(negative_slope = 0.01)
    elif activation == "tanh":
        f = torch.tanh
    elif activation == "softplus":
        f = F.softplus
    elif activation == "sigmoid":
        f = torch.sigmoid
    elif activation == "selu":
        f = F.selu
    elif activation == "elu":
        f = F.elu
    elif activation == "sign":
        f = lambda x: torch.sign(x)
    elif activation == "heaviside":
        f = lambda x: (torch.sign(x) + 1) / 2.
    elif activation == "softmax":
        f = lambda x: nn.Softmax(dim=-1)(x)
    elif activation == "negLogSoftmax":
        f = lambda x: -torch.log(nn.Softmax(dim=-1)(x))
    elif activation == "naturalLogSoftmax":
        def natlogsoftmax(x):
            x = torch.cat((x, torch.zeros_like(x[..., :1])), axis=-1)
            norm = torch.logsumexp(x, axis=-1, keepdim=True)
            return -(x - norm)
        f = natlogsoftmax
    elif act_name == "silu":
        f = F.silu
    elif act_name == "selu":
        f = F.selu
    elif act_name == "prelu":
        f = F.prelu
    elif act_name == "rrelu":
        f = F.rrelu
    elif act_name == "mish":
        f = F.mish
    elif act_name == "celu":
        f = F.celu
    else:
        raise Exception("activation {0} not recognized!".format(activation))
    return f


def get_activation_noise(act_noise):
    noise_type = act_noise["type"]
    if noise_type == "identity":
        f = lambda x: x
    elif noise_type == "gaussian":
        f = lambda x: x + torch.randn(x.shape) * act_noise["scale"]
    elif noise_type == "uniform":
        f = lambda x: x + (torch.rand(x.shape) - 0.5) * act_noise["scale"]
    else:
        raise Exception("act_noise {} is not valid!".format(noise_type))
    return f


class MAELoss(_Loss):
    """Mean absolute loss"""
    def __init__(self, size_average=None, reduce=None):
        super(MAELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
        loss = (input - target).abs()
        if self.reduce:
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
        return loss


class L2Loss(nn.Module):
    """L2 loss, square root of MSE on each example."""
    def __init__(self, reduction="mean", first_dim=1, keepdims=False, epsilon=1e-10):
        super().__init__()
        self.reduction = reduction
        self.first_dim = first_dim
        self.keepdims = keepdims
        self.epsilon = epsilon

    def forward(self, pred, y):
        reduction_dims = tuple(range(self.first_dim,len(pred.shape)))
        loss = ((pred - y).square().sum(reduction_dims) + self.epsilon).sqrt()
        if self.reduction == "mean":
            loss = loss.mean(dim=tuple(range(self.first_dim)), keepdims=self.keepdims)
        elif self.reduction == "sum":
            loss = loss.sum(dim=tuple(range(self.first_dim)), keepdim=self.keepdims)
        elif self.reduction == "none":
            pass
        else:
            raise
        return loss


class MultihotBinaryCrossEntropy(_Loss):
    """Multihot cross-entropy loss."""
    def __init__(self, size_average=None, reduce=None):
        super(MultihotBinaryCrossEntropy, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
        target = torch.tensor(np.concatenate(([np.eye(10)[target[:,c]] for c in range(target.shape[1])]), axis=1))
        return F.binary_cross_entropy_with_logits(input, target, reduce=self.reduce) 


def get_criterion(loss_type, reduce=None, **kwargs):
    """Get loss function"""
    if loss_type == "huber":
        criterion = nn.SmoothL1Loss(reduce=reduce)
    elif loss_type == "mse":
        criterion = nn.MSELoss(reduce=reduce)
    elif loss_type == "mae":
        criterion = MAELoss(reduce=reduce)
    elif loss_type == "DL":
        criterion = Loss_Fun(core="DL", 
            loss_precision_floor=kwargs["loss_precision_floor"] if "loss_precision_floor" in kwargs and kwargs["loss_precision_floor"] is not None else PrecisionFloorLoss, 
            DL_sum=kwargs["DL_sum"] if "DL_sum" in kwargs else False,
        )
    elif loss_type == "DLs":
        criterion = Loss_Fun(core="DLs", 
            loss_precision_floor=kwargs["loss_precision_floor"] if "loss_precision_floor" in kwargs and kwargs["loss_precision_floor"] is not None else PrecisionFloorLoss,
            DL_sum = kwargs["DL_sum"] if "DL_sum" in kwargs else False,
        )
    elif loss_type == "mlse":
        epsilon = 1e-10
        criterion = lambda pred, target: torch.log(nn.MSELoss(reduce=reduce)(pred, target) + epsilon)
    elif loss_type == "mse+mlse":
        epsilon = 1e-10
        criterion = lambda pred, target: torch.log(nn.MSELoss(reduce=reduce)(pred, target) + epsilon) + nn.MSELoss(reduce=reduce)(pred, target).mean()
    elif loss_type == "cross-entropy":
        criterion = nn.CrossEntropyLoss(reduce=reduce)
    elif loss_type == "multihot-bce":
        criterion = MultihotBinaryCrossEntropy(reduce=reduce)
    elif loss_type == "Loss_with_uncertainty":
        criterion = Loss_with_uncertainty(core=kwargs["loss_core"] if "loss_core" in kwargs else "mse", epsilon = 1e-6)
    elif loss_type[:11] == "Contrastive":
        criterion_name = loss_type.split("-")[1]
        beta = eval(loss_type.split("-")[2])
        criterion = ContrastiveLoss(get_criterion(criterion_name, reduce=reduce, **kwargs), beta=beta)
    else:
        raise Exception("loss_type {0} not recognized!".format(loss_type))
    return criterion



def get_criteria_value(model, X, y, criteria_type, criterion, **kwargs):
    loss_precision_floor = kwargs["loss_precision_floor"] if "loss_precision_floor" in kwargs else PrecisionFloorLoss
    pred = forward(model, X, **kwargs)
    
    # Get loss:
    loss = to_np_array(criterion(pred, y))
    result = {"loss": loss}
    
    # Get DL:
    DL_type = criteria_type if "DL" in criteria_type else "DLs"
    data_DL = get_criterion(loss_type = DL_type, loss_precision_floor = loss_precision_floor, DL_sum = True)(pred, y)
    data_DL = to_np_array(data_DL)
    if not isinstance(model, list):
        model = [model]
    model_DL = np.sum([model_ele.DL for model_ele in model])
    DL = data_DL + model_DL
    result["DL"] = DL
    result["model_DL"] = model_DL
    result["data_DL"] = data_DL
    
    # Specify criteria_value:
    if criteria_type == "loss":
        criteria_value = loss
    elif "DL" in criteria_type:
        criteria_value = DL
    else:
        raise Exception("criteria type {0} not valid".format(criteria_type))
    print(result)
    return criteria_value, result


def get_optimizer(optim_type, lr, parameters, **kwargs):
    """Get optimizer"""
    momentum = kwargs["momentum"] if "momentum" in kwargs else 0
    if optim_type == "adam":
        amsgrad = kwargs["amsgrad"] if "amsgrad" in kwargs else False
        optimizer = optim.Adam(parameters, lr=lr, amsgrad=amsgrad)
    elif optim_type == "sgd":
        nesterov = kwargs["nesterov"] if "nesterov" in kwargs else False
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=nesterov)
    elif optim_type == "adabound":
        import adabound
        optimizer = adabound.AdaBound(parameters, lr=lr, final_lr=0.1 if "final_lr" not in kwargs else kwargs["final_lr"])
    elif optim_type == "RMSprop":
        optimizer = optim.RMSprop(parameters, lr=lr, momentum=momentum)
    elif optim_type == "LBFGS":
        optimizer = optim.LBFGS(parameters, lr=lr)
    else:
        raise Exception("optim_type {0} not recognized!".format(optim_type))
    return optimizer


def get_full_struct_param_ele(struct_param, settings):
    struct_param_new = deepcopy(struct_param)
    for i, layer_struct_param in enumerate(struct_param_new):
        if settings is not None and layer_struct_param[1] != "Symbolic_Layer":
            layer_struct_param[2] = {key: value for key, value in deepcopy(settings).items() if key in ["activation"]}
            layer_struct_param[2].update(struct_param[i][2])
        else:
            layer_struct_param[2] = deepcopy(struct_param[i][2])
    return struct_param_new


def get_full_struct_param(struct_param, settings):
    struct_param_new_list = []
    if isinstance(struct_param, tuple):
        for i, struct_param_ele in enumerate(struct_param):
            if isinstance(settings, tuple):
                settings_ele = settings[i]
            else:
                settings_ele = settings
            struct_param_new_list.append(get_full_struct_param_ele(struct_param_ele, settings_ele))
        return tuple(struct_param_new_list)
    else:
        return get_full_struct_param_ele(struct_param, settings)


class Early_Stopping(object):
    """Class for monitoring and suggesting early stopping"""
    def __init__(self, patience=100, epsilon=0, mode="min"):
        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
        self.best_value = None
        self.wait = 0

    def reset(self, value=None):
        self.best_value = value
        self.wait = 0
        
    def monitor(self, value):
        if self.patience == -1:
            self.wait += 1
            return False
        to_stop = False
        if self.patience is not None:
            if self.best_value is None:
                self.best_value = value
                self.wait = 0
            else:
                if (self.mode == "min" and value < self.best_value - self.epsilon) or \
                   (self.mode == "max" and value > self.best_value + self.epsilon):
                    self.best_value = value
                    self.wait = 0
                else:
                    if self.wait >= self.patience:
                        to_stop = True
                    else:
                        self.wait += 1
        return to_stop

    def __repr__(self):
        return "Early_Stopping(patience={}, epsilon={}, mode={}, wait={})".format(self.patience, self.epsilon, self.mode, self.wait)
    
    
class Performance_Monitor(object):
    def __init__(self, patience = 100, epsilon = 0, compare_mode = "absolute"):
        self.patience = patience
        self.epsilon = epsilon
        self.compare_mode = compare_mode
        self.reset()    
    
    def reset(self):
        self.best_value = None
        self.model_list = []
        self.wait = 0
        self.pivot_id = 0

    def monitor(self, value, **kwargs):
        to_stop = False
        is_accept = False
        if self.best_value is None:
            self.best_value = value
            self.wait = 0
            self.model_list = [deepcopy(kwargs)]
            log = deepcopy(self.model_list)
            is_accept = True
        else:
            self.model_list.append(deepcopy(kwargs))
            log = deepcopy(self.model_list)
            if self.compare_mode == "absolute":
                is_better = (value <= self.best_value + self.epsilon)
            elif self.compare_mode == "relative":
                is_better = (value <= self.best_value * (1 + self.epsilon))
            else:
                raise
            if is_better:
                self.best_value = value
                self.pivot_id = self.pivot_id + 1 + self.wait
                self.wait = 0
                self.model_list = [deepcopy(kwargs)]
                is_accept = True
            else:
                if self.wait >= self.patience:
                    to_stop = True
                else:
                    self.wait += 1
        return to_stop, deepcopy(self.model_list[0]), log, is_accept, deepcopy(self.pivot_id)

    
def flatten(*tensors):
    """Flatten the tensor except the first dimension"""
    new_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            new_tensor = tensor.contiguous().view(tensor.shape[0], -1)
        elif isinstance(tensor, np.ndarray):
            new_tensor = tensor.reshape(tensor.shape[0], -1)
        else:
            print(new_tensor)
            raise Exception("tensors must be either torch.Tensor or np.ndarray!")
        new_tensors.append(new_tensor)
    if len(new_tensors) == 1:
        new_tensors = new_tensors[0]
    return new_tensors


def expand_indices(vector, expand_size):
    """Expand each element ele in the vector to range(ele * expand_size, (ele + 1) * expand_size)"""
    assert isinstance(vector, torch.Tensor)
    vector *= expand_size
    vector_expand = [vector + i for i in range(expand_size)]
    vector_expand = torch.stack(vector_expand, 0)
    vector_expand = vector_expand.transpose(0, 1).contiguous().view(-1)
    return vector_expand


def to_one_hot(idx, num):
    """Transform a 1D vector into a one-hot vector with num classes"""
    if len(idx.size()) == 1:
        idx = idx.unsqueeze(-1)
    if not isinstance(idx, Variable):
        if isinstance(idx, np.ndarray):
            idx = torch.LongTensor(idx)
        idx = Variable(idx, requires_grad=False)
    onehot = Variable(torch.zeros(idx.size(0), num), requires_grad=False)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot


def train_test_split(*args, test_size = 0.1):
    """Split the dataset into training and testing sets"""
    import torch
    num_examples = len(args[0])
    train_list = []
    test_list = []
    if test_size is not None:
        num_test = int(num_examples * test_size)
        num_train = num_examples - num_test
        idx_train = np.random.choice(range(num_examples), size = num_train, replace = False)
        idx_test = set(range(num_examples)) - set(idx_train)
        device = args[0].device
        idx_train = torch.LongTensor(list(idx_train)).to(device)
        idx_test = torch.LongTensor(list(idx_test)).to(device)
        for arg in args:
            train_list.append(arg[idx_train])
            test_list.append(arg[idx_test])
    else:
        train_list = args
        test_list = args
    return train_list, test_list


def make_dir(filename):
    """Make directory using filename if the directory does not exist"""
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


        
def get_accuracy(pred, target):
    """Get accuracy from prediction and target"""
    assert len(pred.shape) == len(target.shape) == 1
    assert len(pred) == len(target)
    pred, target = to_np_array(pred, target)
    accuracy = ((pred == target).sum().astype(float) / len(pred))
    return accuracy


def get_model_accuracy(model, X, y, **kwargs):
    """Get accuracy from model, X and target"""
    is_tensor = kwargs["is_tensor"] if "is_tensor" in kwargs else False
    pred = model(X)
    assert len(pred.shape) == 2
    assert isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor)
    assert len(y.shape) == 1
    pred_max = pred.max(-1)[1]
    acc = (y == pred_max).float().mean()
    if not is_tensor:
        acc = to_np_array(acc)
    return acc


def normalize_tensor(X, new_range = None, mean = None, std = None):
    """Normalize the tensor's value range to new_range"""
    X = X.float()
    if new_range is not None:
        assert mean is None and std is None
        X_min, X_max = X.min().item(), X.max().item()
        X_normalized = (X - X_min) / float(X_max - X_min)
        X_normalized = X_normalized * (new_range[1] - new_range[0]) + new_range[0]
    else:
        X_mean = X.mean().item()
        X_std = X.std().item()
        X_normalized = (X - X_mean) / X_std
        X_normalized = X_normalized * std + mean
    return X_normalized


def try_eval(string):
    """Try to evaluate a string. If failed, use original string."""
    try:
        return eval(string)
    except:
        return string


def try_remove(List, item, is_copy=True):
    """Try to remove an item from the List. If failed, return the original List."""
    if is_copy:
        List = deepcopy(List)
    try:
        List.remove(item)
    except:
        pass
    return List


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def get_args(arg, arg_id = 1, type = "str"):
    """get sys arguments from either command line or Jupyter"""
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        arg_return = arg
    except:
        import sys
        try:
            arg_return = sys.argv[arg_id]
            if type == "int":
                arg_return = int(arg_return)
            elif type == "float":
                arg_return = float(arg_return)
            elif type == "bool":
                arg_return = eval(arg_return)
            elif type == "eval":
                arg_return = eval(arg_return)
            elif type == "tuple":
                arg_return = eval_tuple(arg_return)
            elif type == "str":
                pass
            else:
                raise Exception("type {0} not recognized!".format(type))
        except:
#             raise
            arg_return = arg
    return arg_return


def forward(model, X, **kwargs):
    autoencoder = kwargs["autoencoder"] if "autoencoder" in kwargs else None
    is_Lagrangian = kwargs["is_Lagrangian"] if "is_Lagrangian" in kwargs else False
    output = X
    if not is_Lagrangian:
        if isinstance(model, list) or isinstance(model, tuple):
            for model_ele in model:
                output = model_ele(output)
        else:
            output = model(output)
    else:
        if isinstance(model, list) or isinstance(model, tuple):
            for i, model_ele in enumerate(model):
                if i != len(model) - 1:
                    output = model_ele(output)
                else:
                    output = get_Lagrangian_loss(model_ele, output)
        else:
            output = get_Lagrangian_loss(model, output)
    if autoencoder is not None:
        output = autoencoder.decode(output)
    return output


def logplus(x):
    return torch.clamp(torch.log(torch.clamp(x, 1e-9)) / np.log(2), 0)


class Loss_Fun(nn.Module):
    def __init__(self, core = "mse", epsilon = 1e-10, loss_precision_floor = PrecisionFloorLoss, DL_sum = False):
        super(Loss_Fun, self).__init__()
        self.name = "Loss_Fun"
        self.core = core
        self.epsilon = epsilon
        self.loss_precision_floor = loss_precision_floor
        self.DL_sum = DL_sum

    def forward(self, pred, target, sample_weights = None, is_mean = True):
        if len(pred.size()) == 3:
            pred = pred.squeeze(1)
        assert pred.size() == target.size(), "pred and target must have the same size!"
        if self.core == "huber":
            loss = nn.SmoothL1Loss(reduce = False)(pred, target)
        elif self.core == "mse":
            loss = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "mae":
            loss = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "mse-conv":
            loss = get_criterion(self.core, reduce = False)(pred, target).mean(-1).mean(-1)
        elif self.core == "mlse":
            loss = torch.log(nn.MSELoss(reduce = False)(pred, target) + self.epsilon)
        elif self.core == "mse+mlse":
            loss = torch.log(nn.MSELoss(reduce = False)(pred, target) + self.epsilon) + nn.MSELoss(reduce = False)(pred, target).mean()
        elif self.core == "DL":
            loss = logplus(MAELoss(reduce = False)(pred, target) / self.loss_precision_floor)
        elif self.core == "DLs":
            loss = torch.log(1 + nn.MSELoss(reduce = False)(pred, target) / self.loss_precision_floor ** 2) / np.log(4)
        else:
            raise Exception("loss mode {0} not recognized!".format(self.core))
        if len(loss.shape) == 4:  # Dealing with pixel inputs of (num_examples, channels, height, width)
            loss = loss.mean(-1).mean(-1)
        if loss.size(-1) > 1:
            loss = loss.sum(-1, keepdim = True)
        if sample_weights is not None:
            assert tuple(loss.size()) == tuple(sample_weights.size())
            loss = loss * sample_weights
        if is_mean:
            if sample_weights is not None:
                loss = loss * len(sample_weights) / sample_weights.sum()
            if self.DL_sum:
                loss = loss.sum()
            else:
                loss = loss.mean()
        return loss


class Loss_with_uncertainty(nn.Module):
    def __init__(self, core = "mse", epsilon = 1e-6):
        super(Loss_with_uncertainty, self).__init__()
        self.name = "Loss_with_uncertainty"
        self.core = core
        self.epsilon = epsilon
    
    def forward(self, pred, target, log_std = None, std = None, sample_weights = None, is_mean = True):
        if self.core == "mse":
            loss_core = get_criterion(self.core, reduce = False)(pred, target) / 2
        elif self.core == "mae":
            loss_core = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "huber":
            loss_core = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "mlse":
            loss_core = torch.log((target - pred) ** 2 + 1e-10)
        elif self.core == "mse+mlse":
            loss_core = (target - pred) ** 2 / 2 + torch.log((target - pred) ** 2 + 1e-10)
        else:
            raise Exception("loss's core {0} not recognized!".format(self.core))
        if std is not None:
            assert log_std is None
            loss = loss_core / (self.epsilon + std ** 2) + torch.log(std + 1e-7)
        else:
            loss = loss_core / (self.epsilon + torch.exp(2 * log_std)) + log_std
        if sample_weights is not None:
            sample_weights = sample_weights.view(loss.size())
            loss = loss * sample_weights
        if is_mean:
            loss = loss.mean()
        return loss


def expand_tensor(tensor, dim, times):
    """Repeat the value of a tensor locally along the given dimension"""
    if isinstance(times, int) and times == 1:
        return tensor
    if dim < 0:
        dim += len(tensor.size())
    assert dim >= 0
    size = list(tensor.size())
    repeat_times = [1] * (len(size) + 1)
    repeat_times[dim + 1] = times
    size[dim] = size[dim] * times
    return tensor.unsqueeze(dim + 1).repeat(repeat_times).view(*size)


def get_repeat_interleave(input_size, output_size, dim):
    assert output_size % input_size == 0
    repeats = output_size // input_size
    def repeat_interleave(tensor):
        return tensor.repeat_interleave(repeats, dim=dim)
    return repeat_interleave


def sample(dist, n=None):
    """Sample n instances from distribution dist"""
    if n is None:
        return dist.rsample()
    else:
        return dist.rsample((n,))


def set_variational_output_size(model_dict, reparam_mode, latent_size):
    if reparam_mode.startswith("full"):
        model_dict["struct_param"][-1][0] = int((latent_size + 3) * latent_size / 2)
    elif reparam_mode.startswith("diag"):
        if not reparam_mode.startswith("diagg"):
            if model_dict["type"] == "Mixture_Model":
                for i in range(model_dict["num_components"]):
                    if isinstance(model_dict["model_dict_list"], list):
                        model_dict["model_dict_list"][i]["struct_param"][-1][0] = 2 * latent_size
                    elif isinstance(model_dict["model_dict_list"], dict):
                        model_dict["model_dict_list"]["struct_param"][-1][0] = 2 * latent_size
                    else:
                        raise
            else:
                model_dict["struct_param"][-1][0] = 2 * latent_size
    else:
        raise


def shrink_tensor(tensor, dim, shrink_ratio, mode = "any"):
    """Shrink a tensor along certain dimension using neighboring sites"""
    is_tensor = isinstance(tensor, torch.Tensor)
    shape = tuple(tensor.shape)
    if dim < 0:
        dim += len(tensor.shape)
    assert shape[dim] % shrink_ratio == 0
    new_dim = int(shape[dim] / shrink_ratio)
    new_shape = shape[:dim] + (new_dim, shrink_ratio) + shape[dim+1:]
    if is_tensor:
        new_tensor = tensor.view(*new_shape)
    else:
        new_tensor = np.reshape(tensor, new_shape)
    if mode == "any":
        assert tensor.dtype == "bool" or isinstance(tensor, torch.ByteTensor)
        return new_tensor.any(dim + 1)
    elif mode == "all":
        assert tensor.dtype == "bool" or isinstance(tensor, torch.ByteTensor)
        return new_tensor.all(dim + 1)
    elif mode == "sum":
        return new_tensor.sum(dim + 1)
    elif mode == "mean":
        return new_tensor.mean(dim + 1)
    else:
        raise


def permute_dim(X, dim, idx, group_sizes, mode = "permute"):
    from copy import deepcopy
    assert dim != 0
    device = X.device
    if isinstance(idx, tuple) or isinstance(idx, list):
        k, ll = idx
        X_permute = X[:, k, ll * group_sizes: (ll + 1) * group_sizes]
        num = X_permute.size(0)
        if mode == "permute":
            new_idx = torch.randperm(num).to(device)
        elif mode == "resample":
            new_idx = torch.randint(num, size = (num,)).long().to(device)
        else:
            raise
        X_permute = X_permute.index_select(0, new_idx)
        X_new = deepcopy(X)
        X_new[:, k, ll * group_sizes: (ll + 1) * group_sizes] = X_permute
    else:
        X_permute = X.index_select(dim, torch.arange(idx * group_sizes, (idx + 1) * group_sizes).long().to(device))
        num = X_permute.size(0)
        if mode == "permute":
            new_idx = torch.randperm(num).to(device)
        elif mode == "resample":
            new_idx = torch.randint(num, size = (num,)).long().to(device)
        else:
            raise
        X_permute = X_permute.index_select(0, new_idx)
        X_new = deepcopy(X)
        if dim == 1:
            X_new[:,idx*group_sizes: (idx+1)*group_sizes] = X_permute
        elif dim == 2:
            X_new[:,:,idx*group_sizes: (idx+1)*group_sizes] = X_permute
        elif dim == 3:
            X_new[:,:,:,idx*group_sizes: (idx+1)*group_sizes] = X_permute
        else:
            raise
    return X_new


def fill_triangular(vec, dim, mode="lower"):
    """Fill an lower or upper triangular matrices with given vectors.

    Specifically, it transform the examples in the vector form [n_examples, size]
        into the lower-triangular matrix with shape [n_examples, dim, dim]
    
    Args:
        vec: [n_examples, size], where the size
        dim: the dimension of the lower triangular matrix.
            the size == dim * (dim + 1) // 2 must be satisfied.

    Returns:
        matrix: with shape [n_examples, dim, dim]
    """
#     num_examples, size = vec.shape
#     assert size == dim * (dim + 1) // 2
#     matrix = torch.zeros(num_examples, dim, dim).to(vec.device)
#     if mode == "lower":
#         idx = (torch.tril(torch.ones(dim, dim)) == 1)[None]
#     elif mode == "upper":
#         idx = (torch.triu(torch.ones(dim, dim)) == 1)[None]
#     else:
#         raise Exception("mode {} not recognized!".format(mode))
#     idx = idx.repeat(num_examples,1,1)
#     matrix[idx] = vec.contiguous().view(-1)
    num_examples, size = vec.shape
    assert size == dim * (dim + 1) // 2
    if mode == "lower":
        rows, cols = torch.tril_indices(dim, dim)
    elif mode == "upper":
        rows, cols = torch.triu_indices(dim, dim)
    else:
        raise Exception("mode {} not recognized!".format(mode))
    matrix = torch.zeros(num_examples, dim, dim).type(vec.dtype).to(vec.device)
    matrix[:, rows, cols] = vec
    return matrix


def get_tril_block(size, block_size, diagonal=0):
    """Get indices of a lower-triangular block matrix."""
    n = int(np.ceil(size / block_size))
    mesh1, mesh2 = torch.meshgrid(torch.arange(1,1+size), torch.arange(1,1+size), indexing="ij")
    mesh1 = torch.tensor(mesh1.numpy())
    mesh2 = torch.tensor(mesh2.numpy())
    for k in range(n):
        mesh1[k*block_size:(k+1)*block_size, (k+1+diagonal)*block_size:] = 0
        mesh2[k*block_size:(k+1)*block_size, (k+1+diagonal)*block_size:] = 0
    rows = mesh1.view(-1)[torch.where(mesh1.view(-1))[0]] - 1
    cols = mesh2.view(-1)[torch.where(mesh2.view(-1))[0]] - 1
    return rows, cols


def get_triu_block(size, block_size, diagonal=0):
    """Get indices of a upper-triangular block matrix."""
    n = int(np.ceil(size / block_size))
    mesh1, mesh2 = torch.meshgrid(torch.arange(1,1+size), torch.arange(1,1+size), indexing="ij")
    mesh1 = torch.tensor(mesh1.numpy())
    mesh2 = torch.tensor(mesh2.numpy())
    for k in range(n):
        mesh1[k*block_size:(k+1)*block_size, :(k+diagonal)*block_size] = 0
        mesh2[k*block_size:(k+1)*block_size, :(k+diagonal)*block_size] = 0
    rows = mesh1.view(-1)[torch.where(mesh1.view(-1))[0]] - 1
    cols = mesh2.view(-1)[torch.where(mesh2.view(-1))[0]] - 1
    return rows, cols


def get_triu_3D(size):
    """Get the upper triangular tensor for a 3D tensor with given size."""
    rows1 = []
    rows2 = []
    rows3 = []
    for i in range(size):
        rows_ele, cols_ele = torch.triu_indices(size-i, size-i)
        rows2 += rows_ele
        rows3 += cols_ele
        rows1 += [torch.tensor(i)] * len(rows_ele)
    rows1, rows2, rows3 = torch.stack(rows1), torch.stack(rows2), torch.stack(rows3)
    return rows1, rows2, rows3


def get_loss_cumu(loss_dict, cumu_mode):
    """Combine different losses to obtain a single scalar loss.

    Args:
        loss_dict: A dictionary or list of loss values, each of which is a torch scalar.
        cumu_mode: a 2-tuple. Choose from:
            ("generalized-mean"/"gm", {order}): generalized mean with order
            "harmonic": harmonic mean
            "geometric": geometric mean
            "mean": arithmetic mean
            "sum": summation
            "min": minimum
            "original": returns the original loss_dict.

    Returns:
        loss: the combined loss scalar computed according to cumu_mode.
    """
    if cumu_mode == "original":
        return loss_dict
    if isinstance(loss_dict, dict):
        loss_list = torch.stack([loss for loss in loss_dict.values()])
    elif isinstance(loss_dict, list):
        loss_list = torch.stack(loss_dict)
    elif isinstance(loss_dict, torch.Tensor):
        loss_list = loss_dict
        if len(loss_list.shape) == 0:
            return loss_list
    else:
        raise
    N = len(loss_list)
    if N == 1:
        return loss_list[0]
    epsilon = 1e-20  # to prevent NaN
    if isinstance(cumu_mode, str) and cumu_mode.startswith("gm"):
        cumu_mode_str, num = cumu_mode.split("-")
        cumu_mode = (cumu_mode_str, eval(num))
    if isinstance(cumu_mode, tuple) and cumu_mode[0] in ["generalized-mean", "gm"]:
        if cumu_mode[1] == -1:
            cumu_mode = "harmonic"
        elif cumu_mode[1] == 0:
            cumu_mode = "geometric"
        elif cumu_mode[1] == 1:
            cumu_mode = "mean"
        elif cumu_mode[1] == "min":
            cumu_mode = "min"
        elif cumu_mode[1] == "max":
            cumu_mode = "max"
    
    if cumu_mode == "harmonic":
        loss = N / (1 / (loss_list + epsilon)).sum()
    elif cumu_mode == "geometric":
        loss = (loss_list + epsilon).prod() ** (1 / float(N))
    elif cumu_mode == "mean":
        loss = loss_list.mean()
    elif cumu_mode == "sum":
        loss = loss_list.sum()
    elif cumu_mode == "min":
        loss = loss_list.min()
    elif cumu_mode[0] in ["generalized-mean", "gm"]:
        order = cumu_mode[1]
        loss = (((loss_list + epsilon) ** order).mean()) ** (1 / float(order))
    else:
        raise
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


def loss_op_core(pred_core, y_core, reduction="mean", loss_type="mse", normalize_mode="None", **kwargs):
    """Compute the loss. Here pred_core and y_core must both be tensors and have the same shape. 
    Generically they have the shape of [n_nodes, pred_steps, dyn_dims].
    For hybrid loss_type, e.g. "mse+huberlog#1e-3", will recursively call itself.
    """
    if "+" in loss_type:
        loss_list = []
        precision_floor = get_precision_floor(loss_type)
        for loss_component in loss_type.split("+"):
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
                **kwargs
            )
            loss_list.append(loss_ele)
        loss = torch.stack(loss_list).sum()
        return loss

    if normalize_mode != "None":
        assert normalize_mode in ["targetindi", "target"]
        dims_to_reduce = list(np.arange(2, len(y_core.shape)))  # [2, ...]
        if normalize_mode == "target":
            dims_to_reduce.insert(0, 0)  # [0, 2, ...]

    if loss_type.lower() == "mse":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.mse_loss(pred_core, y_core, reduction='none')
            loss = loss / reduce_tensor(y_core.square(), "mean", dims_to_reduce, keepdims=True)
            loss = reduce_tensor(loss, reduction)
        else:
            loss = F.mse_loss(pred_core, y_core, reduction=reduction)
    elif loss_type.lower() == "huber":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.smooth_l1_loss(pred_core, y_core, reduction='none')
            loss = loss / reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True)
            loss = reduce_tensor(loss, reduction)
        else:
            loss = F.smooth_l1_loss(pred_core, y_core, reduction=reduction)
    elif loss_type.lower() == "l1":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.l1_loss(pred_core, y_core, reduction='none')
            loss = loss / reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True)
            loss = reduce_tensor(loss, reduction)
        else:
            loss = F.l1_loss(pred_core, y_core, reduction=reduction)
    elif loss_type.lower() == "l2":
        first_dim = kwargs["first_dim"] if "first_dim" in kwargs else 2
        if normalize_mode in ["target", "targetindi"]:
            loss = L2Loss(reduction='none', first_dim=first_dim)(pred_core, y_core)
            y_L2 = L2Loss(reduction='none', first_dim=first_dim)(torch.zeros(y_core.shape), y_core)
            if normalize_mode == "target":
                y_L2 = y_L2.mean(0, keepdims=True)
            loss = loss / y_L2
            loss = reduce_tensor(loss, reduction)
        else:
            loss = L2Loss(reduction=reduction, first_dim=first_dim)(pred_core, y_core)
    elif loss_type.lower() == "dl":
        loss = DLLoss(pred_core, y_core, reduction=reduction, **kwargs)
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


def matrix_diag_transform(matrix, fun):
    """Return the matrices whose diagonal elements have been executed by the function 'fun'."""
    num_examples = len(matrix)
    idx = torch.eye(matrix.size(-1)).bool().unsqueeze(0)
    idx = idx.repeat(num_examples, 1, 1)
    new_matrix = matrix.clone()
    new_matrix[idx] = fun(matrix.diagonal(dim1=1, dim2=2).contiguous().view(-1))
    return new_matrix


def sort_two_lists(list1, list2, reverse = False):
    """Sort two lists according to the first list."""
    if reverse:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=operator.itemgetter(0), reverse=True))])
    else:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=operator.itemgetter(0)))])
    if len(List) == 0:
        return [], []
    else:
        return List[0], List[1]
    

def sort_dict(Dict, reverse = False):
    """Return an ordered dictionary whose values are sorted"""
    from collections import OrderedDict
    orderedDict = OrderedDict()
    keys, values = list(Dict.keys()), list(Dict.values())
    values_sorted, keys_sorted = sort_two_lists(values, keys, reverse = reverse)
    for key, value in zip(keys_sorted, values_sorted):
        orderedDict[key] = value
    return orderedDict


def get_dict_items(Dict, idx):
    """Obtain dictionary items with the current ordering of dictionary keys"""
    from collections import OrderedDict
    from copy import deepcopy
    keys = list(Dict.keys())
    new_dict = OrderedDict()
    for id in idx:
        new_dict[keys[id]] = deepcopy(Dict[keys[id]])
    return new_dict
    

def to_string(List, connect = "-", num_digits = None, num_strings = None):
    """Turn a list into a string, with specified format"""
    if not isinstance(List, list) and not isinstance(List, tuple):
        return List
    if num_strings is None:
        if num_digits is None:
            return connect.join([str(element) for element in List])
        else:
            return connect.join(["{0:.{1}f}".format(element, num_digits) for element in List])
    else:
        if num_digits is None:
            return connect.join([str(element)[:num_strings] for element in List])
        else:
            return connect.join(["{0:.{1}f}".format(element, num_digits)[:num_strings] for element in List])


def split_string(string):
    """Given a string, return the core string and the number suffix.
    If there is no number suffix, the num_core will be None.
    """
    # Get the starting index for the number suffix:
    assert isinstance(string, str)
    i = 1
    for i in range(1, len(string) + 1):
        if string[-i] in [str(k) for k in range(10)]:
            continue
        else:
            break
    idx = len(string) - i + 1
    # Obtain string_core and num_core:
    string_core = string[:idx]
    if len(string[idx:]) > 0:
        num_core = eval(string[idx:])
    else:
        num_core = None

    return string_core, num_core


def canonicalize_strings(operators):
    """Given a list of strings, return the canonical version.
    
    Example:
        operators = ["EqualRow1", "EqualRow2", "EqualWidth3"]
    
        Returns: mapping = {'EqualRow1': 'EqualRow',
                            'EqualRow2': 'EqualRow1',
                            'EqualWidth3': 'EqualWidth'}
    """
    operators_core = [split_string(ele)[0] for ele in operators]
    counts = Counter(operators_core)
    new_counts = {key: 0 for key in counts}
    mapping = {}
    for operator, operator_core in zip(operators, operators_core):
        count = new_counts[operator_core]
        if count == 0:
            mapping[operator] = operator_core
        else:
            mapping[operator] = "{}{}".format(operator_core, count)
        new_counts[operator_core] += 1
    return mapping


def get_rename_mapping(base_keys, adding_keys):
    """Given a list of base_keys and adding keys, return a mapping of how to 
    rename adding_keys s.t. adding_keys do not have name conflict with base_keys.
    """
    mapping = {}
    for key in adding_keys:
        if key in base_keys:
            string_core, num_core = split_string(key)
            num_proposed = num_core + 1 if num_core is not None else 1
            proposed_name = "{}{}".format(string_core, num_proposed)
            while proposed_name in base_keys:
                num_proposed += 1
                proposed_name = "{}{}".format(string_core, num_proposed)
            mapping[key] = proposed_name
    return mapping


def view_item(dict_list, key):
    if not isinstance(key, tuple):
        return [element[key] for element in dict_list]
    else:
        return [element[key[0]][key[1]] for element in dict_list]

        
def filter_filename(dirname, include=[], exclude=[], array_id=None):
    """Filter filename in a directory"""
    def get_array_id(filename):
        array_id = filename.split("_")[-2]
        try:
            array_id = eval(array_id)
        except:
            pass
        return array_id
    filename_collect = []
    if array_id is None:
        filename_cand = [filename for filename in os.listdir(dirname)]
    else:
        filename_cand = [filename for filename in os.listdir(dirname) if get_array_id(filename) == array_id]
    
    if not isinstance(include, list):
        include = [include]
    if not isinstance(exclude, list):
        exclude = [exclude]
    
    for filename in filename_cand:
        is_in = True
        for element in include:
            if element not in filename:
                is_in = False
                break
        for element in exclude:
            if element in filename:
                is_in = False
                break
        if is_in:
            filename_collect.append(filename)
    return filename_collect


def display_image(dirname, include=[], exclude=[], width=800, height=None):
    from IPython.display import Image, display
    filenames = sorted(filter_filename(dirname, include=include, exclude=exclude))
    for filename in filenames:
        print("{}:".format(filename))
        display(Image(filename=dirname + filename))
        print()


def sort_filename(filename_list):
    """Sort the files according to the id at the end. The filename is in the form of *_NUMBER.p """
    iter_list = []
    for filename in filename_list:
        iter_num = eval(filename.split("_")[-1].split(".")[0])
        iter_list.append(iter_num)
    iter_list_sorted, filename_list_sorted = sort_two_lists(iter_list, filename_list, reverse = True)
    return filename_list_sorted


def remove_files_in_directory(directory, is_remove_subdir = False):
    """Remove files in a directory"""
    import os, shutil
    if directory is None:
        return
    if not os.path.isdir(directory):
        print("Directory {0} does not exist!".format(directory))
        return
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif is_remove_subdir and os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def softmax(tensor, dim):
    assert isinstance(tensor, np.ndarray)
    tensor = tensor - tensor.mean(dim, keepdims = True)
    tensor = np.exp(tensor)
    tensor = tensor / tensor.sum(dim, keepdims = True)
    return tensor


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
          From https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).type_as(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            gm = grad_output.data.numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return grad_input


sqrtm = MatrixSquareRoot.apply



def get_flat_function(List, idx):
    """Get the idx index of List. If idx >= len(List), return the last element"""
    if idx < 0:
        return List[0]
    elif idx < len(List):
        return List[idx]
    else:
        return List[-1]


def Beta_Function(x, alpha, beta):
    """Beta function"""
    from scipy.special import gamma
    return gamma(alpha + beta) / gamma(alpha) / gamma(beta) * x ** (alpha - 1) * (1 - x) ** (beta - 1)


def Zip(*data, **kwargs):
    """Recursive unzipping of data structure
    Example: Zip(*[(('a',2), 1), (('b',3), 2), (('c',3), 3), (('d',2), 4)])
    ==> [[['a', 'b', 'c', 'd'], [2, 3, 3, 2]], [1, 2, 3, 4]]
    Each subtree in the original data must be in the form of a tuple.
    In the **kwargs, you can set the function that is applied to each fully unzipped subtree.
    """
    import collections
    function = kwargs["function"] if "function" in kwargs else None
    if len(data) == 1 and function is None:
        return data[0]
    data = [list(element) for element in zip(*data)]
    for i, element in enumerate(data):
        if isinstance(element[0], tuple):
            data[i] = Zip(*element, **kwargs)
        elif isinstance(element, list):
            if function is not None:
                data[i] = function(element)
    return data


class Gradient_Noise_Scale_Gen(object):
    def __init__(
        self,
        epochs, 
        gamma = 0.55,
        eta = 0.01,
        noise_scale_start = 1e-2,
        noise_scale_end = 1e-6,
        gradient_noise_interval_epoch = 1,
        fun_pointer = "generate_scale_simple",
        ):
        self.epochs = epochs
        self.gradient_noise_interval_epoch = gradient_noise_interval_epoch
        self.max_iter = int(self.epochs / self.gradient_noise_interval_epoch) + 1
        self.gamma = gamma
        self.eta = eta
        self.noise_scale_start = noise_scale_start
        self.noise_scale_end = noise_scale_end
        self.generate_scale = getattr(self, fun_pointer) # Sets the default function to generate scale
    
    def generate_scale_simple(self, verbose = True):     
        gradient_noise_scale = np.sqrt(self.eta * (np.array(range(self.max_iter)) + 1) ** (- self.gamma))
        if verbose:
            print("gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(gradient_noise_scale[0], gradient_noise_scale[-1], self.gamma, self.max_iter))
        return gradient_noise_scale

    def generate_scale_fix_ends(self, verbose = True):
        ratio = (self.noise_scale_start / float(self.noise_scale_end)) ** (1 / self.gamma) - 1
        self.bb = self.max_iter / ratio
        self.aa = self.noise_scale_start * self.bb ** self.gamma
        gradient_noise_scale = np.sqrt(self.aa * (np.array(range(self.max_iter)) + self.bb) ** (- self.gamma))
        if verbose:
            print("gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(gradient_noise_scale[0], gradient_noise_scale[-1], self.gamma, self.max_iter))
        return gradient_noise_scale


def load_model(filename, mode="pickle"):
    if mode == "pickle":
        model_dict = pickle.load(open(filename, "rb"))
    elif mode == "json":
        with open(filename, 'r') as outfile:
            json_dict = json.load(outfile)
            model_dict = deserialize(json_dict)
    else:
        raise Exception("mode {} is not valid!".format(mode))
    return model_dict


def save_model(model_dict, filename, mode="pickle"):
    if mode == "pickle":
        pickle.dump(model_dict, open(filename, "wb"))
    elif mode == "json":
        with open(filename, 'w') as outfile:
            json.dump(serialize(model_dict), outfile)
    else:
        raise Exception("mode {} is not valid!".format(mode))


def to_cpu_recur(item, to_target=None):
    if isinstance(item, dict):
        return {key: to_cpu_recur(value, to_target=to_target) for key, value in item.items()}
    elif isinstance(item, list):
        return [to_cpu_recur(element, to_target=to_target) for element in item]
    elif isinstance(item, tuple):
        return tuple(to_cpu_recur(element, to_target=to_target) for element in item)
    elif isinstance(item, set):
        return {to_cpu_recur(element, to_target=to_target) for element in item}
    else:
        if isinstance(item, torch.Tensor):
            if item.is_cuda:
                item = item.cpu()
            if to_target is not None and to_target == "np":
                item = item.detach().numpy()
            return item
        if to_target is not None and to_target == "torch":
            if isinstance(item, np.ndarray):
                item = torch.FloatTensor(item)
                return item
        return item


def to_cpu(state_dict):
    state_dict_cpu = {}
    for k, v in state_dict.items():
        state_dict_cpu[k] = v.cpu()
    return state_dict_cpu


def serialize(item):
    if isinstance(item, dict):
        return {str(key): serialize(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [serialize(element) for element in item]
    elif isinstance(item, tuple):
        return tuple(serialize(element) for element in item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    else:
        return str(item)


def deserialize(item):
    if isinstance(item, dict):
        try:
            return {eval(key): deserialize(value) for key, value in item.items()}
        except:
            return {key: deserialize(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [deserialize(element) for element in item]
    elif isinstance(item, tuple):
        return tuple(deserialize(element) for element in item)
    else:
        if isinstance(item, str) and item in CLASS_TYPES:
            return item
        else:
            try:
                return eval(item)
            except:
                return item


def deserialize_list_str(string):
    """Parse a string version of recursive lists into recursive lists of strings."""
    assert string.startswith("[") and string.endswith("]")
    string = string[1:-1]
    bracket_count = 0
    punctuations = [0]
    for i, letter in enumerate(string):
        if letter == "," and bracket_count == 0:
            punctuations.append(i)
        elif letter == "[":
            bracket_count += 1
        elif letter == "]":
            bracket_count -= 1

    if len(punctuations) == 1:
        return [string]
    List = []
    for i in range(len(punctuations)):
        if i == 0:
            element = string[:punctuations[1]]
        elif i == len(punctuations) - 1:
            element = string[punctuations[i] + 1:].strip()
        else:
            element = string[punctuations[i] + 1: punctuations[i+1]].strip()
        if element.startswith("[") and element.endswith("]"):
            List.append(deserialize_list_str(element))
        else:
            List.append(element)
    return List


def flatten_list(graph):
    """Flatten a recursive list of lists into a flattened list."""
    if not isinstance(graph, list):
        return [deepcopy(graph)]
    else:
        flattened_graph = []
        for subgraph in graph:
            flattened_graph += flatten_list(subgraph)
        return flattened_graph


def get_num_params(model, is_trainable = None):
    """Get number of parameters of the model, specified by 'None': all parameters;
    True: trainable parameters; False: non-trainable parameters.
    """
    num_params = 0
    for param in list(model.parameters()):
        nn=1
        if is_trainable is None \
            or (is_trainable is True and param.requires_grad is True) \
            or (is_trainable is False and param.requires_grad is False):
            for s in list(param.size()):
                nn = nn * s
            num_params += nn
    return num_params


def set_subtract(list1, list2):
    list1 = list(list1)
    list2 = list(list2)
    assert isinstance(list1, list)
    assert isinstance(list2, list)
    return list(set(list1) - set(list2))


def find_nearest(array, value, mode = "abs"):
    array = deepcopy(array)
    array = np.asarray(array)
    if mode == "abs":
        idx = (np.abs(array - value)).argmin()
    elif mode == "le":
        array[array > value] = -np.Inf
        idx = array.argmax()
    elif mode == "ge":
        array[array < value] = np.Inf
        idx = array.argmin()
    else:
        raise
    return idx, array[idx]


def sort_matrix(matrix, dim, reverse = False):
    if dim == 0:
        _, idx_sort = sort_two_lists(matrix[:,0], range(len(matrix[:,0])), reverse = reverse)
        return matrix[idx_sort]
    elif dim == 1:
        _, idx_sort = sort_two_lists(matrix[0,:], range(len(matrix[0,:])), reverse = reverse)
        return matrix[:,idx_sort]
    else:
        raise


def hashing(X, width = 128):
    import hashlib
    def hash_ele(x):
        return np.array([int(element) for element in np.binary_repr(int(hashlib.md5(x.view(np.uint8)).hexdigest(), 16), width = 128)])[-width:]
    is_torch = isinstance(X, torch.Tensor)
    if is_torch:
        is_cuda = X.is_cuda
    X = to_np_array(X)
    hash_list = np.array([hash_ele(x) for x in X])
    if is_torch:
        hash_list = to_Variable(hash_list, is_cuda = is_cuda)
    
    # Check collision:
    string =["".join([str(e) for e in ele]) for ele in to_np_array(hash_list)]
    uniques = np.unique(np.unique(string, return_counts=True)[1], return_counts = True)
    return hash_list, uniques


def pplot(
    X,
    y,
    markers=".",
    label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(10,8),
    fontsize=18,
    plt = None,
    is_show=True,
    ):
    if plt is None:
        import matplotlib.pylab as plt
        plt.figure(figsize=figsize)
    plt.plot(to_np_array(X), to_np_array(y), markers, label=label)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    if label is not None:
        plt.legend(fontsize=fontsize)
    if is_show:
        plt.show()
    return plt


def formalize_value(value, precision):
    """Formalize value with floating or scientific notation, depending on its absolute value."""
    if 10 ** (-(precision - 1)) <= np.abs(value) <= 10 ** (precision - 1):
        return "{0:.{1}f}".format(value, precision)
    else:
        return "{0:.{1}e}".format(value, precision)


def plot1D_3(X_mesh, Z_mesh, target, view_init=[(30, 50), (90, -90), (0, 0)], zlabel=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    X_mesh, Z_mesh, target = to_np_array(X_mesh, Z_mesh, target)
    fig = plt.figure(figsize=(22,7))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X_mesh, Z_mesh, target, alpha=0.5,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False,
                   )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel(zlabel)
    ax.view_init(elev=view_init[0][0], azim=view_init[0][1])

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(X_mesh, Z_mesh, target, alpha=0.5,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False,
                   )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel(zlabel)
    ax.view_init(elev=view_init[1][0], azim=view_init[1][1])

    ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(X_mesh, Z_mesh, target, alpha=0.5,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False,
                   )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel(zlabel)
    ax.view_init(elev=view_init[2][0], azim=view_init[2][1])
    plt.show()
    

class RampupLR(_LRScheduler):
    """Ramp up the learning rate in exponential steps."""
    def __init__(self, optimizer, num_steps=200, last_epoch=-1):
        self.num_steps = num_steps
        super(RampupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * np.logspace(-12, 0, self.num_steps + 1)[self.last_epoch]
                for base_lr in self.base_lrs]

    
def set_cuda(tensor, is_cuda):
    if isinstance(is_cuda, str):
        return tensor.cuda(is_cuda)
    else:
        if is_cuda:
            return tensor.cuda()
        else:
            return tensor


def isin(ar1, ar2):
    ar2 = torch.LongTensor(ar2)
    ar2 = ar2.to(ar1.device)
    return (ar1[..., None] == ar2).any(-1)


def filter_labels(X, y, labels):
    idx = isin(y, labels)
    return X[idx], y[idx]


def argmin_random(tensor):
    """Returns the flattened argmin of the tensor, and tie-breaks using random choice."""
    argmins = np.flatnonzero(tensor == tensor.min())
    if len(argmins) > 0:
        return np.random.choice(argmins)
    else:
        return np.NaN


def argmax_random(tensor):
    """Returns the flattened argmax of the tensor, and tie-breaks using random choice."""
    argmaxs = np.flatnonzero(tensor == tensor.max())
    if len(argmins) > 0:
        return np.random.choice(argmaxs)
    else:
        return np.NaN


class Transform_Label(object):
    def __init__(self, label_noise_matrix=None, is_cuda=False):
        self.label_noise_matrix = label_noise_matrix
        if self.label_noise_matrix is not None:
            assert ((self.label_noise_matrix.sum(0) - 1) < 1e-10).all()
        self.device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")

    def __call__(self, y):
        if self.label_noise_matrix is None:
            return y
        else:
            noise_matrix = self.label_noise_matrix
            dim = len(noise_matrix)
            y_tilde = []
            for y_ele in y:
                flip_rate = noise_matrix[:, y_ele]
                y_ele = np.random.choice(dim, p = flip_rate)
                y_tilde.append(y_ele)
            y_tilde = torch.LongTensor(y_tilde).to(self.device)
            return y_tilde

        
def base_repr(n, base, length):
    assert n < base ** length, "n should be smaller than b ** length"
    base_repr_str = np.base_repr(n, base, padding = length)[-length:]
    return [int(ele) for ele in base_repr_str]


def base_repr_2_int(List, base):
    if len(List) == 1:
        return List[0]
    elif len(List) == 0:
        return 0
    else:
        return base * base_repr_2_int(List[:-1], base) + List[-1]


def get_variable_name_list(expressions):
    """Get variable names from a given expressions list"""
    return sorted(list({symbol.name for expression in expressions for symbol in expression.free_symbols if "x" in symbol.name}))


def get_param_name_list(expressions):
    """Get parameter names from a given expressions list"""
    return sorted(list({symbol.name for expression in expressions for symbol in expression.free_symbols if "x" not in symbol.name}))


def get_function_name_list(symbolic_expression):
    from sympy import Function
    from sympy.utilities.lambdify import implemented_function
    symbolic_expression = standardize_symbolic_expression(symbolic_expression)
    function_name_list = list({element.func.__name__ for expression in symbolic_expression for element in expression.atoms(Function) if element.func.__name__ not in ["linear"]})
    implemented_function = {}
    for function_name in function_name_list:
        try:
            implemented_function[function_name] = implemented_function(Function(function_name), get_activation(function_name))
        except:
            pass
    return function_name_list, implemented_function


def substitute(expressions, param_dict):
    """Substitute each expression in the expression using the param_dict"""
    new_expressions = []
    has_param_list = []
    for expression in expressions:
        has_param = len(get_param_name_list([expression])) > 0
        has_param_list.append(has_param)
        new_expressions.append(expression.subs(param_dict))
    return new_expressions, has_param_list


def get_coeffs(expression):
    """Get coefficients as a list from an expression, w.r.t. its sorted variable name list"""
    from sympy import Poly
    variable_names = get_variable_name_list([expression])
    variables = standardize_symbolic_expression(variable_names)
    if len(variables) > 0:
        # Peel the outmost activation:
        function_name_list, _ = get_function_name_list(expression)
        if len(function_name_list) > 0:
            expression = expression.args[0]
        poly = Poly(expression, *variables)
        return poly.coeffs(), variable_names
    else:
        return [expression], []

    
def get_coeffs_tree(exprs, param_dict):
    """Get snapped coefficients by traversing the whole expression tree."""
    snapped_list = []
    length = 0
    for expr in exprs:
        length += get_coeffs_recur(expr, param_dict, snapped_list)
    return length, snapped_list


def get_coeffs_recur(expr, param_dict, snapped_list):
    import sympy
    if isinstance(expr, sympy.numbers.Float) or isinstance(expr, sympy.numbers.Integer):
        snapped_list.append(float(expr))
        return 1
    elif isinstance(expr, sympy.symbol.Symbol):
        if not expr.name.startswith("x"):
            if expr.name not in param_dict:
                raise Exception("Non-snapped parameter did not appear in param_dict! Check implementation.")
        return 1
    else:
        length = 0
        for sub_expr in expr.args:
            length += get_coeffs_recur(sub_expr, param_dict, snapped_list)
        length += 1
        return length


def standardize_symbolic_expression(symbolic_expression):
    """Standardize symbolic expression to be a list of SymPy expressions"""
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.utilities.lambdify import implemented_function
    if isinstance(symbolic_expression, str):
        symbolic_expression = parse_expr(symbolic_expression)
    if not (isinstance(symbolic_expression, list) or isinstance(symbolic_expression, tuple)):
        symbolic_expression = [symbolic_expression]
    parsed_symbolic_expression = []
    for expression in symbolic_expression:
        parsed_expression = parse_expr(expression) if isinstance(expression, str) else expression
        if hasattr(parsed_expression.func, "name") and parsed_expression.func.name in ACTIVATION_LIST:
            activation = parsed_expression.func.name
            f = implemented_function(activation, get_activation(activation))
            parsed_expression = f(*parsed_expression.args)
        parsed_symbolic_expression.append(parsed_expression)
    return parsed_symbolic_expression


def get_number_DL(n, status):
#     def rank(n):
#         assert isinstance(n, int)
#         if n == 0:
#             return 1
#         else:
#             return 2 * abs(n) + int((1 - np.sign(n)) / 2)
    epsilon = 1e-10
    n = float(n)
    if status == "snapped":
        if np.abs(n - int(n)) < epsilon:
            return np.log2(1 + abs(int(n)))
        else:
            snapped = snap_core([n], "rational")[0][1]
            if snapped is not None and abs(n - snapped) < epsilon:
                _, numerator, denominator, _ = bestApproximation(n, 100)
                return np.log2((1 + abs(numerator)) * abs(denominator))
            else:
                if (n - np.pi) < 1e-10:
                    # It is pi:
                    return np.log2((1 + 3))
                elif (n - np.e) < 1e-10:
                    # It is e:
                    return np.log2((1 + 2))
                else:
                    raise Exception("The snapped numbers should be a rational number! However, {} is not a rational number.".format(n)) 
    elif status == "non-snapped":
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2
    else:
        raise Exception("status {0} not valid!".format(status))


def get_list_DL(List, status):
    if not (isinstance(List, list) or (isinstance(List, np.ndarray) and (len(List.shape) > 1 or (len(List.shape) ==1 and List.shape[0] > 1)))):
        return get_number_DL(List, status)
    else:
        return np.sum([get_list_DL(element, status) for element in List])


def get_model_DL(model):
    if not(isinstance(model, list) or isinstance(model, tuple)):
        model = [model]
    return np.sum([model_ele.DL for model_ele in model])


def zero_grad_hook_multi(rows, cols):
    """
    Args:
        rows, cols: the joint rows and cols that we want to freeze the parameters.
    """
    def hook_function(grad):
        grad[rows,cols] = 0
        return grad
    return hook_function


def zero_grad_hook(idx):
    def hook_function(grad):
        grad[idx] = 0
        return grad
    return hook_function


## The following are snap functions for finding a best approximated integer or rational number for a real number:
def bestApproximation(x,imax):
    # The input is a numpy parameter vector p.
    # The output is an integer specifying which parameter to change,
    # and a float specifying the new value.
    def float2contfrac(x,nmax):
        c = [np.floor(x)];
        y = x - np.floor(x)
        k = 0
        while np.abs(y)!=0 and k<nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c

    def contfrac2frac(seq):
        ''' Convert the simple continued fraction in `seq` 
            into a fraction, num / den
        '''
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num*u, num
        return num, den

    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[:i+1]) for i in range(len(c))))

    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:,0] / float(q[:,1])

    def truncateContFrac(q,imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k,0]), q[k,1]) <= imax:
            k = k + 1
        return q[:k]

    def pval(p):
        return 1 - np.exp(-p ** 0.87 / 0.36)

    xsign = np.sign(x)
    q = truncateContFrac(contFracRationalApproximations(float2contfrac(abs(x),20)),imax)
    
    if len(q) > 0:
        p = np.abs(q[:,0] / q[:,1] - abs(x)).astype(float) * (1 + np.abs(q[:,0])) * q[:,1]
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i,0] / float(q[i,1]), xsign* q[i,0], q[i,1], p[i])
    else:
        return (None, 0, 0, 1)


def integerSnap(p, top=1):
    metric = np.abs(p - np.round(p))
    chosen = np.argsort(metric)[:top]
    return list(zip(chosen, np.round(p)[chosen]))


def zeroSnap(p, top=1):
    metric = np.abs(p)
    chosen = np.argsort(metric)[:top]
    return list(zip(chosen, np.zeros(len(chosen))))
        

def rationalSnap(p, top=1):
    """Snap to nearest rational number using continued fraction."""
    snaps = np.array(list(bestApproximation(x,100) for x in p))
    chosen = np.argsort(snaps[:, 3])[:top]
    return list(zip(chosen, snaps[chosen, 0]))

def vectorSnap(param_dict, top=1):
    """Divide p by its smallest number, and then perform integer snap.

    Args:
        param_dict: dictionary of parameter name and its values for snapping.
        top: number of parameters to snap.

    Returns:
        param_dict_subs: dictionary for substitution.
        new_param_dict: initial value for the remaining parameters.
    """
    tiny = 0.01
    huge = 1000000.
    print(param_dict)
    p = list(param_dict.values())
    symbs = list(param_dict.keys())
    ap = np.abs(p)
    for i in range(len(p)):
        if ap[i] < tiny:
            ap[i] = huge
    i = np.argmin(ap)
    apmin = ap[i] * np.sign(p[i])
    if apmin >= huge:
        return {}, None
    else:
        q = p / apmin.astype(float)
        snap_targets = integerSnap(q, top=top + 1)
        param_dict_subs = {}
        new_param_dict = {}
        newsymb = symbs[i]
        for k in range(len(snap_targets)):
            if snap_targets[k][0] != i:
                if snap_targets[k][1] != 1:
                    param_dict_subs[symbs[snap_targets[k][0]]] = "{}".format(snap_targets[k][1]) + "*" + newsymb
                else:
                    param_dict_subs[symbs[snap_targets[k][0]]] = newsymb
        new_param_dict[newsymb] = param_dict[newsymb]
        return param_dict_subs, new_param_dict

def pairSnap(p, snap_mode = "integer"):
    p = np.array(p)
    n = p.shape[0]
    pairs = []
    for i in range(1,n):
        for j in range(i):
            pairs.append([i,j,p[i]/p[j]])
            pairs.append([j,i,p[j]/p[i]])
    pairs = np.array(pairs)
    if snap_mode == "integer":
        (k,ratio) = integerSnap(pairs[:,2])
    elif snap_mode == "rational":
        (k,ratio) = rationalSnap(pairs[:,2])
    else:
        raise Exception("snap_mode {0} not recognized!".format(snap_mode))
    return (int(pairs[k,0]), int(pairs[k,1])), int(ratio)

# Finds best separable approximation of a matrix as M=ab^t.
# There's a degeneracy between rescaling a and rescaling b, which we fix by setting the smallest non-zero element of a equal to 1.
def separableSnap(M):
    (U,w,V) = np.linalg.svd(M)
    Mnew = w[0]*np.matmul(U[:,:1],V[:1,:])
    a = U[:,0]*w[0]
    b = V[0,:]
    tiny = 0.001
    huge = 1000000.
    aa = np.abs(a)
    for i in range(len(aa)):
        if aa[i] < tiny:
            aa[i] = huge
    i = np.argmin(aa)
    aamin = aa[i] * np.sign(aa[i])
    aa = aa/aamin
    return (a/aamin,b*aamin)


def snap_core(p, snap_mode, top=1):
    if len(p) == 0:
        return []
    else:
        if snap_mode == "zero":
            return zeroSnap(p, top=top)
        elif snap_mode == "integer":
            return integerSnap(p, top=top)
        elif snap_mode == "rational":
            return rationalSnap(p, top=top)
        elif snap_mode == "vector":
            return vectorSnap(p, top=top)
        elif snap_mode == "pair_integer":
            return pairSnap(p, snap_mode = "integer")
        elif snap_mode == "pair_rational":
            return pairSnap(p, snap_mode = "rational")
        elif snap_mode == "separable":
            return separableSnap(p)
        else:
            raise Exception("Snap mode {0} not recognized!".format(snap_mode))


def snap(param, snap_mode, excluded_idx=None, top=1):
    if excluded_idx is None or len(excluded_idx) == 0:
        return snap_core(param, snap_mode=snap_mode, top=top)
    else:
        full_idx = list(range(len(param)))
        valid_idx = sorted(list(set(full_idx) - set(excluded_idx)))
        valid_dict = list(enumerate(valid_idx))
        param_valid = [param[i] for i in valid_idx]
        snap_targets_valid = snap_core(param_valid, snap_mode=snap_mode, top=top)
        snap_targets = []
        for idx_valid, new_value in snap_targets_valid:
            snap_targets.append((valid_dict[idx_valid][1], new_value))
        return snap_targets


def unsnap(exprs, param_dict):
    """Unsnap a symbolic expression, tranforming all numerical values to learnable parameters."""
    unsnapped_param_dict = {}
    unsnapped_exprs = []
    for expr in exprs:
        unsnapped_expr = unsnap_recur(expr, param_dict, unsnapped_param_dict)
        unsnapped_exprs.append(unsnapped_expr)
    return unsnapped_exprs, unsnapped_param_dict


def unsnap_recur(expr, param_dict, unsnapped_param_dict):
    """Recursively transform each numerical value into a learnable parameter."""
    import sympy
    from sympy import Symbol
    if isinstance(expr, sympy.numbers.Float) or isinstance(expr, sympy.numbers.Integer):
        used_param_names = list(param_dict.keys()) + list(unsnapped_param_dict)
        unsnapped_param_name = get_next_available_key(used_param_names, "p", is_underscore=False)
        unsnapped_param_dict[unsnapped_param_name] = float(expr)
        unsnapped_expr = Symbol(unsnapped_param_name)
        return unsnapped_expr
    elif isinstance(expr, sympy.symbol.Symbol):
        return expr
    else:
        unsnapped_sub_expr_list = []
        for sub_expr in expr.args:
            unsnapped_sub_expr = unsnap_recur(sub_expr, param_dict, unsnapped_param_dict)
            unsnapped_sub_expr_list.append(unsnapped_sub_expr)
        return expr.func(*unsnapped_sub_expr_list)


def get_next_available_key(iterable, key, midfix="", suffix="", is_underscore=True, start_from_null=False):
    """Get the next available key that does not collide with the keys in the dictionary."""
    if start_from_null and key + suffix not in iterable:
        return key + suffix
    else:
        i = 0
        underscore = "_" if is_underscore else ""
        while "{}{}{}{}{}".format(key, underscore, midfix, i, suffix) in iterable:
            i += 1
        new_key = "{}{}{}{}{}".format(key, underscore, midfix, i, suffix)
        return new_key
    
    

def update_dictionary(dictionary, key, item):
    """Update the key: item in the dictionary. If key collision happens, 
    rename the key if the items do not refer to the same thing."""
    if key in dictionary:
        if dictionary[key] is item:
            result = 2
            return result, key
        else:
            result = 0
            new_key = get_next_available_key(dictionary, key)
            dictionary[new_key] = item
            return result, new_key
    else:
        result = 1
        dictionary[key] = item
        return result, key

    
def pprint_dict(dictionary):
    string = ""
    for key, item in dictionary.items():
        string += "{}: {}; ".format(key, item)
    string = string[:-2]
    return string

    
class Batch_Generator(object):
    def __init__(self, X, y, batch_size = 50, target_one_hot_off = False):
        """
        Initilize the Batch_Generator class
        """
        if isinstance(X, tuple):
            self.binary_X = True
        else:
            self.binary_X = False
        if isinstance(X, Variable):
            X = X.cpu().data.numpy()
        if isinstance(y, Variable):
            y = y.data.numpy()
        self.target_one_hot_off = target_one_hot_off

        if self.binary_X:
            X1, X2 = X
            self.sample_length = len(y)
            assert len(X1) == len(X2) == self.sample_length, "X and y must have the same length!"
            assert batch_size <= self.sample_length, "batch_size must not be larger than \
                    the number of samples!"
            self.batch_size = int(batch_size)

            X1_len = [element.shape[-1] for element in X1]
            X2_len = [element.shape[-1] for element in X2]
            y_len = [element.shape[-1] for element in y]
            if len(np.unique(X1_len)) == 1 and len(np.unique(X2_len)) == 1:
                assert len(np.unique(y_len)) == 1, \
                    "when X1 and X2 has only one size, the y should also have only one size!"
                self.combine_size = False
                self.X1 = np.array(X1)
                self.X2 = np.array(X2)
                self.y = np.array(y)
                self.index = np.array(range(self.sample_length))
                self.idx_batch = 0
                self.idx_epoch = 0
            else:
                self.combine_size = True
                self.X1 = X1
                self.X2 = X2
                self.y = y
                self.input_dims_list = zip(X1_len, X2_len)
                self.input_dims_dict = {}
                for i, input_dims in enumerate(self.input_dims_list):
                    if input_dims not in self.input_dims_dict:
                        self.input_dims_dict[input_dims] = {"idx":[i]}
                    else:
                        self.input_dims_dict[input_dims]["idx"].append(i)
                for input_dims in self.input_dims_dict:
                    idx = np.array(self.input_dims_dict[input_dims]["idx"])
                    self.input_dims_dict[input_dims]["idx"] = idx
                    self.input_dims_dict[input_dims]["X1"] = [X1[i] for i in idx]
                    self.input_dims_dict[input_dims]["X2"] = [X2[i] for i in idx]
                    self.input_dims_dict[input_dims]["y"] = [y[i] for i in idx]
                    self.input_dims_dict[input_dims]["idx_batch"] = 0
                    self.input_dims_dict[input_dims]["idx_epoch"] = 0
                    self.input_dims_dict[input_dims]["index"] = np.array(range(len(idx))).astype(int)
        else:
            self.sample_length = len(y)
            assert batch_size <= self.sample_length, "batch_size must not be larger than \
                    the number of samples!"
            self.batch_size = int(batch_size)

            X_len = [element.shape[-1] for element in X]
            y_len = [element.shape[-1] for element in y]

            if len(np.unique(X_len)) == 1 and len(np.unique(y_len)) == 1:
                assert len(np.unique(y_len)) == 1, \
                    "when X has only one size, the y should also have only one size!"
                self.combine_size = False
                self.X = np.array(X)
                self.y = np.array(y)
                self.index = np.array(range(self.sample_length))
                self.idx_batch = 0
                self.idx_epoch = 0
            else:
                self.combine_size = True
                self.X = X
                self.y = y
                self.input_dims_list = zip(X_len, y_len)
                self.input_dims_dict = {}
                for i, input_dims in enumerate(self.input_dims_list):
                    if input_dims not in self.input_dims_dict:
                        self.input_dims_dict[input_dims] = {"idx":[i]}
                    else:
                        self.input_dims_dict[input_dims]["idx"].append(i)
                for input_dims in self.input_dims_dict:
                    idx = np.array(self.input_dims_dict[input_dims]["idx"])
                    self.input_dims_dict[input_dims]["idx"] = idx
                    self.input_dims_dict[input_dims]["X"] = [X[i] for i in idx]
                    self.input_dims_dict[input_dims]["y"] = [y[i] for i in idx]
                    self.input_dims_dict[input_dims]["idx_batch"] = 0
                    self.input_dims_dict[input_dims]["idx_epoch"] = 0
                    self.input_dims_dict[input_dims]["index"] = np.array(range(len(idx))).astype(int)


    def reset(self):
        """Reset the index and batch iteration to the initialization state"""
        if not self.combine_size:
            self.index = np.array(range(self.sample_length))
            self.idx_batch = 0
            self.idx_epoch = 0
        else:
            for input_dims in self.input_dims_dict:
                self.input_dims_dict[input_dims]["idx_batch"] = 0
                self.input_dims_dict[input_dims]["idx_epoch"] = 0
                self.input_dims_dict[input_dims]["index"] = np.array(range(len(idx))).astype(int)


    def next_batch(self, mode = "random", isTorch = False, is_cuda = False, given_dims = None):
        """Generate each batch with the same size (even if the examples and target may have variable size)"""
        if self.binary_X:
            if not self.combine_size:
                start = self.idx_batch * self.batch_size
                end = (self.idx_batch + 1) * self.batch_size

                if end > self.sample_length:
                    self.idx_epoch += 1
                    self.idx_batch = 0
                    np.random.shuffle(self.index)
                    start = 0
                    end = self.batch_size

                self.idx_batch += 1
                chosen_index = self.index[start:end]
                y_batch = deepcopy(self.y[chosen_index])
                if self.target_one_hot_off:
                    y_batch = y_batch.argmax(-1)
                X1_batch = deepcopy(self.X1[chosen_index])
                X2_batch = deepcopy(self.X2[chosen_index])
            else: # If the input_dims have variable size
                if mode == "random":
                    if given_dims is None:
                        rand = np.random.choice(self.sample_length)
                        input_dims = self.input_dims_list[rand]
                    else:
                        input_dims = given_dims
                    length = len(self.input_dims_dict[input_dims]["idx"])
                    if self.batch_size >= length:
                        chosen_index = np.random.choice(length, size = self.batch_size, replace = True)
                    else:
                        start = self.input_dims_dict[input_dims]["idx_batch"] * self.batch_size
                        end = (self.input_dims_dict[input_dims]["idx_batch"] + 1) * self.batch_size
                        if end > length:
                            self.input_dims_dict[input_dims]["idx_epoch"] += 1
                            self.input_dims_dict[input_dims]["idx_batch"] = 0
                            np.random.shuffle(self.input_dims_dict[input_dims]["index"])
                            start = 0
                            end = self.batch_size
                        self.input_dims_dict[input_dims]["idx_batch"] += 1
                        chosen_index = self.input_dims_dict[input_dims]["index"][start:end]
                    y_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["y"][j] for j in chosen_index]))
                    if self.target_one_hot_off:
                        y_batch = y_batch.argmax(-1)
                    X1_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["X1"][j] for j in chosen_index]))
                    X2_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["X2"][j] for j in chosen_index]))

            if isTorch:
                X1_batch = Variable(torch.from_numpy(X1_batch), requires_grad = False).type(torch.FloatTensor)
                X2_batch = Variable(torch.from_numpy(X2_batch), requires_grad = False).type(torch.FloatTensor)
                if self.target_one_hot_off:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.LongTensor)
                else:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.FloatTensor)

                if is_cuda:
                    X1_batch = X1_batch.cuda()
                    X2_batch = X2_batch.cuda()
                    y_batch = y_batch.cuda()

            return (X1_batch, X2_batch), y_batch
        else:
            if not self.combine_size:
                start = self.idx_batch * self.batch_size
                end = (self.idx_batch + 1) * self.batch_size

                if end > self.sample_length:
                    self.idx_epoch += 1
                    self.idx_batch = 0
                    np.random.shuffle(self.index)
                    start = 0
                    end = self.batch_size

                self.idx_batch += 1
                chosen_index = self.index[start:end]
                y_batch = deepcopy(self.y[chosen_index])
                if self.target_one_hot_off:
                    y_batch = y_batch.argmax(-1)
                X_batch = deepcopy(self.X[chosen_index])
            else: # If the input_dims have variable size
                if mode == "random":
                    if given_dims is None:
                        rand = np.random.choice(self.sample_length)
                        input_dims = self.input_dims_list[rand]
                    else:
                        input_dims = given_dims
                    length = len(self.input_dims_dict[input_dims]["idx"])
                    if self.batch_size >= length:
                        chosen_index = np.random.choice(length, size = self.batch_size, replace = True)
                    else:
                        start = self.input_dims_dict[input_dims]["idx_batch"] * self.batch_size
                        end = (self.input_dims_dict[input_dims]["idx_batch"] + 1) * self.batch_size
                        if end > length:
                            self.input_dims_dict[input_dims]["idx_epoch"] += 1
                            self.input_dims_dict[input_dims]["idx_batch"] = 0
                            np.random.shuffle(self.input_dims_dict[input_dims]["index"])
                            start = 0
                            end = self.batch_size
                        self.input_dims_dict[input_dims]["idx_batch"] += 1
                        chosen_index = self.input_dims_dict[input_dims]["index"][start:end]
                    y_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["y"][j] for j in chosen_index]))
                    if self.target_one_hot_off:
                        y_batch = y_batch.argmax(-1)
                    X_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["X"][j] for j in chosen_index]))
            if isTorch:
                X_batch = Variable(torch.from_numpy(X_batch), requires_grad = False).type(torch.FloatTensor)
                if self.target_one_hot_off:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.LongTensor)
                else:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.FloatTensor)

                if is_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

            return X_batch, y_batch

        
def logmeanexp(tensor, axis, keepdims=False):
    """Calculate logmeanexp of tensor."""
    return torch.logsumexp(tensor, axis=axis, keepdims=keepdims) - np.log(tensor.shape[axis])


def str2bool(v):
    """used for argparse, 'type=str2bool', so that can pass in string True or False."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_class_correspondence(predicted_domain, true_domain, verbose=True):
    """Compute the best match of two lists of labels among all permutations."""
    def mismatches(domain1, domain2):
        return (domain1 != domain2).sum()

    def associate(domain, current_ids, new_ids):
        new_domain = deepcopy(domain)
        for i, current_id in enumerate(current_ids):
            new_domain[domain == current_id] = new_ids[i]
        return new_domain

    assert len(predicted_domain.shape) == len(true_domain.shape) == 1
    assert isinstance(predicted_domain, np.ndarray)
    assert isinstance(true_domain, np.ndarray)
    predicted_domain = predicted_domain.flatten().astype(int)
    true_domain = true_domain.flatten().astype(int)
    predicted_ids = np.unique(predicted_domain)
    true_ids = np.unique(true_domain)

    union_ids = np.sort(list(set(predicted_ids).union(set(true_ids))))
    if len(union_ids) > 10:
        if verbose:
            print("num_domains = {0}, too large!".format(len(union_ids)))
        return None, None, None
    min_num_mismatches = np.inf
    argmin_permute = None
    predicted_domain_argmin = None

    for i, union_ids_permute in enumerate(itertools.permutations(union_ids)):
        predicted_domain_permute = associate(predicted_domain, union_ids, union_ids_permute)
        num_mismatches = mismatches(predicted_domain_permute, true_domain)
        if num_mismatches < min_num_mismatches:
            min_num_mismatches = num_mismatches
            argmin_permute = union_ids_permute
            predicted_domain_argmin = predicted_domain_permute
        if verbose and i % 100000 == 0:
            print(i)
        if min_num_mismatches == 0:
            break
    return predicted_domain_argmin, min_num_mismatches, list(zip(union_ids, argmin_permute))


def compute_class_correspondence_greedy(predicted_domain, true_domain, verbose=True, threshold=7):
    """Compute the best match of two lists of labels among all permutations; if number of labels
        is greater than the threshold, the labels with top threshold + 1 numbers perform its own permutation.
    """
    def mismatches(domain1, domain2):
        return (domain1 != domain2).sum()

    def associate(domain, current_ids, new_ids):
        new_domain = deepcopy(domain)
        for i, current_id in enumerate(current_ids):
            new_domain[domain == current_id] = new_ids[i]
        return new_domain

    assert len(predicted_domain.shape) == len(true_domain.shape) == 1
    assert isinstance(predicted_domain, np.ndarray)
    assert isinstance(true_domain, np.ndarray)
    predicted_domain = predicted_domain.flatten().astype(int)
    true_domain = true_domain.flatten().astype(int)
    predicted_ids, _ = get_sorted_counts(predicted_domain)
    true_ids, _ = get_sorted_counts(true_domain)
    assert len(predicted_ids) == len(true_ids)

    if len(true_ids) > threshold:
        num_greedy_ids = max(2, len(true_ids) - threshold)
    else:
        num_greedy_ids = -1

    min_num_mismatches = np.inf
    argmin_permute = None
    predicted_domain_argmin = None  

    total = 0
    for i, ids_permute_greedy in enumerate(itertools.permutations(true_ids[:num_greedy_ids + 1])):
        ids_inner = ids_permute_greedy[-1:] + tuple(true_ids[num_greedy_ids + 1:])
        for j, ids_permute_all in enumerate(itertools.permutations(ids_inner)):
            permuted_id = ids_permute_greedy[:-1] + ids_permute_all
            predicted_domain_permute = associate(predicted_domain, predicted_ids, permuted_id)
            try:
                num_mismatches = mismatches(predicted_domain_permute, true_domain)
            except:
                import pdb; pdb.set_trace()
            if num_mismatches < min_num_mismatches:
                min_num_mismatches = num_mismatches
                argmin_permute = (predicted_ids, permuted_id)
                predicted_domain_argmin = predicted_domain_permute
            if verbose and total % 100000 == 0:
                print(total)
            total += 1
            if min_num_mismatches == 0:
                break
    return predicted_domain_argmin, min_num_mismatches, list(zip(*argmin_permute))


def acc_spectral_clutering(
    X, y,
    n_neighbors=10,
    num_runs=1,
    affinity="nearest_neighbors",
    matching_method="full_permute",
    **kwargs
):
    X, y = to_np_array(X, y)
    n_clusters = len(np.unique(y))
    n_components = n_clusters
    
    acc_list = []
    for i in range(num_runs):
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            n_components=n_components,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
        ).fit(X)
        cluster_labels = clustering.labels_

        if matching_method == "full_permute":
            _, min_num_mismatches, _ = compute_class_correspondence(cluster_labels, y, **kwargs)
        elif matching_method == "greedy_permute":
            _, min_num_mismatches, _ = compute_class_correspondence_greedy(cluster_labels, y, **kwargs)
        else:
            raise
        acc = 1 - min_num_mismatches / float(len(y))
        acc_list.append(acc)
    return np.max(acc_list), cluster_labels


def get_sorted_counts(array):
    array_unique, counts = np.unique(array, return_counts=True)
    counts, array_unique = sort_two_lists(counts, array_unique, reverse=True)
    return array_unique, counts


def filter_kwargs(kwargs, param_names=None, contains=None):
    """Build a new dictionary based on the filtering criteria.

    Args:
        param_names: if not None, will find the keys that are in the list of 'param_names'.
        contains: if not None, will find the keys that contain the substrings in the list of 'contains'.

    Returns:
        new_kwargs: new kwargs dictionary.
    """
    new_kwargs = {}
    if param_names is not None:
        assert contains is None
        if not isinstance(param_names, list):
            param_names = [param_names]
        for key, item in kwargs.items():
            if key in param_names:
                new_kwargs[key] = item
    else:
        assert contains is not None
        if not isinstance(contains, list):
            contains = [contains]
        for key, item in kwargs.items():
            for ele in contains:
                if ele in key:
                    new_kwargs[key] = item
                    break
    return new_kwargs


def get_key_of_largest_value(Dict):
    return max(Dict.items(), key=operator.itemgetter(1))[0]


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def compose_two_keylists(keys1, keys2):
    """Return a fully broadcast list of keys, combining keys1 and keys2."""
    length1 = [len(key) for key in keys1]
    length2 = [len(key) for key in keys2]
    assert len(np.unique(length1)) <= 1
    assert len(np.unique(length2)) <= 1
    if len(length1) == 0:
        return keys2
    if len(length2) == 0:
        return keys1
    length1 = length1[0]
    length2 = length2[0]

    # Find the maximum length where both keys share the same sub_keys:
    is_same = True
    for i in range(1, min(length1, length2) + 1):
        sub_keys1 = [key[:i] for key in keys1]
        sub_keys2 = [key[:i] for key in keys2]
        if set(sub_keys1) != set(sub_keys2):
            assert len(set(sub_keys1).intersection(set(sub_keys2))) == 0
            is_same = False
            break
    if is_same:
        largest_common_length = i
    else:
        largest_common_length = i - 1

    new_keys = []
    keys_common = remove_duplicates([key[:largest_common_length] for key in keys1])
    keys1_reduced = remove_duplicates([key[largest_common_length:] for key in keys1])
    keys2_reduced = remove_duplicates([key[largest_common_length:] for key in keys2])
    for key_common in keys_common:
        if length1 > largest_common_length:
            for key1 in keys1_reduced:
                if length2 > largest_common_length:
                    for key2 in keys2_reduced:
                        key = key_common + key1 + key2
                        new_keys.append(key)
                else:
                    key = key_common + key1
                    new_keys.append(key)

        else:
            if length2 > largest_common_length:
                for key2 in keys2_reduced:
                    key = key_common + key2
                    new_keys.append(key)
            else:
                key = key_common
                new_keys.append(key)
    return new_keys


def broadcast_keys(key_list_all):
    """Return a fully broadcast {new_broadcast_key: list of Arg keys}

    key_list_all: a list of items, where each item is a list of keys.
    For example, key_list_all = [[(0, "s"), (0, "d"), (1, "d)], [0, 1], None], it will return:

    key_dict = {(0, "s"): [(0, "s"), 0, None],
                (0, "d"): [(0, "d"), 0, None],
                (1, "d"): [(1, "d"), 1, None]}
    Here None denotes that there is only one input, which will be broadcast to all.
    """
    # First: get all the combinations
    new_key_list = []
    for i, keys in enumerate(key_list_all):
        if keys is not None:
            keys = [(ele,) if not isinstance(ele, tuple) else ele for ele in keys]
            new_key_list = compose_two_keylists(new_key_list, keys)

    ## new_key_list: a list of fully broadcast keys
    ## key_list_all: a list of original_key_list, each of which corresponds to
    ##               the keys of an OrderedDict()
    key_dict = {}
    for new_key in new_key_list:
        new_key_map_list = []
        is_match_all = True
        for original_key_list in key_list_all:
            if original_key_list is None:
                new_key_map_list.append(None)
                is_match = True
            else:
                is_match = False
                for key in original_key_list:
                    key = (key,) if not isinstance(key, tuple) else key
                    if set(key).issubset(set(new_key)):
                        is_match = True
                        if len(key) == 1:
                            key = key[0]
                        new_key_map_list.append(key)
                        break
            is_match_all = is_match_all and is_match
        if is_match_all:
            if len(new_key) == 1:
                new_key = new_key[0]
            key_dict[new_key] = new_key_map_list
    return key_dict


def split_bucket(dictionary, num_common):
    """Split the dictionary into multiple buckets, determined by key[num_common:]."""
    from multiset import Multiset
    keys_common = remove_duplicates([key[:num_common] for key in dictionary.keys()])
    # Find the different keys:
    keys_diff = []
    for key in dictionary.keys():
        if Multiset(keys_common[0]).issubset(Multiset(key)):
            if key[num_common:]  not in keys_diff:
                keys_diff.append(key[num_common:])
    keys_diff = sorted(keys_diff)

    buckets = [OrderedDict() for _ in range(len(keys_diff))]
    for key, item in dictionary.items():
        id = keys_diff.index(key[num_common:])
        buckets[id][key[:num_common]] = item
    return buckets


def get_list_elements(List, string_idx):
    """Select elements of the list based on string_idx.
    
    Format of string_idx:
        if starting with "r", means first performs random permutation.
        "100:200": the 100th to 199th elements
        "100:" : the 100th elements and onward
        ":200" : the 0th to 199th elements
        "150" : the 150th element
        "::" : all elements
    """
    # Permute if starting with "r":
    if string_idx.startswith("r"):
        List = np.random.permutation(List).tolist()
        string_idx = string_idx[1:]
    # Select indices:
    if string_idx == "::":
        return List
    elif ":" in string_idx:
        string_split = string_idx.split(":")
        string_split = [string for string in string_split if len(string) != 0]
        if len(string_split) == 2:
            start_idx, end_idx = string_idx.split(":")
            start_idx, end_idx = eval(start_idx), eval(end_idx)
            if end_idx > len(List):
                raise Exception("The end index exceeds the length of the list!")
            list_selected = List[start_idx: end_idx]
        elif len(string_split) == 1:
            if string_idx.startswith(":"):
                list_selected = List[:eval(string_idx[1:])]
            else:
                list_selected = List[eval(string_idx[:-1]):]
        else:
            raise
    else:
        string_idx = eval(string_idx)
        list_selected = [List[string_idx]]
    return list_selected


def get_generalized_mean(List, cumu_mode="mean", epsilon=1e-10):
    """Get generalized-mean of elements in the list"""
    List = np.array(list(List))
    assert len(List.shape) == 1

    if cumu_mode[0] == "gm" and cumu_mode[1] == 1:
        cumu_mode = "mean"
    elif cumu_mode[0] == "gm" and cumu_mode[1] == 0:
        cumu_mode = "geometric"
    elif cumu_mode[0] == "gm" and cumu_mode[1] == -1:
        cumu_mode = "harmonic"

    # Obtain mean:
    if cumu_mode == "mean":
        mean = List.mean()
    elif cumu_mode == "min":
        mean = np.min(List)
    elif cumu_mode == "max":
        mean = np.max(List)
    elif cumu_mode == "harmonic":
        mean = len(List) / (1 / (List + epsilon)).sum()
    elif cumu_mode == "geometric":
        mean = (List + epsilon).prod() ** (1 / float(len(List)))
    elif cumu_mode[0] == "gm":
        order = cumu_mode[1]
        mean = (np.minimum((List + epsilon) ** order, 1e30).mean()) ** (1 / float(order))
    else:
        raise
    return mean



def upper_first(string):
    """Return a new string with the first letter capitalized."""
    if len(string) > 0:
        return string[0].upper() + string[1:]
    else:
        return string


def lower_first(string):
    """Return a new string with the first letter capitalized."""
    if len(string) > 0:
        return string[0].lower() + string[1:]
    else:
        return string

    
def update_dict(Dict, key, value):
    """Return a new dictionary with the item with key updated by the corresponding value"""
    if Dict is None:
        return None
    new_dict = deepcopy(Dict)
    new_dict[key] = value
    return new_dict


def get_root_dir(suffix):
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index(suffix)
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def check_same_set(List_of_List):
    """Return True if each element list has the same set of elements."""
    if len(List_of_List) == 0:
        return None
    List = List_of_List[0]
    for List_ele in List_of_List:
        if set(List) != set(List_ele):
            return False
    return True


def check_same_dict(Dict, value_list, key_list):
    """Check if the value stored is the same as the newly given value_list.
    Return a list of keys whose values are different from the stored ones.
    """
    if len(Dict) == 0:
        for key, value in zip(key_list, value_list):
            Dict[key] = value
        return []
    else:
        not_equal_list = []
        for key, value in zip(key_list, value_list):
            value_stored = Dict[key]
            if isinstance(value, Number) or isinstance(value, tuple) or isinstance(value, list):
                is_equal = value == value_stored
                if not is_equal:
                    not_equal_list.append(key)
            else:
                if tuple(value.shape) != tuple(value_stored.shape):
                    not_equal_list.append(key)
                else:
                    is_equal = (value == value_stored).all()
                    if not is_equal:
                        not_equal_list.append(key)
        return not_equal_list


def check_same_model_dict(model_dict1, model_dict2):
    """Check if two model_dict are the same."""
    assert set(model_dict1.keys()) == set(model_dict2.keys()), "model_dict1 and model_dict2 has different keys!"
    for key, item1 in model_dict1.items():
        item2 = model_dict2[key]
        if not isinstance(item1, dict):
            assert item1 == item2, "key '{}' has different values of '{}' and '{}'.".format(key, item1, item2)
    return True


def print_banner(string, banner_size=100, n_new_lines=0):
    """Pring the string sandwidched by two lines."""
    for i in range(n_new_lines):
        print()
    print("\n" + "=" * banner_size + "\n" + string + "\n" + "=" * banner_size + "\n")


class Printer(object):
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))


def switch_dict_keys(Dict, key1, key2):
    inter = Dict[key1]
    Dict[key1] = Dict[key2]
    Dict[key2] = inter
    return Dict


def get_hashing(string_repr, length=None):
    """Get the hashing of a string."""
    import hashlib, base64
    hashing = base64.b64encode(hashlib.md5(string_repr.encode('utf-8')).digest()).decode().replace("/", "a")[:-2]
    if length is not None:
        hashing = hashing[:length]
    return hashing


def get_filename(short_str_dict, args_dict, suffix=".p"):
    """Get the filename using given short_str_dict and info of args_dict.

    Args:
        short_str_dict: mapping of long args name and short string, e.g. {"dataset": "data", "epoch": "ep", "lr": "lr#1"} (the #{Number} means multiple
            args share the same short_str).
        args_dict: args.__dict__.
        suffix: suffix for the filename.
    """
    string_list = []
    for k, v in short_str_dict.items():
        if args_dict[k] is None:
            continue
        elif v == "":
            string_list.append(args_dict[k])
        else:
            if len(v.split("#")) == 2:
                id = eval(v.split("#")[1])
                if id == 1:
                    string_list.append("{}_{}".format(v.split("#")[0], args_dict[k]))
                else:
                    string_list.append("{}".format(args_dict[k]))
            elif k == "gpuid":
                string_list.append("{}:{}".format(v, args_dict[k]))
            else:
                string_list.append("{}_{}".format(v, args_dict[k]))
    return "_".join(string_list) + suffix


def get_filename_short(
    args_shown,
    short_str_dict,
    args_dict,
    hash_exclude_shown=False,
    hash_length=8,
    print_excluded_args=False,
    suffix=".p"
):
    """Get the filename using given short_str_dict, args_shown and info of args_dict.
        The args not in args_shown will not be put explicitly in the filename, but the full
        args_dict will be turned into a unique hash.

    Args:
        args_shown: fields of the args that need to appear explicitly in the filename
        short_str_dict: mapping of long args name and short string, e.g. {"dataset": "data", "epoch": "ep"}
        args_dict: args.__dict__.
        hash_exclude_shown: if True, will exclude the args that are in the args_shown when computing the hash.
        hash_length: length of the hash.
        suffix: suffix for the filename.
    """
    # Get the short name from short_str_dict:
    str_dict = {}
    for key in args_shown:
        if key in args_dict:
            str_dict[key] = short_str_dict[key]
        else:
            raise Exception("'{}' not in the short_str_dict. Need to add its short name into it.".format(key))
    short_filename = get_filename(str_dict, args_dict, suffix="")

    # Get the hashing for the full args_dict:
    args_dict_excluded = deepcopy(args_dict)
    for key in args_shown:
        args_dict_excluded.pop(key)
    if print_excluded_args:
        print("Excluded args in explicit filename: {}".format(list(args_dict_excluded)))
    hashing = get_hashing(str(args_dict_excluded) if hash_exclude_shown else str(args_dict), length=hash_length)
    return short_filename + "_Hash_{}{}".format(hashing, suffix)


def write_to_config(args, filename):
    """Write to a yaml configuration file. The filename contains path to that file.
    """
    import yaml
    dirname = "/".join(filename.split("/")[:-1])
    config_filename = os.path.join(dirname, "config", filename.split("/")[-1][:-2] + ".yaml")
    make_dir(config_filename)
    with open(config_filename, "w") as f:
        yaml.dump(args.__dict__, f)


def load_config(config_filename):
    """Load a configuration file."""
    import yaml
    with open(config_filename, "r") as f:
        Dict = yaml.load(f, Loader=yaml.FullLoader)
    return Dict


def argparser_to_yaml(parser, filename=None, comment_column=40, is_sort_keys=False):
    """Convert an argparser into yaml file with help as comments."""
    import ruamel.yaml
    from ruamel.yaml.compat import StringIO
    from ruamel.yaml import YAML
    class MyYAML(YAML):
        def dump(self, data, stream=None, **kw):
            inefficient = False
            if stream is None:
                inefficient = True
                stream = StringIO()
            YAML.dump(self, data, stream, **kw)
            if inefficient:
                return stream.getvalue()

    data_dict = {}
    help_dict = {}
    for i, action in enumerate(parser._actions):
        if action.__class__.__name__ == "_StoreAction":
            key = action.option_strings[0].split("--")[1]
            default = action.default
            helptxt = action.help
            data_dict[key] = default
            help_dict[key] = helptxt

    if is_sort_keys:
        keys_sorted = sorted(data_dict.keys())
        data_dict_sorted = {}
        for key in keys_sorted:
            data_dict_sorted[key] = data_dict[key]
        data_dict = data_dict_sorted

    yaml = MyYAML()
    inp = yaml.dump(data_dict)
    data = yaml.load(inp)
    for key in data_dict:
        if help_dict[key] is not None:
            data.yaml_add_eol_comment(help_dict[key], key, column=comment_column)

    if filename is None:
        yaml.dump(data, sys.stdout)
    else:
        with open(filename, "w") as f:
            yaml.dump(data, f)


def check_injective(Dict, exclude=[""]):
    """Check if the value of a dictionary is injective, excluding certain values as given by 'exclude'."""
    List = []
    for k, v in Dict.items():
        if v in List and v not in exclude:
            raise Exception("the value {} of {} has duplicates!".format(v, k))
        else:
            List.append(v)


def init_args(args_dict):
    """Init argparse from dictionary."""
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.__dict__ = args_dict
    return args


def update_args(args, key, value):
    args_update = deepcopy(args)
    setattr(args_update, key, value)
    return args_update


def lcm(denominators):
    """Get least common multiplier"""
    return reduce(lambda a,b: a*b // gcd(a,b), denominators)


def get_triu_index(size, order):
    """Obtain generalized upper triangular tensor mask with specific order."""
    matrices = torch.meshgrid((torch.arange(size),) * order)
    for i in range(len(matrices) - 1):
        if i == 0:
            idx = matrices[i] <= matrices[i+1]
        else:
            idx = idx & (matrices[i] <= matrices[i+1])
    idx = idx[None, ...]
    return idx


def get_poly_basis_tensor(x, order):
    """Obtain generalized upper triangular tensor with specific order, using Einsum."""
    strings = "abcdefghijklmn"
    if isinstance(order, Iterable):
        x_list = []
        for order_ele in order:
            x_list.append(get_poly_basis_tensor(x, order_ele))
        return torch.cat(x_list, -1)
    elif order == 1:
        return x
    elif order == 0 or order > len(strings):
        raise Exception("Order must be an integer between 1 and {}".format(len(string)))
    batch_size, size = x.shape
    # Generate einsum string:
    einsum_str = ""
    for i in range(order):
        einsum_str += "z{},".format(strings[i])
    einsum_str = einsum_str[:-1] + "->z" + strings[:order]
    # Compute the outer product:
    tensor = torch.einsum(einsum_str, *((x,) * order))
    # Obtain generalized upper triangular tensor with specific order:
    idx = get_triu_index(size=size, order=order).expand(batch_size, *((size,)*order))
    out = tensor[idx].reshape(batch_size,-1)
    return out


def get_poly_basis(x, order):
    """Obtain generalized upper triangular tensor with specific order."""
    x_list = []
    size = x.shape[-1]
    if isinstance(order, Iterable):
        x_list = []
        for order_ele in order:
            x_list.append(get_poly_basis(x, order_ele))
        return torch.cat(x_list, -1)
    elif order == 1:
        return x
    elif order == 2:
        for i in range(size):
            for j in range(i, size):
                x_list.append(x[..., i] * x[..., j])
    elif order == 3:
        for i in range(size):
            for j in range(i, size):
                for k in range(j, size):
                    x_list.append(x[..., i] * x[..., j] * x[..., k])
    elif order == 4:
        for i in range(size):
            for j in range(i, size):
                for k in range(j, size):
                    for l in range(k, size):
                        x_list.append(x[..., i] * x[..., j] * x[..., k] * x[..., l])
    elif order == 5:
        for i in range(size):
            for j in range(i, size):
                for k in range(j, size):
                    for l in range(k, size):
                        for m in range(l, size):
                            x_list.append(x[..., i] * x[..., j] * x[..., k] * x[..., l] * x[..., m])
    else:
        raise Exception("Order can only be integers from 1 to 5.")
    x_list = torch.stack(x_list, -1)
    return x_list


class Dictionary(object):
    """Custom dictionary that can avoid the collate_fn in pytorch's dataloader."""
    def __init__(self, Dict=None):
        if Dict is not None:
            for k, v in Dict.items():
                self.__dict__[k] = v

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))

class Attr_Dict(dict):
    def __init__(self, *a, **kw):
        dict.__init__(self, *a, **kw)
        self.__dict__ = self

    def update(self, *a, **kw):
        dict.update(self, *a, **kw)
        return self

    def __getattribute__(self, key):
        if key in self:
            return self[key]
        else:
            return object.__getattribute__(self, key)

    def to(self, device):
        self["device"] = device
        Dict = to_device_recur(self, device)
        return Dict

    def copy(self, detach=True):
        return copy_data(self, detach=detach)

    def clone(self, detach=True):
        return self.copy(detach=detach)

    def type(self, dtype):
        return to_type(self, dtype)

    def detach(self):
        return detach_data(self)


def to_type(data, dtype):
    if isinstance(data, dict):
        dct = {key: to_type(value, dtype=dtype) for key, value in data.items()}
        if data.__class__.__name__ == "Attr_Dict":
            dct = Attr_Dict(dct)
        return dct
    elif isinstance(data, list):
        return [to_type(ele, dtype=dtype) for ele in data]
    elif isinstance(data, torch.Tensor):
        return data.type(dtype)
    elif isinstance(data, tuple):
        return tuple(to_type(ele, dtype=dtype) for ele in data)
    elif data.__class__.__name__ in ['HeteroGraph', 'Data']:
        dct = Attr_Dict({key: to_type(value, dtype=dtype) for key, value in vars(data).items()})
        assert len(dct) > 0, "Did not clone anything. Check that your PyG version is below 1.8, preferablly 1.7.1. Follow the the ./design/multiscale/README.md to install the correct version of PyG."
        return dct
    elif data is None:
        return data
    elif isinstance(data, np.ndarray):
        if dtype == torch.float64:
            return data.astype(np.float64)
        elif dtype == torch.float32:
            return data.astype(np.float32)
        else:
            raise
    else:
        return data.type(dtype)


def detach_data(data):
    """Copy Data instance, and detach from source Data."""
    if isinstance(data, dict):
        dct = {key: detach_data(value) for key, value in data.items()}
        if data.__class__.__name__ == "Attr_Dict":
            dct = Attr_Dict(dct)
        return dct
    elif isinstance(data, list):
        return [detach_data(ele) for ele in data]
    elif isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, tuple):
        return tuple(detach_data(ele) for ele in data)
    elif data.__class__.__name__ in ['HeteroGraph', 'Data']:
        dct = Attr_Dict({key: detach_data(value) for key, value in vars(data).items()})
        assert len(dct) > 0, "Did not clone anything. Check that your PyG version is below 1.8, preferablly 1.7.1. Follow the the ./design/multiscale/README.md to install the correct version of PyG."
        return dct
    else:
        return data

def copy_data(data, detach=True):
    """Copy Data instance, and detach from source Data."""
    if isinstance(data, dict):
        dct = {key: copy_data(value, detach=detach) for key, value in data.items()}
        if data.__class__.__name__ == "Attr_Dict":
            dct = Attr_Dict(dct)
        return dct
    elif isinstance(data, list):
        return [copy_data(ele, detach=detach) for ele in data]
    elif isinstance(data, torch.Tensor):
        if detach:
            return data.detach().clone()
        else:
            return data.clone()
    elif isinstance(data, tuple):
        return tuple(copy_data(ele, detach=detach) for ele in data)
    elif data.__class__.__name__ in ['HeteroGraph', 'Data']:
        dct = Attr_Dict({key: copy_data(value, detach=detach) for key, value in vars(data).items()})
        assert len(dct) > 0, "Did not clone anything. Check that your PyG version is below 1.8, preferablly 1.7.1. Follow the the ./design/multiscale/README.md to install the correct version of PyG."
        return dct
    else:
        return deepcopy(data)


class Cache_Dict(dict):
    def __init__(self, *args, cache_num_limit=-1, cache_fraction_limit=-1):
        """
        A dictionary that limits the number of keys and the total cache percentage used by all programs. It will pop the oldest keys when any limit is exceeded.

        Args:
            cache_num_limit: Number of keys. Default -1 means no limit.
            cache_fraction_limit: limit of total cache fraction used by all programs. Default -1 means no limit.
        """
        super().__init__(*args)
        self._cache_num_limit = cache_num_limit
        self._cache_fraction_limit = cache_fraction_limit

    def __setitem__(self, key, item):
        is_pop = False
        if self._cache_num_limit != -1 and len(self) >= self._cache_num_limit:
            is_pop = True
        if not is_pop:
            if self._cache_fraction_limit != -1:
                import psutil
                if psutil.virtual_memory()[2] / 100 > self._cache_fraction_limit:
                    is_pop = True
        if len(self) > 0 and is_pop:
            import gc
            first_key = next(iter(self))
            self.pop(first_key)
            gc.collect()
        self.__dict__[key] = item

    @property
    def core_dict(self):
        return {key: item for key, item in self.__dict__.items() if not isinstance(key, str) or not key.startswith("_")}

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.core_dict)

    def __len__(self):
        return len(self.core_dict)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.core_dict

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.core_dict.keys()

    def values(self):
        return self.core_dict.values()

    def items(self):
        return self.core_dict.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.core_dict

    def __iter__(self):
        return iter(self.core_dict)

    def __unicode__(self):
        return unicode(repr(self.core_dict))


class MineSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        chunk_size (int): Each chunk contains chunk_size of blocks randomly sampled, so that it will finish the chunk of blocks before going on to next chunk. Default -1 will ignore the chunks.
        min_block_size (int): minimum block size to keep. Inside the block the order is also randomly permuted.

        For example, for a dataset with size = 45, chunk_size = 5 and min_block_size of 2, it will
            each time take a chunk of 5 blocks, e.g. ([0,1], [11,10], [33,32], [14,15], [40,41]), permute it randomly
            for yielding, then go on to the next chunk of blocks.
        If min_block_size == 1, then it is fully random and has the same effect as chunk_size == -1.
    """
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        chunk_size: int = -1,
        min_block_size: int = 1,
    ) -> None:
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.min_block_size = min_block_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        if self.chunk_size == -1:
            yield from torch.randperm(n, generator=generator).tolist()
        else:
            """
            1. Build a list of blocks
            2. Permute the list of blocks
            3. Partition into chunks
            4. Permute fully inside chunk
            """
            n_ceil = int(np.ceil(n / self.min_block_size) * self.min_block_size)
            idx_blocks = np.arange(n_ceil).reshape(-1, self.min_block_size)
            block_perm = torch.randperm(len(idx_blocks), generator=generator)
            idx_blocks = idx_blocks[block_perm]
            n_chunks = int(np.ceil(len(idx_blocks) / self.chunk_size))
            all_list = []
            for i in range(n_chunks):
                idx_chunk = idx_blocks[i*self.chunk_size: (i+1)*self.chunk_size].reshape(-1)
                chunk_perm = torch.randperm(len(idx_chunk), generator=generator).numpy()
                idx_chunk_permute = idx_chunk[chunk_perm]
                all_list.append(idx_chunk_permute[idx_chunk_permute < n])
            yield from np.concatenate(all_list).tolist()

    def __len__(self) -> int:
        return self.num_samples


class MineDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Sized,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        chunk_size: int = -1,
        min_block_size: int = 1,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.chunk_size = chunk_size
        self.min_block_size = min_block_size
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.chunk_size == -1:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            """
            1. Build a list of blocks
            2. Permute the list of blocks
            3. Partition into chunks
            4. Permute fully inside chunk
            """
            n = len(self.dataset)
            n_ceil = int(np.ceil(n / self.min_block_size) * self.min_block_size)
            idx_blocks = np.arange(n_ceil).reshape(-1, self.min_block_size)
            block_perm = torch.randperm(len(idx_blocks), generator=generator)
            idx_blocks = idx_blocks[block_perm]
            n_chunks = int(np.ceil(len(idx_blocks) / self.chunk_size))
            all_list = []
            for i in range(n_chunks):
                idx_chunk = idx_blocks[i*self.chunk_size: (i+1)*self.chunk_size].reshape(-1)
                chunk_perm = torch.randperm(len(idx_chunk), generator=generator).numpy()
                idx_chunk_permute = idx_chunk[chunk_perm]
                all_list.append(idx_chunk_permute[idx_chunk_permute < n])
            indices = np.concatenate(all_list).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class My_Tuple(tuple):
    def to(self, device):
        self[0].to(device)
        return self
    
    def __getattribute__(self, key):
        if hasattr(My_Tuple, key):
            return object.__getattribute__(self, key)
        else: 
            return self[0].__getattribute__(key)

class My_Freeze_Tuple(tuple):
    pass

class MineDataParallel(nn.parallel.DataParallel):
    def __getattribute__(self, key):
        module_attrs = [
            'training',
        ]
        if key in module_attrs:
            return object.__getattribute__(self.module, key)
        else:
            if hasattr(MineDataParallel, key):
                return object.__getattribute__(self, key)
            else:
                return super().__getattribute__(key)


class Batch(object):
    def __init__(self, is_absorb_batch=False, is_collate_tuple=False):
        """
        
        Args:
            is_collate_tuple: if True, will collate inside the tuple.
        """
        self.is_absorb_batch = is_absorb_batch
        self.is_collate_tuple = is_collate_tuple

    def collate(self):
        import re
        if torch.__version__.startswith("1.9") or torch.__version__.startswith("1.10") or torch.__version__.startswith("1.11") or torch.__version__.startswith("1.12"):
            from torch._six import string_classes
            from collections import abc as container_abcs
        else:
            from torch._six import container_abcs, string_classes, int_classes
        from pstar import pdict, plist
        default_collate_err_msg_format = (
            "collate_fn: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        np_str_obj_array_pattern = re.compile(r'[SaUO]')
        def default_convert(data):
            r"""Converts each NumPy array data field into a tensor"""
            elem_type = type(data)
            if isinstance(data, torch.Tensor):
                return data
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                # array of string classes and object
                if elem_type.__name__ == 'ndarray' \
                        and np_str_obj_array_pattern.search(data.dtype.str) is not None:
                    return data
                return torch.as_tensor(data)
            elif isinstance(data, container_abcs.Mapping):
                return {key: default_convert(data[key]) for key in data}
            elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
                return elem_type(*(default_convert(d) for d in data))
            elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
                return [default_convert(d) for d in data]
            else:
                return data

        def collate_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                if self.is_absorb_batch:
                    tensor = torch.cat(batch, 0, out=out)
                else:
                    tensor = torch.stack(batch, 0, out=out)
                return tensor
            elif elem is None:
                return None
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
                if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                    return collate_fn([torch.as_tensor(b) for b in batch])
                elif elem.shape == ():  # scalars
                    return torch.as_tensor(batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(elem, int):
#             elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, My_Freeze_Tuple):
                return batch
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple:
                return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
            elif isinstance(elem, My_Tuple):
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return elem.__class__([collate_fn(samples) for samples in transposed])
            elif isinstance(elem, tuple) and not self.is_collate_tuple:
                return batch
            elif isinstance(elem, container_abcs.Sequence):
                # check to make sure that the elements in batch have consistent size
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return  [collate_fn(samples) for samples in transposed]
            elif elem.__class__.__name__ == 'Dictionary':
                return batch
            elif elem.__class__.__name__ == 'DGLHeteroGraph':
                import dgl
                return dgl.batch(batch)
            raise TypeError(default_collate_err_msg_format.format(elem_type))
        return collate_fn


def ddeepcopy(item):
    """Deepcopy with certain custom classes."""
    from pstar import pdict
    if isinstance(item, pdict) or isinstance(item, Attr_Dict):
        return item.copy()
    else:
        return deepcopy(item)

def get_pdict():
    """Obtain pdict with additional functionalities."""
    from pstar import pdict
    class Pdict(pdict):
        def to(self, device):
            self["device"] = device
            return to_device_recur(self, device)

        def copy(self):
            return Pdict(dict.copy(self))
    return Pdict


def to_device_recur(iterable, device, is_detach=False):
    if isinstance(iterable, list):
        return [to_device_recur(item, device, is_detach=is_detach) for item in iterable]
    elif isinstance(iterable, tuple):
        return tuple(to_device_recur(item, device, is_detach=is_detach) for item in iterable)
    elif isinstance(iterable, dict):
        Dict = {key: to_device_recur(item, device, is_detach=is_detach) for key, item in iterable.items()}
        if iterable.__class__.__name__ == "Pdict":
            from pstar import pdict
            class Pdict(pdict):
                def to(self, device):
                    self["device"] = device
                    return to_device_recur(self, device)

                def copy(self):
                    return Pdict(dict.copy(self))
            Dict = Pdict(Dict)
        elif iterable.__class__.__name__ == "Attr_Dict":
            Dict = Attr_Dict(Dict)
        return Dict
    elif hasattr(iterable, "to"):
        iterable = iterable.to(device)
        if is_detach:
            iterable = iterable.detach()
        return iterable
    else:
        if hasattr(iterable, "detach"):
            iterable = iterable.detach()
        return iterable


def get_boundary_locations(size, sector_size, stride):
    """Get a list of 1D sector boundary positions.
    Args:
        size: length of the full domain.
        sector_size: length of the sector.
        stride: how far each sector moves to the right
    Returns:
        boundaries: a list of 1D sector boundary positions
    """
    boundaries = []
    sector_l, sector_r = 0, sector_size  # left and right pos of the sector
    while sector_l < size:
        if sector_l < size and sector_r > size:
            boundaries.append((size - sector_size, size))
            break
        else:
            boundaries.append((sector_l, sector_r))
            if (sector_l, sector_r) == (size - sector_size, size):
                break
        sector_l += stride
        sector_r += stride
    return boundaries


def set_seed(seed):
    """Set up seed."""
    if seed == -1:
        seed = None
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)


def reshape_weight_to_matrix(weight, dim=0):
    if dim != 0:
        # permute dim to front
        weight = weight.permute(dim, *[d for d in range(weight.dim()) if d != dim])
    height = weight.size(0)
    weight = weight.reshape(height, -1)
    return weight


def slice_divmod(slice_item, denom):
    """Given a slice(start, stop, step) and a denominator, return
    a single quotient and a remainder slice.
    """
    if isinstance(slice_item, Number):
        return divmod(slice_item, denom)
    start, stop, step = slice_item.start, slice_item.stop, slice_item.step
    if start is None:
        start = 0
    if step is None:
        step = 1
    idx = np.arange(start, stop, step)
    sim_id = idx // denom
    sim_id_set = np.unique(sim_id)
    assert len(sim_id_set) == 1, "The number of returned sim_id must be 1. Here the sim_id are: {}".format(sim_id_set)
    sim_id = sim_id_set[0]
    start_new = start - sim_id * denom
    stop_new = stop - sim_id * denom
    slice_item_new = slice(start_new, stop_new, step)
    return sim_id, slice_item_new


def slice_add(slice_item, num):
    """Perform addition of a slice by a scaler."""
    if isinstance(slice_item, Number):
        return slice_item + num
    start, stop, step = slice_item.start, slice_item.stop, slice_item.step
    if start is None:
        start = 0
    if step is None:
        step = 1
    return slice(start+num, stop+num, step)


def slice_mul(slice_item, num):
    """Perform multiplication of a slice by a scaler."""
    if isinstance(slice_item, Number):
        return slice_item * num
    start, stop, step = slice_item.start, slice_item.stop, slice_item.step
    if start is None:
        start = 0
    if step is None:
        step = 1
    return slice(start*num, stop*num, step*num)


def tuple_mul(tuple_item, num):
    """Perform element-wise multiplication by a scaler for a tuple or a number."""
    if not isinstance(tuple_item, tuple):
        return tuple_item * num
    List = []
    for item in tuple_item:
        if item is not None:
            List.append(item * num)
        else:
            List.append(None)
    return tuple(List)


def tuple_divide(tuple_item, num):
    """Perform element-wise division by a scaler for a tuple or a number."""
    return tuple_mul(tuple_item, 1/num)


def tuple_add(*tuples):
    """Perform element-wise addition of multiple tuples or numbers."""
    # Number addition:
    if not isinstance(tuples[0], tuple):
        for item in tuples:
            assert not isinstance(item, tuple)
        return sum(tuples)

    # Check if all the tuples have the same length:
    length = len(tuples[0])
    for tuple_item in tuples:
        assert len(tuple_item) == length, "All the tuples must have the same length."

    # Perform element-wise addition:
    List = []
    for k in range(length):
        # k is the k'th element in each tuple:
        is_None = False  # Whether the k'th element is None for all tuples
        item_sum = 0
        for tuple_item in tuples:
            # for each tuple:
            if tuple_item[k] is None or is_None:
                assert tuple_item[k] is None, "Either all {}th elements are None or are not None".format(k)
                is_None = True
            else:
                item_sum = item_sum + tuple_item[k]
        if is_None:
            List.append(None)
        else:
            List.append(item_sum)
    return tuple(List)


def tuple_subtract(tuple1, tuple2):
    """Perform element-wise subtraction of two tuples or numbers."""
    return tuple_add(tuple1, tuple_mul(tuple2, -1))


def tuple_shape_length_equal(tuple1, tuple2):
    """Check if the number of dimensions for each element of tuples or numbers are the same."""
    if not isinstance(tuple1, tuple):
        assert not isinstance(tuple2, tuple)
        return len(tuple1.shape) == len(tuple2.shape)
    else:
        is_equal = True
        for tuple_item1, tuple_item2 in zip(tuple1, tuple2):
            if tuple_item1 is None:
                assert tuple_item2 is None
            else:
                if len(tuple_item1.shape) != len(tuple_item2.shape):
                    is_equal = False
                    break
        return is_equal


def forward_Runge_Kutta(model, x, mode="RK4"):
    """Perform forward prediction using Runge-Kutta scheme."""
    if mode.startswith("RK"):
        k1 = model.forward_core(x)
        assert tuple_shape_length_equal(k1, x), "the number of dimensions for the output of model and input must be the same!"
        if mode == "RK2":
            k2 = model.forward_core(tuple_add(x, tuple_divide(k1, 2)))
            x = tuple_add(x, k2)
        elif mode == "RK3":
            k2 = model.forward_core(tuple_add(x, tuple_divide(k1, 2)))
            k3 = model.forward_core(tuple_add(x, tuple_mul(k1, -1), tuple_mul(k2, 2)))
            x = tuple_add(x, tuple_divide(tuple_add(k1, tuple_mul(k2, 4), k3), 6))
        elif mode == "RK4":
            k2 = model.forward_core(tuple_add(x, tuple_divide(k1, 2)))
            k3 = model.forward_core(tuple_add(x, tuple_divide(k2, 2)))
            k4 = model.forward_core(tuple_add(x, k3))
            x = tuple_add(x, tuple_divide(tuple_add(k1, tuple_mul(k2, 2), tuple_mul(k3, 2), k4), 6))
        else:
            raise Exception("mode '{}' is not supported!".format(mode))
    else:
        raise Exception("mode '{}' is not supported!".format(mode))
    return x


def get_machine_name():
    return os.uname()[1].split('.')[0]


def is_diagnose(loc, filename, diagnose_filename="/experiments/diagnose.yml"):
    """If the given loc and filename matches that of the diagose.yml, will return True and (later) call an pde.set_trace()."""
    try:
        with open(get_root_dir() + diagnose_filename, "r") as f:
            Dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        return False
    Dict.pop(None, None)
    if not ("loc" in Dict and "dirname" in Dict and "filename" in Dict):
        return False
    if loc == Dict["loc"] and filename == Dict["dirname"] + Dict["filename"]:
        return True
    else:
        return False


def get_device(args):
    """Initialize PyTorch device.

    Args:
        args.gpuid choose from an integer or True or False.
    """
    is_cuda = eval(args.gpuid)
    if not isinstance(is_cuda, bool):
        is_cuda = "cuda:{}".format(is_cuda)
    device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")
    return device


def get_elements(src, string_idx, dim=0):
    """Get slices of elements from tensor src.

    Args:
        string_idx: has the same convention as index-select with Python List. E.g.
            '4'  -> src[4]
            ':3' -> src[1, 2, 3]
            '2:' -> src[2, 3, ... max_t]
            '2:4', dim=1 -> src[:, [2,3,4]]
            '1:4+6:8', dim=1 -> src[:, [1,2,3,6,7]]
    """
    if "+" in string_idx:
        # If there is multiple slices concatenated by "+":
        string_idx_split = string_idx.split("+")
        List = []
        for idx in string_idx_split:
            List.append(get_elements(src, idx, dim=dim))
        List = torch.cat(List, dim=dim)
        return List

    if string_idx == "":
        return src
    elif ":" not in string_idx:
        idx = eval(string_idx)
        if dim == 0:
            if idx < 0:
                idx += len(src)
            return src[idx:idx+1]
        elif dim == 1:
            if idx < 0:
                idx += src.shape[1]
            return src[:, idx:idx+1]
        else:
            raise
    else:
        if string_idx.startswith(":"):
            if dim == 0:
                return src[:eval(string_idx[1:])]
            elif dim == 1:
                return src[:, :eval(string_idx[1:])]
            else:
                raise
        elif string_idx.endswith(":"):
            if dim == 0:
                return src[eval(string_idx[:-1]):]
            elif dim == 1:
                return src[:, eval(string_idx[:-1]):]
            else:
                raise
        else:
            string_idx_split = string_idx.split(":")
            assert len(string_idx_split) == 2
            if dim == 0:
                return src[eval(string_idx_split[0]): eval(string_idx_split[1])]
            elif dim == 1:
                return src[:, eval(string_idx_split[0]): eval(string_idx_split[1])]
            else:
                raise


def build_optimizer(args, params):
    """
    Build optimizer and scheduler.

    Required argparse:
        parser.add_argument('--opt', type=str,
                            help='Optimizer such as adam, sgd, rmsprop or adagrad.')
        parser.add_argument('--lr', type=float,
                            help='Learning rate.')
        parser.add_argument('--lr_scheduler_type', type=str,
                            help='type of the lr-scheduler. Choose from "rop", "cos", "cos-re" and "None".')
        parser.add_argument('--lr_scheduler_T0', type=int, default=50,
                            help='T0 for CosineAnnealingWarmRestarts (cos-re) scheduler')
        parser.add_argument('--lr_scheduler_T_mult', type=int, default=1,
                            help='Multiplication factor for increasing T_i after a restart, for CosineAnnealingWarmRestarts (cos-re) scheduler.')
        parser.add_argument('--lr_scheduler_factor', type=float, default=0.1,
                            help='Multiplication factor for ReduceOnPlateau lr-scheduler.')
        parser.add_argument('--weight_decay', type=float,
                            help='Weight decay.')
    """
    from torch.optim import lr_scheduler
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)

    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.lr_scheduler_type == "rop":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_scheduler_factor, verbose=True)
    elif args.lr_scheduler_type == "cos":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler_type == "cos-re":
        epochs_new = get_epochs_T_mult(epochs=args.epochs, T_0=args.lr_scheduler_T0, T_mult=args.lr_scheduler_T_mult)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_scheduler_T0, T_mult=args.lr_scheduler_T_mult)
        args.epochs = epochs_new
        print("Reset epochs to {} to locate on the cumulative of geometric series.".format(args.epochs))
    elif args.lr_scheduler_type == "None":
        scheduler = None
    else:
        raise
    return optimizer, scheduler


def get_epochs_T_mult(epochs, T_0, T_mult):
    """Select the maximum number within [T_0, epochs] that is T_0*(1+T_mul+T_mul**2+ ...)."""
    assert T_mult >= 1, "T_mult must be greater than or equal to 1!"
    if T_mult == 1:
        return epochs // T_0 * T_0
    T_exponent = int(np.floor(np.log(epochs//T_0 * (T_mult-1) + 1) / np.log(T_mult)))
    T_factor = (T_mult ** T_exponent - 1) // (T_mult - 1)
    epochs = T_0 * T_factor
    return epochs


def get_cosine_decay(value_start, value_end, n_iter):
    array = value_end + (1 + torch.cos(torch.arange(n_iter+1)/n_iter * torch.pi)) * (value_start - value_end) / 2
    return array


def get_decay_list(start_value, end_value, steps, mode="linear"):
    if mode == "linear":
        beta_list = np.linspace(start_value, end_value, steps)
    elif mode == "exp":
        beta_list = (np.logspace(1, 0, steps) - 1) / 9 * (start_value - end_value) + end_value
    elif mode == "square":
        beta_list = np.linspace(1, 0, steps) ** 2 * (start_value - end_value) + end_value
    elif mode == "cos":
        beta_list = get_cosine_decay(start_value, end_value, steps)
    else:
        raise
    return beta_list


def get_string_slice(List):
    """Given a list of integers, return a string representation of the slices.
    E.g. List = [1,2,2,4,7,8,9,11]  => string = '1:3+4:5+7:10+11:12'
    """
    List = sorted(List)
    string = ""
    start_id = None
    end_id = None
    for ele in List:
        if start_id is None:
            start_id = ele
            end_id = start_id + 1
        elif ele == end_id:
            end_id += 1
        elif ele > end_id:
            # there is a discontinuity
            string += "+{}:{}".format(start_id, end_id)
            start_id = ele
            end_id = ele + 1
    string += "+{}:{}".format(start_id, end_id)
    return string[1:]


def get_softplus_offset(offset):
    """Get offsetted softplus to reduce initial amplitude."""
    def softplus_offset(x):
        return F.softplus(x-offset, beta=1)
    return softplus_offset


def get_inverse_softplus_offset(offset):
    """Get inverse offsetted softplus."""
    def inverse_softplus_offset(x):
        return torch.log(x.exp() - 1) + offset
    return inverse_softplus_offset


class MineDataset(Dataset):
    def __init__(
        self,
        data=None,
        idx_list=None,
        transform=None,
    ):
        """User defined dataset that can be used for PyTorch DataLoader"""
        self.data = data
        self.transform = transform
        if idx_list is None:
            self.idx_list = torch.arange(len(self.data))
        else:
            self.idx_list = idx_list

    def __len__(self):
        return len(self.idx_list)

    def process_sample(self, sample):
        return sample

    def __getitem__(self, idx):
        is_list = True
        if isinstance(idx, torch.Tensor):
            if len(idx.shape) == 0 or (len(idx.shape) == 1 and len(idx) == 1):
                idx = idx.item()
                is_list = False
        elif isinstance(idx, list):
            pass
        elif isinstance(idx, Number):
            is_list = False

        if isinstance(idx, slice) or is_list:
            Dict = self.__dict__.copy()
            Dict["idx_list"] = self.idx_list[idx]
            return self.__class__(**Dict)

        sample = self.process_sample(self.data[self.idx_list[idx]])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"({len(self)})"


def clip_grad(optimizer):
    """Clip gradient."""
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def gather_broadcast(tensor, dim, index):
    """
    Given tensor, gather the index along the dimension dim.
        For example, if tensor has shape [3,4,5,8,9], dim=2, then index must have
        the shape of [3,4], whose value is inside range(5), and returns a tensor_gather
        of size [3,4,8,9].
    """
    dim_size = tensor.shape[dim]
    assert len(index.shape) == dim and tensor.shape[:dim] == index.shape and index.max() < dim_size
    assert dim >= 1, "dim must >= 1!"
    index_onehot = torch.eye(dim_size, dim_size)[index].bool()
    tensor_gathered = tensor[index_onehot].reshape(*index.shape, *tensor.shape[dim+1:])
    return tensor_gathered


def show_warning():
    import warnings
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        import traceback
        import warnings
        import sys
        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))
    warnings.showwarning = warn_with_traceback
    return warnings


def copy_with_model_dict(model, other_attr=None):
    """Copy a model based on its model_dict."""
    if other_attr is None:
        other_attr = []
    kwargs = model.model_dict
    state_dict = kwargs.pop("state_dict")
    assert kwargs.pop("type") == model.__class__.__name__
    other_attr_dict = {}
    for key in other_attr:
        other_attr_dict[key] = kwargs.pop(key)
    new_model = model.__class__(**kwargs)
    for key, value in other_attr_dict.items():
        if isinstance(value, np.ndarray):
            value = torch.FloatTensor(value)
        setattr(new_model, key, value)
    new_model.load_state_dict(state_dict)
    assert new_model.model_dict.keys() == model.model_dict.keys()
    return new_model


class TopKList(list):
    """A list that stores the top K dictionaries that has the lowest/highest {sort_key} values."""
    def __init__(self, K, sort_key, duplicate_key, mode="max"):
        """
        Args:
            K: top K elements will be saved in the list.
            sort_key: the key to use for the top-K ranking.
            duplicate_key: the key to check, and if there are already elements that has the same value to the duplicate, do not append.
            mode: choose from "max" (the larger the better) and "min" (the smaller the better).
        """
        self.K = K
        self.sort_key = sort_key
        self.duplicate_key = duplicate_key
        self.mode = mode

    def append(self, item):
        """Insert an item into the list, if the length is less then self.K, simply insert.
        Otherwise if it is better than the worst element in terms of sort_key, replace it.

        Returns:
            is_update: whether the TopKList is updated.
        """
        assert isinstance(item, dict), "item must be a dictionary!"
        assert self.sort_key in item, "item must have sort_key of '{}'!".format(self.sort_key)
        is_update = False

        # If there are already elements that has the same value to the duplicate, do not append:
        is_duplicate = False
        for element in self:
            if element[self.duplicate_key] == item[self.duplicate_key]:
                is_duplicate = True
                break
        if is_duplicate:
            return is_update

        # Append if still space or the value for self.sort_key is better than the worst:
        if len(self) < self.K:
            super().append(item)
            is_update = True
        elif len(self) == self.K:
            sort_value = np.array([to_np_array(ele[self.sort_key]) for ele in self])
            if self.mode == "max":
                argmin = sort_value.argmin()
                if sort_value[argmin] < item[self.sort_key]:
                    self.pop(argmin)
                    super().append(item)
                    is_update = True
            elif self.mode == "min":
                argmax = sort_value.argmax()
                if sort_value[argmax] > item[self.sort_key]:
                    self.pop(argmax)
                    super().append(item)
                    is_update = True
            else:
                raise Exception("mode must be either 'min' or 'max'.")
        else:
            raise Exception("Cannot exceed K={} items".format(self.K))
        return is_update

    def get_items(self, key):
        """Obtain the item corresponding to the key for each element."""
        return [item[key] for item in self]

    def is_available(self):
        """Return True if the number of elements is less than self.K."""
        return len(self) < self.K


def pdump(file, filename):
    """Dump a file via pickle."""
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def pload(filename):
    """Load a filename saved as pickle."""
    with open(filename, "rb") as f:
        file = pickle.load(f)
    return file


def cmd_to_args_dict(cmd, str_args=["gpuid"]):
    """Transform a command into a dictionary for args.

    Args:
        cmd: e.g. python script.py --exp_id="exp1.0" --date_time="1-1" --gpuid="0"
        str_args: args that should keep the str format.

    Returns:
        Dict: a dictionary where the keys are the args keys and the values are the values corresponding to the keys.
    """
    from numbers import Number
    Dict = {}
    for string_item in cmd.split(" "):
        if string_item.startswith("--"):
            key, value = string_item[2:].split("=")
            try:
                value_candidate = eval(value)
                if (isinstance(value_candidate, Number) or isinstance(value_candidate, bool)) and key not in str_keys:
                    value = value_candidate
            except:
                pass
            Dict[key] = value
    return Dict


def try_call(fun, args=None, kwargs=None, time_interval=5, max_n_trials=20, max_exp_time=None):
    """Try executing some function fun with *args and **kwargs for {max_n_trials} number of times
        each separate by time interval of {time_interval} seconds.
    """
    if args is None:
        args = []
    if not isinstance(args, list):
        args = [args]
    if kwargs is None:
        kwargs = {}
    if max_exp_time is None:
        time_interval_list = [time_interval] * max_n_trials
    else:
        time_interval_list = [2 ** k for k in range(20) if 2 ** (k + 1) <= max_exp_time]
    for i, time_interval in enumerate(time_interval_list):
        is_succeed = False
        try:
            output = fun(*args, **kwargs)
            is_succeed = True
        except Exception as e:
            error = str(e)
        if is_succeed:
            break
        else:
            print("Fail to execute function {} for the {}th time, with error: {}".format(fun, i+1, error))
        time.sleep(time_interval)
    if not is_succeed:
        raise Exception("Fail to execute function {} for the {}th time, same as the max_n_trials of {}. Check error!".format(fun, i+1, max_n_trials))
    return output


def get_instance_keys(class_instance):
    """Get the instance keys of a class"""
    return [key for key in vars(class_instance) if key[:1] != "_"]


class Model_Wrapper(nn.Module):
    """Wrapping a nn.Module inside the class."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def set_c(self, c_repr):
        self.c_repr = c_repr
        return self

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def __getattribute__(self, item):
        """Obtain the attributes. Prioritize the instance attributes in self.model."""
        if item == "model":
            return object.__getattribute__(self, "model")
        elif item.startswith("_"):
            return object.__getattribute__(self, item)
        elif item in get_instance_keys(self.model):
            return getattr(self.model, item)
        else:
            return object.__getattribute__(self, item)

    @property
    def model_dict(self):
        return self.model.model_dict


def fill_matrix_with_triu(array, size):
    """array: [B, size*(size+1)/2]

    tensor: a symmetric matrix of [B, size, size]
    """
    assert len(array.shape) == 2
    rows, cols = torch.triu_indices(size, size)
    device = array.device
    assert len(rows) == array.shape[-1]
    tensor_triu = torch.zeros(array.shape[0], size, size).type(array.dtype).to(device)
    tensor_triu[:,rows,cols] = array
    tensor = tensor_triu + torch.triu(tensor_triu, diagonal=1).transpose(1,2)
    return tensor


def remove_elements(List, elements):
    """Remove elements in the List if they exist in the List, and return the new list."""
    NewList = deepcopy(List)
    for element in elements:
        if element in NewList:
            NewList.remove(element)
    return NewList


def get_soft_IoU(mask1, mask2, dim, epsilon=1):
    """Get soft IoU score for two masks.

    Args:
        mask1, mask2: two masks with the same shape and value between [0, 1]
        dim: dimensions over which to aggregate.
    """
    if isinstance(mask1, np.ndarray):
        soft_IoU = (mask1 * mask2).sum(dim) / (mask1 + mask2 - mask1 * mask2).sum(dim).clip(epsilon, None)
    else:
        soft_IoU = (mask1 * mask2).sum(dim) / (mask1 + mask2 - mask1 * mask2).sum(dim).clamp(epsilon)
    return soft_IoU


def get_soft_Jaccard_distance(mask1, mask2, dim, epsilon=1):
    """Get soft Jaccard distance for two masks."""
    return 1 - get_soft_IoU(mask1, mask2, dim=dim, epsilon=epsilon)


def get_triu_ids(array, is_triu=True):
    if isinstance(array, Number):
        array = np.arange(array)
    rows_matrix, col_matrix = np.meshgrid(array, array)
    matrix_cat = np.stack([rows_matrix, col_matrix], -1)
    rr, cc = np.triu_indices(len(matrix_cat), k=1)
    rows, cols = matrix_cat[cc, rr].T
    return rows, cols


def get_nx_graph(graph, isplot=False):
    import networkx as nx
    g = nx.DiGraph()
    graph_dict = dict([ele[:2] for ele in graph])
    for item in graph:
        if isinstance(item[0], Number) or isinstance(item[0], str):
            g.add_node("{}:{}".format(item[0], item[1]), type=item[1], E=item[2] if len(item) > 2 else None)
        elif isinstance(item[0], tuple):
            src, dst = item[0]
            g.add_edge(
                "{}:{}".format(src, graph_dict[src]),
                "{}:{}".format(dst, graph_dict[dst]),
                type=item[1],
                E=item[2] if len(item) > 2 else None,
            )
    if isplot:
        draw_nx_graph(g)
    return g


def draw_nx_graph(g):
    import networkx as nx
    import matplotlib.pylab as plt
    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos, with_labels=True, edge_color="#E115DA")
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels={(edge_src, edge_dst): item["type"] for edge_src, edge_dst, item in g.edges(data=True)} if "type" in list(g.edges(data=True))[0][2] else None,
        font_color='red',
    )
    plt.show()


def to_nx_graph(item, to_undirected=False):
    """Recursively transform an item or iterable into nx Graph."""
    if isinstance(item, MineList):
        g = get_nx_graph(item)
        if to_undirected:
            g = g.to_undirected(reciprocal=False)
        return g
    elif isinstance(item, list):
        return [to_nx_graph(ele, to_undirected=to_undirected) for ele in item]
    elif isinstance(item, dict):
        return {key: to_nx_graph(value, to_undirected=to_undirected) for key, value in item.items()}
    elif isinstance(item, tuple):
        return tuple(to_nx_graph(ele, to_undirected=to_undirected) for ele in item)
    else:
        raise


def func_recursive(item, func, atomic_class=None, *args, **kwargs):
    """Recursively apply a function to an item (iterable)"""
    if atomic_class is not None and isinstance(item, atomic_class):
        return func(item)
    elif isinstance(item, list):
        return [func_recursive(ele, func=func, atomic_class=atomic_class, *args, **kwargs) for ele in item]
    elif isinstance(item, dict):
        return {key: func_recursive(value, func=func, atomic_class=atomic_class, *args, **kwargs) for key, value in item.items()}
    elif isinstance(item, tuple):
        return tuple(func_recursive(ele, func=func, atomic_class=atomic_class, *args, **kwargs) for ele in item)
    else:
        raise


def nx_to_graph(g):
    """Transform nx graph into graph list format."""
    from networkx import DiGraph
    graph = MineList()
    for node in g.nodes:
        node, node_type = node.split(":")
        graph.append((int(node), node_type))
    for node1, node2, edge_info in g.edges(data=True):
        graph.append(((int(node1.split(":")[0]), int(node2.split(":")[0])), edge_info["type"]))
        if not isinstance(g, DiGraph):
            graph.append(((int(node2.split(":")[0]), int(node1.split(":")[0])), edge_info["type"]))
    return graph


def to_line_graph(graph):
    """
    Transform a graph into line_graph.
    
    Example:
        graph: 
            [(0, 'line'),
             (1, 'line'),
             (2, 'line'),
             ((0, 1), 're1'),
             ((1, 2), 're2'),
            ]
        line_graph: 
            [('0,1', 're1'),
             ('1,2', 're2'),
             (('0,1', '1,2'), 'line'),
             (('1,2', '0,1'), 'line'),
            ]
    """
    edges = [item for item in graph if not isinstance(item[0], Number)]
    line_graph = MineList([])
    for edge in edges:
        edge_node = f"{edge[0][0]},{edge[0][1]}"
        line_graph.append((edge_node, edge[1]))
    for edge1 in edges:
        for edge2 in edges:
            if edge1 != edge2:
                edge1_nodes = set(edge1[0])
                edge2_nodes = set(edge2[0])
                common_nodes = list(edge1_nodes.intersection(edge2_nodes))
                if len(common_nodes) > 0:
                    assert len(common_nodes) == 1
                    common_node = common_nodes[0]
                    line_graph.append(((f"{edge1[0][0]},{edge1[0][1]}", f"{edge2[0][0]},{edge2[0][1]}"), dict(graph)[common_node]))
    return line_graph


def get_all_graphs(graph):
    """
    Args:
        graph, e.g. 
            [(0, ('green', 'cube', 'small')),
             (1, ('red', 'cube', 'small')),
             (2, ('green', 'cube', 'small')),
             ((0, 1), ('SameShape', 'SameSize')),
             ((0, 2), ('SameColor', 'SameShape', 'SameSize')),
             ((1, 2), ('SameShape', 'SameSize'))]
    
    Returns: 
        graphs_all, e.g.:
            [[(0, ('green', 'cube', 'small')),
              (1, ('red', 'cube', 'small')),
              (2, ('green', 'cube', 'small')),
              ((0, 1), 'SameShape'),
              ((0, 2), 'SameColor'),
              ((1, 2), 'SameShape')],
             [(0, ('green', 'cube', 'small')),
              (1, ('red', 'cube', 'small')),
              (2, ('green', 'cube', 'small')),
              ((0, 1), 'SameShape'),
              ((0, 2), 'SameColor'),
              ((1, 2), 'SameSize')],
              
             ...
             
             [(0, ('green', 'cube', 'small')),
              (1, ('red', 'cube', 'small')),
              (2, ('green', 'cube', 'small')),
              ((0, 1), 'SameSize'),
              ((0, 2), 'SameSize'),
              ((1, 2), 'SameSize')]]
    """
    nodes = [ele for ele in graph if not (isinstance(ele[0], tuple) or isinstance(ele[0], list))]
    edges = [ele for ele in graph if isinstance(ele[0], tuple) or isinstance(ele[0], list)]
    edges_dict = dict(edges)
    id_dict = {key: list(range(len(item))) for key, item in edges_dict.items()}
    id_values = list(id_dict.values())
    id_keys = list(id_dict)
    all_combinations = list(itertools.product(*id_values))
    graphs_all = []
    for combination in all_combinations:
        edges_entity = []
        for i, item in enumerate(combination):
            key = id_keys[i]
            value = edges_dict[key][item]
            edges_entity.append((key, value))
        graphs_all.append(nodes + edges_entity)
    return graphs_all


def get_matching_mapping(graph, subgraph_cand):
    """
    Args:
        graph, e.g.:
            [(0, ('Green', 'Cube', 'Small')),
             (1, ('Blue', 'Cube', 'Small')),
             (2, ('Red', 'Cube', 'Small')),
             ((0, 1), 'SameShape'),
             ((0, 2), 'SameColor'),
             ((1, 2), 'SameShape')]
        subgraph_cand, e.g.
            [(0, 'Red'),
             (1, ""),
             (2, ""),
             ((0, 1), 'SameColor'),
             ((1, 2), 'SameShape')]

    Returns:
        node_reverse_mappings_final, e.g. {1: 0, 0: 2, 2: 1} (from subgraph to graph)
    """
    def get_common_diff_nodes(edge):
        line1 = eval(edge[0].split(":")[0])
        line2 = eval(edge[1].split(":")[0])
        common_node = list(set(line1).intersection(set(line2)))[0]
        diff1 = list(set(line1).difference({common_node}))[0]
        diff2 = list(set(line2).difference({common_node}))[0]
        return common_node, (diff1, diff2)
    from networkx.algorithms import isomorphism
    graph_linegraph = get_nx_graph(to_line_graph(graph))
    subgraph_linegraph = get_nx_graph(to_line_graph(subgraph_cand))
    DiGM = isomorphism.DiGraphMatcher(graph_linegraph, subgraph_linegraph)
    valid_linegraph_mappings = []
    for match in DiGM.subgraph_isomorphisms_iter():
        """
        edge_match: {graph_node: subgraph_cand_node}, e.g.
                    {'0,1:SameShape': '0,1:SameColor', '0,2:SameColor': '1,2:SameShape'}
        """
        is_valid = True
        for graph_node, subgraph_node in match.items():
            if subgraph_node.split(":")[1] != graph_node.split(":")[1]:
                is_valid = False
                break
        reverse_match = {value: key for key, value in match.items()}

        if is_valid:
            for edge in subgraph_linegraph.edges:
                """
                edge, e.g. ('0,1:SameColor', '1,2:SameShape')
                """
                subgraph_edge_type = subgraph_linegraph.edges[edge]["type"]
                graph_edge_type = graph_linegraph.edges[(reverse_match[edge[0]], reverse_match[edge[1]])]["type"]
                if subgraph_edge_type != "" and subgraph_edge_type not in graph_edge_type:
                    is_valid = False
                    break
        if is_valid:
            valid_linegraph_mappings.append(match)
    node_reverse_mappings = []
    if len(valid_linegraph_mappings) > 0:
        for valid_linegraph_mapping in valid_linegraph_mappings:
            node_reverse_mapping = []
            reverse_match = {value: key for key, value in valid_linegraph_mapping.items()}
            for edge in subgraph_linegraph.edges:
                subgraph_common_node, subgraph_diff = get_common_diff_nodes(edge)
                graph_common_node, graph_diff = get_common_diff_nodes((reverse_match[edge[0]], reverse_match[edge[1]]))
                node_reverse_mapping.append((subgraph_common_node, graph_common_node))
                node_reverse_mapping.append((subgraph_diff[0], graph_diff[0]))
                node_reverse_mapping.append((subgraph_diff[1], graph_diff[1]))
            node_reverse_mapping = dict(remove_duplicates(node_reverse_mapping))
            node_reverse_mappings.append(node_reverse_mapping)

    graph_dict = dict(graph)
    subgraph_dict = dict(subgraph_cand)
    node_reverse_mappings_final = []
    for node_reverse_mapping in node_reverse_mappings:
        is_valid = True
        for subgraph_node, graph_node in node_reverse_mapping.items():
            if subgraph_dict[subgraph_node] != "" and subgraph_dict[subgraph_node] not in graph_dict[graph_node]:
                is_valid = False
                break
        if is_valid:
            node_reverse_mappings_final.append(node_reverse_mapping)
    return node_reverse_mappings_final


def filter_sub_linegraph(alpha_item, task_linegraph_item):
    """
    Args:
        alpha_item: e.g. [True, True, False]
        task_linegraph_item: e.g. 
            [('0,1', 're1'),
             ('1,2', 're2'),
             ('0,2', 're3'),
             (('0,1', '1,2'), 'line'),
             (('0,1', '0,2'), 'line'),
             (('1,2', '0,1'), 'line'),
             (('1,2', '0,2'), 'line'),
             (('0,2', '0,1'), 'line'),
             (('0,2', '1,2'), 'line'),
            ]

    Returns:
        linegraph:
            [('0,1', 're1'),
             ('1,2', 're2'),
             (('0,1', '1,2'), 'line'),
             (('1,2', '0,1'), 'line'),
            ]
    """
    assert len(alpha_item.shape) == 1
    linegraph = []
    nodes_all = [ele[0] for ele in task_linegraph_item if not isinstance(ele[0], tuple)]
    nodes_chosen = [ele for i, ele in enumerate(nodes_all) if alpha_item[i] == True]
    linegraph = [ele for ele in task_linegraph_item if not isinstance(ele[0], tuple) and ele[0] in nodes_chosen]
    for item in task_linegraph_item:
        if isinstance(item[0], tuple):
            if item[0][0] in nodes_chosen and item[0][1] in nodes_chosen:
                linegraph.append(item)
    return linegraph


def get_graph_edit_distance(g1, g2, to_undirected=False):
    """Get the edit distance of two graphs considering their node and edge types.

    Args:
        g1, g2: has the format of 
        [
            [(0, 'Line', ...),
             (2, 'Line', ...),
             (3, 'Line', ...),
             ((0, 2), 'VerticalEdge', ...),
             ((0, 3), 'Parallel', ...),
             ((2, 3), 'VerticalEdge', ...),
        ]
        to_undirected: if True, will first transform the nx graph into an undirected graph.

    Returns:
        edit_distance: the edit distance between the two graphs.
    """
    import networkx as nx
    def node_match(node_dict1, node_dict2):
        return node_dict1["type"] == node_dict2["type"]
    def edge_match(edge_dict1, edge_dict2):
        return edge_dict1["type"] == edge_dict2["type"]
    def standardize_graph(graph):
        new_graph = []
        for ele in graph:
            if isinstance(ele[0], list):
                ele = (tuple(ele[0]), ele[1])
            new_graph.append(ele)
        return new_graph
    if not isinstance(g1, nx.Graph):
        g1 = get_nx_graph(standardize_graph(g1))
    if not isinstance(g2, nx.Graph):
        g2 = get_nx_graph(standardize_graph(g2))
    if to_undirected:
        g1 = g1.to_undirected(reciprocal=False)
        g2 = g2.to_undirected(reciprocal=False)
    edit_distance = nx.graph_edit_distance(g1, g2, node_match=node_match, edge_match=edge_match)
    return edit_distance


def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string


def filter_df(df, filter_dict):
    """Filter a pandas DataFrame according to a dictionary.

    Args:
        filter_dict, e.g. 
        filter_dict = {
            "dataset": "c-Line",
            "lr": 0.0001,
        }
    """
    mask = None
    for key, item in filter_dict.items():
        if mask is None:
            mask = df[key] == item
        else:
            mask = mask & (df[key] == item)
    return df[mask]


def get_unique_keys_df(df, types="all", exclude=None, exclude_str=None):
    """Get the unique keys in a pandas Dataframe."""
    if exclude is None:
        exclude = []
    if not isinstance(exclude, list):
        exclude = [exclude]
    if types == "all":
        keys_str = list(df.keys())
    elif types == "str":
        keys_str = [key for key, value in zip(df.keys(), df.iloc[0].values) if isinstance(value, str)]
    elif types == "number":
        keys_str = [key for key, value in zip(df.keys(), df.iloc[0].values) if isinstance(value, Number)]
    keys_unique = []
    for key in keys_str:
        try:
            if len(df[key].unique()) > 1 and key not in exclude:
                is_exclude = False
                if exclude_str is not None:
                    for string in exclude_str:
                        if string in key:
                            is_exclude = True
                            break
                if is_exclude:
                    continue
                keys_unique.append(key)
        except:
            pass
    return keys_unique


def groupby_add_keys(df, by, add_keys, other_keys=None, mode="mean"):
    """
    Group the df by the "by" argument, and also add the keys of "add_keys" (e.g. "hash", "filename") 
        if there is only one instance corresponding to the row.

    Args:
        add_keys: list of keys to add at the rightmost, e.g. ["hash", "filename"]
        other_keys: other keys to show. If None, will use all keys in df.
        mode: how to aggregate the values if there are more than one instance for the groupby.

    Returns:
        df_group: the desired DataFrame.
    """
    import pandas as pd
    def groupby_df(df, by, mode):
        if mode == "mean":
            df_group = df.groupby(by=by).mean()
        elif mode == "median":
            df_group = df.groupby(by=by).median()
        elif mode == "max":
            df_group = df.groupby(by=by).max()
        elif mode == "min":
            df_group = df.groupby(by=by).min()
        elif mode == "var":
            df_group = df.groupby(by=by).var()
        elif mode == "std":
            df_group = df.groupby(by=by).std()
        elif mode == "count":
            df_group = df.groupby(by=by).count()
        else:
            raise
        return df_group
    df = deepcopy(df)
    if isinstance(mode, str):
        df_group = groupby_df(df, by=by, mode=mode)
    else:
        assert isinstance(mode, dict)
        df_list = []
        for mode_ele, keys in mode.items():
            if mode_ele != "count":
                df_list.append(groupby_df(df[by + keys], by=by, mode=mode_ele))
            else:
                df["count"] = 1
                df_list.append(groupby_df(df[by + ["count"]], by=by, mode=mode_ele))
        df_group = pd.concat(df_list, axis=1)
    if other_keys is None:
        other_keys = list(df_group.keys())
    if not isinstance(add_keys, list):
        add_keys = [add_keys]
    if not isinstance(other_keys, list):
        other_keys = [other_keys]
    if isinstance(mode, dict) and "count" in mode and "count" not in other_keys:
        other_keys.append("count")
    df_group[add_keys] = None
    for i in range(len(df_group)):
        for k, key in enumerate(reversed(add_keys)):
            df_group_ele = df_group.iloc[i]
            filter_dict = dict(zip(df_group.T.keys().names, df_group.T.keys()[i]))
            df_filter = filter_df(df, filter_dict)
            if len(df_filter) == 1:
                df_group.iat[i, -(k+1)] = df_filter[key].values[0]
    df_group = df_group[other_keys + add_keys]
    return df_group


def enumerate_binary_array(dim, filter_fn=None):
    """Enumerate all binary arrays that satisfies the condition of filter_fn.

    Args:
        filter_fn: a list of functions which the binary array must satisfy (return True).
    """
    List = []
    for i in range(2 ** dim):
        string = np.binary_repr(i, width=dim)
        item = np.array([int(ele) for ele in string])
        if filter_fn is not None:
            if not isinstance(filter_fn, list):
                filter_fn = [filter_fn]
            is_skip = False
            for func in filter_fn:
                if not func(item):
                    is_skip = True
                    break
            if is_skip:
                continue
        List.append(item)
    return np.array(List)


def check_continuous(array):
    """Check if the array is continuous, where 0 is deemed as a breaking point."""
    if not isinstance(array, np.ndarray):
        array = np.array([int(ele) for ele in array])
    num_locs = np.where(array)[0]
    is_continuous = False
    if len(num_locs) == 1:
        is_continuous = True
    elif len(num_locs) > 1:
        if num_locs[-1] - num_locs[0] == len(num_locs) - 1:
            is_continuous = True
    return is_continuous


class Interp1d_torch(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (B, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (B, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (B, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        B = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (B, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


def extend_dims(tensor, n_dims, loc="right"):
    """Extends the dimensions by appending 1 at the right or left of the shape.

    E.g. if tensor has shape of (4, 6), then 
        extend_dims(tensor, 4, "right") has shape of (4,6,1,1);
        extend_dims(tensor, 4, "left")  has shape of (1,1,4,6).
    """
    if loc == "right":
        while len(tensor.shape) < n_dims:
            tensor = tensor[..., None]
    elif loc == "left":
        while len(tensor.shape) < n_dims:
            tensor = tensor[None]
    else:
        raise
    return tensor


def diff_recursive(d1, d2, level='root'):
    """Recursively compare two objects (dictionary, list, tensor, number, etc.)"""
    from numbers import Number
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            print('{:<20} + {} - {}'.format(level, s1-s2, s2-s1))
            common_keys = s1 & s2
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            diff_recursive(d1[k], d2[k], level='{}.{}'.format(level, k))

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print('{:<20} len1={}; len2={}'.format(level, len(d1), len(d2)))
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            diff_recursive(d1[i], d2[i], level='{}[{}]'.format(level, i))

    else:
        result = (d1 == d2)
        if not isinstance(result, Number):
            result = result.all()
        if result == False:
            print('{:<20} {} != {}'.format(level, d1, d2))


def requires_grad(parameters, flag=True):
    """Make the parameters of a module or list require gradient or not require gradient"""
    for p in parameters:
        p.requires_grad = flag
        

class MineList(list):
    pass


def get_norm(tensor, norm_type, dim=-1, epsilon=1e-10):
    """Get the norm for a batch of vectors, on the "dim" dimension.

    Args:
        norm_type: choose from "l2", "l1", "max" (L_infinity).
    """
    if norm_type == "l2":
        norm = (tensor.square().sum(dim) + epsilon).sqrt()
    elif norm_type == "l1":
        norm = tensor.abs().sum(dim)
    elif norm_type == "max":
        norm = tensor.max(dim)[0]
    else:
        raise
    return norm


def tsne_torch(embeddings, num_components, verbose=True):
    """
    Perform tsne embedding.

    Args:
        num_components: dimensions of the mapped space.
    """
    from tsne_torch import TorchTSNE as TSNE
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])
    embeddings = TSNE(n_components=num_components, perplexity=30, n_iter=1000, verbose=verbose).fit_transform(embeddings)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.reshape(new_shape)
    return embeddings


def leaky_clamp(x, min, max, slope=0.01):
    """Between min and max, use the value x. Outside, use slope. Also the full function is continuous."""
    negative_slope = 2*slope - 1
    return ((nn.LeakyReLU(negative_slope=negative_slope)(x-min)+min) + (max-nn.LeakyReLU(negative_slope=negative_slope)(max-x))) / 2


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
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        if not isinstance(n_neurons, Number):
            assert n_layers == len(n_neurons), "If n_neurons is not a number, then its length must be equal to n_layers={}".format(n_layers)
        self.n_layers = n_layers
        self.act_name = act_name
        self.output_size = output_size
        self.last_layer_linear = last_layer_linear
        self.is_res = is_res
        self.normalization_type = normalization_type
        self.normalization_n_groups = 2
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

    def forward(self, x):
        if self.act_name != "siren":
            u = x
            for i in range(1, self.n_layers + 1):
                u = getattr(self, "layer_{}".format(i))(u)
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


def get_normalization(normalization_type, n_channels, n_groups=2):
    """Get normalization layer."""
    if normalization_type == "bn1d":
        layer = nn.BatchNorm1d(n_channels)
    elif normalization_type == "bn2d":
        layer = nn.BatchNorm2d(n_channels)
    elif normalization_type == "gn":
        layer = nn.GroupNorm(num_groups=n_groups, num_channels=n_channels)
    elif normalization_type == "None":
        layer = nn.Identity()
    else:
        raise Exception("normalization_type '{}' is not valid!".format(normalization_type))
    return layer


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


def first_key(Dict):
    """Get the first key of a dictionary."""
    return next(iter(Dict))


def first_item(Dict):
    """Get the first item of a dictionary."""
    return Dict[first_key(Dict)]


def get_cap(string):
    """Get the string where the first letter is capitalized."""
    return string[0].upper() + string[1:]


def clear_dir(dirname):
    import os
    import glob
    files = glob.glob(dirname)
    for f in files:
        os.remove(f)


def scatter_add_grid_(grid, indices, src):
    """
    Scatter add to a 2D grid, in place.

    Args:
        grid: [H, W, (...)]
        indices: [B, 2] where each row is an index on H, W
        src:  [B, (...)]

    Returns:
        grid: [H, W, (...)] where the corresponding indices have added the src, also allowing 
            duplication of indices. Inplace operation
    """
    height, width = grid.shape[:2]
    grid_flatten = grid.view(-1, *grid.shape[2:])
    indices_flatten = indices[:,0] * width + indices[:,1]
    for _ in range(len(grid_flatten.shape) - len(indices_flatten.shape)):
        indices_flatten = indices_flatten[...,None]
    indices_flatten = indices_flatten.expand_as(src)
    grid_flatten.scatter_add_(0, index=indices_flatten, src=src)
    grid = grid_flatten.view(height, width, *grid.shape[2:])
    return grid
