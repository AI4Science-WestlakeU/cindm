#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from collections import Counter
import numpy as np
from numbers import Number
from functools import reduce
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from pytorch_net.util import get_activation, get_activation_noise, init_weight, init_bias, init_module_weights, init_module_bias, to_np_array, to_Variable, zero_grad_hook, ACTIVATION_LIST
from pytorch_net.util import standardize_symbolic_expression, get_param_name_list, get_variable_name_list, get_list_DL, get_coeffs_tree, snap, unsnap
AVAILABLE_REG = ["L1", "L2", "param"]
Default_Activation = "linear"


# ## Register all layer types:

# In[2]:


def get_Layer(layer_type, input_size, output_size, W_init = None, b_init = None, settings = {}, is_cuda = False):
    """Obtain layer from specifications."""
    if layer_type == "Simple_Layer":
        layer = Simple_Layer(input_size=input_size,
                             output_size=output_size,
                             W_init=W_init,
                             b_init=b_init,
                             settings=settings,
                             is_cuda=is_cuda,
                            )
    elif layer_type == "SuperNet_Layer":
        layer = SuperNet_Layer(input_size=input_size,
                               output_size=output_size,
                               W_init=W_init,
                               b_init=b_init,
                               settings=settings,
                               is_cuda=is_cuda,
                              )
    elif layer_type == "Symbolic_Layer":
        layer = Symbolic_Layer(input_size=input_size,
                               output_size=output_size,
                               W_init=W_init,
                               b_init=b_init,
                               settings=settings,
                               is_cuda=is_cuda,
                              )
    elif layer_type == "Utility_Layer":
        layer = Utility_Layer(input_size=input_size,
                              output_size=output_size,
                              settings=settings,
                              is_cuda=is_cuda,
                             )
    else:
        raise Exception("layer_type '{}' not recognized!".format(layer_type))
    return layer


def load_layer_dict(layer_dict, layer_type, is_cuda=False):
    """Load layer from layer_dict."""
    new_layer = get_Layer(layer_type="Symbolic_Layer",
                          input_size=layer_dict["input_size"],
                          output_size=layer_dict["output_size"],
                          W_init=layer_dict["weights"],
                          b_init=layer_dict["bias"],
                          settings=layer_dict["settings"],
                          is_cuda=is_cuda,
                         )
    return new_layer


def Simple_2_Symbolic(simple_layer, settings={}, mode="normal", prefix=""):
    """Transform Simple Layer to Symbolic Layer."""
    from sympy import Symbol, Function
    input_size = simple_layer.input_size
    output_size = simple_layer.output_size
    symbolic_expression = []
    W_core, b_core = simple_layer.get_weights_bias()
    W_init = {}

    if mode == "normal":
        for j in range(output_size):
            expression = 0
            if W_core is not None:
                for i in range(input_size):
                    expression += Symbol("{0}W{1}{2}".format(prefix, i, j)) * Symbol("x{0}".format(i))
                    W_init["{0}W{1}{2}".format(prefix, i, j)] = W_core[i, j]
            if b_core is not None:
                expression += Symbol("{0}b{1}".format(prefix, j))
                W_init["{0}b{1}".format(prefix, j)] = b_core[j]
            if "activation" in simple_layer.settings:
                activation_name = simple_layer.settings["activation"]
            elif "activation" in settings:
                activation_name = settings["activation"]
            else:
                activation_name = Default_Activation
            if activation_name != "linear":
                expression = Function(activation_name)(expression)
            symbolic_expression.append(expression)
    elif mode == "separable":
        vector_p, vector_q = snap(W_core, "separable")[0]
        for j in range(output_size):
            expression = 0
            for i in range(input_size):
                expression += Symbol("x{}".format(i)) * Symbol("{0}p{1}".format(prefix, i)) * Symbol("{0}q{1}".format(prefix, j))
                W_init["{0}p{1}".format(prefix, i)] = vector_p[i]
            expression += Symbol("{0}b{1}".format(prefix, j))
            W_init["{0}q{1}".format(prefix, j)] = vector_q[j]
            W_init["{0}b{1}".format(prefix, j)] = b_core[j]
            if "activation" in simple_layer.settings:
                activation_name = simple_layer.settings["activation"]
            elif "activation" in settings:
                activation_name = settings["activation"]
            else:
                activation_name = Default_Activation        
            if activation_name != "linear":
                expression = Function(activation_name)(expression)
            symbolic_expression.append(expression)   

    return get_Layer(layer_type="Symbolic_Layer",
                     input_size=input_size,
                     output_size=output_size,
                     W_init=W_init,
                     b_init=None,
                     settings={"symbolic_expression": str(symbolic_expression)},
                     is_cuda=simple_layer.is_cuda,
                    )


# ## Simple Layer:

# In[1]:


class Simple_Layer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        W_init=None,     # initialization for weights
        b_init=None,     # initialization for bias
        settings={},     # Other settings that are relevant to this specific layer
        is_cuda=False,
        ):
        # Firstly, must perform this step:
        super(Simple_Layer, self).__init__()
        # Saving the attribuites:
        if isinstance(input_size, tuple):
            self.input_size = reduce(lambda x, y: x * y, input_size)
            self.input_size_original = input_size
        else:
            self.input_size = input_size
        if isinstance(output_size, tuple):
            self.output_size = reduce(lambda x, y: x * y, output_size)
            self.output_size_original = output_size
        else:
            self.output_size = output_size

        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")
        self.settings = settings

        # Other attributes that are specific to this layer:
        self.activation = settings["activation"] if "activation" in settings else Default_Activation
        self.weight_on = settings["weight_on"] if "weight_on" in settings else True
        self.bias_on = settings["bias_on"] if "bias_on" in settings else True
        self.reg_on = settings["reg_on"] if "reg_on" in settings else True

        # Define the learnable parameters in the module (use any name you like). 
        # use nn.Parameter() so that the parameters is registered in the module and can be gradient-updated:
        # self.W_init, self.b_init can be a numpy array, or a string like "glorot-normal":
        if self.weight_on:
            self.W_core = nn.Parameter(torch.randn(self.input_size, self.output_size))
            init_weight(self.W_core, init=W_init)
        if self.bias_on:
            self.b_core = nn.Parameter(torch.zeros(self.output_size))
            init_bias(self.b_core, init=b_init)
        # Dropout:
        if "dropout_rate" in settings:
            self.dropout = nn.Dropout(p=settings["dropout_rate"])
        self.set_cuda(is_cuda)

        # Initialize parameter freeze if stipulated:
        if "snap_dict" in self.settings:
            # Clear snapping if either self.weight_on is False or self.bias_on is False
            pop_snapping = []
            for pos, idx in self.settings["snap_dict"]:
                if (self.weight_on is False and pos == "weight") or (self.bias_on is False and pos == "bias"):
                    pop_snapping.append((pos, idx))
            for key in pop_snapping:
                self.settings["snap_dict"].pop(key)
        
            # Initialize freeze:
            self.snap_dict = self.settings["snap_dict"]
            self.initialize_param_freeze(update_values=True)
        else:
            self.snap_dict = {}


    def __repr__(self):
        string = ""
        if not self.weight_on:
            string += ", weight_on=False"
        if not self.bias_on:
            string += ", bias_on=False"
        if not self.reg_on:
            string += ", reg_on=False"
        if "dropout_rate" in self.settings:
            string += ", dropout_rate={}".format(self.settings["dropout_rate"])
        if "act_noise" in self.settings:
            string += ", act_noise={}".format(self.settings["act_noise"])
        return 'Simple_Layer({}, "{}"{})'.format(self.output_size, self.activation, string)


    def change(self, target, new_property):
        if target == "weight":
            if self.weight_on:
                old_property = "on"
                if new_property == "off":
                    self.settings["weight_on"] = False
                    self.weight_on = False
                    delattr(self, "W_core")
            else:
                old_property = "off"
                if new_property == "on":
                    self.settings.pop("weight_on")
                    self.weight_on = True
                    self.W_core = nn.Parameter(torch.randn(self.input_size, self.output_size))
                    init_weight(self.W_core, init=None)

        elif target == "bias":
            if self.bias_on:
                old_property = "on"
                if new_property == "off":
                    self.settings["bias_on"] = False
                    self.bias_on = False
                    delattr(self, "b_core")
            else:
                old_property = "off"
                if new_property == "on":
                    self.settings.pop("bias_on")
                    self.bias_on = True
                    self.b_core = nn.Parameter(torch.zeros(self.output_size))
                    init_bias(self.b_core, init=None)
                
        elif target == "activation":
            old_property = self.settings["activation"]
            self.settings["activation"] = new_property
            self.activation = self.settings["activation"]  
        else:
            raise Exception("target can only be activation!")
        return old_property


    @property
    def struct_param(self):
        output_size = self.output_size_original if hasattr(self, "output_size_original") else self.output_size
        if len(self.snap_dict) > 0:
            self.settings["snap_dict"] = self.snap_dict
        return [output_size, "Simple_Layer", self.settings]


    @property
    def layer_dict(self):
        input_size = self.input_size_original if hasattr(self, "input_size_original") else self.input_size
        output_size = self.output_size_original if hasattr(self, "output_size_original") else self.output_size
        Layer_dict =  {
            "input_size": input_size,
            "output_size": output_size,
            "settings": self.settings,
        }
        if len(self.snap_dict) > 0:
            Layer_dict["settings"]["snap_dict"] = self.snap_dict
        Layer_dict["weights"], Layer_dict["bias"] = self.get_weights_bias()
        return Layer_dict


    @property
    def DL(self):
        non_snapped_list = []
        snapped_list = []
        # Weights:
        if self.weight_on:
            shape = self.W_core.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if ("weight", (i, j)) in self.snap_dict:
                        snapped_list.append(self.snap_dict[("weight", (i, j))]["new_value"])
                    else:
                        non_snapped_list.append(to_np_array(self.W_core[i, j]))
        # Bias:
        if self.bias_on:
            for i in range(len(self.b_core)):
                if ("bias", i) in self.snap_dict:
                    snapped_list.append(self.snap_dict[("bias", i)]["new_value"])
                else:
                    non_snapped_list.append(to_np_array(self.b_core[i]))
        return get_list_DL(snapped_list, "snapped") + get_list_DL(non_snapped_list, "non-snapped")


    def load_layer_dict(self, layer_dict):
        new_layer = load_layer_dict(layer_dict, "Simple_Layer", self.is_cuda)
        self.__dict__.update(new_layer.__dict__)


    def forward(self, input, p_dict=None):
        output = input
        if hasattr(self, "input_size_original"):
            output = output.view(-1, self.input_size)
        # Dropout:
        if hasattr(self, "dropout"):
            output = self.dropout(output)

        # Perform dot(X, W) + b:
        if self.weight_on:
            output = torch.matmul(output, self.W_core)
        if self.bias_on:
            output = output + self.b_core
        
        # If p_dict is not None, update the first neuron's activation according to p_dict:
        if p_dict is not None:
            p_dict = p_dict.view(-1)
            if len(p_dict) == 2:
                output_0 = output[:,:1] * p_dict[1] + p_dict[0]
            elif len(p_dict) == 1:
                output_0 = output[:,:1] + p_dict[0]
            else:
                raise
            if output.size(1) > 1:
                output = torch.cat([output_0, output[:,1:]], 1)
            else:
                output = output_0

        # Perform activation function:
        output = get_activation(self.activation)(output)

        # Add activation noise:
        if "act_noise" in self.settings:
            output = get_activation_noise(self.settings["act_noise"])(output)

        if hasattr(self, "output_size_original"):
            output = output.view(*((-1,) + self.output_size_original))
        assert output.size(0) == input.size(0), "output_size {0} must have same length as input_size {1}. Check shape!".format(output.size(0), input.size(0))
        return output


    def prune_output_neurons(self, neuron_ids):
        if not isinstance(neuron_ids, list):
            neuron_ids = [neuron_ids]
        preserved_ids = torch.LongTensor(np.array(list(set(range(self.output_size)) - set(neuron_ids)))).to(self.device)
        if self.weight_on:
            self.W_core = nn.Parameter(self.W_core.data[:, preserved_ids])
            self.output_size = self.W_core.shape[1]
        if self.bias_on:
            self.b_core = nn.Parameter(self.b_core.data[preserved_ids])
            self.output_size = self.b_core.shape[0]
    
    
    def prune_input_neurons(self, neuron_ids):
        if self.weight_on:
            if not isinstance(neuron_ids, list):
                neuron_ids = [neuron_ids]
            preserved_ids = torch.LongTensor(np.array(list(set(range(self.input_size)) - set(neuron_ids))))
            self.W_core = nn.Parameter(self.W_core.data[preserved_ids, :])
            self.input_size = self.W_core.size(0)
        else:
            print("Cannot shrink input neurons since weight_on=False")

    
    def add_output_neurons(self, num_neurons, mode="imitation"):
        if mode == "imitation":
            if self.weight_on:
                W_core_mean = to_np_array(self.W_core.mean())
                W_core_std = to_np_array(self.W_core.std())
                new_W_core = torch.randn(self.input_size, num_neurons) * W_core_std + W_core_mean
            if self.bias_on:
                b_core_mean = to_np_array(self.b_core.mean())
                b_core_std = to_np_array(self.b_core.std())
                new_b_core = torch.randn(num_neurons) * b_core_std + b_core_mean
        elif mode == "zeros":
            if self.weight_on:
                new_W_core = torch.zeros(self.input_size, num_neurons)
            if self.bias_on:
                new_b_core = torch.zeros(num_neurons)
        elif mode[0] == "copy":
            neuron_id = mode[1]
            if self.weight_on:
                new_W_core = self.W_core[:, neuron_id: neuron_id + 1].detach().data
            if self.bias_on:
                new_b_core = self.b_core[neuron_id: neuron_id + 1].detach().data
        else:
            raise Exception("mode {0} not recognized!".format(mode))
        if self.weight_on:
            self.W_core = nn.Parameter(torch.cat([self.W_core.data, new_W_core.to(self.device)], 1))
        if self.bias_on:
            self.b_core = nn.Parameter(torch.cat([self.b_core.data, new_b_core.to(self.device)], 0))
        self.output_size += num_neurons
        
    
    def add_input_neurons(self, num_neurons, mode="imitation", position="end"):
        if self.weight_on:
            if mode == "imitation":
                W_core_mean = self.W_core.mean().item()
                W_core_std = self.W_core.std().item()
                new_W_core = torch.randn(num_neurons, self.output_size) * W_core_std + W_core_mean
            elif mode == "zeros":
                new_W_core = torch.zeros(num_neurons, self.output_size)
            else:
                raise Exception("mode {} not recognized!".format(mode))
            if position == "end":
                self.W_core = nn.Parameter(torch.cat([self.W_core.data, new_W_core], 0))
            else:
                assert isinstance(position, Number)
                self.W_core = nn.Parameter(torch.cat([self.W_core.data[:position], new_W_core, self.W_core.data[position:]], 0))
            self.input_size += num_neurons
        else:
            print("Cannot add input neurons since weight_on=False")
        

    def standardize(self, mode="b_mean_zero"):
        if mode == "b_mean_zero":
            if self.bias_on:
                b_mean = to_np_array(self.b_core.mean())
                self.b_core.data.copy_(self.b_core.data - b_mean)
        else:
            raise Exception("mode {0} not recognized!".format(mode))


    def simplify(self, mode="snap", excluded_idx=[], top=1, **kwargs):
        def get_idx_list(key_list, input_size, output_size, weight_on):
            """Transform (pos, true_idx) list to idx list."""
            idx_list = []
            for pos, true_idx in key_list:
                if pos == "weight":
                    assert self.weight_on is True
                    idx_list.append(true_idx[0] * output_size + true_idx[1])
                elif pos == "bias":
                    assert self.bias_on is True
                    if weight_on:
                        idx_list.append(true_idx + input_size * output_size)
                    else:
                        idx_list.append(true_idx)
                else:
                    raise
            return sorted(idx_list)

        def get_true_idx(idx, input_size, output_size, weight_on):
            """Get (pos, true_idx) from idx"""
            if weight_on:
                if idx < input_size * output_size:
                    pos = "weight"
                    true_idx = (int(idx / output_size), idx % output_size)
                else:
                    pos = "bias"
                    true_idx = idx - input_size * output_size
            else:
                pos = "bias"
                true_idx = idx
            return pos, true_idx

        if mode == "snap":
            snap_mode = kwargs["snap_mode"] if "snap_mode" in kwargs else "integer"
            if snap_mode == "unsnap":
                self.remove_param_freeze()
                return ["unsnap"]
            elif snap_mode == "vector":
                return []
            else:
                # Identify the parameters to freeze:
                param = []
                if self.weight_on:
                    param.append(to_np_array(self.W_core.view(-1)))
                if self.bias_on:
                    param.append(to_np_array(self.b_core.view(-1), full_reduce=False))
                param = np.concatenate(param)
                if "snap_targets" in kwargs and kwargs["snap_targets"] is not None:
                    snap_targets = kwargs["snap_targets"]
                    is_target_given = True
                else:
                    excluded_idx_combined = get_idx_list(set([element[0] for element in excluded_idx] + list(self.snap_dict.keys())), 
                                                         self.input_size, self.output_size, self.weight_on)
                    snap_targets = snap(param, snap_mode=snap_mode, excluded_idx=excluded_idx_combined, top=top)
                    is_target_given = False

                info_list = []
                for idx, new_value in snap_targets:
                    if new_value is not None:
                        if is_target_given:
                            pos, true_idx = idx
                            new_value = float(new_value)
                        else:
                            pos, true_idx = get_true_idx(idx, self.input_size, self.output_size, self.weight_on)
                            new_value = new_value.astype(float)
                        info_list.append(((pos, true_idx), new_value))
                        if pos == "weight":
                            new_W_core = self.W_core.data
                            new_W_core[true_idx] = new_value
                            self.W_core = nn.Parameter(new_W_core)
                        elif pos == "bias":
                            new_b_core = self.b_core.data
                            new_b_core[true_idx] = new_value
                            self.b_core = nn.Parameter(new_b_core)
                        self.snap_dict[(pos, true_idx)] = {"new_value": new_value}
                        self.initialize_param_freeze(update_values=False)
        else:
            raise Exception("mode {0} not recognized!".format(mode))
        return info_list


    def initialize_param_freeze(self, update_values=True):
        if update_values:
            if self.weight_on:
                new_W_core = self.W_core.data
                for (pos, true_idx), item in self.snap_dict.items():
                    if pos == "weight":
                        new_W_core[true_idx] = item["new_value"]
                self.W_core = nn.Parameter(new_W_core)
            if self.bias_on:
                new_b_core = self.b_core.data
                for (pos, true_idx), item in self.snap_dict.items():
                    if pos == "bias":
                        new_b_core[true_idx] = item["new_value"]
                self.b_core = nn.Parameter(new_b_core)
        
        # Initialize hook:
        for pos, true_idx in self.snap_dict.keys():
            hook_function = zero_grad_hook(true_idx)
            if self.weight_on and pos == "weight":
                h = self.W_core.register_hook(hook_function)
            elif self.bias_on and pos == "bias":
                h = self.b_core.register_hook(hook_function)
    

    def remove_param_freeze(self, index_list=None):
        if index_list is None:
            self.snap_dict = {}
            self.settings.pop("snap_dict")
        else:
            for key in index_list:
                self.snap_dict.pop(key)
            self.initialize_param_freeze(update_values=True)


    def get_param_names(self, source):
        if source == "modules":
            if self.weight_on:
                param_names = ["W_core"]
            if self.bias_on:
                param_names.append("b_core")
        if source == "attention":
            param_names = []
        return param_names


    def get_weights_bias(self, is_grad=False):
        if not is_grad:
            W_core = deepcopy(to_np_array(self.W_core, full_reduce=False)) if self.weight_on else None
            b_core = deepcopy(to_np_array(self.b_core, full_reduce=False)) if self.bias_on else None
            return W_core, b_core
        else:
            W_grad = self.W_core.grad if self.weight_on else None
            b_grad = self.b_core.grad if self.bias_on else None
            W_grad = deepcopy(to_np_array(W_grad, full_reduce=False)) if W_grad is not None else None
            b_grad = deepcopy(to_np_array(b_grad, full_reduce=False)) if b_grad is not None else None
            return W_grad, b_grad

    
    def get_regularization(self, mode, source=["weight", "bias"]):
        if not isinstance(source, list):
            source = [source]
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad=False).to(self.device)
        if self.reg_on:
            for source_ele in source:
                if self.weight_on:
                    if source_ele == "weight":
                        if mode == "L1":
                            reg = reg + self.W_core.abs().sum()
                        elif mode == "L2":
                            reg = reg + (self.W_core ** 2).sum()
                        elif mode in AVAILABLE_REG:
                            pass
                        else:
                            raise Exception("mode '{}' not recognized!".format(mode))
                elif source_ele == "bias":
                    if self.bias_on:
                        if mode == "L1":
                            reg = reg + self.b_core.abs().sum()
                        elif mode == "L2":
                            reg = reg + (self.b_core ** 2).sum()
                        elif mode in AVAILABLE_REG:
                            pass
                        else:
                            raise Exception("mode '{}' not recognized!".format(mode))
        return reg


    def set_cuda(self, is_cuda):
        if isinstance(is_cuda, str):
            self.cuda(is_cuda)
        else:
            if is_cuda:
                self.cuda()
            else:
                self.cpu()
        self.is_cuda = is_cuda


    def set_trainable(self, is_trainable):
        if is_trainable:
            if self.weight_on:
                self.W_core.requires_grad = True
            if self.bias_on:
                self.b_core.requires_grad = True
        else:
            if self.weight_on:
                self.W_core.requires_grad = False
            if self.bias_on:
                self.b_core.requires_grad = False


# ## Utility layers:

# In[ ]:


class Utility_Layer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        settings={},
        is_cuda=False,
    ):
        super(Utility_Layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.is_cuda = is_cuda
        self.settings = settings


    def forward(self, input, **kwargs):
        layer_type = self.settings["type"]
        if layer_type == "reshape":
            return input.reshape(self.output_size)
        elif layer_type == "flatten":
            return input.view(-1)
        else:
            raise Exception("layer_type {} is not valid".format(layer_type))

    @property
    def model_dict(self):
        model_dict = {}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["settings"] = self.settings
        return model_dict


    @property
    def struct_param(self):
        return [self.output_size, "Utility_Layer", self.settings]


    def get_regularization(self, mode, source=["weight"], **kwargs):
        reg = to_Variable([0], is_cuda=self.is_cuda)
        return reg


    def set_trainable(self, is_trainable):
        pass


    def set_cuda(self, is_cuda):
        self.is_cuda = is_cuda


# ## Symbolic Layer:

# In[4]:


class Symbolic_Layer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        W_init=None,
        b_init=None,
        settings={},
        is_cuda=False,
        ):
        super(Symbolic_Layer, self).__init__()
        from sympy.parsing.sympy_parser import parse_expr
        self.input_size = input_size
        self.output_size = output_size
        self.W_init = W_init # Here we use W_init to represent all parameter initial values
        self.is_cuda = is_cuda
        self.is_numerical = False
        self.set_symbolic_expression(str(settings["symbolic_expression"]), p_init=self.W_init)


    @property
    def layer_dict(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weights": self.get_param_dict(),
            "bias": None,
            "settings": {"symbolic_expression": str(self.symbolic_expression)},
        }


    @property
    def struct_param(self):
        return [self.output_size, "Symbolic_Layer", {"symbolic_expression": str(self.symbolic_expression)}]


    @property
    def settings(self):
        return {"symbolic_expression": str(self.symbolic_expression)}
    
    
    @property
    def activation(self):
        activation_list = []
        for expression in self.symbolic_expression:
            if hasattr(expression.func, "name") and expression.func.name in ACTIVATION_LIST:
                act_ele = expression.func.name
            else:
                act_ele = "linear"
            activation_list.append(act_ele)
        if len(Counter(activation_list)) > 1:
            return "linear"
        else:
            return list(Counter(activation_list).keys())[0]
    
    
    def change(self, target, new_property):
        from sympy import Add
        from sympy.utilities.lambdify import implemented_function
        assert target == "activation"
        prev_activation = self.activation
        activation = new_property
        if activation != prev_activation:
            if prev_activation == "linear":
                f = implemented_function(activation, get_activation(activation))
                new_symbolic_expression = [f(expression) for expression in self.symbolic_expression]
                self.set_symbolic_expression(new_symbolic_expression)
            else:
                if activation != "linear":
                    f = implemented_function(activation, get_activation(activation))
                    new_symbolic_expression = [f(*expression.args) for expression in self.symbolic_expression]
                else:
                    new_symbolic_expression = [Add(*expression.args) for expression in self.symbolic_expression]
                self.set_symbolic_expression(new_symbolic_expression)


    @property
    def numerical_expression(self):
        from sympy import Symbol
        """Replace the parameter in symbolic_expression by their numerical values"""
        substitution = [(Symbol(param_name), to_np_array(getattr(self, param_name))) for param_name in self.param_name_list]
        return [expression.subs(substitution) for expression in self.symbolic_expression]
    
    
    def set_numerical(self, is_numerical):
        self.is_numerical = is_numerical


    @property
    def DL(self):
        param_dict = self.get_param_dict()
        expr_length, snapped_list = get_coeffs_tree(self.symbolic_expression, param_dict)
        non_snapped_list = list(param_dict.values())
        return get_list_DL(snapped_list, "snapped") + get_list_DL(non_snapped_list, "non-snapped")


    def load_layer_dict(self, layer_dict):
        new_layer = load_layer_dict(layer_dict, "Symbolic_Layer", self.is_cuda)
        self.__dict__.update(new_layer.__dict__)


    def prune_output_neurons(self, neuron_ids):
        if not isinstance(neuron_ids, list):
            neuron_ids = [neuron_ids]
        variable_names = self.get_variable_name_list()
        assert "x" not in variable_names, "In order to prune output_neurons, 'x' cannot be in the symbolic_expression!"
        symbolic_expression = [expression for i, expression in enumerate(self.symbolic_expression) if i not in neuron_ids]
        self.output_size = sum(self.get_expression_length(symbolic_expression))
        self.set_symbolic_expression(symbolic_expression)
    
    
    def standardize(self, mode="b_mean_zero"):
        from sympy import Function
        if mode == "b_mean_zero":
            param_dict = self.get_param_dict()
            bias_list = []
            for expression in self.symbolic_expression:
                fun_name_list = self.get_function_name_list(expression)
                if len(fun_name_list) == 1:
                    expr = expression.args[0]
                elif len(fun_name_list) == 0:
                    expr = expression
                else:
                    raise Exception("There must be at most one activation function")

                vars_subs = {element: 0 for element in self.get_variable_name_list(expr)}
                bias = expr.subs(vars_subs).subs(param_dict)
                bias_list.append(bias)
            bias_mean = np.mean(bias_list)

            new_symbolic_expression = []
            for expression in self.symbolic_expression:
                fun_name_list = self.get_function_name_list(expression)
                if len(fun_name_list) == 1:
                    fun = Function(fun_name_list[0])
                    expr = expression.args[0]
                elif len(fun_name_list) == 0:
                    expr = expression
                else:
                    raise Exception("There must be at most one activation function")

                expr = expr - bias_mean
                if len(fun_name_list) == 1:
                    expr = fun(expr)
                new_symbolic_expression.append(expr)
            self.set_symbolic_expression(new_symbolic_expression)            
        else:
            raise Exception("mode {0} not recognized!".format(mode))
    
    
    def init_with_p_dict(self, p_dict):
        self.set_param_values(p_dict)


    def init_bias_with_input(self, input, mode="std_sqrt"):
        pass


    def get_param_name_list(self, symbolic_expression=None):
        """Get parameter names from a given symbolic expression"""
        # Here in the Sympy_Net we assume that the input is always represented by Symbol("x"), so "x" is excluded from param_name_list:
        symbolic_expression = self.symbolic_expression if symbolic_expression is None else symbolic_expression
        symbolic_expression = standardize_symbolic_expression(symbolic_expression)
        return get_param_name_list(symbolic_expression)


    def get_variable_name_list(self, symbolic_expression=None):
        symbolic_expression = self.symbolic_expression if symbolic_expression is None else symbolic_expression
        symbolic_expression = standardize_symbolic_expression(symbolic_expression)
        return get_variable_name_list(symbolic_expression)


    def get_function_name_list(self, symbolic_expression=None):
        from sympy import Function
        from sympy.utilities.lambdify import implemented_function
        symbolic_expression = self.symbolic_expression if symbolic_expression is None else symbolic_expression
        symbolic_expression = standardize_symbolic_expression(symbolic_expression)
        function_name_list = list({element.func.__name__ for expression in symbolic_expression for element in expression.atoms(Function) if element.func.__name__ not in ["linear"]})
        self.implemented_function = {}
        for function_name in function_name_list:
            try:
                self.implemented_function[function_name] = implemented_function(Function(function_name), get_activation(function_name))
            except:
                pass
        return function_name_list


    def get_param_dict(self):
        param_names = self.get_param_name_list()
        return {param_name: to_np_array(getattr(self, param_name)) for param_name in param_names}


    def set_param_values(self, new_param_values):
        param_names = self.get_param_name_list()
        for key, value in new_param_values.items():
            if key in param_names:
                if isinstance(value, Variable):
                    value_core = value.data
                elif isinstance(value, float) or isinstance(value, int):
                    value_core = torch.FloatTensor(np.array([value]))
                getattr(self, key).data.copy_(value_core.view(-1))


    def get_weights_bias(self, is_grad=False):
        if not is_grad:
            return deepcopy(self.get_param_dict()), None
        else:
            param_names = self.get_param_name_list()
            param_grad_dict = {}
            for param_name in param_names:
                grad = getattr(self, param_name).grad
                if grad is not None:
                    param_grad_dict[param_name] = grad.item()
                else:
                    param_grad_dict[param_name] = None
            return param_grad_dict, None


    def get_expression_length(self, symbolic_expression=None):
        symbolic_expression = self.symbolic_expression if symbolic_expression is None else symbolic_expression
        symbolic_expression = standardize_symbolic_expression(symbolic_expression)
        length_list = []
        for expression in symbolic_expression:
            variable_list = self.get_variable_name_list([expression])
            if "x" in variable_list:
                assert len(variable_list) == 1, "x cannot coexist with x1, x2, etc. in a single expression, since the dimension is not compatible"
                length = self.input_size
            else:
                length = 1
            length_list.append(length)
        return length_list


    def set_symbolic_expression(self, symbolic_expression, p_init=None):
        """Set a new symbolic expression and update the parameterss"""
        symbolic_expression = standardize_symbolic_expression(symbolic_expression)
        assert sum(self.get_expression_length(symbolic_expression)) == self.output_size, "symbolic_expression's combined output length must be equal to self.output_size!"
        self.old_param_name_list = self.get_param_name_list(self.symbolic_expression) if hasattr(self, "symbolic_expression") else []
        self.symbolic_expression = symbolic_expression
        self.param_name_list = self.get_param_name_list(symbolic_expression)
        self.variable_name_list = self.get_variable_name_list(symbolic_expression)
        self.get_function_name_list()
        
        # If the new expression has parameter names that did not appear in previous expression, create it:
        for param_name in self.param_name_list:            
            if not hasattr(self, param_name):
                if p_init is not None:
                    param_init = p_init[param_name] if param_name in p_init else None
                else:
                    param_init = None

                if param_init is None:
                    setattr(self, param_name, nn.Parameter(torch.randn(1)))
                else:
                    setattr(self, param_name, nn.Parameter(torch.FloatTensor(np.array([param_init]))))

        # Delete class parameters that do not appear in the new symbolic expression:
        param_name_to_delete = set(self.old_param_name_list) - set(self.param_name_list)
        for param_name in param_name_to_delete:
            delattr(self, param_name)

        self.set_cuda(self.is_cuda)


    def set_trainable(self, is_trainable):
        param_name_list = self.get_param_name_list()
        for param_name in param_name_list:
            if is_trainable:
                getattr(self, param_name).requires_grad = True
            else:
                getattr(self, param_name).requires_grad = False
    
    
    def set_cuda(self, is_cuda):
        if isinstance(is_cuda, str):
            self.cuda(is_cuda)
        else:
            if is_cuda:
                self.cuda()
            else:
                self.cpu()
        self.is_cuda = is_cuda


    def forward(self, input, p_dict=None):
        from sympy import Symbol, lambdify, N
        symbols = [Symbol(variable_name) for variable_name in self.variable_name_list]
        if p_dict is None:
            symbols = tuple(symbols + [Symbol(param_name) for param_name in self.param_name_list])  # Get symbolic variables
        else:
            symbols = tuple(symbols + [Symbol(param_name) for param_name in sorted(list(p_dict.keys())) if "x" not in param_name])
        f_list = [lambdify(symbols, N(expression), torch) for expression in self.symbolic_expression]    # Obtain the lambda function f(x0, x1,..., param0, param1, ...)
        # Obtain the data that will be fed into (x0, x1,..., param0, param1, ...):
        variables_feed = []
        for variable_name in self.variable_name_list:
            if variable_name == "x":
                variable_feed = input
            else:
                idx = int(variable_name[1:])
                variable_feed = input[:, idx: idx + 1]
            variables_feed.append(variable_feed)
        if p_dict is None:
            symbols_feed = variables_feed + [getattr(self, param_name) for param_name in self.param_name_list]
        else:
            symbols_feed = variables_feed + [p_dict[param_name] for param_name in sorted(list(p_dict.keys())) if "x" not in param_name]
        output_list = []
        for f in f_list:
            output_ele = f(*symbols_feed)
            if not isinstance(output_ele, Variable):
                output_ele = to_Variable(torch.ones(input.shape[0], 1), is_cuda=self.is_cuda) * output_ele
            elif len(output_ele.shape) < 2 or output_ele.shape[0] != input.shape[0]:
                multiplier = to_Variable(torch.ones(input.shape[0], 1), is_cuda=self.is_cuda)
                output_ele = output_ele * multiplier
            output_list.append(output_ele)
        return torch.cat(output_list, 1)


    def simplify(self, mode="form", **kwargs):
        from sympy import simplify, Symbol
        verbose = kwargs["verbose"] if "verbose" in kwargs else 0
        info_list = []
        if not isinstance(mode, list):
            mode = [mode]
        for mode_ele in mode:
            if mode_ele == "form":
                prev_expression = self.symbolic_expression
                new_expression = [simplify(expression) for expression in self.symbolic_expression]
                self.set_symbolic_expression(new_expression)
                if verbose > 0:
                    print("Original expression:\tsymbolic: {0}; \t numerical: {1}".format(prev_expression, self.numerical_expression))
                    print("New  expression: \tsymbolic: {0}; \t numerical: {1}".format(self.symbolic_expression, self.numerical_expression))                
            elif mode_ele == "snap":
                snap_mode = kwargs["snap_mode"] if "snap_mode" in kwargs else "integer"
                top = kwargs["top"] if "top" in kwargs else 1
                if snap_mode == "unsnap":
                    unsnapped_expression, new_param_dict = unsnap(self.symbolic_expression, self.get_param_dict())
                    self.set_symbolic_expression(unsnapped_expression, new_param_dict)
                    info_list = info_list + [(mode_ele, snap_mode, unsnapped_expression)]
                elif snap_mode in ["vector"]:
                    param_dict = self.get_param_dict()
                    param_dict_subs, new_param_dict = snap(param_dict, snap_mode=snap_mode, top=top)
                    param_dict.update(new_param_dict)
                    if verbose > 0:
                        print("Original expression:\tsymbolic: {0}; \t numerical: {1}".format(self.symbolic_expression, self.numerical_expression))
                        print("Substitution:  \t{0}, with value: {1}".format(pprint_dict(param_dict_subs), pprint_dict(new_param_dict)))
                    new_expression = [expression.subs(param_dict_subs) for expression in self.symbolic_expression]
                    self.set_symbolic_expression(new_expression, param_dict)
                    info_list = info_list + [(deepcopy(param_dict_subs), deepcopy(new_param_dict))]
                    if verbose > 0:
                        print("New  expression: \tsymbolic: {0}; \t numerical: {1}".format(self.symbolic_expression, self.numerical_expression))
                else:
                    param_names = list(self.get_param_dict().keys())
                    param_array = np.array(list(self.get_param_dict().values()))
                    snap_targets = snap(param_array, snap_mode=snap_mode, top=top)
                    if not (len(snap_targets) == 1 and snap_targets[0][1] is None):
                        subs_targets = [(Symbol(param_names[idx]), new_value) for idx, new_value in snap_targets]
                        prev_expression = self.symbolic_expression
                        if verbose > 0:
                            print("Original expression:\tsymbolic: {0}; \t numerical: {1}".format(prev_expression, self.numerical_expression))
                            print("Substitution:  \t{0}".format(subs_targets))
                        new_expression = [expression.subs(subs_targets) for expression in self.symbolic_expression]
                        self.set_symbolic_expression(new_expression)
                        info_list = info_list + [(param_names[idx], new_value) for idx, new_value in snap_targets]
                        if verbose > 0:
                            print("New  expression: \tsymbolic: {0}; \t numerical: {1}".format(self.symbolic_expression, self.numerical_expression))
            elif mode_ele == "pair_snap":
                if len(self.get_param_dict()) < 2:
                    raise Exception("Less than 2 parameters. Cannot pair_snap!")
                else:
                    def get_param_inverse_dict(Dict):
                        inverse_dict = {}
                        i = 0
                        param_list = []
                        for key, value in Dict.items():
                            param_list.append(value)
                            inverse_dict[i] = key
                            i += 1
                        return param_list, inverse_dict
                    snap_mode = kwargs["snap_mode"] if "snap_mode" in kwargs else "integer"
                    top = kwargs["top"] if "top" in kwargs else 1
                    if snap_mode == "integer":
                        snap_mode_whole = "pair_integer"
                    elif snap_mode == "rational":
                        snap_mode_whole = "pair_rational"
                    else:
                        raise Exception("snap_mode {} not recognized!".format(snap_mode))
                    param_list, inverse_dict = get_param_inverse_dict(self.get_param_dict())

                    snap_targets = snap(param_list, snap_mode=snap_mode_whole, top=top)
                    subs_targets = [(Symbol(inverse_dict[replace_id]), Symbol(inverse_dict[ref_id]) * ratio) for (replace_id, ref_id), ratio in snap_targets]
                    prev_expression = self.symbolic_expression
                    new_expression = [expression.subs(subs_targets) for expression in self.symbolic_expression]
                    self.set_symbolic_expression(new_expression)
                    info_list = info_list + [(inverse_dict[replace_id], "{} * ".format(ratio) + inverse_dict[ref_id]) for (replace_id, ref_id), ratio in snap_targets]
                    if verbose > 0:
                        print("Original expression:\tsymbolic: {}; \t numerical: {}".format(prev_expression, self.numerical_expression))
                        print("Substitution:  \t{}".format(subs_targets))
                        print("New  expression: \tsymbolic: {}; \t numerical: {}".format(self.symbolic_expression, self.numerical_expression))
            else:
                raise Exception("mode {} not recognized!".format(mode_ele))
        return info_list


    def get_regularization(self, mode, source=["weight"], **kwargs):
        reg = to_Variable([0], is_cuda=self.is_cuda)
        if not isinstance(source, list):
            source = [source]
        param_list = [param for param in self.parameters()]
        if len(param_list) > 0 and "weight" in source:
            params = torch.cat(param_list)
            scale_factor = kwargs["reg_scale_factor"] if "reg_scale_factor" in kwargs else None
            if mode == "L1":
                if scale_factor is not None:
                    reg_indi = (params * to_Variable(scale_factor, is_cuda = self.is_cuda)).abs().sum()
                else:
                    reg_indi = params.abs().sum()
                reg = reg + reg_indi                        
            elif mode == "L2":
                if scale_factor is not None:
                    reg_indi = torch.sum((params * to_Variable(scale_factor, is_cuda = self.is_cuda)) ** 2)
                else:
                    reg_indi = torch.sum(params ** 2)
                reg = reg + reg_indi
        return reg


# ## SuperNet Layer:

# In[5]:


class SuperNet_Layer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        W_init=None,     # initialization for weights
        b_init=None,     # initialization for bias
        settings={},
        is_cuda=False,
        ):
        super(SuperNet_Layer, self).__init__()
        # Saving the attribuites:
        if isinstance(input_size, tuple):
            self.input_size = reduce(lambda x, y: x * y, input_size)
            self.input_size_original = input_size
        else:
            self.input_size = input_size
        if isinstance(output_size, tuple):
            self.output_size = reduce(lambda x, y: x * y, output_size)
            self.output_size_original = output_size
        else:
            self.output_size = output_size
        self.W_init = W_init
        self.b_init = b_init
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")
        
        # Obtain additional initialization settings if provided:
        self.W_available = settings["W_available"] if "W_available" in settings else ["dense", "Toeplitz"]
        self.b_available = settings["b_available"] if "b_available" in settings else ["dense", "None"]
        self.A_available = settings["A_available"] if "A_available" in settings else ["linear", "relu"]
        self.W_sig_init  = settings["W_sig_init"] if "W_sig_init" in settings else None # initialization for the significance for the weights
        self.b_sig_init  = settings["b_sig_init"] if "b_sig_init" in settings else None # initialization for the significance for the bias
        self.A_sig_init  = settings["A_sig_init"] if "A_sig_init" in settings else None # initialization for the significance for the activations
        for W_candidate in self.W_available:
            if "2D-in" in W_candidate:
                self.input_size_2D = settings["input_size_2D"]
            if "2D-out" in W_candidate:
                self.output_size_2D = settings["output_size_2D"]
        for b_candidate in self.b_available:
            if "2D" in b_candidate:
                self.output_size_2D = settings["output_size_2D"]
        
        # Initialize layer:
        self.init_layer()
        self.set_cuda(is_cuda)
    
    
    @property
    def settings(self):
        layer_settings = {}
        layer_settings["W_available"] = deepcopy(self.W_available)
        layer_settings["b_available"] = deepcopy(self.b_available)
        layer_settings["A_available"] = deepcopy(self.A_available)
        layer_settings["W_sig_init"] = to_np_array(self.W_sig)
        layer_settings["b_sig_init"] = to_np_array(self.b_sig)
        layer_settings["A_sig_init"] = to_np_array(self.A_sig)
        return layer_settings
    
    @property
    def struct_param(self):
        return [self.output_size, "SuperNet_Layer", self.settings]

        
    def init_layer(self):
        self.W_layer_seed = nn.Parameter(torch.FloatTensor(np.random.randn(self.input_size, self.output_size)))
        self.b_layer_seed = nn.Parameter(torch.zeros(self.output_size))
        init_weight(self.W_layer_seed, init = self.W_init)
        init_bias(self.b_layer_seed, init = self.b_init)
        if "arithmetic-series-in" in self.W_available:
            self.W_interval_j = nn.Parameter(torch.randn(self.output_size) / np.sqrt(self.input_size + self.output_size))
        if "arithmetic-series-out" in self.W_available:
            self.W_interval_i = nn.Parameter(torch.randn(self.input_size) / np.sqrt(self.input_size + self.output_size))
        if "arithmetic-series-2D-in" in self.W_available:
            self.W_mean_2D_in = nn.Parameter(torch.randn(self.output_size) / np.sqrt(self.input_size_2D[0] + self.input_size_2D[1] + self.output_size))
            self.W_interval_2D_in = nn.Parameter(torch.randn(2, self.output_size) / np.sqrt(self.input_size_2D[0] + self.input_size_2D[1] + self.output_size))
        if "arithmetic-series-2D-out" in self.W_available:
            self.W_mean_2D_out = nn.Parameter(torch.randn(self.input_size) / np.sqrt(self.input_size + self.output_size_2D[0] + self.output_size_2D[1]))
            self.W_interval_2D_out = nn.Parameter(torch.randn(2, self.input_size) / np.sqrt(self.input_size + self.output_size_2D[0] + self.output_size_2D[1]))
        if "arithmetic-series" in self.b_available:
            self.b_interval = nn.Parameter(torch.randn(1) / np.sqrt(self.output_size))
        if "arithmetic-series-2D" in self.b_available:
            self.b_mean_2D = nn.Parameter(torch.randn(1) / np.sqrt(self.output_size))
            self.b_interval_2D = nn.Parameter(torch.randn(2) / np.sqrt(self.output_size_2D[0] + self.output_size_2D[1]))
        
        if self.W_sig_init is None:
            self.W_sig = nn.Parameter(torch.zeros(len(self.W_available)))
        else:
            self.W_sig = nn.Parameter(torch.FloatTensor(self.W_sig_init))
        if self.b_sig_init is None:
            self.b_sig = nn.Parameter(torch.zeros(len(self.b_available)))
        else:
            self.b_sig = nn.Parameter(torch.FloatTensor(self.b_sig_init))
        if self.A_sig_init is None:
            self.A_sig = nn.Parameter(torch.zeros(len(self.A_available)))
        else:
            self.A_sig = nn.Parameter(torch.FloatTensor(self.A_sig_init))


    def get_layers(self, source=["weight", "bias"]):
        """All the different SuperNet layers are based on the same W_seed matrices. 
        For example, W_seed is based on the full self.W_layer_seed; "Toeplitz" is based on
        the first row and first column of self.W_layer_seed to construct the Toeplitz matrix, etc.
        """
        # Superimpose different weights:
        if "weight" in source:
            self.W_list = []
            for weight_type in self.W_available:
                if weight_type == "dense":
                    W_layer = self.W_layer_seed
                elif weight_type == "Toeplitz":
                    W_layer_stacked = []
                    if self.output_size > 1:
                        inv_idx = torch.arange(self.output_size - 1, 0, -1).long().to(self.device)
                        W_seed = torch.cat([self.W_layer_seed[0][inv_idx], self.W_layer_seed[:,0]])
                    else:
                        W_seed = self.W_layer_seed[:,0]
                    for j in range(self.output_size):
                        W_layer_stacked.append(W_seed[self.output_size - j - 1: self.output_size - j - 1 + self.input_size])
                    W_layer = torch.stack(W_layer_stacked, 1)
                elif weight_type == "arithmetic-series-in":
                    mean_j = self.W_layer_seed.mean(0)
                    idx_i = torch.FloatTensor(np.repeat(np.arange(self.input_size), self.output_size)).to(self.device)
                    idx_j = torch.LongTensor(range(self.output_size) * self.input_size).to(self.device)
                    offset = self.input_size / float(2) - 0.5
                    W_layer = (mean_j[idx_j] + self.W_interval_j[idx_j] * Variable(idx_i - offset, requires_grad = False)).view(self.input_size, self.output_size)
                elif weight_type == "arithmetic-series-out":
                    mean_i = self.W_layer_seed.mean(1)
                    idx_i = torch.LongTensor(np.repeat(np.arange(self.input_size), self.output_size)).to(self.device)
                    idx_j = torch.FloatTensor(range(self.output_size) * self.input_size).to(self.device)
                    offset = self.output_size / float(2) - 0.5
                    W_layer = (mean_i[idx_i] + self.W_interval_i[idx_i] * Variable(idx_j - offset, requires_grad = False)).view(self.input_size, self.output_size)
                elif weight_type == "arithmetic-series-2D-in":
                    idx_i, idx_j, idx_k = np.meshgrid(range(self.input_size_2D[0]), range(self.input_size_2D[1]), range(self.output_size), indexing = "ij")
                    idx_i = torch.from_numpy(idx_i).float().view(-1).to(self.device)
                    idx_j = torch.from_numpy(idx_j).float().view(-1).to(self.device)
                    idx_k = torch.from_numpy(idx_k).long().view(-1).to(self.device)
                    offset_i = self.input_size_2D[0] / float(2) - 0.5
                    offset_j = self.input_size_2D[1] / float(2) - 0.5
                    W_layer = (self.W_mean_2D_in[idx_k] +                                self.W_interval_2D_in[:, idx_k][0] * Variable(idx_i - offset_i, requires_grad = False) +                                self.W_interval_2D_in[:, idx_k][1] * Variable(idx_j - offset_j, requires_grad = False)).view(self.input_size, self.output_size)
                elif weight_type == "arithmetic-series-2D-out":
                    idx_k, idx_i, idx_j = np.meshgrid(range(self.input_size), range(self.output_size_2D[0]), range(self.output_size_2D[1]), indexing = "ij")
                    idx_k = torch.from_numpy(idx_k).long().view(-1).to(self.device)
                    idx_i = torch.from_numpy(idx_i).float().view(-1).to(self.device)
                    idx_j = torch.from_numpy(idx_j).float().view(-1).to(self.device)
                    offset_i = self.output_size_2D[0] / float(2) - 0.5
                    offset_j = self.output_size_2D[1] / float(2) - 0.5
                    W_layer = (self.W_mean_2D_out[idx_k] +                                self.W_interval_2D_out[:, idx_k][0] * Variable(idx_i - offset_i, requires_grad = False) +                                self.W_interval_2D_out[:, idx_k][1] * Variable(idx_j - offset_j, requires_grad = False)).view(self.input_size, self.output_size)
                else:
                    raise Exception("weight_type '{0}' not recognized!".format(weight_type))
                self.W_list.append(W_layer)

            if len(self.W_available) == 1:
                self.W_core = W_layer
            else:
                self.W_list = torch.stack(self.W_list, dim = 2)
                W_sig_softmax = nn.Softmax(dim = -1)(self.W_sig.unsqueeze(0))
                self.W_core = torch.matmul(self.W_list, W_sig_softmax.transpose(1,0)).squeeze(2)
    
        # Superimpose different biases:
        if "bias" in source:
            self.b_list = []
            for bias_type in self.b_available:
                if bias_type == "None":
                    b_layer = Variable(torch.zeros(self.output_size).to(self.device), requires_grad = False)
                elif bias_type == "constant":
                    b_layer = self.b_layer_seed[0].repeat(self.output_size)
                elif bias_type == "arithmetic-series":
                    mean = self.b_layer_seed.mean()
                    offset = self.output_size / float(2) - 0.5
                    idx = Variable(torch.FloatTensor(range(self.output_size)).to(self.device), requires_grad = False)
                    b_layer = mean + self.b_interval * (idx - offset)
                elif bias_type == "arithmetic-series-2D":
                    idx_i, idx_j = np.meshgrid(range(self.output_size_2D[0]), range(self.output_size_2D[1]), indexing = "ij")
                    idx_i = torch.from_numpy(idx_i).float().view(-1).to(self.device)
                    idx_j = torch.from_numpy(idx_j).float().view(-1).to(self.device)
                    offset_i = self.output_size_2D[0] / float(2) - 0.5
                    offset_j = self.output_size_2D[1] / float(2) - 0.5
                    b_layer = (self.b_mean_2D +                                self.b_interval_2D[0] * Variable(idx_i - offset_i, requires_grad = False) +                                self.b_interval_2D[1] * Variable(idx_j - offset_j, requires_grad = False)).view(-1)
                elif bias_type == "dense":
                    b_layer = self.b_layer_seed
                else:
                    raise Exception("bias_type '{0}' not recognized!".format(bias_type))
                self.b_list.append(b_layer)

            if len(self.b_available) == 1:
                self.b_core = b_layer
            else:
                self.b_list = torch.stack(self.b_list, dim = 1)
                b_sig_softmax = nn.Softmax(dim = -1)(self.b_sig.unsqueeze(0))
                self.b_core = torch.matmul(self.b_list, b_sig_softmax.transpose(1,0)).squeeze(1)


    def forward(self, X, p_dict=None):
        del p_dict
        output = X
        if hasattr(self, "input_size_original"):
            output = output.view(-1, self.input_size)
        # Get superposition of layers:
        self.get_layers(source = ["weight", "bias"])

        # Perform dot(X, W) + b:
        output = torch.matmul(output, self.W_core) + self.b_core
        
        # Exert superposition of activation functions:
        if len(self.A_available) == 1:
            output = get_activation(self.A_available[0])(output)
        else:
            self.A_list = []
            A_sig_softmax = nn.Softmax(dim = -1)(self.A_sig.unsqueeze(0))
            for i, activation in enumerate(self.A_available):
                A = get_activation(activation)(output)
                self.A_list.append(A)
            self.A_list = torch.stack(self.A_list, 2)
            output = torch.matmul(self.A_list, A_sig_softmax.transpose(1,0)).squeeze(2)

        if hasattr(self, "output_size_original"):
            output = output.view(*((-1,) + self.output_size_original))
        return output
    
    
    def get_param_names(self, source):
        if source == "modules":
            param_names = ["W_layer_seed", "b_layer_seed"]
            if "arithmetic-series-in" in self.W_available:
                param_names.append("W_interval_j")
            if "arithmetic-series-out" in self.W_available:
                param_names.append("W_interval_i")
            if "arithmetic-series-2D-in" in self.W_available:
                param_names = param_names + ["W_mean_2D_in", "W_interval_2D_in"]
            if "arithmetic-series-2D-out" in self.W_available:
                param_names = param_names + ["W_mean_2D_out", "W_interval_2D_out"]
            if "arithmetic-series" in self.b_available:
                param_names.append("b_interval")
        if source == "attention":
            param_names = ["W_sig", "b_sig", "A_sig"]
        return param_names
    
    
    def get_weights_bias(self):
        self.get_layers(source = ["weight", "bias"])
        return to_np_array(self.W_layer_seed), to_np_array(self.b_layer_seed)


    def get_regularization(self, mode, source = ["weight", "bias"]):
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad = False).to(self.device)
        if not isinstance(source, list):
            source = [source]
        if mode == "L1":
            if "weight" in source:
                reg = reg + self.W_core.abs().sum()
            if "bias" in source:
                reg = reg + self.b_core.abs().sum()
        elif mode == "layer_L1":
            if "weight" in source:
                self.get_layers(source = ["weight"])
                reg = reg + self.W_list.abs().sum()
            if "bias" in source:
                self.get_layers(source = ["bias"])
                reg = reg + self.b_list.abs().sum()
        elif mode == "L2":
            if "weight" in source:
                reg = reg + torch.sum(self.W_core ** 2)
            if "bias" in source:
                reg = reg + torch.sum(self.b_core ** 2)
        elif mode == "S_entropy":
            if "weight" in source:
                W_sig_softmax = nn.Softmax(dim = -1)(self.W_sig.unsqueeze(0))
                reg = reg - torch.sum(W_sig_softmax * torch.log(W_sig_softmax))
            if "bias" in source:
                b_sig_softmax = nn.Softmax(dim = -1)(self.b_sig.unsqueeze(0))
                reg = reg - torch.sum(b_sig_softmax * torch.log(b_sig_softmax))
        elif mode == "S_entropy_activation":
            A_sig_softmax = nn.Softmax(dim = -1)(self.A_sig.unsqueeze(0))
            reg = reg - torch.sum(A_sig_softmax * torch.log(A_sig_softmax))
        elif mode in AVAILABLE_REG:
            pass
        else:
            raise Exception("mode '{}' not recognized!".format(mode))
        return reg


    def set_cuda(self, is_cuda):
        if isinstance(is_cuda, str):
            self.cuda(is_cuda)
        else:
            if is_cuda:
                self.cuda()
            else:
                self.cpu()
        self.is_cuda = is_cuda

