#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import numpy as np
import pprint as pp
from copy import deepcopy
import pickle
from numbers import Number
from collections import OrderedDict
import itertools
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch_net.modules import get_Layer, load_layer_dict, Simple_2_Symbolic
from pytorch_net.util import forward, get_epochs_T_mult, Loss_Fun, get_activation, get_criterion, get_criteria_value, get_optimizer, get_full_struct_param, plot_matrices, get_model_DL, PrecisionFloorLoss, get_list_DL, init_weight
from pytorch_net.util import Early_Stopping, Performance_Monitor, record_data, to_np_array, to_Variable, make_dir, formalize_value, RampupLR, Transform_Label, view_item, load_model, save_model, to_cpu_recur, filter_kwargs


# ## Training functionality:

# In[ ]:


def train(
    model,
    X=None,
    y=None,
    train_loader=None,
    validation_data=None,
    validation_loader=None,
    criterion=nn.MSELoss(),
    inspect_interval=10,
    isplot=False,
    is_cuda=None,
    **kwargs
    ):
    """Training function for generic models. "model" can be a single model or a ordered list of models"""
    def get_regularization(model, loss_epoch, **kwargs):
        """Compute regularization."""
        reg_dict = kwargs["reg_dict"] if "reg_dict" in kwargs else None
        reg = to_Variable([0], is_cuda = is_cuda)
        if reg_dict is not None:
            for reg_type, reg_coeff in reg_dict.items():
                # Setting up regularization strength:
                if isinstance(reg_coeff, Number):
                    reg_coeff_ele = reg_coeff
                else:
                    if loss_epoch < len(reg_coeff):
                        reg_coeff_ele = reg_coeff[loss_epoch]
                    else:
                        reg_coeff_ele = reg_coeff[-1]
                # Accumulate regularization:
                reg = reg + model.get_regularization(source=[reg_type], mode=reg_mode, **kwargs) * reg_coeff_ele
        return reg

    if is_cuda is None:
        if X is None and y is None:
            assert train_loader is not None
            is_cuda = train_loader.dataset.tensors[0].is_cuda
        else:
            is_cuda = X.is_cuda

    # Optimization kwargs:
    epochs = kwargs["epochs"] if "epochs" in kwargs else 10000
    lr = kwargs["lr"] if "lr" in kwargs else 5e-3
    lr_rampup_steps = kwargs["lr_rampup"] if "lr_rampup" in kwargs else 200
    optim_type = kwargs["optim_type"] if "optim_type" in kwargs else "adam"
    optim_kwargs = kwargs["optim_kwargs"] if "optim_kwargs" in kwargs else {}
    scheduler_type = kwargs["scheduler_type"] if "scheduler_type" in kwargs else "ReduceLROnPlateau"
    gradient_noise = kwargs["gradient_noise"] if "gradient_noise" in kwargs else None
    data_loader_apply = kwargs["data_loader_apply"] if "data_loader_apply" in kwargs else None

    # Inspection kwargs:
    inspect_step = kwargs["inspect_step"] if "inspect_step" in kwargs else None  # Whether to inspect each step
    inspect_items = kwargs["inspect_items"] if "inspect_items" in kwargs else None
    inspect_items_train = get_inspect_items_train(inspect_items)
    inspect_functions = kwargs["inspect_functions"] if "inspect_functions" in kwargs else None
    if inspect_functions is not None:
        for inspect_function_key in inspect_functions:
            if inspect_function_key not in inspect_items:
                inspect_items.append(inspect_function_key)
    inspect_items_interval = kwargs["inspect_items_interval"] if "inspect_items_interval" in kwargs else 1000
    inspect_image_interval = kwargs["inspect_image_interval"] if "inspect_image_interval" in kwargs else None
    inspect_loss_precision = kwargs["inspect_loss_precision"] if "inspect_loss_precision" in kwargs else 4
    callback = kwargs["callback"] if "callback" in kwargs else None

    # Saving kwargs:
    record_keys = kwargs["record_keys"] if "record_keys" in kwargs else ["loss"]
    filename = kwargs["filename"] if "filename" in kwargs else None
    if filename is not None:
        make_dir(filename)
    save_interval = kwargs["save_interval"] if "save_interval" in kwargs else None
    save_step = kwargs["save_step"] if "save_step" in kwargs else None
    logdir = kwargs["logdir"] if "logdir" in kwargs else None
    data_record = {key: [] for key in record_keys}
    info_to_save = kwargs["info_to_save"] if "info_to_save" in kwargs else None
    if info_to_save is not None:
        data_record.update(info_to_save)
    patience = kwargs["patience"] if "patience" in kwargs else 20
    if patience is not None:
        early_stopping_epsilon = kwargs["early_stopping_epsilon"] if "early_stopping_epsilon" in kwargs else 0
        early_stopping_monitor = kwargs["early_stopping_monitor"] if "early_stopping_monitor" in kwargs else "loss"
        early_stopping = Early_Stopping(patience = patience, epsilon = early_stopping_epsilon, mode = "max" if early_stopping_monitor in ["accuracy"] else "min")
    if logdir is not None:
        from pytorch_net.logger import Logger
        batch_idx = 0
        logger = Logger(logdir)
    logimages = kwargs["logimages"] if "logimages" in kwargs else None
    reg_mode = kwargs["reg_mode"] if "reg_mode" in kwargs else "L1"

    if validation_loader is not None:
        assert validation_data is None
        X_valid, y_valid = None, None
    elif validation_data is not None:
        X_valid, y_valid = validation_data
    else:
        X_valid, y_valid = X, y

    # Setting up dynamic label noise:
    label_noise_matrix = kwargs["label_noise_matrix"] if "label_noise_matrix" in kwargs else None
    transform_label = Transform_Label(label_noise_matrix = label_noise_matrix, is_cuda=is_cuda)

    # Setting up cotrain optimizer:
    co_kwargs = kwargs["co_kwargs"] if "co_kwargs" in kwargs else None
    if co_kwargs is not None:
        co_optimizer = co_kwargs["co_optimizer"]
        co_model = co_kwargs["co_model"]
        co_criterion = co_kwargs["co_criterion"] if "co_criterion" in co_kwargs else None
        co_multi_step = co_kwargs["co_multi_step"] if "co_multi_step" in co_kwargs else 1

    # Get original loss:
    if len(inspect_items_train) > 0:
        loss_value_train = get_loss(model, train_loader, X, y, criterion=criterion, loss_epoch=-1, transform_label=transform_label, **kwargs)
        info_dict_train = prepare_inspection(model, train_loader, X, y, transform_label=transform_label, **kwargs)
        if "loss" in record_keys:
            record_data(data_record, [loss_value_train], ["loss_tr"])
    loss_original = get_loss(model, validation_loader, X_valid, y_valid, criterion=criterion, loss_epoch=-1, transform_label=transform_label, **kwargs)
    if "loss" in record_keys:
        record_data(data_record, [-1, loss_original], ["iter", "loss"])
    if "reg" in record_keys and "reg_dict" in kwargs and len(kwargs["reg_dict"]) > 0:
        reg_value = get_regularization(model, loss_epoch=0, **kwargs)
        record_data(data_record, [reg_value], ["reg"])
    if "param" in record_keys:
        record_data(data_record, [model.get_weights_bias(W_source="core", b_source="core")], ["param"])
    if "param_grad" in record_keys:
        record_data(data_record, [model.get_weights_bias(W_source="core", b_source="core", is_grad=True)], ["param_grad"])
    if co_kwargs is not None:
        co_loss_original = get_loss(co_model, validation_loader, X_valid, y_valid, criterion=criterion, loss_epoch=-1, transform_label=transform_label, **co_kwargs)
        if "co_loss" in record_keys:
            record_data(data_record, [co_loss_original], ["co_loss"])
    if filename is not None and save_interval is not None:
        record_data(data_record, [{}], ["model_dict"])

    # Setting up optimizer:
    parameters = model.parameters()
    num_params = len(list(model.parameters()))
    if num_params == 0:
        print("No parameters to optimize!")
        loss_value = get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion, loss_epoch = -1, transform_label=transform_label, **kwargs)
        if "loss" in record_keys:
            record_data(data_record, [0, loss_value], ["iter", "loss"])
        if "param" in record_keys:
            record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core")], ["param"])
        if "param_grad" in record_keys:
            record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
        if co_kwargs is not None:
            co_loss_value = get_loss(co_model, validation_loader, X_valid, y_valid, criterion = criterion, loss_epoch = -1, transform_label=transform_label, **co_kwargs)
            record_data(data_record, [co_loss_value], ["co_loss"])
        return loss_original, loss_value, data_record
    optimizer = get_optimizer(optim_type, lr, parameters, **optim_kwargs) if "optimizer" not in kwargs or ("optimizer" in kwargs and kwargs["optimizer"] is None) else kwargs["optimizer"]

    # Initialize inspect_items:
    if inspect_items is not None:
        print("{}:".format(-1), end = "")
        print("\tlr: {0:.3e}\t loss:{1:.{2}f}".format(optimizer.param_groups[0]["lr"], loss_original, inspect_loss_precision), end = "")
        info_dict = prepare_inspection(model, validation_loader, X_valid, y_valid, transform_label=transform_label, **kwargs)
        if len(inspect_items_train) > 0:
            print("\tloss_tr: {0:.{1}f}".format(loss_value_train, inspect_loss_precision), end = "")
            info_dict_train = update_key_train(info_dict_train, inspect_items_train)
            info_dict.update(info_dict_train)
        if "reg" in record_keys and "reg_dict" in kwargs and len(kwargs["reg_dict"]) > 0:
            print("\treg:{0:.{1}f}".format(to_np_array(reg_value), inspect_loss_precision), end="")
        if len(info_dict) > 0:
            for item in inspect_items:
                if item in info_dict:
                    print(" \t{0}: {1:.{2}f}".format(item, info_dict[item], inspect_loss_precision), end = "")
                    if item in record_keys and item not in ["loss", "reg"]:
                        record_data(data_record, [to_np_array(info_dict[item])], [item])

        if co_kwargs is not None:
            co_info_dict = prepare_inspection(co_model, validation_loader, X_valid, y_valid, transform_label=transform_label, **co_kwargs)
            if "co_loss" in inspect_items:
                co_loss_value = get_loss(co_model, validation_loader, X_valid, y_valid, criterion=criterion, loss_epoch=-1, transform_label=transform_label, **co_kwargs)
                print("\tco_loss: {}".format(formalize_value(co_loss_value, inspect_loss_precision)), end="")
            if len(co_info_dict) > 0:
                for item in inspect_items:
                    if item in co_info_dict:
                        print(" \t{0}: {1}".format(item, formalize_value(co_info_dict[item], inspect_loss_precision)), end="")
                        if item in record_keys and item != "loss":
                            record_data(data_record, [to_np_array(co_info_dict[item])], [item])
        print("\n")

    # Setting up gradient noise:
    if gradient_noise is not None:
        from pytorch_net.util import Gradient_Noise_Scale_Gen
        scale_gen = Gradient_Noise_Scale_Gen(epochs=epochs,
                                             gamma=gradient_noise["gamma"],  # decay rate
                                             eta=gradient_noise["eta"],      # starting variance
                                             gradient_noise_interval_epoch=1,
                                            )
        gradient_noise_scale = scale_gen.generate_scale(verbose=True)

    # Set up learning rate scheduler:
    if scheduler_type is not None:
        if scheduler_type == "ReduceLROnPlateau":
            scheduler_patience = kwargs["scheduler_patience"] if "scheduler_patience" in kwargs else 40
            scheduler_factor = kwargs["scheduler_factor"] if "scheduler_factor" in kwargs else 0.1
            scheduler_verbose = kwargs["scheduler_verbose"] if "scheduler_verbose" in kwargs else False
            scheduler = ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience, verbose=scheduler_verbose)
        elif scheduler_type == "LambdaLR":
            scheduler_lr_lambda = kwargs["scheduler_lr_lambda"] if "scheduler_lr_lambda" in kwargs else (lambda epoch: 0.97 ** (epoch // 2))
            scheduler = LambdaLR(optimizer, lr_lambda=scheduler_lr_lambda)
        elif scheduler_type == "cos":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == "coslr":
            T_0 = max(min(25, epochs//31), 1)
            T_mult = kwargs["scheduler_T_mult"] if "scheduler_T_mult" in kwargs else 2
            epochs = get_epochs_T_mul(epochs, T_0=T_0, T_mult=T_mult)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
        else:
            raise
    # Ramping or learning rate for the first lr_rampup_steps steps:
    if lr_rampup_steps is not None and train_loader is not None:
        scheduler_rampup = RampupLR(optimizer, num_steps=lr_rampup_steps)
        if hasattr(train_loader, "dataset"):
            data_size = len(train_loader.dataset)
        else:
            data_size = kwargs["data_size"]

    # Initialize logdir:
    if logdir is not None:
        if logimages is not None:
            for tag, image_fun in logimages["image_fun"].items():
                image = image_fun(model, logimages["X"], logimages["y"])
                logger.log_images(tag, image, -1)

    # Training:
    to_stop = False

    for i in range(epochs + 1):
        model.train()

        # Updating gradient noise:
        if gradient_noise is not None:
            hook_handle_list = []
            if i % scale_gen.gradient_noise_interval_epoch == 0:
                for h in hook_handle_list:
                    h.remove()
                hook_handle_list = []
                scale_idx = int(i / scale_gen.gradient_noise_interval_epoch)
                if scale_idx >= len(gradient_noise_scale):
                    current_gradient_noise_scale = gradient_noise_scale[-1]
                else:
                    current_gradient_noise_scale = gradient_noise_scale[scale_idx]
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.requires_grad:
                            h = param.register_hook(lambda grad: grad + Variable(torch.normal(mean=torch.zeros(grad.size()),
                                                                                              std=current_gradient_noise_scale * torch.ones(grad.size()))))
                            hook_handle_list.append(h)

        if X is not None and y is not None:
            if optim_type != "LBFGS":
                optimizer.zero_grad()
                reg = get_regularization(model, loss_epoch=i, **kwargs)
                loss = model.get_loss(X, transform_label(y), criterion=criterion, loss_epoch=i, **kwargs) + reg
                loss.backward()
                optimizer.step()
            else:
                # "LBFGS" is a second-order optimization algorithm that requires a slightly different procedure:
                def closure():
                    optimizer.zero_grad()
                    reg = get_regularization(model, loss_epoch=i, **kwargs)
                    loss = model.get_loss(X, transform_label(y), criterion=criterion, loss_epoch=i, **kwargs) + reg
                    loss.backward()
                    return loss
                optimizer.step(closure)
            
            # Cotrain step:
            if co_kwargs is not None:
                if "co_warmup_epochs" not in co_kwargs or "co_warmup_epochs" in co_kwargs and i >= co_kwargs["co_warmup_epochs"]:
                    for _ in range(co_multi_step):
                        co_optimizer.zero_grad()
                        co_reg = get_regularization(co_model, loss_epoch=i, **co_kwargs)
                        co_loss = co_model.get_loss(X, transform_label(y), criterion=co_criterion, loss_epoch=i, **co_kwargs) + co_reg
                        co_loss.backward()
                        co_optimizer.step()
        else:
            if inspect_step is not None:
                info_dict_step = {key: [] for key in inspect_items}

            if "loader_process" in kwargs and kwargs["loader_process"] is not None:
                train_loader = kwargs["loader_process"]("train")
            for k, data_batch in enumerate(train_loader):
                if isinstance(data_batch, tuple) or isinstance(data_batch, list):
                    X_batch, y_batch = data_batch
                    if data_loader_apply is not None:
                        X_batch, y_batch = data_loader_apply(X_batch, y_batch)
                else:
                    X_batch, y_batch = data_loader_apply(data_batch)
                if optim_type != "LBFGS":
                    optimizer.zero_grad()
                    reg = get_regularization(model, loss_epoch=i, **kwargs)
                    loss = model.get_loss(X_batch, transform_label(y_batch), criterion=criterion, loss_epoch=i, loss_step=k, **kwargs) + reg
                    loss.backward()
                    if logdir is not None:
                        batch_idx += 1
                        if len(model.info_dict) > 0:
                            for item in inspect_items:
                                if item in model.info_dict:
                                    logger.log_scalar(item, model.info_dict[item], batch_idx)
                    optimizer.step()
                else:
                    def closure():
                        optimizer.zero_grad()
                        reg = get_regularization(model, loss_epoch=i, **kwargs)
                        loss = model.get_loss(X_batch, transform_label(y_batch), criterion=criterion, loss_epoch=i, loss_step=k, **kwargs) + reg
                        loss.backward()
                        return loss
                    if logdir is not None:
                        batch_idx += 1
                        if len(model.info_dict) > 0:
                            for item in inspect_items:
                                if item in model.info_dict:
                                    logger.log_scalar(item, model.info_dict[item], batch_idx)
                    optimizer.step(closure)

                # Rampup scheduler:
                if lr_rampup_steps is not None and i * data_size // len(X_batch) + k < lr_rampup_steps:
                    scheduler_rampup.step()

                # Cotrain step:
                if co_kwargs is not None:
                    if "co_warmup_epochs" not in co_kwargs or "co_warmup_epochs" in co_kwargs and i >= co_kwargs["co_warmup_epochs"]:
                        for _ in range(co_multi_step):
                            co_optimizer.zero_grad()
                            co_reg = get_regularization(co_model, loss_epoch=i, **co_kwargs)
                            co_loss = co_model.get_loss(X_batch, transform_label(y_batch), criterion=co_criterion, loss_epoch=i, loss_step=k, **co_kwargs) + co_reg
                            co_loss.backward()
                            if logdir is not None:
                                if len(co_model.info_dict) > 0:
                                    for item in inspect_items:
                                        if item in co_model.info_dict:
                                            logger.log_scalar(item, co_model.info_dict[item], batch_idx)
                            co_optimizer.step()

                # Inspect at each step:
                if inspect_step is not None:
                    if k % inspect_step == 0:
                        print("s{}:".format(k), end = "")
                        info_dict = prepare_inspection(model, validation_loader, X_valid, y_valid, transform_label=transform_label, **kwargs) 
                        if "loss" in inspect_items:
                            info_dict_step["loss"].append(loss.item())
                            print("\tloss: {0:.{1}f}".format(loss.item(), inspect_loss_precision), end="")
                        if len(info_dict) > 0:
                            for item in inspect_items:
                                if item in info_dict:
                                    info_dict_step[item].append(info_dict[item])
                                    print(" \t{0}: {1}".format(item, formalize_value(info_dict[item], inspect_loss_precision)), end = "")
                        if co_kwargs is not None:
                            if "co_warmup_epochs" not in co_kwargs or "co_warmup_epochs" in co_kwargs and i >= co_kwargs["co_warmup_epochs"]:
                                co_info_dict = prepare_inspection(co_model, validation_loader, X_valid, y_valid, transform_label=transform_label, **co_kwargs)
                                if "co_loss" in inspect_items:
                                    print("\tco_loss: {0:.{1}f}".format(co_loss.item(), inspect_loss_precision), end="")
                                    info_dict_step["co_loss"].append(co_loss.item())
                                if len(co_info_dict) > 0:
                                    for item in inspect_items:
                                        if item in co_info_dict and item != "co_loss":
                                            info_dict_step[item].append(co_info_dict[item])
                                            print(" \t{0}: {1}".format(item, formalize_value(co_info_dict[item], inspect_loss_precision)), end="")
                        print()
                    if k % save_step == 0:
                        if filename is not None:
                            pickle.dump(model.model_dict, open(filename[:-2] + "_model.p", "wb"))

        if logdir is not None:
            # Log values and gradients of the parameters (histogram summary)
#             for tag, value in model.named_parameters():
#                 tag = tag.replace('.', '/')
#                 logger.log_histogram(tag, to_np_array(value), i)
#                 logger.log_histogram(tag + '/grad', to_np_array(value.grad), i)
            if logimages is not None:
                for tag, image_fun in logimages["image_fun"].items():
                    image = image_fun(model, logimages["X"], logimages["y"])
                    logger.log_images(tag, image, i)

        if i % inspect_interval == 0:
            model.eval()
            if inspect_items is not None and i % inspect_items_interval == 0 and len(inspect_items_train) > 0:
                loss_value_train = get_loss(model, train_loader, X, y, criterion = criterion, loss_epoch = i, transform_label=transform_label, **kwargs)
                info_dict_train = prepare_inspection(model, train_loader, X, y, transform_label=transform_label, **kwargs)
            loss_value = get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion, loss_epoch = i, transform_label=transform_label, **kwargs)
            reg_value = get_regularization(model, loss_epoch = i, **kwargs)
            if scheduler_type is not None:
                if lr_rampup_steps is None or train_loader is None or (lr_rampup_steps is not None and i * data_size // len(X_batch) + k >= lr_rampup_steps):
                    if scheduler_type == "ReduceLROnPlateau":
                        scheduler.step(loss_value)
                    else:
                        scheduler.step()
            if callback is not None:
                assert callable(callback)
                callback(model = model,
                         X = X_valid,
                         y = y_valid,
                         iteration = i,
                         loss = loss_value,
                        )
            if patience is not None:
                if early_stopping_monitor == "loss":
                    to_stop = early_stopping.monitor(loss_value)
                else:
                    info_dict = prepare_inspection(model, validation_loader, X_valid, y_valid, transform_label=transform_label, **kwargs)
                    to_stop = early_stopping.monitor(info_dict[early_stopping_monitor])
            if inspect_items is not None:
                if i % inspect_items_interval == 0:
                    # Get loss:
                    print("{}:".format(i), end = "")
                    print("\tlr: {0:.3e}\tloss: {1:.{2}f}".format(optimizer.param_groups[0]["lr"], loss_value, inspect_loss_precision), end = "")
                    info_dict = prepare_inspection(model, validation_loader, X_valid, y_valid, transform_label=transform_label, **kwargs)
                    if len(inspect_items_train) > 0:
                        print("\tloss_tr: {0:.{1}f}".format(loss_value_train, inspect_loss_precision), end = "")
                        info_dict_train = update_key_train(info_dict_train, inspect_items_train)
                        info_dict.update(info_dict_train)
                    if "reg" in inspect_items and "reg_dict" in kwargs and len(kwargs["reg_dict"]) > 0:
                        print("\treg:{0:.{1}f}".format(to_np_array(reg_value), inspect_loss_precision), end="")
                    
                    # Print and record:
                    if len(info_dict) > 0:
                        for item in inspect_items:
                            if item + "_val" in info_dict:
                                print(" \t{0}: {1}".format(item, formalize_value(info_dict[item + "_val"], inspect_loss_precision)), end = "")
                                if item in record_keys and item not in ["loss", "reg"]:
                                    record_data(data_record, [to_np_array(info_dict[item + "_val"])], [item])

                        # logger:
                        if logdir is not None:
                            for item in inspect_items:
                                if item + "_val" in info_dict:
                                    logger.log_scalar(item + "_val", info_dict[item + "_val"], i)

                    # Co_model:
                    if co_kwargs is not None:
                        co_loss_value = get_loss(co_model, validation_loader, X_valid, y_valid, criterion = criterion, loss_epoch = i, transform_label=transform_label, **co_kwargs)
                        co_info_dict = prepare_inspection(co_model, validation_loader, X_valid, y_valid, transform_label=transform_label, **co_kwargs)
                        if "co_loss" in inspect_items:
                            print("\tco_loss: {0:.{1}f}".format(co_loss_value, inspect_loss_precision), end="")
                        if len(co_info_dict) > 0:
                            for item in inspect_items:
                                if item + "_val" in co_info_dict:
                                    print(" \t{0}: {1}".format(item, formalize_value(co_info_dict[item + "_val"], inspect_loss_precision)), end="")
                                    if item in record_keys and item != "co_loss":
                                        record_data(data_record, [to_np_array(co_info_dict[item + "_val"])], [item])
                        if "co_loss" in record_keys:
                            record_data(data_record, [co_loss_value], ["co_loss"])

                    # Training metrics:
                    if inspect_step is not None:
                        for item in info_dict_step:
                            if len(info_dict_step[item]) > 0:
                                print(" \t{0}_s: {1}".format(item, formalize_value(np.mean(info_dict_step[item]), inspect_loss_precision)), end = "")
                                if item in record_keys and item != "loss":
                                    record_data(data_record, [np.mean(info_dict_step[item])], ["{}_s".format(item)])

                    # Record loss:
                    if "loss" in record_keys:
                        record_data(data_record, [i, loss_value], ["iter", "loss"])
                    if "reg" in record_keys and "reg_dict" in kwargs and len(kwargs["reg_dict"]) > 0:
                        record_data(data_record, [reg_value], ["reg"])
                    if "param" in record_keys:
                        record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core")], ["param"])
                    if "param_grad" in record_keys:
                        record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
                    print("\n")
                    try:
                        sys.stdout.flush()
                    except:
                        pass
            if isplot:
                if inspect_image_interval is not None and hasattr(model, "plot"):
                    if i % inspect_image_interval == 0:
                        if gradient_noise is not None:
                            print("gradient_noise: {0:.9f}".format(current_gradient_noise_scale))
                        plot_model(model, data_loader = validation_loader, X = X_valid, y = y_valid, transform_label=transform_label, data_loader_apply=data_loader_apply)
                if co_kwargs is not None and "inspect_image_interval" in co_kwargs and co_kwargs["inspect_image_interval"] and hasattr(co_model, "plot"):
                    if i % co_kwargs["inspect_image_interval"] == 0:
                        plot_model(co_model, data_loader = validation_loader, X = X_valid, y = y_valid, transform_label=transform_label, data_loader_apply=data_loader_apply)
        if save_interval is not None:
            if i % save_interval == 0:
                record_data(data_record, [model.model_dict], ["model_dict"])
                if co_kwargs is not None:
                    record_data(data_record, [co_model.model_dict], ["co_model_dict"])
                if filename is not None:
                    pickle.dump(data_record, open(filename, "wb"))
        if to_stop:
            break

    loss_value = get_loss(model, validation_loader, X_valid, y_valid, criterion=criterion, loss_epoch=epochs, transform_label=transform_label, **kwargs)
    if isplot:
        import matplotlib.pylab as plt
        for key, item in data_record.items():
            if isinstance(item, Number) or len(data_record["iter"]) != len(item):
                continue
            if key not in ["iter", "model_dict"]:
                if key in ["accuracy"]:
                    plt.figure(figsize = (8,6))
                    plt.plot(data_record["iter"], data_record[key])
                    plt.xlabel("epoch")
                    plt.ylabel(key)
                    plt.title(key)
                    plt.show()
                else:
                    plt.figure(figsize = (8,6))
                    plt.semilogy(data_record["iter"], data_record[key])
                    plt.xlabel("epoch")
                    plt.ylabel(key)
                    plt.title(key)
                    plt.show()
    return loss_original, loss_value, data_record


def train_simple(model, X, y, validation_data = None, inspect_interval = 5, **kwargs):
    """minimal version of training. "model" can be a single model or a ordered list of models"""
    def get_regularization(model, **kwargs):
        reg_dict = kwargs["reg_dict"] if "reg_dict" in kwargs else None
        reg = to_Variable([0], is_cuda = X.is_cuda)
        for model_ele in model:
            if reg_dict is not None:
                for reg_type, reg_coeff in reg_dict.items():
                    reg = reg + model_ele.get_regularization(source = [reg_type], mode = "L1", **kwargs) * reg_coeff
        return reg
    if not(isinstance(model, list) or isinstance(model, tuple)):
        model = [model]
    epochs = kwargs["epochs"] if "epochs" in kwargs else 2000
    lr = kwargs["lr"] if "lr" in kwargs else 5e-3
    optim_type = kwargs["optim_type"] if "optim_type" in kwargs else "adam"
    optim_kwargs = kwargs["optim_kwargs"] if "optim_kwargs" in kwargs else {}
    loss_type = kwargs["loss_type"] if "loss_type" in kwargs else "mse"
    early_stopping_epsilon = kwargs["early_stopping_epsilon"] if "early_stopping_epsilon" in kwargs else 0
    patience = kwargs["patience"] if "patience" in kwargs else 40
    record_keys = kwargs["record_keys"] if "record_keys" in kwargs else ["loss", "mse", "data_DL", "model_DL"]
    scheduler_type = kwargs["scheduler_type"] if "scheduler_type" in kwargs else "ReduceLROnPlateau"
    loss_precision_floor = kwargs["loss_precision_floor"] if "loss_precision_floor" in kwargs else PrecisionFloorLoss
    autoencoder = kwargs["autoencoder"] if "autoencoder" in kwargs else None
    data_record = {key: [] for key in record_keys}
    isplot = kwargs["isplot"] if "isplot" in kwargs else False
    if patience is not None:
        early_stopping = Early_Stopping(patience = patience, epsilon = early_stopping_epsilon)
    
    if validation_data is not None:
        X_valid, y_valid = validation_data
    else:
        X_valid, y_valid = X, y
    
    # Get original loss:
    criterion = get_criterion(loss_type, loss_precision_floor = loss_precision_floor)
    DL_criterion = Loss_Fun(core = "DLs", loss_precision_floor = loss_precision_floor, DL_sum = True)
    DL_criterion_absolute = Loss_Fun(core = "DLs", loss_precision_floor = PrecisionFloorLoss, DL_sum = True)
    pred_valid = forward(model, X_valid, **kwargs)
    loss_original = to_np_array(criterion(pred_valid, y_valid))
    if "loss" in record_keys:
        record_data(data_record, [-1, loss_original], ["iter","loss"])
    if "mse" in record_keys:
        record_data(data_record, [to_np_array(nn.MSELoss()(pred_valid, y_valid))], ["mse"])
    if "data_DL" in record_keys:
        record_data(data_record, [to_np_array(DL_criterion(pred_valid, y_valid))], ["data_DL"])
    if "data_DL_absolute" in record_keys:
        record_data(data_record, [to_np_array(DL_criterion_absolute(pred_valid, y_valid))], ["data_DL_absolute"])
    if "model_DL" in record_keys:
        record_data(data_record, [get_model_DL(model)], ["model_DL"])
    if "param" in record_keys:
        record_data(data_record, [model[0].get_weights_bias(W_source = "core", b_source = "core")], ["param"])
    if "param_grad" in record_keys:
        record_data(data_record, [model[0].get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
    if "param_collapse_layers" in record_keys:
        record_data(data_record, [simplify(deepcopy(model[0]), X, y, "collapse_layers", verbose = 0)[0]                                  .get_weights_bias(W_source = "core", b_source = "core")], ["param"])

    # Setting up optimizer:
    parameters = itertools.chain(*[model_ele.parameters() for model_ele in model])
    num_params = np.sum([[len(list(model_ele.parameters())) for model_ele in model]])
    if num_params == 0:
        print("No parameters to optimize!")
        pred_valid = forward(model, X_valid, **kwargs)
        loss_value = to_np_array(criterion(pred_valid, y_valid))
        if "loss" in record_keys:
            record_data(data_record, [0, loss_value], ["iter", "loss"])
        if "mse" in record_keys:
            record_data(data_record, [to_np_array(nn.MSELoss()(pred_valid, y_valid))], ["mse"])
        if "data_DL" in record_keys:
            record_data(data_record, [to_np_array(DL_criterion(pred_valid, y_valid))], ["data_DL"])
        if "data_DL_absolute" in record_keys:
            record_data(data_record, [to_np_array(DL_criterion_absolute(pred_valid, y_valid))], ["data_DL_absolute"])
        if "model_DL" in record_keys:
            record_data(data_record, [get_model_DL(model)], ["model_DL"])
        if "param" in record_keys:
            record_data(data_record, [model[0].get_weights_bias(W_source = "core", b_source = "core")], ["param"])
        if "param_grad" in record_keys:
            record_data(data_record, [model[0].get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
        if "param_collapse_layers" in record_keys:
            record_data(data_record, [simplify(deepcopy(model[0]), X, y, "collapse_layers", verbose = 0)[0]                                      .get_weights_bias(W_source = "core", b_source = "core")], ["param"])
        return loss_original, loss_value, data_record
    optimizer = get_optimizer(optim_type, lr, parameters, **optim_kwargs)
    
    # Set up learning rate scheduler:
    if scheduler_type is not None:
        if scheduler_type == "ReduceLROnPlateau":
            scheduler_patience = kwargs["scheduler_patience"] if "scheduler_patience" in kwargs else 10
            scheduler_factor = kwargs["scheduler_factor"] if "scheduler_factor" in kwargs else 0.1
            scheduler = ReduceLROnPlateau(optimizer, factor = scheduler_factor, patience = scheduler_patience)
        elif scheduler_type == "LambdaLR":
            scheduler_lr_lambda = kwargs["scheduler_lr_lambda"] if "scheduler_lr_lambda" in kwargs else (lambda epoch: 1 / (1 + 0.01 * epoch))
            scheduler = LambdaLR(optimizer, lr_lambda = scheduler_lr_lambda)
        else:
            raise

    # Training:
    to_stop = False
    for i in range(epochs + 1):
        if optim_type != "LBFGS":
            optimizer.zero_grad()
            pred = forward(model, X, **kwargs)
            reg = get_regularization(model, **kwargs)
            loss = criterion(pred, y) + reg
            loss.backward()
            optimizer.step()
        else:
            # "LBFGS" is a second-order optimization algorithm that requires a slightly different procedure:
            def closure():
                optimizer.zero_grad()
                pred = forward(model, X, **kwargs)
                reg = get_regularization(model, **kwargs)
                loss = criterion(pred, y) + reg
                loss.backward()
                return loss
            optimizer.step(closure)
        if i % inspect_interval == 0:
            pred_valid = forward(model, X_valid, **kwargs)
            loss_value = to_np_array(criterion(pred_valid, y_valid))
            if scheduler_type is not None:
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(loss_value)
                else:
                    scheduler.step()
            if "loss" in record_keys:
                record_data(data_record, [i, loss_value], ["iter", "loss"])
            if "mse" in record_keys:
                record_data(data_record, [to_np_array(nn.MSELoss()(pred_valid, y_valid))], ["mse"])
            if "data_DL" in record_keys:
                record_data(data_record, [to_np_array(DL_criterion(pred_valid, y_valid))], ["data_DL"])
            if "data_DL_absolute" in record_keys:
                record_data(data_record, [to_np_array(DL_criterion_absolute(pred_valid, y_valid))], ["data_DL_absolute"])
            if "model_DL" in record_keys:
                record_data(data_record, [get_model_DL(model)], ["model_DL"])
            if "param" in record_keys:
                record_data(data_record, [model[0].get_weights_bias(W_source = "core", b_source = "core")], ["param"])
            if "param_grad" in record_keys:
                record_data(data_record, [model[0].get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
            if "param_collapse_layers" in record_keys:
                record_data(data_record, [simplify(deepcopy(model[0]), X, y, "collapse_layers", verbose = 0)[0]                                          .get_weights_bias(W_source = "core", b_source = "core")], ["param"])
            if patience is not None:
                to_stop = early_stopping.monitor(loss_value)
        if to_stop:
            break

    pred_valid = forward(model, X_valid, **kwargs)
    loss_value = to_np_array(criterion(pred_valid, y_valid))
    if isplot:
        import matplotlib.pylab as plt
        if "mse" in data_record:
            plt.semilogy(data_record["iter"], data_record["mse"])
            plt.xlabel("epochs")
            plt.title("MSE")
            plt.show()
        if "loss" in data_record:
            plt.plot(data_record["iter"], data_record["loss"])
            plt.xlabel("epochs")
            plt.title("Loss")
            plt.show()
    return loss_original, loss_value, data_record


def load_model_dict_net(model_dict, is_cuda = False):
    net_type = model_dict["type"]
    if net_type.startswith("MLP"):
        return MLP(input_size = model_dict["input_size"],
                   struct_param = model_dict["struct_param"] if "struct_param" in model_dict else None,
                   W_init_list = model_dict["weights"] if "weights" in model_dict else None,
                   b_init_list = model_dict["bias"] if "bias" in model_dict else None,
                   settings = model_dict["settings"] if "settings" in model_dict else {},
                   is_cuda = is_cuda,
                  )
    elif net_type == "Labelmix_MLP":
        model = Labelmix_MLP(input_size=model_dict["input_size"],
                             struct_param=model_dict["struct_param"],
                             idx_label=model_dict["idx_label"] if "idx_label" in model_dict else None,
                             is_cuda=is_cuda,
                            )
        if "state_dict" in model_dict:
            model.load_state_dict(model_dict["state_dict"])
        return model
    elif net_type == "Multi_MLP":
        return Multi_MLP(input_size = model_dict["input_size"],
                   struct_param = model_dict["struct_param"],
                   W_init_list = model_dict["weights"] if "weights" in model_dict else None,
                   b_init_list = model_dict["bias"] if "bias" in model_dict else None,
                   settings = model_dict["settings"] if "settings" in model_dict else {},
                   is_cuda = is_cuda,
                  )
    elif net_type == "Branching_Net":
        return Branching_Net(net_base_model_dict = model_dict["net_base_model_dict"],
                             net_1_model_dict = model_dict["net_1_model_dict"],
                             net_2_model_dict = model_dict["net_2_model_dict"],
                             is_cuda = is_cuda,
                            )
    elif net_type == "Fan_in_MLP":
        return Fan_in_MLP(model_dict_branch1=model_dict["model_dict_branch1"],
                          model_dict_branch2=model_dict["model_dict_branch2"],
                          model_dict_joint=model_dict["model_dict_joint"],
                          is_cuda=is_cuda,
                         )
    elif net_type == "Net_reparam":
        return Net_reparam(model_dict=model_dict["model"],
                           reparam_mode=model_dict["reparam_mode"],
                           is_cuda=is_cuda,
                          )
    elif net_type == "Wide_ResNet":
        model = Wide_ResNet(depth=model_dict["depth"],
                            widen_factor=model_dict["widen_factor"],
                            input_channels=model_dict["input_channels"],
                            output_size=model_dict["output_size"],
                            dropout_rate=model_dict["dropout_rate"],
                            is_cuda=is_cuda,
                           )
        if "state_dict" in model_dict:
            model.load_state_dict(model_dict["state_dict"])
        return model
    elif net_type.startswith("ConvNet"):
        return ConvNet(input_channels = model_dict["input_channels"],
                       struct_param = model_dict["struct_param"],
                       W_init_list = model_dict["weights"] if "weights" in model_dict else None,
                       b_init_list = model_dict["bias"] if "bias" in model_dict else None,
                       settings = model_dict["settings"] if "settings" in model_dict else {},
                       return_indices = model_dict["return_indices"] if "return_indices" in model_dict else False,
                       is_cuda = is_cuda,
                      )
    elif net_type == "Conv_Autoencoder":
        model = Conv_Autoencoder(input_channels_encoder = model_dict["input_channels_encoder"],
                                 input_channels_decoder = model_dict["input_channels_decoder"],
                                 struct_param_encoder = model_dict["struct_param_encoder"],
                                 struct_param_decoder = model_dict["struct_param_decoder"],
                                 settings = model_dict["settings"],
                                 is_cuda = is_cuda,
                                )
        if "encoder" in model_dict:
            model.encoder.load_model_dict(model_dict["encoder"])
        if "decoder" in model_dict:
            model.decoder.load_model_dict(model_dict["decoder"])
        return model
    elif model_dict["type"] == "Conv_Model":
        is_generative = model_dict["is_generative"] if "is_generative" in model_dict else False
        return Conv_Model(encoder_model_dict = model_dict["encoder_model_dict"] if not is_generative else None,
                          core_model_dict = model_dict["core_model_dict"],
                          decoder_model_dict = model_dict["decoder_model_dict"],
                          latent_size = model_dict["latent_size"],
                          is_generative = model_dict["is_generative"] if is_generative else False,
                          is_res_block = model_dict["is_res_block"] if "is_res_block" in model_dict else False,
                          is_cuda = is_cuda,
                         )
    else:
        raise Exception("net_type {} not recognized!".format(net_type))
        

def load_model_dict(model_dict, is_cuda = False):
    net_type = model_dict["type"]
    if net_type not in ["Model_Ensemble", "LSTM", "Model_with_Uncertainty", "Mixture_Model", "Mixture_Gaussian"]:
        return load_model_dict_net(model_dict, is_cuda = is_cuda)
    elif net_type == "Model_Ensemble":
        if model_dict["model_type"] == "MLP":
            model_ensemble = Model_Ensemble(
                num_models = model_dict["num_models"],
                input_size = model_dict["input_size"],
                model_type = model_dict["model_type"],
                output_size = model_dict["output_size"],
                is_cuda = is_cuda,
                # Here we just create some placeholder network. The model will be overwritten in the next steps:
                struct_param = [[1, "Simple_Layer", {}]],
            )
        elif model_dict["model_type"] == "LSTM":
            model_ensemble = Model_Ensemble(
                num_models = model_dict["num_models"],
                input_size = model_dict["input_size"],
                model_type = model_dict["model_type"],
                output_size = model_dict["output_size"],
                is_cuda = is_cuda,
                # Here we just create some placeholder network. The model will be overwritten in the next steps:
                hidden_size = 3,
                output_struct_param = [[1, "Simple_Layer", {}]],
            )
        else:
            raise
        for k in range(model_ensemble.num_models):
            setattr(model_ensemble, "model_{}".format(k), load_model_dict(model_dict["model_{}".format(k)], is_cuda = is_cuda))
        return model_ensemble
    elif net_type == "Model_with_Uncertainty":
        return Model_with_Uncertainty(model_pred = load_model_dict(model_dict["model_pred"], is_cuda = is_cuda),
                                      model_logstd = load_model_dict(model_dict["model_logstd"], is_cuda = is_cuda))
    elif net_type == "Mixture_Model":
        return Mixture_Model(model_dict_list=model_dict["model_dict_list"],
                             weight_logits_model_dict=model_dict["weight_logits_model_dict"],
                             num_components=model_dict["num_components"],
                             is_cuda=is_cuda,
                            )
    elif net_type == "Mixture_Gaussian":
        return load_model_dict_Mixture_Gaussian(model_dict, is_cuda = is_cuda)
    else:
        raise Exception("net_type {} not recognized!".format(net_type))


## Helper functions:
def get_accuracy(pred, target):
    """Get accuracy from prediction and target"""
    assert len(pred.shape) == len(target.shape) == 1
    assert len(pred) == len(target)
    pred, target = to_np_array(pred, target)
    accuracy = ((pred == target).sum().astype(float) / len(pred))
    return accuracy


def flatten(*tensors):
    """Flatten the tensor except the first dimension"""
    new_tensors = []
    for tensor in tensors:
        new_tensors.append(tensor.view(tensor.size(0), -1))
    if len(new_tensors) == 1:
        new_tensors = new_tensors[0]
    return new_tensors


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


def matrix_diag_transform(matrix, fun):
    """Return the matrices whose diagonal elements have been executed by the function 'fun'."""
    num_examples = len(matrix)
    idx = torch.eye(matrix.size(-1)).bool().unsqueeze(0)
    idx = idx.repeat(num_examples, 1, 1)
    new_matrix = matrix.clone()
    new_matrix[idx] = fun(matrix.diagonal(dim1=1, dim2=2).contiguous().view(-1))
    return new_matrix


def Zip(*data, **kwargs):
    """Recursive unzipping of data structure
    Example: Zip(*[(('a',2), 1), (('b',3), 2), (('c',3), 3), (('d',2), 4)])
    ==> [[['a', 'b', 'c', 'd'], [2, 3, 3, 2]], [1, 2, 3, 4]]
    Each subtree in the original data must be in the form of a tuple.
    In the **kwargs, you can set the function that is applied to each fully unzipped subtree.
    """
    import collections
    function = kwargs["function"] if "function" in kwargs else None
    if len(data) == 1:
        return data[0]
    data = [list(element) for element in zip(*data)]
    for i, element in enumerate(data):
        if isinstance(element[0], tuple):
            data[i] = Zip(*element, **kwargs)
        elif isinstance(element, list):
            if function is not None:
                data[i] = function(element)
    return data


def get_loss(model, data_loader=None, X=None, y=None, criterion=None, transform_label=None, **kwargs):
    """Get loss using the whole data or data_loader. Return the average validation loss with np.ndarray format"""
    max_validation_iter = kwargs["max_validation_iter"] if "max_validation_iter" in kwargs else None
    if transform_label is None:
        transform_label = Transform_Label()
    if "loader_process" in kwargs and kwargs["loader_process"] is not None:
        data_loader = kwargs["loader_process"]("test")
    if data_loader is not None:
        assert X is None and y is None
        loss_record = 0
        count = 0
        # Taking the average of all metrics:
        for j, data_batch in enumerate(data_loader):
            if isinstance(data_batch, tuple) or isinstance(data_batch, list):
                X_batch, y_batch = data_batch
                if "data_loader_apply" in kwargs and kwargs["data_loader_apply"] is not None:
                    X_batch, y_batch = kwargs["data_loader_apply"](X_batch, y_batch)
            else:
                X_batch, y_batch = kwargs["data_loader_apply"](data_batch)
            loss_ele = to_np_array(model.get_loss(X_batch, transform_label(y_batch), criterion = criterion, **kwargs))
            if j == 0:
                all_info_dict = {key: 0 for key in model.info_dict.keys()}
            loss_record = loss_record + loss_ele
            count += 1
            for key in model.info_dict:
                all_info_dict[key] = all_info_dict[key] + model.info_dict[key]

            if max_validation_iter is not None and count > max_validation_iter:
                break

        for key in model.info_dict:
            all_info_dict[key] = all_info_dict[key] / count
        loss = loss_record / count
        model.info_dict = deepcopy(all_info_dict)
    else:
        assert X is not None and y is not None
        loss = to_np_array(model.get_loss(X, transform_label(y), criterion = criterion, **kwargs))
    return loss


def plot_model(model, data_loader=None, X=None, y=None, transform_label=None, **kwargs):
    data_loader_apply = kwargs["data_loader_apply"] if "data_loader_apply" in kwargs else None
    max_validation_iter = kwargs["max_validation_iter"] if "max_validation_iter" in kwargs else None
    if transform_label is None:
        transform_label = Transform_Label()
    if "loader_process" in kwargs and kwargs["loader_process"] is not None:
        data_loader = kwargs["loader_process"]("test")
    if data_loader is not None:
        assert X is None and y is None
        X_all = []
        y_all = []
        for i, data_batch in enumerate(data_loader):
            if isinstance(data_batch, tuple) or isinstance(data_batch, list):
                X_batch, y_batch = data_batch
                if data_loader_apply is not None:
                    X_batch, y_batch = data_loader_apply(X_batch, y_batch)
            else:
                X_batch, y_batch = data_loader_apply(data_batch)
            X_all.append(X_batch)
            y_all.append(y_batch)
            if max_validation_iter is not None and i >= max_validation_iter:
                break
        if not isinstance(X_all[0], torch.Tensor):
            X_all = Zip(*X_all, function = torch.cat)
        else:
            X_all = torch.cat(X_all, 0)
        y_all = torch.cat(y_all)
        model.plot(X_all, transform_label(y_all))
    else:
        assert X is not None and y is not None
        model.plot(X, transform_label(y))


def prepare_inspection(model, data_loader=None, X=None, y=None, transform_label=None, **kwargs):
    inspect_functions = kwargs["inspect_functions"] if "inspect_functions" in kwargs else None
    max_validation_iter = kwargs["max_validation_iter"] if "max_validation_iter" in kwargs else None
    verbose = kwargs["verbose"] if "verbose" in kwargs else False
    if transform_label is None:
        transform_label = Transform_Label()
    if "loader_process" in kwargs and kwargs["loader_process"] is not None:
        data_loader = kwargs["loader_process"]("test")
    if data_loader is None:
        assert X is not None and y is not None
        all_dict_summary = model.prepare_inspection(X, transform_label(y), **kwargs)
        if inspect_functions is not None:
            for inspect_function_key, inspect_function in inspect_functions.items():
                all_dict_summary[inspect_function_key] = inspect_function(model, X, y, **kwargs)
    else:
        assert X is None and y is None
        all_dict = {}
        for j, data_batch in enumerate(data_loader):
            if verbose is True:
                print("valid step: {}".format(j))
            if isinstance(data_batch, tuple) or isinstance(data_batch, list):
                X_batch, y_batch = data_batch
                if "data_loader_apply" in kwargs and kwargs["data_loader_apply"] is not None:
                    X_batch, y_batch = kwargs["data_loader_apply"](X_batch, y_batch)
            else:
                X_batch, y_batch = kwargs["data_loader_apply"](data_batch)
            info_dict = model.prepare_inspection(X_batch, transform_label(y_batch), valid_step=j, **kwargs)
            for key, item in info_dict.items():
                if key not in all_dict:
                    all_dict[key] = [item]
                else:
                    all_dict[key].append(item)
            if inspect_functions is not None:
                for inspect_function_key, inspect_function in inspect_functions.items():
                    inspect_function_result = inspect_function(model, X_batch, transform_label(y_batch), **kwargs)
                    if inspect_function_key not in all_dict:
                        all_dict[inspect_function_key] = [inspect_function_result]
                    else:
                        all_dict[inspect_function_key].append(inspect_function_result)
            if max_validation_iter is not None and j >= max_validation_iter:
                break
        all_dict_summary = {}
        for key, item in all_dict.items():
            all_dict_summary[key + "_val"] = np.mean(all_dict[key])
    return all_dict_summary


def get_inspect_items_train(inspect_items):
    if inspect_items is None:
        return []
    inspect_items_train = []
    for item in inspect_items:
        if item.endswith("_tr"):
            inspect_items_train.append("_".join(item.split("_")[:-1]))
    return inspect_items_train


def update_key_train(info_dict_train, inspect_items_train):
    info_dict_train_new = {}
    for key, item in info_dict_train.items():
        if key in inspect_items_train:
            info_dict_train_new[key + "_tr"] = item
    return deepcopy(info_dict_train_new)


# ## Simplification functionality:

# In[ ]:


def simplify(
    model,
    X=None,
    y=None,
    mode="full",
    isplot=False,
    target_name=None,
    validation_data=None,
    **kwargs
):
    """Simplify a neural network model in various ways. "model" can be a single model or a ordered list of models"""
    verbose = kwargs["verbose"] if "verbose" in kwargs else 1
    if validation_data is None:
        X_valid, y_valid = X, y
    else:
        X_valid, y_valid = validation_data
    simplify_criteria = kwargs["simplify_criteria"] if "simplify_criteria" in kwargs else ("DLs", 0.05, 3, "relative") # the first argument choose from "DL", "loss"
    simplify_epsilon = simplify_criteria[1]
    simplify_patience = simplify_criteria[2]
    simplify_compare_mode = simplify_criteria[3]
    performance_monitor = Performance_Monitor(patience = simplify_patience, epsilon = simplify_epsilon, compare_mode = simplify_compare_mode)
    record_keys = kwargs["record_keys"] if "record_keys" in kwargs else ["mse"]
    loss_precision_floor = kwargs["loss_precision_floor"] if "loss_precision_floor" in kwargs else PrecisionFloorLoss
    if X is not None:
        if y is None:
            y = Variable(forward(model, X, **kwargs).data, requires_grad = False)
    if not (isinstance(model, list) or isinstance(model, tuple)):
        model = [model]
        is_list = False
    else:
        is_list = True
    if mode == "full":
        mode = ["collapse_layers", "snap"]
    if not isinstance(mode, list):
        mode = [mode]

    # Obtain the original loss and setup criterion:
    loss_type = kwargs["loss_type"] if "loss_type" in kwargs else "mse"
    criterion = get_criterion(loss_type, loss_precision_floor = loss_precision_floor)
    DL_criterion = Loss_Fun(core = "DLs", loss_precision_floor = loss_precision_floor, DL_sum = True)
    loss_dict = OrderedDict()

    for mode_ele in mode:
        if verbose >= 1:
            print("\n" + "=" * 48 + "\nSimplifying mode: {}".format(mode_ele), end = "")
            if mode_ele == "snap":
                snap_mode = kwargs["snap_mode"] if "snap_mode" in kwargs else "integer"
                print(" {}".format(snap_mode), end = "")
            if target_name is not None:
                print(" for {}".format(target_name))
            else:
                print()
            print("=" * 48)
        
        # Record the loss before simplification:
        if X is not None:
            pred_valid = forward(model, X_valid, **kwargs)
            loss_original = to_np_array(criterion(pred_valid, y_valid))
            loss_list = [loss_original]
            if verbose >= 1:
                print("original_loss: {}".format(loss_original))
            mse_record_whole = [to_np_array(nn.MSELoss()(pred_valid, y_valid))]
            data_DL_whole = [to_np_array(DL_criterion(pred_valid, y_valid))]
        model_DL_whole = [get_model_DL(model)]
        event_list = ["before simplification"]
        iter_end_whole = [1]
        is_accept_whole = []
        if "param" in record_keys:
            param_record_whole = [model[0].get_weights_bias(W_source = "core", b_source = "core")]
        if "param_grad" in record_keys:
            param_grad_record_whole = [model[0].get_weights_bias(W_source = "core", b_source = "core", is_grad = True)]
        
        # Begin simplification:
        if mode_ele == "collapse_layers":
            all_collapse_dict = {}
            for model_id, model_ele in enumerate(model):
                # Obtain activations for each layer:
                activation_list = []
                for k in range(len(model_ele.struct_param)):
                    if "activation" in model_ele.struct_param[k][2]:
                        activation_list.append(model_ele.struct_param[k][2]["activation"])
                    elif "activation" in model_ele.settings:
                        activation_list.append(model_ele.settings["activation"])
                    else:
                        activation_list.append("default")
                
                # Build the collapse_list that stipulates which layers to collapse:
                collapse_dict = {}
                current_start = None
                current_layer_type = None
                for k, activation in enumerate(activation_list):
                    if activation == "linear" and k != len(activation_list) - 1:
                        if k not in collapse_dict and current_start is None:
                            # Create a new bunch:
                            if model_ele.struct_param[k + 1][1] == model_ele.struct_param[k][1]: # The current layer must have the same layer_type as the next layer
                                current_start = k
                                collapse_dict[current_start] = [k]
                                current_layer_type = model_ele.struct_param[k][1]
                        else:
                            # Adding to current bunch:
                            if model_ele.struct_param[k + 1][1] == model_ele.struct_param[k][1] == current_layer_type:
                                collapse_dict[current_start].append(k)
                            else:
                                collapse_dict[current_start].append(k)
                                current_start = None
                    else:
                        if current_start is not None:
                            collapse_dict[current_start].append(k)
                        current_start = None

                # Build new layer:
                new_layer_info = {}
                for current_start, layer_ids in collapse_dict.items():
                    for i, layer_id in enumerate(layer_ids):
                        layer = getattr(model_ele, "layer_{}".format(layer_id))
                        if i == 0:
                            W_accum = layer.W_core
                            b_accum = layer.b_core
                        else:
                            W_accum = torch.matmul(W_accum, layer.W_core)
                            b_accum = torch.matmul(b_accum, layer.W_core) + layer.b_core
                    if model_ele.is_cuda:
                        W_accum = W_accum.cpu()
                        b_accum = b_accum.cpu()
                    last_layer_id = collapse_dict[current_start][-1]
                    new_layer_info[current_start] = {"W_init": W_accum.data.numpy(), "b_init": b_accum.data.numpy(),
                                                     "layer_struct_param": [b_accum.size(0), model_ele.struct_param[last_layer_id][1], deepcopy(model_ele.struct_param[last_layer_id][2])],
                                                    }
                    new_layer_info[current_start].pop("snap_dict", None)
                if verbose >= 1:
                    print("model_id {}, layers collapsed: {}".format(model_id, collapse_dict))
                
                # Rebuild the Net:
                if len(collapse_dict) > 0:
                    all_collapse_dict[model_id] = {"collapse_dict": collapse_dict, 
                                                   "new_layer_info": new_layer_info, 
                                                   "collapse_layer_ids": [idx for item in collapse_dict.values() for idx in item],
                                                  }

            # Rebuild the list of models:
            if len(all_collapse_dict) > 0:
                model_new = []
                for model_id, model_ele in enumerate(model):
                    if model_id in all_collapse_dict:
                        W_list, b_list = model_ele.get_weights_bias(W_source = "core", b_source = "core")
                        W_init_list = []
                        b_init_list = []
                        struct_param = []
                        for k in range(len(model_ele.struct_param)):
                            if k not in all_collapse_dict[model_id]["collapse_layer_ids"]:
                                struct_param.append(model_ele.struct_param[k])
                                W_init_list.append(W_list[k])
                                b_init_list.append(b_list[k])
                            else:
                                if k in all_collapse_dict[model_id]["collapse_dict"].keys():
                                    struct_param.append(all_collapse_dict[model_id]["new_layer_info"][k]["layer_struct_param"])
                                    W_init_list.append(all_collapse_dict[model_id]["new_layer_info"][k]["W_init"])
                                    b_init_list.append(all_collapse_dict[model_id]["new_layer_info"][k]["b_init"])
                        model_ele_new = MLP(input_size = model_ele.input_size,
                                            struct_param = struct_param,
                                            W_init_list = W_init_list,
                                            b_init_list = b_init_list,
                                            settings = model_ele.settings,
                                            is_cuda = model_ele.is_cuda,
                                           )
                    else:
                        model_ele_new = model_ele
                    model_new.append(model_ele_new)               
                model = model_new

                # Calculate the loss again:
                pred_valid = forward(model, X_valid, **kwargs)
                loss_new = to_np_array(criterion(pred_valid, y_valid))
                if verbose >= 1:
                    print("after collapsing linear layers in all models, new loss {}".format(loss_new))
                loss_list.append(loss_new)
                mse_record_whole.append(to_np_array(nn.MSELoss()(pred_valid, y_valid)))
                data_DL_whole.append(to_np_array(DL_criterion(pred_valid, y_valid)))
                model_DL_whole.append(get_model_DL(model))
                if "param" in record_keys:
                    param_record_whole.append(model[0].get_weights_bias(W_source = "core", b_source = "core"))
                if "param_grad" in record_keys:
                    param_grad_record_whole.append(model[0].get_weights_bias(W_source = "core", b_source = "core", is_grad = True))
                iter_end_whole.append(1)
                event_list.append({mode_ele: all_collapse_dict})

        elif mode_ele in ["local", "snap"]:
            # 'local': greedily try reducing the input dimension by removing input dimension from the beginning;
            # 'snap': greedily snap each float parameter into an integer or rational number. Set argument 'snap_mode' == 'integer' or 'rational'.
            if mode_ele == "snap":
                target_params = [[(model_id, layer_id), "snap"] for model_id, model_ele in enumerate(model) for layer_id in range(len(model_ele.struct_param))]
            elif mode_ele == "local":
                for model_id, model_ele in enumerate(model):
                    if len(model_ele.struct_param) > 0:
                        first_model_id = model_id
                        break
                first_layer = getattr(model[first_model_id], "layer_0")
                target_params = [[(first_model_id, 0), [[(("weight", (i, j)), 0.) for j in range(first_layer.output_size)]                                                             for i in range(first_layer.input_size)]]]
            else:
                raise

            excluded_idx_dict = {item[0]: [] for item in target_params}
            target_layer_ids_exclude = []
            for (model_id, layer_id), target_list in target_params:
                layer = getattr(model[model_id], "layer_{}".format(layer_id))
                if isinstance(target_list, list):
                    max_passes = len(target_list)
                elif target_list == "snap":
                    max_passes = (layer.input_size + 1) * layer.output_size
                    if "max_passes" in kwargs:
                        max_passes = min(max_passes, kwargs["max_passes"])
                else:
                    raise Exception("target_list {} not recognizable!".format(target_list))
                if verbose >= 2:
                    print("\n****starting model:****")
                    model[model_id].get_weights_bias(W_source = "core", b_source = "core", verbose = True)
                    print("********\n" )
                
                
                performance_monitor.reset()
                criteria_value, criteria_result = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                to_stop, pivot_dict, log, _, pivot_id = performance_monitor.monitor(criteria_value, model_dict = model[model_id].model_dict, criteria_result = criteria_result)
                for i in range(max_passes):
                    # Perform tentative simplification
                    if isinstance(target_list, list):
                        info = layer.simplify(mode = "snap", excluded_idx = excluded_idx_dict[(model_id, layer_id)], snap_targets = target_list[i], **kwargs)
                    else:
                        info = layer.simplify(mode = "snap", excluded_idx = excluded_idx_dict[(model_id, layer_id)], **kwargs)
                    if len(info) == 0:
                        target_layer_ids_exclude.append((model_id, layer_id))
                        print("Pass {0}, (model {1}, layer {2}) has no parameters to snap. Revert to pivot model. Go to next layer".format(i, model_id, layer_id))
                        break
                    excluded_idx_dict[(model_id, layer_id)] = excluded_idx_dict[(model_id, layer_id)] + info

                    _, loss_new, data_record = train_simple(model, X, y, optim_type = "adam", validation_data = validation_data, **kwargs)
                    if verbose >= 2:
                        print("=" * 8)
                        model[model_id].get_weights_bias(W_source = "core", b_source = "core", verbose = True) 
                    criteria_value, criteria_result = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                    to_stop, pivot_dict, log, is_accept, pivot_id = performance_monitor.monitor(criteria_value, model_dict = model[model_id].model_dict, criteria_result = criteria_result)
                    is_accept_whole.append(is_accept)
                    if is_accept:
                        print('[Accepted] as pivot model!')
                        print()

                    # Check if the criterion after simplification and refit is worse. If it is worse than the simplify_epsilon, revert:
                    if to_stop:
                        target_layer_ids_exclude.append((model_id, layer_id))
                        if verbose >= 1:
                            print("Pass {0}, loss: {1}\tDL: {2}. New snap {3} is do not improve by {4} = {5} for {6} steps. Revert the simplification to pivot model. Go to next layer.".format(
                                i, view_item(log, ("criteria_result", "loss")), view_item(log, ("criteria_result", "DL")), info, simplify_criteria[0], simplify_epsilon, simplify_patience))
                        break
                    mse_record_whole += data_record["mse"]
                    data_DL_whole += data_record["data_DL"]
                    model_DL_whole += data_record["model_DL"]
                    if "param" in record_keys:
                        param_record_whole += data_record["param"]
                    if "param_grad" in record_keys:
                        param_grad_record_whole += data_record["param_grad"]
                    iter_end_whole.append(len(data_record["mse"]))

                    model[model_id].reset_layer(layer_id, layer)
                    loss_list.append(loss_new)
                    event_list.append({mode_ele: ((model_id, layer_id), info)})
                    if verbose >= 1:
                        print("Pass {0}, snap (model {1}, layer {2}), snap {3}. \tloss: {4}\tDL: {5}".format(
                            i, model_id, layer_id, info, view_item(log, ("criteria_result", "loss")), view_item(log, ("criteria_result", "DL"))))

                # Update the whole model's struct_param and snap_dict:
                model[model_id].load_model_dict(pivot_dict["model_dict"])
                model[model_id].synchronize_settings()
                if verbose >= 2:
                    print("\n****pivot model at {}th transformation:****".format(pivot_id))
                    model[model_id].get_weights_bias(W_source = "core", b_source = "core", verbose = True)
                    print("********\n" )

        elif mode_ele == "pair_snap":
            model_new = []
            for model_id, model_ele in enumerate(model):
                for layer_id, layer_struct_param in enumerate(model_ele.struct_param):
                    if layer_struct_param[1] == "Symbolic_Layer":
                        layer = getattr(model_ele, "layer_{}".format(layer_id))
                        max_passes = len(layer.get_param_dict()) - 1
                        if "max_passes" in kwargs:
                            max_passes = min(max_passes, kwargs["max_passes"])
                        if verbose > 1:
                            print("original:")
                            print("symbolic_expression: ", layer.symbolic_expression)
                            print("numerical_expression: ", layer.numerical_expression)
                            print()

                        performance_monitor.reset()
                        criteria_value, criteria_result = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                        to_stop, pivot_dict, log, _, pivot_id = performance_monitor.monitor(criteria_value, model_dict = model[model_id].model_dict, criteria_result = criteria_result)
                        for i in range(max_passes):
                            info = layer.simplify(mode = "pair_snap", **kwargs)
                            if len(info) == 0:
                                target_layer_ids_exclude.append((model_id, layer_id))
                                print("Pass {0}, (model {1}, layer {2}) has no parameters to pair_snap. Revert to pivot model. Go to next layer".format(i, model_id, layer_id))
                                break
                            _, loss, data_record = train_simple(model, X, y, optim_type = "adam", epochs = 1000, validation_data = validation_data, **kwargs)
                            criteria_value, criteria_result = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                            to_stop, pivot_dict, log, is_accept, pivot_id = performance_monitor.monitor(criteria_value, model_dict = model[model_id].model_dict, criteria_result = criteria_result)
                            is_accept_whole.append(is_accept)
                            if to_stop:
                                if verbose >= 1:
                                    print("\nPass {0}, loss: {1}\tDL: {2}. New snap {3} is do not improve by {4} = {5} for {6} steps. Revert the simplification to pivot model. Go to next layer.".format(
                                        i, view_item(log, ("criteria_result", "loss")), view_item(log, ("criteria_result", "DL")), info, simplify_criteria[0], simplify_epsilon, simplify_patience))
                                break

                            mse_record_whole += data_record["mse"]
                            data_DL_whole += data_record["data_DL"]
                            model_DL_whole += data_record["model_DL"]
                            if "param" in record_keys:
                                param_record_whole += data_record["param"]
                            if "param_grad" in record_keys:
                                param_grad_record_whole += data_record["param_grad"]
                            iter_end_whole.append(len(data_record["mse"]))

                            model[model_id].reset_layer(layer_id, layer)
                            loss_list.append(loss)
                            event_list.append({mode_ele: ((model_id, layer_id), info)})
                            if verbose >= 1:
                                print("\nPass {0}, snap (model {1}, layer {2}), snap {3}. \tloss: {4}\tDL: {5}".format(
                                    i, model_id, layer_id, info, view_item(log, ("criteria_result", "loss")), view_item(log, ("criteria_result", "DL"))))
                                print("symbolic_expression: ", layer.symbolic_expression)
                                print("numerical_expression: ", layer.numerical_expression)
                                print()

                        model[model_id].load_model_dict(pivot_dict["model_dict"])
                        print("final: \nsymbolic_expression: ", getattr(model[model_id], "layer_{0}".format(layer_id)).symbolic_expression)
                        print("numerical_expression: ", getattr(model[model_id], "layer_{0}".format(layer_id)).numerical_expression)
                        print()

        elif mode_ele[:11] == "to_symbolic":
            from sympy import Symbol
            force_simplification = kwargs["force_simplification"] if "force_simplification" in kwargs else False
            is_multi_model = True if len(model) > 1 else False
            for model_id, model_ele in enumerate(model):
                for layer_id, layer_struct_param in enumerate(model_ele.struct_param):
                    prefix = "L{}_".format(layer_id)
                    if layer_struct_param[1] == "Simple_Layer":
                        # Obtain loss before simplification:
                        layer = getattr(model_ele, "layer_{}".format(layer_id))
                        if X is not None:
                            criteria_prev, criteria_result_prev = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                        
                        if mode_ele.split("_")[-1] == "separable":
                            new_layer = Simple_2_Symbolic(layer, settings = model_ele.settings, mode = "separable", prefix = prefix)
                        else:
                            new_layer = Simple_2_Symbolic(layer, settings = model_ele.settings, prefix = prefix)
                        model[model_id].reset_layer(layer_id, new_layer)

                        if "snap_dict" in model_ele.settings and layer_id in model_ele.settings["snap_dict"]:
                            subs_targets = []
                            for (pos, true_idx), item in model_ele.settings["snap_dict"][layer_id].items():
                                if pos == "weight":
                                    subs_targets.append((Symbol("W{0}{1}".format(true_idx[0], true_idx[1])), item["new_value"]))
                                elif pos == "bias":
                                    subs_targets.append((Symbol("b{}".format(true_idx)), item["new_value"]))
                                else:
                                    raise Exception("pos {} not recognized!".format(pos))
                            new_expression = [expression.subs(subs_targets) for expression in new_layer.symbolic_expression]
                            new_layer.set_symbolic_expression(new_expression)
                            model_ele.settings["snap_dict"].pop(layer_id)
                            model_ele.struct_param[layer_id][2].update(new_layer.struct_param[2])
                        
                        # Calculate the loss again:
                        if X is not None:
                            criteria_new, criteria_result_new = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                            if verbose >= 1:
                                print("Prev_loss: {0}, new loss: {1}\tprev_DL: {2:.9f}, new DL: {3:.9f}".format(
                                       criteria_result_prev["loss"], criteria_result_new["loss"], criteria_result_prev["DL"], criteria_result_new["DL"]))
                                print()
                            if criteria_new > criteria_prev * (1 + 0.05):
                                print("to_symbolic DL increase more than 5%! ", end = "")
                                if not force_simplification:
                                    print("Reset layer.")
                                    model[model_id].reset_layer(layer_id, layer)
                                else:
                                    print("Nevertheless, force simplification.")

                            loss_list.append(criteria_result_new["loss"])
                            print("{0} succeed. Prev_loss: {1}\tnew_loss: {2}\tprev_DL: {3:.9f}, new_DL: {4:.9f}".format(
                                    mode_ele, criteria_result_prev["loss"], criteria_result_new["loss"],
                                    criteria_result_prev["DL"], criteria_result_new["DL"]))
                        else:
                            print("{0} succeed.".format(mode_ele))
                        event_list.append({mode_ele: (model_id, layer_id)})
                        
                    
                    elif layer_struct_param[1] == "Sneuron_Layer":
                        # Obtain loss before simplification:
                        layer = getattr(model_ele, "layer_{0}".format(layer_id))
                        criteria_prev, criteria_result_prev = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                        
                        new_layer = Sneuron_2_Symbolic(layer, prefix = prefix)
                        model[model_id].reset_layer(layer_id, new_layer)
                        
                        # Calculate the loss again:
                        criteria_new, criteria_result_new = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                        if verbose >= 1:
                            print("Prev_loss: {0}, new loss: {1}\tprev_DL: {2:.9f}, new DL: {3:.9f}".format(
                                   criteria_result_prev["loss"], criteria_result_new["loss"], criteria_result_prev["DL"], criteria_result_new["DL"]))
                            print()
                        if criteria_new > criteria_prev * (1 + 0.05):  
                            print("to_symbolic DL increase more than 5%! ", end = "")
                            if not force_simplification:
                                print("Reset layer.")
                                model[model_id].reset_layer(layer_id, layer)
                            else:
                                print("Nevertheless, force simplification.")
                        
                        loss_list.append(criteria_result_new["loss"])
                        event_list.append({mode_ele: (model_id, layer_id)})
                        print("{0} succeed. Prev_loss: {1}\tnew_loss: {2}\tprev_DL: {3:.9f}, new_DL: {4:.9f}".format(
                                mode_ele, criteria_result_prev["loss"], criteria_result_new["loss"],
                                criteria_result_prev["DL"], criteria_result_new["DL"]))
            if X is not None:
                mse_record_whole.append(to_np_array(nn.MSELoss()(pred_valid, y_valid)))
                data_DL_whole.append(to_np_array(DL_criterion(pred_valid, y_valid)))
            model_DL_whole.append(get_model_DL(model))
            if "param" in record_keys:
                param_record_whole.append(model[0].get_weights_bias(W_source = "core", b_source = "core"))
            if "param_grad" in record_keys:
                param_grad_record_whole.append(model[0].get_weights_bias(W_source = "core", b_source = "core", is_grad = True))
            iter_end_whole.append(1)

        elif mode_ele == "symbolic_simplification":
            """Collapse multi-layer symbolic expression"""
            from sympy import Symbol, Poly, expand, prod
            force_simplification = kwargs["force_simplification"] if "force_simplification" in kwargs else False
            numerical_threshold = kwargs["numerical_threshold"] if "numerical_threshold" in kwargs else None
            is_numerical = kwargs["is_numerical"] if "is_numerical" in kwargs else False
            max_poly_degree = kwargs["max_poly_degree"] if "max_poly_degree" in kwargs else None
            show_before_truncate = kwargs["show_before_truncate"] if "show_before_truncate" in kwargs else False
            for model_id, model_ele in enumerate(model):
                is_all_symbolic = True
                for layer_id, layer_struct_param in enumerate(model_ele.struct_param):
                    if layer_struct_param[1] != "Symbolic_Layer":
                        is_all_symbolic = False
                if is_all_symbolic:
                    criteria_prev, criteria_result_prev = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                    variables = OrderedDict()
                    for i in range(model[0].layer_0.input_size):
                        variables["x{0}".format(i)] = Symbol("x{0}".format(i))
                    expression = list(variables.values())
                    param_dict_all = {}

                    # Collapse multiple layers:
                    for layer_id, layer_struct_param in enumerate(model_ele.struct_param):
                        layer = getattr(model_ele, "layer_{0}".format(layer_id))
                        layer_expression = deepcopy(layer.numerical_expression)
                        layer_expression_new = []
                        for expr in layer_expression:
                            new_expr = expr.subs({"x{0}".format(i): "t{0}".format(i) for i in range(len(expression))})  # Use a temporary variable to prevent overriding
                            new_expr = new_expr.subs({"t{0}".format(i): expression[i] for i in range(len(expression))})
                            layer_expression_new.append(expand(new_expr))
                        expression = layer_expression_new
                    
                    # Show full expression before performing truncation:
                    if show_before_truncate:
                        for i, expr in enumerate(expression):
                            print("Full expression {0}:".format(i))
                            pp.pprint(Poly(expr, *list(variables.values())))
                            print()

                    model_ele_candidate = MLP(input_size = model[0].layer_0.input_size,
                                              struct_param = [[layer.output_size, "Symbolic_Layer", {"symbolic_expression": "x0"}]],
                                              settings = {},
                                              is_cuda = model_ele.is_cuda,
                                             )
                    # Setting maximul degree for polynomial:
                    if max_poly_degree is not None:
                        new_expression = []
                        for expr in expression:
                            expr = Poly(expr, *list(variables.values()))
                            degree_list = []
                            coeff_list = []
                            for degree, coeff in expr.terms():
                                # Only use monomials with degree not larger than max_poly_degree:
                                if sum(degree) <= max_poly_degree: 
                                    degree_list.append(degree)
                                    coeff_list.append(coeff)

                            new_expr = 0
                            for degree, coeff in zip(degree_list, coeff_list):
                                new_expr += prod([variables["x{0}".format(i)] ** degree[i] for i in range(len(degree))]) * coeff
                            new_expression.append(new_expr)
                        expression = new_expression

                    # Update symbolic expression for model_ele_candidate:
                    if not is_numerical:
                        param_dict_all = {}
                        expression_new_all = []
                        for expr in expression:
                            expression_new, param_dict = numerical_2_parameter(expr, idx = len(param_dict_all), threshold = numerical_threshold)
                            expression_new_all.append(expression_new)
                            param_dict_all.update(param_dict)
                        model_ele_candidate.layer_0.set_symbolic_expression(expression_new_all, p_init = param_dict_all)
                    else:
                        model_ele_candidate.layer_0.set_symbolic_expression(expression)
                        model_ele_candidate.layer_0.set_numerical(True)
                    
                    criteria_new, criteria_result_new = get_criteria_value(model_ele_candidate, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                    if criteria_new > criteria_prev * (1 + 0.05):                            
                        print("to_symbolic DL increase more than 5%! ", end = "")
                        if force_simplification:
                            print("Nevertheless, force simplification.")
                            model[model_id] = model_ele_candidate
                        else:
                            print("Revert.")
                    else:
                        model[model_id] = model_ele_candidate

        elif mode_ele == "activation_snap":
            from sympy import Function
            def get_sign_snap_candidate(layer, activation_source, excluded_neurons = None):
                coeff_dict = {}
                for i in range(len(layer.symbolic_expression)):
                    current_expression = [layer.symbolic_expression[i]]
                    func_names = layer.get_function_name_list(current_expression)
                    if activation_source in func_names:
                        coeff = [element for element in layer.get_param_name_list(current_expression) if element[0] == "W"]
                        coeff_dict[i] = np.mean([np.abs(value) for key, value in layer.get_param_dict().items() if key in coeff])
                best_idx = None
                best_value = 0
                for key, value in coeff_dict.items():
                    if value > best_value and key not in excluded_neurons:
                        best_value = value
                        best_idx = key
                return best_idx, best_value

            activation_source = kwargs["activation_source"] if "activation_source" in kwargs else "sigmoid"
            activation_target = kwargs["activation_target"] if "activation_target" in kwargs else "heaviside"
            activation_fun_source = Function(activation_source)
            activation_fun_target = Function(activation_target)

            for model_id, model_ele in enumerate(model):
                for layer_id, layer_struct_param in enumerate(model_ele.struct_param):
                    if layer_struct_param[1] == "Symbolic_Layer":
                        layer = getattr(model_ele, "layer_{0}".format(layer_id))
                        excluded_neurons = []
                        if activation_source not in layer.get_function_name_list():
                            continue

                        performance_monitor.reset()
                        criteria_value, criteria_result = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                        to_stop, pivot_dict, log, _, pivot_id = performance_monitor.monitor(criteria_value, model_dict = model[model_id].model_dict, criteria_result = criteria_result)
                        for i in range(layer_struct_param[0]):
                            # Obtain loss before simplification:
                            layer = getattr(model_ele, "layer_{0}".format(layer_id))
                            best_idx, _ = get_sign_snap_candidate(layer, activation_source, excluded_neurons = excluded_neurons)
                            excluded_neurons.append(best_idx)
                            
                            new_expression = [expression.subs(activation_fun_source, activation_fun_target) if j == best_idx else expression for j, expression in enumerate(layer.symbolic_expression)]
                            print("Pass {0}, candidate new expression: {1}".format(i, new_expression))
                            layer.set_symbolic_expression(new_expression)

                            # Train:
                            _, loss_new, data_record = train_simple(model, X, y, validation_data = validation_data, **kwargs)

                            criteria_value, criteria_result = get_criteria_value(model, X, y, criteria_type = simplify_criteria[0], criterion = criterion, **kwargs)
                            to_stop, pivot_dict, log, is_accept, pivot_id = performance_monitor.monitor(criteria_value, model_dict = model[model_id].model_dict, criteria_result = criteria_result)
                            is_accept_whole.append(is_accept)
                            # Check if the criterion after simplification and refit is worse. If it is worse than the simplify_epsilon, revert:
                            if to_stop:
                                model[model_id].load_model_dict(pivot_dict["model_dict"])
                                if verbose >= 1:
                                    print("Pass {0}, loss: {1}\tDL: {2}. New snap {3} is do not improve by {4} = {5} for {6} steps. Revert the simplification to pivot model. Continue".format(
                                        i, view_item(log, ("criteria_result", "loss")), view_item(log, ("criteria_result", "DL")), info, simplify_criteria[0], simplify_epsilon, simplify_patience))
                                continue   
                                
                            mse_record_whole += data_record["mse"]
                            data_DL_whole += data_record["data_DL"]
                            model_DL_whole += data_record["model_DL"]
                            if "param" in record_keys:
                                param_record_whole += data_record["param"]
                            if "param_grad" in record_keys:
                                param_grad_record_whole += data_record["param_grad"]
                            iter_end_whole.append(len(data_record["mse"]))

                            loss_list.append(loss_new)
                            event_list.append({mode_ele: (model_id, layer_id)})
                            if verbose >= 1:
                                print("{0} succeed at (model {1}, layer {2}). loss: {3}\tDL: {4}".format(
                                    mode_ele, model_id, layer_id, view_item(log, ("criteria_result", "loss")), view_item(log, ("criteria_result", "DL"))))
                                print("symbolic_expression: ", layer.symbolic_expression)
                                print("numerical_expression: ", layer.numerical_expression)
                                print()
                        model[model_id].load_model_dict(pivot_dict["model_dict"])
        
        elif mode_ele == "ramping-L1":
            loss_list_specific = []
            ramping_L1_list = kwargs["ramping_L1_list"] if "ramping_L1_list" in kwargs else np.logspace(-7, -1, 30)
            ramping_mse_threshold = kwargs["ramping_mse_threshold"] if "ramping_mse_threshold" in kwargs else 1e-5
            ramping_final_multiplier = kwargs["ramping_final_multiplier"] if "ramping_final_multiplier" in kwargs else 1e-2
            layer_dict_dict = {}
            for i, L1_amp in enumerate(ramping_L1_list):
                reg_dict = {"weight": L1_amp, "bias": L1_amp, "param": L1_amp}
                _, loss_end, data_record = train_simple(model, X, y, reg_dict = reg_dict, patience = None, validation_data = validation_data, **kwargs)
                layer_dict_dict[i] = model[0].layer_0.layer_dict
                weight, bias = model[0].layer_0.get_weights_bias()
                print("L1-amp: {0}\tloss: {1}\tweight: {2}\tbias: {3}".format(L1_amp, loss_end, weight, bias))
                loss_list_specific.append(loss_end)
                if "param" in record_keys:
                    param_record_whole.append((weight, bias))
                if loss_end > ramping_mse_threshold:
                    if len(loss_list_specific) == 1:
                        print("\nThe MSE after the first L1-amp={0} is already larger than the ramping_mse_threshold. Stop and use current L1-amp. The figures will look empty.".format(ramping_mse_threshold))
                    else:
                        print("\nThe MSE {0} is larger than the ramping_mse_threshold {1}, stop ramping-L1 simplification".format(loss_end, ramping_mse_threshold))
                    break 
                mse_record_whole.append(data_record["mse"][-1])
                data_DL_whole.append(data_record["data_DL"][-1])
                model_DL_whole.append(data_record["model_DL"][-1])
                iter_end_whole.append(1)
            final_L1_amp = L1_amp * ramping_final_multiplier
            final_L1_idx = np.argmin(np.abs(np.array(ramping_L1_list) - final_L1_amp))
            layer_dict_final = layer_dict_dict[final_L1_idx]
            print("Final L1_amp used: {0}".format(ramping_L1_list[final_L1_idx]))
            if "param" in record_keys:
                print("Final param value:\nweights: {0}\nbias{1}".format(param_record_whole[final_L1_idx][0], param_record_whole[final_L1_idx][1]))
            model[0].layer_0.load_layer_dict(layer_dict_final)
            mse_record_whole = mse_record_whole[: final_L1_idx + 2]
            data_DL_whole = data_DL_whole[: final_L1_idx + 2]
            model_DL_whole = model_DL_whole[: final_L1_idx + 2]
            iter_end_whole = iter_end_whole[: final_L1_idx + 2]

            if isplot:
                def dict_to_list(Dict):
                    return np.array([value for value in Dict.values()])
                weights_list = []
                bias_list = []
                for element in param_record_whole:
                    if isinstance(element[0], dict):
                        element_core = dict_to_list(element[0])
                        weights_list.append(element_core)
                    else:
                        element_core = to_np_array(element[0]).squeeze(1)
                        weights_list.append(element_core)
                        bias_list.append(to_np_array(element[1]))
                weights_list = np.array(weights_list)
                bias_list = np.array(bias_list).squeeze(1)

                import matplotlib.pylab as plt
                plt.figure(figsize = (7,5))
                plt.loglog(ramping_L1_list[: len(loss_list_specific)], loss_list_specific)
                plt.xlabel("L1 amp", fontsize = 16)
                plt.ylabel("mse", fontsize = 16)
                plt.show()

                plt.figure(figsize = (7,5))
                plt.semilogx(ramping_L1_list[: len(loss_list_specific)], loss_list_specific)
                plt.xlabel("L1 amp", fontsize = 16)
                plt.ylabel("mse", fontsize = 16)
                plt.show()

                plt.figure(figsize = (7,5))
                for i in range(weights_list.shape[1]):
                    plt.semilogx(ramping_L1_list[: len(loss_list_specific)], weights_list[:,i], label = "weight_{0}".format(i))
                if len(bias_list) > 0:
                    plt.semilogx(ramping_L1_list[: len(loss_list_specific)], bias_list, label = "bias")
                plt.xlabel("L1 amp", fontsize = 16)
                plt.ylabel("parameter_values", fontsize = 16)
                plt.legend()
                plt.show()
                plt.clf()
                plt.close()
        else:
            raise Exception("mode {0} not recognized!".format(mode_ele))

        loss_dict[mode_ele] = {}
        if X is not None:
            loss_dict[mode_ele]["mse_record_whole"] = mse_record_whole
            loss_dict[mode_ele]["data_DL_whole"] = data_DL_whole
            loss_dict[mode_ele]["{0}_test".format(loss_type)] = loss_list
        loss_dict[mode_ele]["model_DL_whole"] = model_DL_whole
        if "param" in record_keys:
            loss_dict[mode_ele]["param_record_whole"] = param_record_whole
        if "param_grad" in record_keys:
            loss_dict[mode_ele]["param_grad_record_whole"] = param_grad_record_whole
        loss_dict[mode_ele]["iter_end_whole"] = iter_end_whole
        loss_dict[mode_ele]["event_list"] = event_list
        loss_dict[mode_ele]["is_accept_whole"] = is_accept_whole
        if mode_ele == "ramping-L1":
            loss_dict[mode_ele]["ramping_L1_list"] = ramping_L1_list
            loss_dict[mode_ele]["loss_list_specific"] = loss_list_specific

    if not is_list:
        model = model[0]
    
    return model, loss_dict


# ## Model architectures:

# ### MLP:

# In[3]:


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        struct_param = None,
        W_init_list = None,     # initialization for weights
        b_init_list = None,     # initialization for bias
        settings = {},          # Default settings for each layer, if the settings for the layer is not provided in struct_param
        is_cuda = False,
        ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.is_cuda = is_cuda
        self.settings = deepcopy(settings)
        if struct_param is not None:
            self.num_layers = len(struct_param)
            self.W_init_list = W_init_list
            self.b_init_list = b_init_list
            self.info_dict = {}

            self.init_layers(deepcopy(struct_param))
        else:
            self.num_layers = 0


    @property
    def struct_param(self):
        return [getattr(self, "layer_{0}".format(i)).struct_param for i in range(self.num_layers)]

    
    @property
    def output_size(self):
        return self.get_layer(-1).output_size


    @property
    def structure(self):
        structure = OrderedDict()
        structure["input_size"] = self.input_size
        structure["output_size"] = self.output_size
        structure["struct_param"] = self.struct_param if hasattr(self, "struct_param") else None
        return structure


    def init_layers(self, struct_param):
        res_forward = self.settings["res_forward"] if "res_forward" in self.settings else False
        for k, layer_struct_param in enumerate(struct_param):
            if res_forward:
                num_neurons_prev = struct_param[k - 1][0] + self.input_size if k > 0 else self.input_size
            else:
                num_neurons_prev = struct_param[k - 1][0] if k > 0 else self.input_size
            num_neurons = layer_struct_param[0]
            W_init = self.W_init_list[k] if self.W_init_list is not None else None
            b_init = self.b_init_list[k] if self.b_init_list is not None else None

            # Get settings for the current layer:
            layer_settings = deepcopy(self.settings) if bool(self.settings) else {}
            layer_settings.update(layer_struct_param[2])            

            # Construct layer:
            layer = get_Layer(layer_type = layer_struct_param[1],
                              input_size = num_neurons_prev,
                              output_size = num_neurons,
                              W_init = W_init,
                              b_init = b_init,
                              settings = layer_settings,
                              is_cuda = self.is_cuda,
                             )
            setattr(self, "layer_{}".format(k), layer)


    def forward(self, *input, p_dict=None, **kwargs):
        kwargs = filter_kwargs(kwargs, ["res_forward", "is_res_block", "act_noise_scale"])  # only allow certain kwargs to be passed
        if isinstance(input, tuple):
            input = torch.cat(input, -1)
        output = input
        res_forward = self.settings["res_forward"] if "res_forward" in self.settings else False
        is_res_block = self.settings["is_res_block"] if "is_res_block" in self.settings else False
        for k in range(len(self.struct_param)):
            p_dict_ele = p_dict[k] if p_dict is not None else None
            if res_forward and k > 0:
                output = getattr(self, "layer_{}".format(k))(torch.cat([output, input], -1), p_dict=p_dict_ele, **kwargs)
            else:
                output = getattr(self, "layer_{}".format(k))(output, p_dict=p_dict_ele, **kwargs)
        if is_res_block:
            output = output + input
        return output
    
    
    def copy(self):
        return deepcopy(self)


    def simplify(self, X=None, y=None, mode="full", isplot=False, target_name=None, validation_data = None, **kwargs):
        new_model, _ = simplify(self, X, y, mode=mode, isplot=isplot, target_name=target_name, validation_data=validation_data, **kwargs)
        self.__dict__.update(new_model.__dict__)
    
    
    def snap(self, snap_mode="integer", top=5, **kwargs):
        """Generate a set of new models whose parameters are snapped, each model with a different number of snapped parameters."""
        if not hasattr(self, "num_layers") or self.num_layers != 1:
            return False, [self]
        else:
            model_list = []
            top = top if snap_mode != "unsnap" else 1
            for top_ele in range(1, top + 1):
                new_model = self.copy()
                layer = new_model.layer_0
                info_list = layer.simplify(mode="snap", top=top_ele, snap_mode=snap_mode)
                if len(info_list) > 0:
                    new_model.reset_layer(0, layer)
                    model_list.append(new_model)
            is_succeed = len(model_list) > 0
            return is_succeed, model_list


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        reg = to_Variable([0], is_cuda=self.is_cuda)
        for k in range(len(self.struct_param)):
            layer = getattr(self, "layer_{}".format(k))
            reg = reg + layer.get_regularization(mode = mode, source = source)
        return reg


    def get_layer(self, layer_id):
        if layer_id < 0:
            layer_id += self.num_layers
        return getattr(self, "layer_{}".format(layer_id))


    def reset_layer(self, layer_id, layer):
        setattr(self, "layer_{}".format(layer_id), layer)


    def insert_layer(self, layer_id, layer):
        if layer_id < 0:
            layer_id += self.num_layers
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{}".format(layer_id + 1))
            if next_layer.struct_param[1] == "Simple_Layer":
                assert next_layer.input_size == layer.output_size, "The inserted layer's output_size {0} must be compatible with next layer_{1}'s input_size {2}!"                    .format(layer.output_size, layer_id + 1, next_layer.input_size)
        for i in range(self.num_layers - 1, layer_id - 1, -1):
            setattr(self, "layer_{}".format(i + 1), getattr(self, "layer_{}".format(i)))
        setattr(self, "layer_{}".format(layer_id), layer)
        self.num_layers += 1
    
    
    def remove_layer(self, layer_id):
        if layer_id < 0:
            layer_id += self.num_layers
        if layer_id < self.num_layers - 1:
            num_neurons_prev = self.struct_param[layer_id - 1][0] if layer_id > 0 else self.input_size
            replaced_layer = getattr(self, "layer_{}".format(layer_id + 1))
            if replaced_layer.struct_param[1] == "Simple_Layer":
                assert replaced_layer.input_size == num_neurons_prev,                     "After deleting layer_{0}, the replaced layer's input_size {1} must be compatible with previous layer's output neurons {2}!"                        .format(layer_id, replaced_layer.input_size, num_neurons_prev)
        for i in range(layer_id, self.num_layers - 1):
            setattr(self, "layer_{}".format(i), getattr(self, "layer_{}".format(i + 1)))
        self.num_layers -= 1


    def prune_neurons(self, layer_id, neuron_ids):
        if layer_id == "input":
            layer = self.get_layer(0)
            layer.prune_input_neurons(neuron_ids)
            self.input_size = layer.input_size
        else:
            if layer_id < 0:
                layer_id = self.num_layers + layer_id
            layer = getattr(self, "layer_{}".format(layer_id))
            layer.prune_output_neurons(neuron_ids)
            self.reset_layer(layer_id, layer)
            if layer_id < self.num_layers - 1:
                next_layer = getattr(self, "layer_{}".format(layer_id + 1))
                next_layer.prune_input_neurons(neuron_ids)
                self.reset_layer(layer_id + 1, next_layer)


    def add_neurons(self, layer_id, num_neurons, mode = ("imitation", "zeros")):
        if not isinstance(mode, list) and not isinstance(mode, tuple):
            mode = (mode, mode)
        if layer_id < 0:
            layer_id = self.num_layers + layer_id
        layer = getattr(self, "layer_{}".format(layer_id))
        layer.add_output_neurons(num_neurons, mode = mode[0])
        self.reset_layer(layer_id, layer)
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{}".format(layer_id + 1))
            next_layer.add_input_neurons(num_neurons, mode = mode[1])
            self.reset_layer(layer_id + 1, next_layer)
        if layer_id == 0:
            self.input_size = self.get_layer(0).input_size

    
    def inspect_operation(self, input, operation_between, p_dict = None, **kwargs):
        output = input
        res_forward = self.settings["res_forward"] if "res_forward" in self.settings else False
        is_res_block = self.settings["is_res_block"] if "is_res_block" in self.settings else False
        for k in range(*operation_between):
            p_dict_ele = p_dict[k] if p_dict is not None else None
            if res_forward and k > 0:
                output = getattr(self, "layer_{}".format(k))(torch.cat([output, input], -1), p_dict = p_dict_ele)
            else:
                output = getattr(self, "layer_{}".format(k))(output, p_dict = p_dict_ele)
        if is_res_block:
            output = output + input
        return output


    def get_weights_bias(self, W_source = "core", b_source = "core", layer_ids = None, is_grad = False, isplot = False, verbose = False, raise_error = True):
        if not hasattr(self, "struct_param"):
            return None, None
        layer_ids = range(len(self.struct_param)) if layer_ids is None else layer_ids
        W_list = []
        b_list = []
        if W_source is not None:
            for k in range(len(self.struct_param)):
                if k in layer_ids:
                    if W_source == "core":
                        try:
                            W, _ = getattr(self, "layer_{}".format(k)).get_weights_bias(is_grad = is_grad)
                        except Exception as e:
                            if raise_error:
                                raise
                            else:
                                print(e)
                            W = np.array([np.NaN])
                    else:
                        raise Exception("W_source '{}' not recognized!".format(W_source))
                    W_list.append(W)
        
        if b_source is not None:
            for k in range(len(self.struct_param)):
                if k in layer_ids:
                    if b_source == "core":
                        try:
                            _, b = getattr(self, "layer_{}".format(k)).get_weights_bias(is_grad = is_grad)
                        except Exception as e:
                            if raise_error:
                                raise
                            else:
                                print(e)
                            b = np.array([np.NaN])
                    else:
                        raise Exception("b_source '{}' not recognized!".format(b_source))
                b_list.append(b)

        if verbose:
            import pprint as pp
            if W_source is not None:
                print("weight:")
                pp.pprint(W_list)
            if b_source is not None:
                print("bias:")
                pp.pprint(b_list)
                
        if isplot:
            if W_source is not None:
                print("weight {}:".format(W_source))
                plot_matrices(W_list)
            if b_source is not None:
                print("bias {}:".format(b_source))
                plot_matrices(b_list)

        return W_list, b_list


    def split_to_model_ensemble(self, mode = "standardize"):
        num_models = self.struct_param[-1][0]
        model_core = deepcopy(self)
        if mode == "standardize":
            last_layer = getattr(model_core, "layer_{}".format(model_core.num_layers - 1))
            last_layer.standardize(mode = "b_mean_zero")
        else:
            raise Exception("mode {} not recognized!".format(mode))
        model_list = [deepcopy(model_core) for i in range(num_models)]
        for i, model in enumerate(model_list):
            to_prune = list(range(num_models))
            to_prune.pop(i)
            model.prune_neurons(-1, to_prune)
        return construct_model_ensemble_from_nets(model_list)
    
    
    @property
    def model_dict(self):
        model_dict = {"type": self.__class__.__name__}
        model_dict["input_size"] = self.input_size
        model_dict["struct_param"] = get_full_struct_param(self.struct_param, self.settings)
        model_dict["weights"], model_dict["bias"] = self.get_weights_bias(W_source = "core", b_source = "core")
        model_dict["settings"] = deepcopy(self.settings)
        model_dict["net_type"] = self.__class__.__name__
        return model_dict


    @property
    def DL(self):
        return np.sum([getattr(self, "layer_{}".format(i)).DL for i in range(self.num_layers)])


    def load_model_dict(self, model_dict):
        new_net = load_model_dict_net(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)
    
    
    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


    def get_loss(self, input, target, criterion, **kwargs):
        y_pred = self(input, **kwargs)
        return criterion(y_pred, target)


    def prepare_inspection(self, X, y, **kwargs):
        return {}


    def set_cuda(self, is_cuda):
        for k in range(self.num_layers):
            getattr(self, "layer_{}".format(k)).set_cuda(is_cuda)
        self.is_cuda = is_cuda


    def set_trainable(self, is_trainable):
        for k in range(self.num_layers):
            getattr(self, "layer_{}".format(k)).set_trainable(is_trainable)


    def get_snap_dict(self):
        snap_dict = {}
        for k in range(len(self.struct_param)):
            layer = getattr(self, "layer_{}".format(k))
            if hasattr(layer, "snap_dict"):
                recorded_layer_snap_dict = {}
                for key, item in layer.snap_dict.items():
                    recorded_layer_snap_dict[key] = {"new_value": item["new_value"]}
                if len(recorded_layer_snap_dict) > 0:
                    snap_dict[k] = recorded_layer_snap_dict
        return snap_dict


    def synchronize_settings(self):
        snap_dict = self.get_snap_dict()
        if len(snap_dict) > 0:
            self.settings["snap_dict"] = snap_dict
        return self.settings


    def get_sympy_expression(self, verbose = True):
        expressions = {i: {} for i in range(self.num_layers)}
        for i in range(self.num_layers):
            layer = getattr(self, "layer_{}".format(i))
            if layer.struct_param[1] == "Symbolic_Layer":
                if verbose:
                    print("Layer {}, symbolic_expression:  {}".format(i, layer.symbolic_expression))
                    print("          numerical_expression: {}".format(layer.numerical_expression))
                expressions[i]["symbolic_expression"] = layer.symbolic_expression
                expressions[i]["numerical_expression"] = layer.numerical_expression
                expressions[i]["param_dict"] = layer.get_param_dict()
                expressions[i]["DL"] = layer.DL
            else:
                if verbose:
                    print("Layer {} is not a symbolic layer.".format(i))
                expressions[i] = None
        return expressions


# ### Labelmix_MLP:

# In[ ]:


class Labelmix_MLP(nn.Module):
    def __init__(
        self,
        input_size,
        struct_param,
        idx_label=None,
        is_cuda=False,
    ):
        super(Labelmix_MLP, self).__init__()
        self.input_size = input_size
        self.struct_param = struct_param
        self.num_layers = len(struct_param)
        if idx_label is not None and len(idx_label) == input_size:
            idx_label = None
        if idx_label is not None:
            self.idx_label = torch.LongTensor(idx_label)
            idx_main = list(set(range(input_size)) - set(to_np_array(idx_label).astype(int).tolist()))
            self.idx_main = torch.LongTensor(idx_main)
        else:
            self.idx_label = None
            self.idx_main = torch.LongTensor(list(range(input_size)))
        num_neurons_prev = len(self.idx_main)
        for i, layer_struct_param in enumerate(struct_param):
            num_neurons = layer_struct_param[0]
            setattr(self, "W_{}_main".format(i), nn.Parameter(torch.randn(num_neurons_prev, num_neurons)))
            setattr(self, "b_{}_main".format(i), nn.Parameter(torch.zeros(num_neurons)))
            init_weight(getattr(self, "W_{}_main".format(i)), init=None)
            num_neurons_prev = num_neurons
            if self.idx_label is not None:
                setattr(self, "W_{}_mul".format(i), nn.Parameter(torch.randn(len(self.idx_label), num_neurons)))
                setattr(self, "W_{}_add".format(i), nn.Parameter(torch.randn(len(self.idx_label), num_neurons)))
                init_weight(getattr(self, "W_{}_mul".format(i)), init=None)
                init_weight(getattr(self, "W_{}_add".format(i)), init=None)
                setattr(self, "b_{}_mul".format(i), nn.Parameter(torch.zeros(num_neurons)))
                setattr(self, "b_{}_add".format(i), nn.Parameter(torch.zeros(num_neurons)))
        self.set_cuda(is_cuda)
    

    def forward(self, input):
        output = input[:, self.idx_main]
        if self.idx_label is not None:
            labels = input[:, self.idx_label]
        for i, layer_struct_param in enumerate(self.struct_param):
            output = torch.matmul(output, getattr(self, "W_{}_main".format(i))) + getattr(self, "b_{}_main".format(i))
            if "activation" in layer_struct_param[2]:
                output = get_activation(layer_struct_param[2]["activation"])(output)
            if self.idx_label is not None:
                A_mul = torch.matmul(labels, getattr(self, "W_{}_mul".format(i))) + getattr(self, "b_{}_mul".format(i))
                A_add = torch.matmul(labels, getattr(self, "W_{}_add".format(i))) + getattr(self, "b_{}_add".format(i))
                output = output * A_mul + A_add
        return output
    
    
    def get_loss(self, X, y, criterion, **kwargs):
        y_pred = self(X)
        return criterion(y_pred, y)


    def set_cuda(self, is_cuda):
        if isinstance(is_cuda, str):
            self.cuda(is_cuda)
        else:
            if is_cuda:
                self.cuda()
            else:
                self.cpu()
        self.is_cuda = is_cuda


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        reg = to_Variable([0], is_cuda=self.is_cuda)
        return reg


    @property
    def model_dict(self):
        model_dict = {"type": "Labelmix_MLP"}
        model_dict["input_size"] = self.input_size
        model_dict["struct_param"] = self.struct_param
        if self.idx_label is not None:
            model_dict["idx_label"] = to_np_array(self.idx_label).astype(int)
        model_dict["state_dict"] = to_cpu_recur(self.state_dict())
        return model_dict


# ### Multi_MLP (MLPs in series):

# In[ ]:


class Multi_MLP(nn.Module):
    def __init__(
        self,
        input_size,
        struct_param,
        W_init_list = None,     # initialization for weights
        b_init_list = None,     # initialization for bias
        settings = None,          # Default settings for each layer, if the settings for the layer is not provided in struct_param
        is_cuda = False,
        ):
        super(Multi_MLP, self).__init__()
        self.input_size = input_size
        self.num_layers = len(struct_param)
        self.W_init_list = W_init_list
        self.b_init_list = b_init_list
        self.settings = deepcopy(settings)
        self.num_blocks = len(struct_param)
        self.is_cuda = is_cuda
        
        for i, struct_param_ele in enumerate(struct_param):
            input_size_block = input_size if i == 0 else struct_param[i - 1][-1][0]
            setattr(self, "block_{0}".format(i), MLP(input_size = input_size_block,
                                                     struct_param = struct_param_ele,
                                                     W_init_list = W_init_list[i] if W_init_list is not None else None,
                                                     b_init_list = b_init_list[i] if b_init_list is not None else None,
                                                     settings = self.settings[i] if self.settings is not None else {},
                                                     is_cuda = self.is_cuda,
                                                    ))
    
    def forward(self, input):
        output = input
        for i in range(self.num_blocks):
            output = getattr(self, "block_{0}".format(i))(output)
        return output


    def get_loss(self, input, target, criterion, **kwargs):
        y_pred = self(input, **kwargs)
        return criterion(y_pred, target)


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for i in range(self.num_blocks):
            reg = reg + getattr(self, "block_{0}".format(i)).get_regularization(mode = mode, source = source)
        return reg


    @property
    def struct_param(self):
        return [getattr(self, "block_{0}".format(i)).struct_param for i in range(self.num_blocks)]


    @property
    def model_dict(self):
        model_dict = {"type": self.__class__.__name__}
        model_dict["input_size"] = self.input_size
        model_dict["struct_param"] = self.struct_param
        model_dict["weights"], model_dict["bias"] = self.get_weights_bias(W_source = "core", b_source = "core")
        model_dict["settings"] = deepcopy(self.settings)
        model_dict["net_type"] = self.__class__.__name__
        return model_dict


    def load_model_dict(self, model_dict):
        new_net = load_model_dict_Multi_MLP(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


    def get_weights_bias(self, W_source = "core", b_source = "core"):
        W_list = []
        b_list = []
        for i in range(self.num_blocks):
            W, b = getattr(self, "block_{0}".format(i)).get_weights_bias(W_source = W_source, b_source = b_source)
            W_list.append(W)
            b_list.append(b)
        return deepcopy(W_list), deepcopy(b_list)


    def prepare_inspection(self, X, y, **kwargs):
        return {}


    def set_cuda(self, is_cuda):
        for i in range(self.num_blocks):
            getattr(self, "block_{0}".format(i)).set_cuda(is_cuda)
        self.is_cuda = is_cuda


    def set_trainable(self, is_trainable):
        for i in range(self.num_blocks):
            getattr(self, "block_{0}".format(i)).set_trainable(is_trainable)


# ### Branching_Net:

# In[ ]:


class Branching_Net(nn.Module):
    """An MLP that consists of a base network, and net_1 and net_2 that branches off from the output of the base network."""
    def __init__(
        self,
        net_base_model_dict,
        net_1_model_dict,
        net_2_model_dict,
        is_cuda = False,
        ):
        super(Branching_Net, self).__init__()
        self.net_base = load_model_dict(net_base_model_dict, is_cuda = is_cuda)
        self.net_1 = load_model_dict(net_1_model_dict, is_cuda = is_cuda)
        self.net_2 = load_model_dict(net_2_model_dict, is_cuda = is_cuda)
        self.info_dict = {}
    
    
    def forward(self, X, **kwargs):
        shared = self.net_base(X)
        shared = shared.max(0, keepdim = True)[0]
        return self.net_1(shared)[0], self.net_2(shared)[0]


    def get_regularization(self, source = ["weights", "bias"], mode = "L1"):
        reg = self.net_base.get_regularization(source = source, mode = mode) +               self.net_1.get_regularization(source = source, mode = mode) +               self.net_2.get_regularization(source = source, mode = mode)
        return reg


    def set_trainable(self, is_trainable):
        self.net_base.set_trainable(is_trainable)
        self.net_1.set_trainable(is_trainable)
        self.net_2.set_trainable(is_trainable)


    def prepare_inspection(self, X, y, **kwargs):
        return deepcopy(self.info_dict)


    @property
    def model_dict(self):
        model_dict = {"type": "Branching_Net"}
        model_dict["net_base_model_dict"] = self.net_base.model_dict
        model_dict["net_1_model_dict"] = self.net_1.model_dict
        model_dict["net_2_model_dict"] = self.net_2.model_dict
        return model_dict


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)

    
class Fan_in_MLP(nn.Module):
    def __init__(
        self,
        model_dict_branch1,
        model_dict_branch2,
        model_dict_joint,
        is_cuda=False,
    ):
        super(Fan_in_MLP, self).__init__()
        if model_dict_branch1 is not None:
            self.net_branch1 = load_model_dict(model_dict_branch1, is_cuda=is_cuda)
        else:
            self.net_branch1 = None
        if model_dict_branch2 is not None:
            self.net_branch2 = load_model_dict(model_dict_branch2, is_cuda=is_cuda)
        else:
            self.net_branch2 = None
        self.net_joint = load_model_dict(model_dict_joint, is_cuda=is_cuda)
        self.is_cuda = is_cuda
        self.info_dict = {}
    
    def forward(self, X1, X2, is_outer=False):
        if is_outer:
            X2 = X2[...,None,:]
        if self.net_branch1 is not None:
            X1 = self.net_branch1(X1)
        if self.net_branch2 is not None:
            X2 = self.net_branch2(X2)
        X1, X2 = broadcast_all(X1, X2)
        out = torch.cat([X1, X2], -1)
        # if is_outer=True, then output dimension: [..., X2dim, X1dim, out_dim]:
        return self.net_joint(out).squeeze(-1)
    
    def get_loss(self, input, target, criterion, **kwargs):
        X1, X2 = input
        y_pred = self(X1, X2)
        return criterion(y_pred, target)
    
    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        if self.net_branch1 is not None:
            reg = reg + self.net_branch1.get_regularization(source=source, mode=mode)
        if self.net_branch2 is not None:
            reg = reg + self.net_branch2.get_regularization(source=source, mode=mode)
        return reg
    
    def prepare_inspection(self, X, y, **kwargs):
        return deepcopy(self.info_dict)
    
    @property
    def model_dict(self):
        model_dict = {'type': self.__class__.__name__}
        model_dict["model_dict_branch1"] = self.net_branch1.model_dict if self.net_branch1 is not None else None
        model_dict["model_dict_branch2"] = self.net_branch2.model_dict if self.net_branch2 is not None else None
        model_dict["model_dict_joint"] = self.net_joint.model_dict
        return model_dict


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


# ###  Mixture_Model:

# In[ ]:


class Mixture_Model(nn.Module):
    def __init__(
        self,
        model_dict_list,
        weight_logits_model_dict,
        num_components,
        is_cuda=False,
    ):
        super(Mixture_Model, self).__init__()
        self.num_components = num_components
        for i in range(self.num_components):
            if isinstance(model_dict_list, list):
                setattr(self, "model_{}".format(i), load_model_dict(model_dict_list[i], is_cuda=is_cuda))
            else:
                assert isinstance(model_dict_list, dict)
                setattr(self, "model_{}".format(i), load_model_dict(model_dict_list, is_cuda=is_cuda))
        self.weight_logits_model = load_model_dict(weight_logits_model_dict, is_cuda=is_cuda)
        self.is_cuda = is_cuda


    def forward(self, input):
        output_list = []
        for i in range(self.num_components):
            output = getattr(self, "model_{}".format(i))(input)
            output_list.append(output)
        output_list = torch.stack(output_list, -1)
        weight_logits = self.weight_logits_model(input)
        return output_list, weight_logits

    @property
    def model_dict(self):
        model_dict = {"type": "Mixture_Model",
                      "model_dict_list": [getattr(self, "model_{}".format(i)).model_dict for i in range(self.num_components)],
                      "weight_logits_model_dict": self.weight_logits_model.model_dict,
                      "num_components": self.num_components,
                     }
        return model_dict


# ### Model_Ensemble:

# In[ ]:


class Model_Ensemble(nn.Module):
    """Model_Ensemble is a collection of models with the same architecture 
       but independent parameters"""
    def __init__(
        self,
        num_models,
        input_size,
        struct_param,
        W_init_list = None,
        b_init_list = None,
        settings = None,
        net_type = "MLP",
        is_cuda = False,
        ):
        super(Model_Ensemble, self).__init__()
        self.num_models = num_models
        self.input_size = input_size
        self.net_type = net_type
        self.is_cuda = is_cuda
        for i in range(self.num_models):
            if settings is None:
                settings_model = {}
            elif isinstance(settings, list) or isinstance(settings, tuple):
                settings_model = settings[i]
            else:
                settings_model = settings
            if isinstance(struct_param, tuple):
                struct_param_model = struct_param[i]
            else:
                struct_param_model = struct_param
            if net_type == "MLP":
                net = MLP(input_size = self.input_size,
                          struct_param = deepcopy(struct_param_model),
                          W_init_list = deepcopy(W_init_list[i]) if W_init_list is not None else None,
                          b_init_list = deepcopy(b_init_list[i]) if b_init_list is not None else None,
                          settings = deepcopy(settings_model),
                          is_cuda = is_cuda,
                         )
            elif net_type == "ConvNet":
                net = ConvNet(input_channels = self.input_size,
                              struct_param = deepcopy(struct_param_model),
                              settings = deepcopy(settings_model),
                              is_cuda = is_cuda,
                             )
            else:
                raise Exception("Net_type {0} not recognized!".format(net_type))
            setattr(self, "model_{0}".format(i), net)


    @property
    def struct_param(self):
        return tuple(getattr(self, "model_{0}".format(i)).struct_param for i in range(self.num_models))


    @property
    def settings(self):
        return [getattr(self, "model_{0}".format(i)).settings for i in range(self.num_models)]
    
    
    def get_all_models(self):
        return [getattr(self, "model_{0}".format(i)) for i in range(self.num_models)]


    def init_bias_with_input(self, input, mode = "std_sqrt", neglect_last_layer = True):
        for i in range(self.num_models):
            model = getattr(self, "model_{0}".format(i))
            model.init_bias_with_input(input, mode = mode, neglect_last_layer = neglect_last_layer)
    
    
    def initialize_param_freeze(self, update_values = True):
        for i in range(self.num_models):
            model = getattr(self, "model_{0}".format(i))
            model.initialize_param_freeze(update_values = update_values)
    
    
    def apply_model(self, input, model_id):
        return fetch_model(self, model_id)(input)


    def fetch_model(self, model_id):
        return getattr(self, "model_{0}".format(model_id))


    def set_trainable(self, is_trainable):
        for i in range(self.num_models):
            getattr(self, "model_{0}".format(i)).set_trainable(is_trainable)


    def forward(self, input):
        output_list = []
        for i in range(self.num_models):
            if self.net_type == "MLP":
                output = getattr(self, "model_{0}".format(i))(input)
            elif self.net_type == "ConvNet":
                output = getattr(self, "model_{0}".format(i))(input)[0]
            else:
                raise Exception("Net_type {0} not recognized!".format(self.net_type))
            output_list.append(output)
        return torch.stack(output_list, 1)


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        if not isinstance(source, list):
            source = [source]
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        model0 = self.model_0
        # Elastic_weight_reg:
        if "elastic_weight" in source or "elastic_bias" in source:
            # Setting up excluded layer:
            excluded_layer = kwargs["excluded_layer"] if "excluded_layer" in kwargs else [-1]
            if not isinstance(excluded_layer, list):
                excluded_layer = [excluded_layer]
            excluded_layer = [element + model0.num_layers if element < 0 else element for element in excluded_layer]
            elastic_mode = kwargs["elastic_mode"] if "elastic_mode" in kwargs else "var"
            
            # Compute the elastic_weight_reg:
            for k in range(model0.num_layers):
                if k in excluded_layer:
                    continue
                W_accum_k = []
                b_accum_k = []
                num_neurons_prev = model0.struct_param[k - 1][0] if k > 0 else self.input_size
                num_neurons = model0.struct_param[k][0]
                for i in range(self.num_models):
                    model = getattr(self, "model_{0}".format(i))
                    assert model0.num_layers == model.num_layers
                    assert num_neurons_prev == model.struct_param[k - 1][0] if k > 0 else model.input_size,                             "all models' input/output size at each layer must be identical!"
                    assert num_neurons == model.struct_param[k][0],                             "all models' input/output size at each layer must be identical!"
                    layer_k = getattr(model, "layer_{0}".format(k))
                    if "elastic_weight" in source:
                        W_accum_k.append(layer_k.W_core)
                    if "elastic_bias" in source:
                        b_accum_k.append(layer_k.b_core)
                if "elastic_weight" in source:
                    if elastic_mode == "var":
                        reg = reg + torch.stack(W_accum_k, -1).var(-1).sum()
                    elif elastic_mode == "std":
                        reg = reg + torch.stack(W_accum_k, -1).std(-1).sum()
                    else:
                        raise
                if "elastic_bias" in source:
                    if elastic_mode == "var":
                        reg = reg + torch.stack(b_accum_k, -1).var(-1).sum()
                    elif elastic_mode == "std":
                        reg = reg + torch.stack(b_accum_k, -1).std(-1).sum()
                    else:
                        raise
            source_core = deepcopy(source)
            if "elastic_weight" in source_core:
                source_core.remove("elastic_weight")
            if "elastic_bias" in source_core:
                source_core.remove("elastic_bias")
        else:
            source_core = source
        
        # Other regularizations:
        for k in range(self.num_models):
            reg = reg + getattr(self, "model_{0}".format(k)).get_regularization(source = source_core, mode = mode, **kwargs)
        return reg
    
    
    def get_weights_bias(self, W_source = None, b_source = None, verbose = False, isplot = False):
        W_list_dict = {}
        b_list_dict = {}
        for i in range(self.num_models):
            if verbose:
                print("\nmodel {0}:".format(i))
            W_list_dict[i], b_list_dict[i] = getattr(self, "model_{0}".format(i)).get_weights_bias(
                W_source = W_source, b_source = b_source, verbose = verbose, isplot = isplot)
        return W_list_dict, b_list_dict
    
    
    def combine_to_net(self, mode = "mean", last_layer_mode = "concatenate"):
        model0 = self.model_0
        if mode == "mean":
            struct_param = deepcopy(model0.struct_param)
            settings = deepcopy(model0.settings)
            W_init_list = []
            b_init_list = []
            for k in range(model0.num_layers):
                num_neurons_prev = model0.struct_param[k - 1][0] if k > 0 else self.input_size
                num_neurons = model0.struct_param[k][0]
                W_accum_k = []
                b_accum_k = []
                for i in range(self.num_models):
                    model = getattr(self, "model_{0}".format(i))
                    assert model0.num_layers == model.num_layers
                    assert num_neurons_prev == model.struct_param[k - 1][0] if k > 0 else model.input_size,                             "If mode == 'mean', all models' input/output size at each layer must be identical!"
                    assert num_neurons == model.struct_param[k][0],                             "If mode == 'mean', all models' input/output size at each layer must be identical!"
                    layer_k = getattr(model, "layer_{0}".format(k))
                    W_accum_k.append(layer_k.W_core)
                    b_accum_k.append(layer_k.b_core)

                if k == model0.num_layers - 1:
                    current_mode = last_layer_mode
                else:
                    current_mode = mode

                if current_mode == "mean":
                    W_accum_k = torch.stack(W_accum_k, -1).mean(-1)
                    b_accum_k = torch.stack(b_accum_k, -1).mean(-1)
                elif current_mode == "concatenate":
                    W_accum_k = torch.cat(W_accum_k, -1)
                    b_accum_k = torch.cat(b_accum_k, -1)
                    struct_param[-1][0] = sum([self.struct_param[i][-1][0] for i in range(self.num_models)])
                else:
                    raise Exception("mode {0} not recognized!".format(last_layer_mode))
                W_init_list.append(W_accum_k.data.numpy())
                b_init_list.append(b_accum_k.data.numpy())
            
            # Build the net:
            net = MLP(input_size = self.input_size,
                      struct_param = struct_param,
                      W_init_list = W_init_list, 
                      b_init_list = b_init_list,
                      settings = settings,
                     )
        else:
            raise Exception("mode {0} not recognized!".format(mode))
        return net
        
    
    def remove_models(self, model_ids):
        if not isinstance(model_ids, list):
            model_ids = [model_ids]
        model_list = []
        k = 0
        for i in range(self.num_models):
            if i not in model_ids:
                if k != i:
                    setattr(self, "model_{0}".format(k), getattr(self, "model_{0}".format(i)))
                k += 1
        num_models_new = k
        for i in range(num_models_new, self.num_models):
            delattr(self, "model_{0}".format(i))
        self.num_models = num_models_new


    def add_models(self, models):
        if not isinstance(models, list):
            models = [models]
        for i, model in enumerate(models):
            setattr(self, "model_{0}".format(i + self.num_models), model)
        self.num_models += len(models)


    def simplify(self, X, y, idx, mode = "full", validation_data = None, isplot = False, **kwargs):
        def process_idx(idx):
            idx = idx.byte()
            if len(idx.size()) == 1:
                idx = idx.unqueeze(1)
            if idx.size(1) == 1:
                idx = idx.repeat(1, self.num_models)
            return idx
        idx = process_idx(idx)
        if validation_data is not None:
            X_valid, y_valid, idx_valid = validation_data
            idx_valid = process_idx(idx_valid)        
        
        loss_dict = {}
        for i in range(self.num_models):
            model = getattr(self, "model_{0}".format(i))
            X_chosen = torch.masked_select(X, idx[:, i:i+1]).view(-1, X.size(1))
            y_chosen = torch.masked_select(y, idx[:, i:i+1]).view(-1, y.size(1))
            if validation_data is not None:
                X_valid_chosen = torch.masked_select(X_valid, idx_valid[:, i:i+1]).view(-1, X_valid.size(1))
                y_valid_chosen = torch.masked_select(y_valid, idx_valid[:, i:i+1]).view(-1, y_valid.size(1))
                if len(X_valid_chosen) == 0:
                    validation_data_chosen = None
                else:
                    validation_data_chosen = (X_valid_chosen, y_valid_chosen)
            else:
                validation_data_chosen = None
            if len(X_chosen) == 0:
                print("The {0}'th model has no corresponding data to simplify with, skip.".format(i))
            else:
                new_model, loss_dict["model_{0}".format(i)] = simplify(model, X_chosen, y_chosen, mode = mode, validation_data = validation_data_chosen, isplot = isplot, target_name = "model_{0}".format(i), **kwargs)
                setattr(self, "model_{0}".format(i), new_model)
        return loss_dict
    
    
    def get_sympy_expression(self):
        expressions = {}
        for k in range(self.num_models):
            print("\nmodel {0}:".format(k))
            expressions["model_{0}".format(k)] = getattr(self, "model_{0}".format(k)).get_sympy_expression()
        return expressions


    @property
    def DL(self):
        return np.sum([getattr(self, "model_{0}".format(i)).DL for i in range(self.num_models)])


    def get_weights_bias(self, W_source = None, b_source = None, verbose = False, isplot = False):
        W_list_dict = {}
        b_list_dict = {}
        for i in range(self.num_models):
            if verbose:
                print("\nmodel {0}:".format(i))
            W_list_dict[i], b_list_dict[i] = getattr(self, "model_{0}".format(i)).get_weights_bias(W_source = W_source, b_source = b_source, verbose = verbose, isplot = isplot)
        return W_list_dict, b_list_dict


    @property
    def model_dict(self):
        model_dict = {"type": "Model_Ensemble"}
        for i in range(self.num_models):
            model_dict["model_{0}".format(i)] = getattr(self, "model_{0}".format(i)).model_dict
        model_dict["input_size"] = self.input_size
        model_dict["struct_param"] = self.struct_param
        model_dict["num_models"] = self.num_models
        model_dict["net_type"] = self.net_type
        return model_dict


    def load_model_dict(self, model_dict):
        new_model_ensemble = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_model_ensemble.__dict__)


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)

        
def load_model_dict_model_ensemble(model_dict, is_cuda = False):
    num_models = len([model_name for model_name in model_dict if model_name[:6] == "model_"])
    return Model_Ensemble(num_models = num_models,
                          input_size = model_dict["input_size"],
                          struct_param = tuple([deepcopy(model_dict["model_{0}".format(i)]["struct_param"]) for i in range(num_models)]),
                          W_init_list = [deepcopy(model_dict["model_{0}".format(i)]["weights"]) for i in range(num_models)],
                          b_init_list = [deepcopy(model_dict["model_{0}".format(i)]["bias"]) for i in range(num_models)],
                          settings = [deepcopy(model_dict["model_{0}".format(i)]["settings"]) for i in range(num_models)],
                          net_type = model_dict["net_type"] if "net_type" in model_dict else "MLP",
                          is_cuda = is_cuda,
                         )


def combine_model_ensembles(model_ensembles, input_size):
    model_ensembles = deepcopy(model_ensembles)
    model_ensemble_combined = None
    model_id = 0
    for k, model_ensemble in enumerate(model_ensembles):
        if model_ensemble.input_size == input_size:
            if model_ensemble_combined is None:
                model_ensemble_combined = model_ensemble
        else:
            continue  
        for i in range(model_ensemble.num_models):
            model = getattr(model_ensemble, "model_{0}".format(i))
            setattr(model_ensemble_combined, "model_{0}".format(model_id), model)
            model_id += 1
    model_ensemble_combined.num_models = model_id
    return model_ensemble_combined


def construct_model_ensemble_from_nets(nets):
    num_models = len(nets)
    if num_models is None:
        return None
    input_size = nets[0].input_size
    struct_param = tuple(net.struct_param for net in nets)
    is_cuda = False
    for net in nets:
        if net.input_size != input_size:
            raise Exception("The input_size for all nets must be the same!")
        if net.is_cuda:
            is_cuda = True
    model_ensemble = Model_Ensemble(num_models = num_models, input_size = input_size, struct_param = struct_param, is_cuda = is_cuda)
    for i, net in enumerate(nets):
        setattr(model_ensemble, "model_{0}".format(i), net)
    return model_ensemble


# In[ ]:


class Model_with_uncertainty(nn.Module):
    def __init__(
        self,
        model_pred,
        model_logstd,
        ):
        super(Model_with_uncertainty, self).__init__()
        self.model_pred = model_pred
        self.model_logstd = model_logstd
        
    def forward(self, input, noise_amp = None, **kwargs):
        return self.model_pred(input, noise_amp = noise_amp, **kwargs), self.model_logstd(input, **kwargs)
    
    def get_loss(self, input, target, criterion, noise_amp = None, **kwargs):
        pred, log_std = self(input, noise_amp = noise_amp, **kwargs)
        return criterion(pred = pred, target = target, log_std = log_std)
    
    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        return self.model_pred.get_regularization(source = source, mode = mode, **kwargs) + self.model_logstd.get_regularization(source = source, mode = mode, **kwargs)
    
    @property
    def model_dict(self):
        model_dict = {}
        model_dict["type"] = "Model_with_Uncertainty"
        model_dict["model_pred"] = self.model_pred.model_dict
        model_dict["model_logstd"] = self.model_logstd.model_dict
        return model_dict

    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)

    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)

    def set_cuda(self, is_cuda):
        self.model_pred.set_cuda(is_cuda)
        self.model_logstd.set_cuda(is_cuda)
        
    def set_trainable(self, is_trainable):
        self.model_pred.set_trainable(is_trainable)
        self.model_logstd.set_trainable(is_trainable)


# ### RNN:

# In[ ]:


class RNNCellBase(nn.Module):
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


# ### LSTM:

# In[ ]:


class LSTM(RNNCellBase):
    """a LSTM class"""
    def __init__(
        self,
        input_size,
        hidden_size,
        output_struct_param,
        output_settings = {},
        bias = True,
        is_cuda = False,
        ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.output_net = MLP(input_size = self.hidden_size, struct_param = output_struct_param, settings = output_settings, is_cuda = is_cuda)
        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)
        self.reset_parameters()
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")
        self.to(self.device)

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_one_step(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return self._backend.LSTMCell(
            input, hx,
            self.W_ih, self.W_hh,
            self.b_ih, self.b_hh,
        )
    
    def forward(self, input, hx = None):
        if hx is None:
            hx = [torch.randn(input.size(0), self.hidden_size).to(self.device),
                  torch.randn(input.size(0), self.hidden_size).to(self.device),
                 ]
        hhx, ccx = hx
        for i in range(input.size(1)):
            hhx, ccx = self.forward_one_step(input[:, i], (hhx, ccx))
        output = self.output_net(hhx)
        return output

    def get_regularization(self, source, mode = "L1", **kwargs):
        if not isinstance(source, list):
            source = [source]
        reg = self.output_net.get_regularization(source = source, mode = mode)
        for source_ele in source:
            if source_ele == "weight":
                if mode == "L1":
                    reg = reg + self.W_ih.abs().sum() + self.W_hh.abs().sum()
                elif mode == "L2":
                    reg = reg + (self.W_ih ** 2).sum() + (self.W_hh ** 2).sum()
                else:
                    raise Exception("mode {0} not recognized!".format(mode))
            elif source_ele == "bias":
                if self.bias:
                    if mode == "L1":
                        reg = reg + self.b_ih.abs().sum() + self.b_hh.abs().sum()
                    elif mode == "L2":
                        reg = reg + (self.b_ih ** 2).sum() + (self.b_hh ** 2).sum()
                    else:
                        raise Exception("mode {0} not recognized!".format(mode))
            else:
                raise Exception("source {0} not recognized!".format(source_ele))
        return reg
    
    def get_weights_bias(self, W_source = None, b_source = None, verbose = False, isplot = False):
        W_dict = OrderedDict()
        b_dict = OrderedDict()
        W_o, b_o = self.output_net.get_weights_bias(W_source = W_source, b_source = b_source)
        if W_source == "core":
            W_dict["W_ih"] = self.W_ih.cpu().detach().numpy()
            W_dict["W_hh"] = self.W_hh.cpu().detach().numpy()
            W_dict["W_o"] = W_o
            if isplot:
                print("W_ih, W_hh:")
                plot_matrices([W_dict["W_ih"], W_dict["W_hh"]])
                print("W_o:")
                plot_matrices(W_o)
        if self.bias and b_source == "core":
            b_dict["b_ih"] = self.b_ih.cpu().detach().numpy()
            b_dict["b_hh"] = self.b_hh.cpu().detach().numpy()
            b_dict["b_o"] = b_o
            if isplot:
                print("b_ih, b_hh:")
                plot_matrices([b_dict["b_ih"], b_dict["b_hh"]])
                print("b_o:")
                plot_matrices(b_o)
        return W_dict, b_dict
    
    def get_loss(self, input, target, criterion, hx = None, **kwargs):
        y_pred = self(input, hx = hx)
        return criterion(y_pred, target)
    
    def prepare_inspection(self, X, y, **kwargs):
        return {}

    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)

    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


# ### Wide ResNet:

# In[ ]:


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate=None, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Wide_ResNet(nn.Module):
    """Adapted from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py"""
    def __init__(
        self,
        depth,
        widen_factor,
        input_channels,
        output_size,
        dropout_rate=None,
        is_cuda=False,
    ):
        super(Wide_ResNet, self).__init__()

        self.depth = depth
        self.widen_factor = widen_factor
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.output_size = output_size

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        nStages = [16*k, 16*k, 32*k, 64*k]
        self.in_planes = nStages[0]

        self.conv1 = conv3x3(self.input_channels,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], output_size)
        self.set_cuda(is_cuda)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = out.mean((-1,-2))  # replacing the out= F.avg_pool2d(out, 8) which is sensitive to the input shape.
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    
    def set_cuda(self, is_cuda):
        if isinstance(is_cuda, str):
            self.cuda(is_cuda)
        else:
            if is_cuda:
                self.cuda()
            else:
                self.cpu()
        self.is_cuda = is_cuda

    
    @property
    def model_dict(self):
        model_dict = {"type": "Wide_ResNet"}
        model_dict["state_dict"] = to_cpu_recur(self.state_dict())
        model_dict["depth"] = self.depth
        model_dict["widen_factor"] = self.widen_factor
        model_dict["input_channels"] = self.input_channels
        model_dict["output_size"] = self.output_size
        model_dict["dropout_rate"] = self.dropout_rate
        return model_dict

    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)

    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)
    
    def get_regularization(self, *args, **kwargs):
        return to_Variable([0], is_cuda = self.is_cuda)
    
    def prepare_inspection(self, *args, **kwargs):
        return {}


# ### CNN:

# In[ ]:


class ConvNet(nn.Module):
    def __init__(
        self,
        input_channels,
        struct_param=None,
        W_init_list=None,
        b_init_list=None,
        settings={},
        return_indices=False,
        is_cuda=False,
        ):
        super(ConvNet, self).__init__()
        self.input_channels = input_channels
        if struct_param is not None:
            self.struct_param = struct_param
            self.W_init_list = W_init_list
            self.b_init_list = b_init_list
            self.settings = settings
            self.num_layers = len(struct_param)
            self.info_dict = {}
            self.param_available = ["Conv2d", "ConvTranspose2d", "BatchNorm2d", "Simple_Layer"]
            self.return_indices = return_indices
            for i in range(len(self.struct_param)):
                if i > 0:
                    k = 1
                    while self.struct_param[i - k][0] is None:
                        k += 1
                    num_channels_prev = self.struct_param[i - k][0]
                else:
                    num_channels_prev = input_channels
                    k = 0
                if self.struct_param[i - k][1] == "Simple_Layer" and isinstance(num_channels_prev, tuple) and len(num_channels_prev) == 3:
                    num_channels_prev = num_channels_prev[0]
                num_channels = self.struct_param[i][0]
                layer_type = self.struct_param[i][1]
                layer_settings = self.struct_param[i][2]
                if "layer_input_size" in layer_settings and isinstance(layer_settings["layer_input_size"], tuple):
                    num_channels_prev = layer_settings["layer_input_size"][0]
                if layer_type == "Conv2d":
                    layer = nn.Conv2d(num_channels_prev, 
                                      num_channels,
                                      kernel_size = layer_settings["kernel_size"],
                                      stride = layer_settings["stride"] if "stride" in layer_settings else 1,
                                      padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                      dilation = layer_settings["dilation"] if "dilation" in layer_settings else 1,
                                     )
                elif layer_type == "ConvTranspose2d":
                    layer = nn.ConvTranspose2d(num_channels_prev,
                                               num_channels,
                                               kernel_size = layer_settings["kernel_size"],
                                               stride = layer_settings["stride"] if "stride" in layer_settings else 1,
                                               padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                               output_padding = layer_settings["output_padding"] if "output_padding" in layer_settings else 0,
                                               dilation = layer_settings["dilation"] if "dilation" in layer_settings else 1,
                                              )
                elif layer_type == "Simple_Layer":
                    layer = get_Layer(layer_type = layer_type,
                                      input_size = layer_settings["layer_input_size"],
                                      output_size = num_channels,
                                      W_init = W_init_list[i] if self.W_init_list is not None and self.W_init_list[i] is not None else None,
                                      b_init = b_init_list[i] if self.b_init_list is not None and self.b_init_list[i] is not None else None,
                                      settings = layer_settings,
                                      is_cuda = is_cuda,
                                     )
                elif layer_type == "MaxPool2d":
                    layer = nn.MaxPool2d(kernel_size = layer_settings["kernel_size"],
                                         stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                         padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                         return_indices = layer_settings["return_indices"] if "return_indices" in layer_settings else False,
                                        )
                elif layer_type == "MaxUnpool2d":
                    layer = nn.MaxUnpool2d(kernel_size = layer_settings["kernel_size"],
                                           stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                           padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                          )
                elif layer_type == "Upsample":
                    layer = nn.Upsample(scale_factor = layer_settings["scale_factor"],
                                        mode = layer_settings["mode"] if "mode" in layer_settings else "nearest",
                                       )
                elif layer_type == "BatchNorm2d":
                    layer = nn.BatchNorm2d(num_features = num_channels)
                elif layer_type == "Dropout2d":
                    layer = nn.Dropout2d(p = 0.5)
                elif layer_type == "Flatten":
                    layer = Flatten()
                else:
                    raise Exception("layer_type {0} not recognized!".format(layer_type))

                # Initialize using provided initial values:
                if self.W_init_list is not None and self.W_init_list[i] is not None and layer_type not in ["Simple_Layer"]:
                    layer.weight.data = torch.FloatTensor(self.W_init_list[i])
                    layer.bias.data = torch.FloatTensor(self.b_init_list[i])

                setattr(self, "layer_{0}".format(i), layer)
            self.set_cuda(is_cuda)


    def forward(self, input, indices_list = None, **kwargs):
        return self.inspect_operation(input, operation_between = (0, self.num_layers), indices_list = indices_list)
    
    
    def inspect_operation(self, input, operation_between, indices_list = None):
        output = input
        if indices_list is None:
            indices_list = []
        start_layer, end_layer = operation_between
        if end_layer < 0:
            end_layer += self.num_layers
        for i in range(start_layer, end_layer):
            if "layer_input_size" in self.struct_param[i][2]:
                output_size_last = output.shape[0]
                layer_input_size = self.struct_param[i][2]["layer_input_size"]
                if not isinstance(layer_input_size, tuple):
                    layer_input_size = (layer_input_size,)
                output = output.view(-1, *layer_input_size)
                assert output.shape[0] == output_size_last, "output_size reshaped to different length. Check shape!"
            if "Unpool" in self.struct_param[i][1]:
                output_tentative = getattr(self, "layer_{0}".format(i))(output, indices_list.pop(-1))
            else:
                output_tentative = getattr(self, "layer_{0}".format(i))(output)
            if isinstance(output_tentative, tuple):
                output, indices = output_tentative
                indices_list.append(indices)
            else:
                output = output_tentative
            if "activation" in self.struct_param[i][2]:
                activation = self.struct_param[i][2]["activation"]
            else:
                if "activation" in self.settings:
                    activation = self.settings["activation"]
                else:
                    activation = "linear"
                if "Pool" in self.struct_param[i][1] or "Unpool" in self.struct_param[i][1] or "Upsample" in self.struct_param[i][1]:
                    activation = "linear"
            output = get_activation(activation)(output)
        if self.return_indices:
            return output, indices_list
        else:
            return output


    def get_loss(self, input, target, criterion, **kwargs):
        y_pred = self(input, **kwargs)
        if self.return_indices:
            y_pred = y_pred[0]
        return criterion(y_pred, target)


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        if not isinstance(source, list):
            source = [source]
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for k in range(self.num_layers):
            if self.struct_param[k][1] not in self.param_available:
                continue
            layer = getattr(self, "layer_{0}".format(k))
            for source_ele in source:
                if source_ele == "weight":
                    if self.struct_param[k][1] not in ["Simple_Layer"]:
                        item = layer.weight
                    else:
                        item = layer.W_core
                elif source_ele == "bias":
                    if self.struct_param[k][1] not in ["Simple_Layer"]:
                        item = layer.bias
                    else:
                        item = layer.b_core
                if mode == "L1":
                    reg = reg + item.abs().sum()
                elif mode == "L2":
                    reg = reg + (item ** 2).sum()
                else:
                    raise Exception("mode {0} not recognized!".format(mode))
        return reg


    def get_weights_bias(self, W_source = "core", b_source = "core"):
        W_list = []
        b_list = []
        for k in range(self.num_layers):
            if self.struct_param[k][1] == "Simple_Layer":
                layer = getattr(self, "layer_{0}".format(k))
                if W_source == "core":
                    W_list.append(to_np_array(layer.W_core))
                if b_source == "core":
                    b_list.append(to_np_array(layer.b_core))
            elif self.struct_param[k][1] in self.param_available:
                layer = getattr(self, "layer_{0}".format(k))
                if W_source == "core":
                    W_list.append(to_np_array(layer.weight))
                if b_source == "core":
                    b_list.append(to_np_array(layer.bias, full_reduce = False))
            else:
                if W_source == "core":
                    W_list.append(None)
                if b_source == "core":
                    b_list.append(None)
        return W_list, b_list


    @property
    def model_dict(self):
        model_dict = {"type": self.__class__.__name__}
        model_dict["net_type"] = self.__class__.__name__
        model_dict["input_channels"] = self.input_channels
        model_dict["struct_param"] = self.struct_param
        model_dict["settings"] = self.settings
        model_dict["weights"], model_dict["bias"] = self.get_weights_bias(W_source = "core", b_source = "core")
        model_dict["return_indices"] = self.return_indices
        return model_dict

    
    @property
    def output_size(self):
        return self.struct_param[-1][0]
    
    
    @property
    def structure(self):
        structure = OrderedDict()
        structure["input_channels"] = self.input_channels
        structure["output_size"] = self.output_size
        structure["struct_param"] = self.struct_param if hasattr(self, "struct_param") else None
        return structure
        


    def get_sympy_expression(self, verbose=True):
        expressions = {i: None for i in range(self.num_layers)}
        return expressions


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)
    
    
    def DL(self):
        DL = 0
        for k in range(self.num_layers):
            layer_type = self.struct_param[k][1]
            if layer_type in self.param_available:
                layer = getattr(self, "layer_{0}".format(k))
                if layer_type == "Simple_Layer":
                    DL += layer.DL
                else:
                    DL += get_list_DL(to_np_array(layer.weight), "non-snapped")
                    DL += get_list_DL(to_np_array(layer.bias), "non-snapped")
        return DL
    

    def load_model_dict(self, model_dict):
        new_net = load_model_dict_net(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)


    def prepare_inspection(self, X, y, **kwargs):
        pred_prob = self(X)
        if self.return_indices:
            pred_prob = pred_prob[0]
        pred = pred_prob.max(1)[1]
#         self.info_dict["accuracy"] = get_accuracy(pred, y)
        return deepcopy(self.info_dict)
    

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
        for k in range(self.num_layers):
            layer = getattr(self, "layer_{0}".format(k))
            if self.struct_param[k][1] == "Simple_Layer":
                layer.set_trainable(is_trainable)
            elif self.struct_param[k][1] in self.param_available:
                for param in layer.parameters():
                    param.requires_grad = is_trainable



class Conv_Model(nn.Module):
    def __init__(
        self,
        encoder_model_dict,
        core_model_dict,
        decoder_model_dict,
        latent_size = 2,
        is_generative = True,
        is_res_block = True,
        is_cuda = False,
        ):
        """Conv_Model consists of an encoder, a core and a decoder"""
        super(Conv_Model, self).__init__()
        self.latent_size = latent_size
        self.is_generative = is_generative
        if not is_generative:
            self.encoder = load_model_dict(encoder_model_dict, is_cuda = is_cuda)
        self.core = load_model_dict(core_model_dict, is_cuda = is_cuda)
        self.decoder = load_model_dict(decoder_model_dict, is_cuda = is_cuda)
        self.is_res_block = is_res_block
        self.is_cuda = is_cuda
        self.info_dict = {}


    @property
    def num_layers(self):
        if self.is_generative:
            return 1
        else:
            return len(self.core.model_dict["struct_param"])


    def forward(
        self,
        X,
        latent = None,
        **kwargs
        ):
        if self.is_generative:
            if len(latent.shape) == 1:
                latent = latent.repeat(len(X), 1)
            latent = self.core(latent)
        else:
            p_dict = {k: latent if k == 0 else None for k in range(self.num_layers)}
            latent = self.encoder(X)
            latent = self.core(latent, p_dict = p_dict)
        output = self.decoder(latent)
        if self.is_res_block:
            output = (X + nn.Sigmoid()(output)).clamp(0, 1)
        return output
    
    
    def forward_multistep(self, X, latents, isplot = False, num_images = 1):
        assert len(latents.shape) == 1
        length = int(len(latents) / 2)
        output = X
        for i in range(length - 1):
            latent = latents[i * self.latent_size: (i + 2) * self.latent_size]
            output = self(output, latent = latent)
            if isplot:
                plot_matrices(output[:num_images,0])
        return output


    def get_loss(self, X, y, criterion, **kwargs):
        return criterion(self(X = X[0], latent = X[1]), y)
    
    
    def plot(self, X, y, num_images = 1):
        y_pred = self(X[0], latent = X[1])
        idx_list = np.random.choice(len(X[0]), num_images)
        for idx in idx_list:
            matrix = torch.cat([X[0][idx], y[idx], y_pred[idx]])
            plot_matrices(matrix, images_per_row = 8)
    
    
    def get_regularization(self, source = ["weights", "bias"], mode = "L1"):
        if self.is_generative:
            return self.core.get_regularization(source = source, mode = mode) +                     self.decoder.get_regularization(source = source, mode = mode)
        else:
            return self.encoder.get_regularization(source = source, mode = mode) +                     self.core.get_regularization(source = source, mode = mode) +                     self.decoder.get_regularization(source = source, mode = mode)


    def prepare_inspection(self, X, y, **kwargs):
        return deepcopy(self.info_dict)
    
    
    def set_trainable(self, is_trainable):
        if not self.is_generative:
            self.encoder.set_trainable(is_trainable)
        self.core.set_trainable(is_trainable)
        self.decoder.set_trainable(is_trainable)
    
    
    @property
    def model_dict(self):
        model_dict = {"type": "Conv_Model"}
        if not self.is_generative:
            model_dict["encoder_model_dict"] = self.encoder.model_dict
        model_dict["latent_size"] = self.latent_size
        model_dict["core_model_dict"] = self.core.model_dict
        model_dict["decoder_model_dict"] = self.decoder.model_dict
        model_dict["is_generative"] = self.is_generative
        model_dict["is_res_block"] = self.is_res_block
        return model_dict


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)



class Conv_Autoencoder(nn.Module):
    def __init__(
        self,
        input_channels_encoder,
        input_channels_decoder,
        struct_param_encoder,
        struct_param_decoder,
        latent_size = (1,2),
        share_model_among_steps = False,
        settings = {},
        is_cuda = False,
        ):
        """Conv_Autoencoder consists of an encoder and a decoder"""
        super(Conv_Autoencoder, self).__init__()
        self.input_channels_encoder = input_channels_encoder
        self.input_channels_decoder = input_channels_decoder
        self.struct_param_encoder = struct_param_encoder
        self.struct_param_decoder = struct_param_decoder
        self.share_model_among_steps = share_model_among_steps
        self.settings = settings
        self.encoder = ConvNet(input_channels = input_channels_encoder, struct_param = struct_param_encoder, settings = settings, is_cuda = is_cuda)
        self.decoder = ConvNet(input_channels = input_channels_decoder, struct_param = struct_param_decoder, settings = settings, is_cuda = is_cuda)
        self.is_cuda = is_cuda
    
    def encode(self, input):
        if self.share_model_among_steps:
            latent = []
            for i in range(input.shape[1]):
                latent_step = self.encoder(input[:, i:i+1])
                latent.append(latent_step)
            return torch.cat(latent, 1)
        else:
            return self.encoder(input)
    
    def decode(self, latent):
        if self.share_model_among_steps:
            latent_size = self.struct_param_encoder[-1][0]
            latent = latent.view(latent.size(0), -1, latent_size)
            output = []
            for i in range(latent.shape[1]):
                output_step = self.decoder(latent[:, i].contiguous())
                output.append(output_step)
            return torch.cat(output, 1)
        else:
            return self.decoder(latent)
    
    def set_trainable(self, is_trainable):
        self.encoder.set_trainable(is_trainable)
        self.decoder.set_trainable(is_trainable)
    
    def forward(self, input):
        return self.decode(self.encode(input))
    
    def get_loss(self, input, target, criterion, **kwargs):
        return criterion(self(input), target)
    
    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        return self.encoder.get_regularization(source = source, mode = mode) +                self.decoder.get_regularization(source = source, mode = mode)
    
    @property
    def model_dict(self):
        model_dict = {"type": "Conv_Autoencoder"}
        model_dict["net_type"] = "Conv_Autoencoder"
        model_dict["input_channels_encoder"] = self.input_channels_encoder
        model_dict["input_channels_decoder"] = self.input_channels_decoder
        model_dict["struct_param_encoder"] = self.struct_param_encoder
        model_dict["struct_param_decoder"] = self.struct_param_decoder
        model_dict["share_model_among_steps"] = self.share_model_among_steps
        model_dict["settings"] = self.settings
        model_dict["encoder"] = self.encoder.model_dict
        model_dict["decoder"] = self.decoder.model_dict
        return model_dict
    
    def load_model_dict(self, model_dict):
        model = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(model.__dict__)

    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)

    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)

    def DL(self):
        return self.encoder.DL + self.decoder.DL



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# ### VAE:

# In[ ]:


class VAE(nn.Module):
    def __init__(
        self,
        encoder_model_dict,
        decoder_model_dict,
        is_cuda = False,
        ):
        super(VAE, self).__init__()
        self.encoder = load_model_dict(encoder_model_dict, is_cuda = is_cuda)
        self.decoder = load_model_dict(decoder_model_dict, is_cuda = is_cuda)
        self.is_cuda = is_cuda
        self.info_dict = {}


    def encode(self, X):
        Z = self.encoder(X)
        latent_size = int(Z.shape[-1] / 2)
        mu = Z[..., :latent_size]
        logvar = Z[..., latent_size:]
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def decode(self, Z):
        return self.decoder(Z)


    def forward(self, X):
        mu, logvar = self.encode(X)
        Z = self.reparameterize(mu, logvar)
        return self.decode(Z), mu, logvar


    def get_loss(self, X, y = None, **kwargs):
        recon_X, mu, logvar = self(X)
        BCE = F.binary_cross_entropy(recon_X.view(recon_X.shape[0], -1), X.view(X.shape[0], -1), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (BCE + KLD) / len(X)
        self.info_dict["KLD"] = KLD.item() / len(X)
        self.info_dict["BCE"] = BCE.item() / len(X)
        return loss


    def model_dict(self):
        model_dict = {"type": "VAE"}
        model_dict["encoder_model_dict"] = self.encoder.model_dict
        model_dict["decoder_model_dict"] = self.decoder.model_dict
        return model_dict


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        return self.encoder.get_regularization(source = source, mode = mode) + self.decoder.get_regularization(source = source, mode = mode)


    def prepare_inspection(self, X, y, **kwargs):
        return deepcopy(self.info_dict)


# ## Reparameterization toolkit:

# In[ ]:


class Net_reparam(nn.Module):
    """Module that uses reparameterization to take into two inputs and gets a scaler"""
    def __init__(
        self,
        model_dict,
        reparam_mode,
        is_cuda=False,
        ):
        super(Net_reparam, self).__init__()
        self.model = load_model_dict(model_dict, is_cuda=is_cuda)
        self.reparam_mode = reparam_mode

    def forward(self, X, Z, is_outer=False):
        """
        Obtaining single value using reparameterization.

        Args:
            X shape: [Bx, ...]
            Z shape: [S, Bz, Z]
            is_outer: whether to use outer product to get a tensor with shape [S, Bz, Bx].
        
        Returns:
            If is_outer==True, return log_prob of shape [S, Bz, Bx]
            If is_outer==False, return log_prob of shape [S, Bz]  (where Bz=Bx)
        """
        dist, _ = reparameterize(self.model, X, mode=self.reparam_mode)
        if is_outer:
            log_prob = dist.log_prob(Z[...,None,:])
        else:
            log_prob = dist.log_prob(Z)
        if self.reparam_mode == 'diag':
            log_prob = log_prob.sum(-1)
        return log_prob

    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        return self.model.get_regularization(source=source, model=mode, **kwargs)

    def prepare_inspection(self, X, y, **kwargs):
        return {}

    @property
    def model_dict(self):
        model_dict = {"type": "Net_reparam"}
        model_dict["model"] = self.model.model_dict
        model_dict["reparam_mode"] = self.reparam_mode
        return model_dict

    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)

    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


def reparameterize(model, input, mode="full", size=None):
    if mode.startswith("diag"):
        if model is not None and model.__class__.__name__ == "Mixture_Model":
            return reparameterize_mixture_diagonal(model, input, mode=mode)
        else:
            return reparameterize_diagonal(model, input, mode=mode)
    elif mode == "full":
        return reparameterize_full(model, input, size=size)
    else:
        raise Exception("Mode {} is not valid!".format(mode))


def reparameterize_diagonal(model, input, mode):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if mode.startswith("diagg"):
        if isinstance(mean_logit, tuple):
            mean = mean_logit[0]
        else:
            mean = mean_logit
        std = torch.ones(mean.shape).to(mean.device)
        dist = Normal(mean, std)
        return dist, (mean, std)
    elif mode.startswith("diag"):
        if isinstance(mean_logit, tuple):
            mean_logit = mean_logit[0]
        size = int(mean_logit.size(-1) / 2)
        mean = mean_logit[:, :size]
        std = F.softplus(mean_logit[:, size:], beta=1) + 1e-10
        dist = Normal(mean, std)
        return dist, (mean, std)
    else:
        raise Exception("mode {} is not valid!".format(mode))


def reparameterize_mixture_diagonal(model, input, mode):
    mean_logit, weight_logits = model(input)
    if mode.startswith("diagg"):
        mean_list = mean_logit
        scale_list = torch.ones(mean_list.shape).to(mean_list.device)
    else:
        size = int(mean_logit.size(-2) / 2)
        mean_list = mean_logit[:, :size]
        scale_list = F.softplus(mean_logit[:, size:], beta=1) + 0.01  # Avoid the std to go to 0
    dist = Mixture_Gaussian_reparam(mean_list=mean_list,
                                    scale_list=scale_list,
                                    weight_logits=weight_logits,
                                   )
    return dist, (mean_list, scale_list)


def reparameterize_full(model, input, size=None):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if isinstance(mean_logit, tuple):
        mean_logit = mean_logit[0]
    if size is None:
        dim = mean_logit.size(-1)
        size = int((np.sqrt(9 + 8 * dim) - 3) / 2)
    mean = mean_logit[:, :size]
    scale_tril = fill_triangular(mean_logit[:, size:], size)
    scale_tril = matrix_diag_transform(scale_tril, F.softplus)
    dist = MultivariateNormal(mean, scale_tril = scale_tril)
    return dist, (mean, scale_tril)


def sample(dist, n=None):
    """Sample n instances from distribution dist"""
    if n is None:
        return dist.rsample()
    else:
        return dist.rsample((n,))


# ## Probability models:
# ### Mixture of Gaussian:

# In[ ]:


class Mixture_Gaussian(nn.Module):
    def __init__(
        self,
        num_components,
        dim,
        param_mode = "full",
        is_cuda = False,
        ):
        super(Mixture_Gaussian, self).__init__()
        self.num_components = num_components
        self.dim = dim
        self.param_mode = param_mode
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")
        self.info_dict = {}


    def initialize(self, model_dict = None, input = None, num_samples = 100, verbose = False):
        if input is not None:
            neg_log_prob_min = np.inf
            loc_init_min = None
            scale_init_min = None
            for i in range(num_samples):
                neg_log_prob, loc_init_list, scale_init_list = self.initialize_ele(input)
                if verbose:
                    print("{0}: neg_log_prob: {1:.4f}".format(i, neg_log_prob))
                if neg_log_prob < neg_log_prob_min:
                    neg_log_prob_min = neg_log_prob
                    loc_init_min = self.loc_list.detach()
                    scale_init_min = self.scale_list.detach()

            self.loc_list = nn.Parameter(loc_init_min.to(self.device))
            self.scale_list = nn.Parameter(scale_init_min.to(self.device))
            print("min neg_log_prob: {0:.6f}".format(to_np_array(neg_log_prob_min)))
        else:
            if model_dict is None:
                self.weight_logits = nn.Parameter((torch.randn(self.num_components) * np.sqrt(2 / (1 + self.dim))).to(self.device))
            else:
                self.weight_logits = nn.Parameter((torch.FloatTensor(model_dict["weight_logits"])).to(self.device))
            if self.param_mode == "full": 
                size = self.dim * (self.dim + 1) // 2
            elif self.param_mode == "diag":
                size = self.dim
            else:
                raise
            
            if model_dict is None:
                self.loc_list = nn.Parameter(torch.randn(self.num_components, self.dim).to(self.device))
                self.scale_list = nn.Parameter((torch.randn(self.num_components, size) / self.dim).to(self.device))
            else:
                self.loc_list = nn.Parameter(torch.FloatTensor(model_dict["loc_list"]).to(self.device))
                self.scale_list = nn.Parameter(torch.FloatTensor(model_dict["scale_list"]).to(self.device))


    def initialize_ele(self, input):
        if self.param_mode == "full":
            size = self.dim * (self.dim + 1) // 2
        elif self.param_mode == "diag":
            size = self.dim
        else:
            raise
        length = len(input)
        self.weight_logits = nn.Parameter(torch.zeros(self.num_components).to(self.device))
        self.loc_list = nn.Parameter(input[torch.multinomial(torch.ones(length) / length, self.num_components)].detach())
        self.scale_list = nn.Parameter((torch.randn(self.num_components, size).to(self.device) * input.std() / 5).to(self.device))
        neg_log_prob = self.get_loss(input)
        return neg_log_prob


    def prob(self, input):
        if len(input.shape) == 1:
            input = input.unsqueeze(1)
        assert len(input.shape) in [0, 2, 3]
        input = input.unsqueeze(-2)
        if self.param_mode == "diag":
            scale_list = F.softplus(self.scale_list)
            logits = (- (input - self.loc_list) ** 2 / 2 / scale_list ** 2 - torch.log(scale_list * np.sqrt(2 * np.pi))).sum(-1)
        else:
            raise
        prob = torch.matmul(torch.exp(logits), nn.Softmax(dim = 0)(self.weight_logits))
#         prob_list = []
#         for i in range(self.num_components):
#             if self.param_mode == "full":
#                 scale_tril = fill_triangular(getattr(self, "scale_{0}".format(i)), self.dim)
#                 scale_tril = matrix_diag_transform(scale_tril, F.softplus)
#                 dist = MultivariateNormal(getattr(self, "loc_{0}".format(i)), scale_tril = scale_tril)
#                 log_prob = dist.log_prob(input)
#             elif self.param_mode == "diag":
#                 dist = Normal(getattr(self, "loc_{0}".format(i)).unsqueeze(0), F.softplus(getattr(self, "scale_{0}".format(i))))
#                 mu = getattr(self, "loc_{0}".format(i)).unsqueeze(0)
#                 sigma = F.softplus(getattr(self, "scale_{0}".format(i)))
#                 log_prob = (- (input - mu) ** 2 / 2 / sigma ** 2 - torch.log(sigma * np.sqrt(2 * np.pi))).sum(-1)
#             else:
#                 raise
#             setattr(self, "component_{0}".format(i), dist)
#             prob = torch.exp(log_prob)
#             prob_list.append(prob)
#         prob_list = torch.stack(prob_list, -1)
#         prob = torch.matmul(prob_list, nn.Softmax(dim = 0)(self.weight_logits))
        return prob


    def log_prob(self, input):
        return torch.log(self.prob(input) + 1e-45)


    def get_loss(self, X, y = None, **kwargs):
        """Optimize negative log-likelihood"""
        neg_log_prob = - self.log_prob(X).mean() / np.log(2)
        self.info_dict["loss"] = to_np_array(neg_log_prob)
        return neg_log_prob


    def prepare_inspection(X, y, criterion, **kwargs):
        return deepcopy(self.info_dict)


    @property
    def model_dict(self):
        model_dict = {"type": "Mixture_Gaussian"}
        model_dict["num_components"] = self.num_components
        model_dict["dim"] = self.dim
        model_dict["param_mode"] = self.param_mode
        model_dict["weight_logits"] = to_np_array(self.weight_logits)
        model_dict["loc_list"] = to_np_array(self.loc_list)
        model_dict["scale_list"] = to_np_array(self.scale_list)
        return model_dict


    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)


    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


    def get_param(self):
        weights = to_np_array(nn.Softmax(dim = 0)(self.weight_logits))
        loc_list = to_np_array(self.loc_list)
        scale_list = to_np_array(self.scale_list)
        print("weights: {0}".format(weights))
        print("loc:")
        pp.pprint(loc_list)
        print("scale:")
        pp.pprint(scale_list)
        return weights, loc_list, scale_list


    def visualize(self, input):
        import scipy
        import matplotlib.pylab as plt
        std = to_np_array(input.std())
        X = np.arange(to_np_array(input.min()) - 0.2 * std, to_np_array(input.max()) + 0.2 * std, 0.1)
        Y_dict = {}
        weights = nn.Softmax(dim = 0)(self.weight_logits)
        plt.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
        for i in range(self.num_components):
            Y_dict[i] = weights[0].item() * scipy.stats.norm.pdf((X - self.loc_list[i].item()) / self.scale_list[i].item())
            plt.plot(X, Y_dict[i])
        Y = np.sum([item for item in Y_dict.values()], 0)
        plt.plot(X, Y, 'k--')
        plt.plot(input.data.numpy(), np.zeros(len(input)), 'k*')
        plt.title('Density of {0}-component mixture model'.format(self.num_components))
        plt.ylabel('probability density');


    def get_regularization(self, source = ["weights", "bias"], mode = "L1", **kwargs):
        reg = to_Variable([0], requires_grad = False).to(self.device)
        return reg


# ### Mixture_Gaussian for reparameterization:

# In[ ]:


class Mixture_Gaussian_reparam(nn.Module):
    def __init__(
        self,
        # Use as reparamerization:
        mean_list=None,
        scale_list=None,
        weight_logits=None,
        # Use as prior:
        Z_size=None,
        n_components=None,
        mean_scale=0.1,
        scale_scale=0.1,
        # Mode:
        is_reparam=True,
        reparam_mode="diag",
        is_cuda=False,
    ):
        super(Mixture_Gaussian_reparam, self).__init__()
        self.is_reparam = is_reparam
        self.reparam_mode = reparam_mode
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")

        if self.is_reparam:
            self.mean_list = mean_list         # size: [B, Z, k]
            self.scale_list = scale_list       # size: [B, Z, k]
            self.weight_logits = weight_logits # size: [B, k]
            self.n_components = self.weight_logits.shape[-1]
            self.Z_size = self.mean_list.shape[-2]
        else:
            self.n_components = n_components
            self.Z_size = Z_size
            self.mean_list = nn.Parameter((torch.rand(1, Z_size, n_components) - 0.5) * mean_scale)
            if reparam_mode == "diag":
                self.scale_list = nn.Parameter(torch.log(torch.exp((torch.rand(1, Z_size, n_components) * 0.2 + 0.9) * scale_scale) - 1))
            elif reparam_mode == "diagg":
                self.register_buffer('scale_list', torch.ones(1, Z_size, n_components))
            else:
                raise
            self.weight_logits = nn.Parameter(torch.zeros(1, n_components))
            if mean_list is not None:
                self.mean_list.data = to_Variable(mean_list)
                self.scale_list.data = to_Variable(scale_list)
                self.weight_logits.data = to_Variable(weight_logits)

        self.to(self.device)


    def log_prob(self, input):
        """Obtain the log_prob of the input."""
        input = input.unsqueeze(-1)  # [S, B, Z, 1]
        if self.reparam_mode in ["diag", "diagg"]:
            if self.is_reparam:
                # logits: [S, B, Z, k]
                logits = - (input - self.mean_list) ** 2 / 2 / self.scale_list ** 2 - torch.log(self.scale_list * np.sqrt(2 * np.pi))
            else:
                scale_list = F.softplus(self.scale_list, beta=1)
                logits = - (input - self.mean_list) ** 2 / 2 / scale_list ** 2 - torch.log(scale_list * np.sqrt(2 * np.pi))
        else:
            raise
        # log_softmax(weight_logits): [B, k]
        # logits: [S, B, Z, k]
        # log_prob: [S, B, Z]
        log_prob = torch.logsumexp(logits + F.log_softmax(self.weight_logits, -1).unsqueeze(-2), axis=-1)  # F(...).unsqueeze(-2): [B, 1, k]
        return log_prob


    def prob(self, Z):
        return torch.exp(self.log_prob(Z))


    def sample(self, n=None):
        if n is None:
            n_core = 1
        else:
            assert isinstance(n, tuple)
            n_core = n[0]
        weight_probs = F.softmax(self.weight_logits, -1)  # size: [B, m]
        idx = torch.multinomial(weight_probs, n_core, replacement=True).unsqueeze(-2).expand(-1, self.mean_list.shape[-2], -1)  # multinomial result: [B, S]; result: [B, Z, S]
        mean_list  = torch.gather(self.mean_list,  dim=-1, index=idx)  # [B, Z, S]
        if self.is_reparam:
            scale_list = torch.gather(self.scale_list, dim=-1, index=idx)  # [B, Z, S]
        else:
            scale_list = F.softplus(torch.gather(self.scale_list, dim=-1, index=idx), beta=1)  # [B, Z, S]
        Z = torch.normal(mean_list, scale_list).permute(2, 0, 1)
        if n is None:
            Z = Z.squeeze(0)
        return Z


    def rsample(self, n=None):
        return self.sample(n=n)


    def __repr__(self):
        return "Mixture_Gaussian_reparam({}, Z_size={})".format(self.n_components, self.Z_size)


    @property
    def model_dict(self):
        model_dict = {"type": "Mixture_Gaussian_reparam"}
        model_dict["is_reparam"] = self.is_reparam
        model_dict["reparam_mode"] = self.reparam_mode
        model_dict["Z_size"] = self.Z_size
        model_dict["n_components"] = self.n_components
        model_dict["mean_list"] = to_np_array(self.mean_list)
        model_dict["scale_list"] = to_np_array(self.scale_list)
        model_dict["weight_logits"] = to_np_array(self.weight_logits)
        return model_dict


# ### Triangular distribution:

# In[ ]:


class Triangular_dist(Distribution):
    """Probability distribution with a Triangular shape."""
    def __init__(self, loc, a, b, validate_args=None):
        self.loc, self.a, self.b = broadcast_all(loc, a, b)
        batch_shape = torch.Size() if isinstance(loc, Number) else self.loc.size()
        super(Triangular_dist, self).__init__(batch_shape, validate_args=validate_args)
        
    @property
    def mean(self):
        return self.loc + (self.b - self.a) / 3
    
    @property
    def variance(self):
        return (self.a ** 2 + self.b ** 2 + self.a * self.b) / 18
    
    @property
    def stddev(self):
        return torch.sqrt(self.variance)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PieceWise, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.a = self.a.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        super(Triangular_dist, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.loc - self.a, self.loc + self.b)
    
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        with torch.no_grad():
            return self.icdf(rand)
    
    def rsample(self, sample_shape=torch.Size()):
        """Sample with reparameterization."""
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(rand)

    def icdf(self, value):
        """Inverse cdf."""
        if self._validate_args:
            self._validate_sample(value)
        assert value.min() >= 0 and value.max() <= 1
        value, loc, a, b = broadcast_all(value, self.loc, self.a, self.b)
        a_plus_b = a + b
        idx = value < a / a_plus_b
        iidx = ~idx
        out = torch.ones_like(value)
        out[idx] = loc[idx] - a[idx] + torch.sqrt(a[idx] * a_plus_b[idx] * value[idx])
        out[iidx] = loc[iidx] + b[iidx] - torch.sqrt(b[iidx] * a_plus_b[iidx] * (1 - value[iidx]) )
        return out

    def prob(self, value):
        """Get probability."""
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        value, loc, a, b = broadcast_all(value, self.loc, self.a, self.b)
        idx1 = (loc - a <= value) & (value <= loc)
        idx2 = (loc < value) & (value <= loc + b)
        a_plus_b = a + b

        out = torch.zeros_like(value)
        out[idx1] = 2 * (value[idx1] - loc[idx1] + a[idx1]) / a[idx1] / a_plus_b[idx1]
        out[idx2] = -2 * (value[idx2] - loc[idx2] - b[idx2]) / b[idx2] / a_plus_b[idx2]
        return out

    def log_prob(self, value):
        """Get log probability."""
        return torch.log(self.prob(value))
    
    @property
    def model_dict(self):
        model_dict = {"type": "Triangular_dist"}
        model_dict["loc"] = to_np_array(self.loc)
        model_dict["a"] = to_np_array(self.a)
        model_dict["b"] = to_np_array(self.b)
        return model_dict

    def load(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        model_dict = load_model(filename, mode=mode)
        self.load_model_dict(model_dict)

    def save(self, filename):
        mode = "json" if filename.endswith(".json") else "pickle"
        save_model(self.model_dict, filename, mode=mode)


# In[ ]:


def load_model_dict_distribution(model_dict, is_cuda = False):
    if model_dict["type"] == "Mixture_Gaussian":
        model = Mixture_Gaussian(
            num_components=model_dict["num_components"],
            dim=model_dict["dim"],
            param_mode=model_dict["param_mode"],
            is_cuda=is_cuda,
        )
        model.initialize(model_dict = model_dict)
    elif model_dict["type"] == "Mixture_Gaussian_reparam":
        model = Mixture_Gaussian_reparam(
            is_reparam=model_dict["is_reparam"],
            reparam_mode=model_dict["reparam_mode"],
            mean_list=model_dict["mean_list"],
            scale_list=model_dict["scale_list"],
            weight_logits=model_dict["weight_logits"],
            Z_size=model_dict["Z_size"],
            n_components=model_dict["n_components"],
            is_cuda=is_cuda,
        )
    elif model_dict["type"] == "Triangular_dist":
        model = Triangular_dist(
            loc=model_dict["loc"],
            a=model_dict["a"],
            b=model_dict["b"],
        )
    else:
        raise Exception("Type {} is not valid!".format(model_dict["type"]))
    return model

