import argparse
import datetime
from deepsnap.batch import Batch as deepsnap_Batch
import numpy as np
import pdb
import pickle
import pprint as pp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb
from torch import optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import time
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from le_pde.datasets.load_dataset import load_data
from le_pde.argparser import arg_parse
from le_pde.models import get_model, load_model, unittest_model, build_optimizer, test
from le_pde.pytorch_net.util import pload, pdump, get_time, update_args, clip_grad, Batch, set_seed, update_dict, plot_matrices, make_dir, to_np_array, record_data, Early_Stopping, str2bool, get_filename_short, get_num_params, ddeepcopy as deepcopy, write_to_config
from le_pde.utils import p, EXP_PATH, get_model_dict, get_elements, is_diagnose, get_keys_values, loss_op, to_tuple_shape, parse_string_idx_to_list, parse_multi_step, get_device, Channel_Gen, process_data_for_CNN, get_activation, get_normalization, Mean, Flatten, Permute, Reshape, to_cpu


# ## Initialization:


args = arg_parse()
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    is_jupyter = True
    args.exp_id = "user-test"
    args.date_time = "{}-{}".format(datetime.datetime.now().month, datetime.datetime.now().day)

    # Train:
    args.epochs = 200
    args.contrastive_rel_coef = 0
    args.n_conv_blocks = 6
    args.latent_noise_amp = 1e-5
    args.hinge = 1
    args.multi_step = "1"
    args.latent_multi_step = "1^2^3^4"
    args.latent_loss_normalize_mode = "targetindi"
    args.channel_mode = "exp-16"
    args.batch_size = 20
    args.val_batch_size = 20
    args.reg_type = "None"
    args.reg_coef = 1e-4
    args.is_reg_anneal = True
    args.lr_scheduler_type = "cos"
    args.id = "test2"
    args.n_workers = 0
    args.plot_interval = 50
    args.temporal_bundle_steps = 1


    # For fno:
    args.dataset = "fno"
    args.algo = "contrast"
    args.encoder_type = "cnn-s"
    args.evo_conv_type = "cnn"
    args.decoder_type = "cnn-tr"
    args.padding_mode = "zeros"
    args.n_conv_layers_latent = 3
    args.n_conv_blocks = 4
    args.n_latent_levs = 1
    args.is_latent_flatten = True
    args.latent_size = 64
    args.decoder_act_name = "rational"
    args.use_grads = False
    args.n_train = ":20"
    args.test_interval = 2
    args.epochs = 2000
    args.use_pos = False
    args.contrastive_rel_coef = 0
    args.is_prioritized_dropout = False
    args.input_steps = 1
    # args.multi_step = "1^2^3^4^5^6^7^8^9^10^11^12^13^14^15^16^17^18^19^20"
    args.multi_step = "1^2^3^4"
    args.loss_type = "lp"
    
    # For fnomodel:
    args.dataset = "movinggas"
    args.algo = "fno"
    args.id = "2"
    
    # args.latent_multi_step = "1^2:0.1^3:0.1^4:0.1"
    args.is_y_variable_length = False
    
    # # For mppde1d:
    # args.dataset = "mppde1d-E3-100"
    # args.algo = "contrast"
    # args.encoder_type = "cnn-s"
    # args.evo_conv_type = "cnn"
    # args.decoder_type = "cnn-tr"
    # args.padding_mode = "zeros"
    # args.n_conv_layers_latent = 3
    # args.n_conv_blocks = 4
    # args.n_latent_levs = 1
    # args.is_latent_flatten = True
    # args.latent_size = 64
    # args.act_name = "elu"
    # args.decoder_act_name = "rational"
    # args.use_grads = False
    # args.n_train = "-1"
    # args.epochs = 2000
    # args.use_pos = False
    # args.contrastive_rel_coef = 0
    # args.is_prioritized_dropout = False
    # args.input_steps = 1
    # # args.multi_step = "1^2^3^4^5^6^7^8^9^10^11^12^13^14^15^16^17^18^19^20"
    # args.multi_step = "1^2"
    # args.latent_multi_step = "1^2"
    # args.temporal_bundle_steps = 25
    # args.static_encoder_type = "param-2-elu"
    # args.static_latent_size = 16

    # For karman3d-large:
    args.dataset = "karman3d-large"
    args.algo = "contrast"
    args.encoder_type = "cnn-s"
    args.evo_conv_type = "cnn"
    args.decoder_type = "cnn-tr"
    args.padding_mode = "zeros"
    args.n_conv_layers_latent = 3
    args.channel_base = "exp-24"
    args.n_conv_blocks = 6
    args.n_latent_levs = 1
    args.is_latent_flatten = True
    args.latent_size = 64
    args.act_name = "elu"
    args.decoder_act_name = "elu"
    args.use_grads = False
    args.n_train = ":10"
    args.test_interval = 2
    args.epochs = 2000
    args.use_pos = False
    args.contrastive_rel_coef = 0
    args.is_prioritized_dropout = False
    args.input_steps = 1
    args.multi_step = "1^2^3^4"
    args.latent_multi_step = "1^2:0.1^4:0.1"
    args.batch_size = 1
    args.n_workers = 0
    args.dp_mode = "None"
    args.time_interval = 2
    # args.is_latent_flatten = False
    # args.no_latent_evo = True
    # args.n_latent_levs = 2

    # Data:
    args.time_interval = 1
    args.dataset_split_type = "random"
    args.train_fraction = 1

    # Model:
    args.evolution_type = "mlp-3-elu-2"
    args.forward_type = "Euler"
    args.act_name = "elu"
    args.evo_groups = 1

    args.gpuid = "7"
    args.is_unittest = True

except:
    is_jupyter = False

if args.dataset.startswith("mppde1d"):
    if args.n_conv_blocks == 4:
        if args.dataset.endswith("-40"):
            args.output_padding_str = "0-0-0-0"
        elif args.dataset.endswith("-50"):
            args.output_padding_str = "1-0-1-0"
        elif args.dataset.endswith("-100"):
            args.output_padding_str = "1-1-0-0"
    else:
        raise

# In[ ]:


set_seed(args.seed)
if "dataset_train_val" not in locals():
    (dataset_train_val, dataset_test), (train_loader, val_loader, test_loader) = load_data(args)
#print(get_device(args))
p.print(f"Minibatches for train: {len(train_loader)}")
p.print(f"Minibatches for val: {len(val_loader)}")
p.print(f"Minibatches for test: {len(test_loader)}")
device = get_device(args)
args.device = device
data = deepcopy(dataset_test[0])
epoch = 0
if args.load_filename != "None":
    assert  args.load_dirname != "None"
    loaded_dirname = EXP_PATH + args.load_dirname
    loaded_filename = os.path.join(loaded_dirname, args.load_filename)
    with open(loaded_filename, "rb") as f:
        data_record = pickle.load(f)
    last_model_dict = data_record["last_model_dict"]
    if args.dataset.startswith("PIL-1Dsmall10"):
        # legacy:
        last_model_dict["normalization_type"] = data_record["args"]["normalization_type"]
    p.print("Loading the last model from {}, at epoch {}.".format(loaded_filename, data_record["last_epoch"]))
    epoch = data_record["last_epoch"] + 1 # Resume training from next epoch
    is_same_model_setting = True
    for key in last_model_dict:
        if hasattr(args, key):
            if last_model_dict[key] != getattr(args, key) and key not in ["loss_type"]:
                p.print("\tDiff in {}: model_dict: {}, args: {}".format(key, last_model_dict[key], getattr(args, key)))
                is_same_model_setting = False
            else:
                p.print("\tSame {}".format(key))
    if is_same_model_setting:
        p.print("All model settings are the same with the given args.")
    else:
        raise Exception("Not all model settings are the same with the given args!")
    model = load_model(last_model_dict, device)
else:
    model = get_model(args, data, device)
if args.dataset.startswith("PIL-1Dsmall10") or args.dataset.startswith("PHIL-1Dhybrid"):
    model.type(torch.float64)
if args.dataset.startswith("mppde1d") and not args.algo.startswith("fno"):
    model.type(torch.float32)
print()
p.print("model:")
print(model)
kwargs = {}


# ## Training:

# In[ ]:


opt, scheduler = build_optimizer(args, model.parameters())

if args.load_filename != "None":
    if args.load_exp_renew is True:
        epoch = 0
        data_record = {}
    else:
        opt.load_state_dict(data_record["last_optimizer_dict"])
        scheduler.load_state_dict(data_record["last_scheduler_dict"])

n_params_model = get_num_params(model)
p.print("n_params_model: {}".format(n_params_model), end="")
machine_name = os.uname()[1].split('.')[0]
if args.load_filename == "None" or args.load_exp_renew is True:
    data_record = {"n_params_model": n_params_model, "args": update_dict(args.__dict__, "machine_name", machine_name),
               "best_train_model_dict": [], "best_train_loss": [], "best_train_loss_history":[]}
print()
early_stopping = Early_Stopping(patience=args.early_stopping_patience)

# pp.pprint(data_record)
short_str_dict = {
    "dataset": "",
    "n_train": "train",
    "algo": "algo",
    "encoder_type": "enc",
    "evo_conv_type": "evo",
    "act_name": "act",
    "latent_size": "hid",
    "loss_type": "lo",
    "recons_coef": "recef",
    "consistency_coef": "conef",
    "n_conv_blocks": "nconv",
    "n_latent_levs": "nlat",
    "n_conv_layers_latent": "clat",
    "no_latent_evo": "nl",
    "is_latent_flatten": "lf",
    "reg_type": "reg",
    "gpuid": "gpu",
    "id": "id",
}


filename_short = get_filename_short(
    short_str_dict.keys(),
    short_str_dict,
    args_dict=args.__dict__,
)
filename = EXP_PATH + "{}_{}/".format(args.exp_id, args.date_time) + filename_short[:-2] + "_{}.p".format(machine_name)
write_to_config(args, filename)
args.filename = filename
p.print(filename, banner_size=100)
make_dir(filename)
best_val_loss = np.Inf
if args.load_filename != "None":
    val_loss = np.Inf
collate_fn = deepsnap_Batch.collate() if data.__class__.__name__ == "HeteroGraph" else Batch(is_absorb_batch=True).collate()
if args.is_unittest:
    unittest_model(model,
                   collate_fn([data, data]), args, device, use_grads=args.use_grads, use_pos=args.use_pos,
                   test_cases="all" if not (args.dataset.startswith("PIL") or args.dataset.startswith("PHIL")) else "model_dict", algo=args.algo,
                   **kwargs
                  )
if args.dp_mode != "None":
    if args.dp_mode == "dp":
        model = torch.nn.DataParallel(model, device_ids=[int(ele) for ele in args.gpuid.split(",")])
if args.is_tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(EXP_PATH + '/log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
pp.pprint(args.__dict__)

if args.is_pretrain_autoencode:
    # Pretrain encoder and decoder:
    model.requires_grad(True, ["encoder", "decoder"])
    model.requires_grad(False, ["evolution", "static-encoder"])
    args_pretrain = deepcopy(args)
    args_pretrain.recons_coef = 1
    args_pretrain.consistency_coef = 0
    args_pretrain.contrastive_rel_coef = 0
    opt_pretrain, scheduler_pretrain = build_optimizer(args_pretrain, model.parameters())
    p.print("Begin doing pretrain...")
    assert args.epochs_pretrain > 0, "If is_pretrain_autoencode is True, then args.epochs_pretrain should be greater than 0."
    for epoch_pretrain in range(args.epochs_pretrain):
        total_loss_pretrain = 0
        count_pretrain = 0
        for j_pretrain, data in enumerate(train_loader):
            data.to(device)
            opt_pretrain.zero_grad()
            loss_pretrain = model.get_loss(
                data,
                args_pretrain,
                current_epoch=epoch_pretrain,
                current_minibatch=j_pretrain,
                **kwargs
            )
            loss_pretrain.backward()
            if args_pretrain.is_clip_grad:
                clip_grad(opt_pretrain)
            opt_pretrain.step()
            total_loss_pretrain += loss_pretrain.item()
            count_pretrain += 1
            del loss_pretrain
            del data
        loss_pretrain_mean = total_loss_pretrain / count_pretrain
        record_data(data_record, [epoch_pretrain, loss_pretrain_mean], ["epoch_pretrain", "loss_pretrain"])
        p.print(f"Epoch_pretrain: {epoch_pretrain}  \t loss_pretrain: {loss_pretrain_mean:.6f}")

    model.requires_grad(False, ["encoder", "decoder"])
    model.requires_grad(True, ["evolution", "static-encoder"])
    args.recons_coef = 0
    args.multi_step = ""
    opt, scheduler = build_optimizer(args, model.parameters())

while epoch < args.epochs:
    total_loss = 0
    count = 0
    model.train()
    train_info = {}
    best_train_loss = np.Inf
    last_few_losses = []
    num_losses = 20
    t_start = time.time()
    for j, data in enumerate(train_loader):
        t_end = time.time()
        if args.verbose >= 2 and j % 100 == 0:
            p.print(f"Data loading time: {t_end - t_start:.6f}")
        if args.dp_mode == "None":
            data.to(device)
        if args.algo.startswith("supn") and j % int(args.reinit_mode.split("-")[1]) == 0:
            kwargs["G_s"] = reinit_G_s(data, args, G_s=kwargs["G_s"] if "G_s" in kwargs else None)
        if args.algo.startswith("supn") and j % 20 == 0:
            kwargs["isplot"] = is_jupyter
        else:
            kwargs["isplot"] = False
        opt.zero_grad()

        args_core = update_args(args, "multi_step", "1") if epoch < args.multi_step_start_epoch else args
        if args.dp_mode == "dp" :
            loss = get_loss(
                model,
                data,
                args_core,
                current_epoch=epoch,
                current_minibatch=j,
                **kwargs
            )
        else:
            loss = model.get_loss(
                data,
                args_core,
                current_epoch=epoch,
                current_minibatch=j,
                **kwargs
            )
        last_few_losses.append(loss.item())
        if len(last_few_losses)>num_losses:
            last_few_losses.pop(0)
        if is_diagnose(loc="1", filename=filename):
            pdb.set_trace()
        loss.backward()
        if args.is_clip_grad:
            clip_grad(opt)
        opt.step()
        total_loss = total_loss + loss.item()
        count += 1
        if args.is_tensorboard:
            writer.add_scalar("loss", total_loss, epoch)
        if args.verbose >= 2 and args.save_iterations!=-1 and j % args.save_iterations == 0:
            # Printing results inside an epoch:
            print(f"Data loading time: {t_end - t_start:.6f}")
            p.print("    Epoch {}   minibatch {}:    loss {:.4e}".format(epoch, str(j).rjust(4), loss.item()), end="")
            keys, values = get_keys_values(model.info, exclude=["pred"])
            for key, value in zip(keys, values):
                print("                {}: {:.4e}".format(key, value), end="")
            if args.dataset.startswith("PIL") and len(data.t) <= 8:
                print("                 t: {}".format(to_np_array(data.t)))
            else:
                print()

            updated_record = False
            if len(last_few_losses)==num_losses and (sum(last_few_losses)/num_losses < best_train_loss):
                best_train_loss = sum(last_few_losses)/num_losses
                best_train_model_dict = get_model_dict(model)
                if len(data_record["best_train_model_dict"]) <= epoch:
                    data_record["best_train_model_dict"].append(best_train_model_dict)
                    data_record["best_train_loss"].append(best_train_loss)
                    data_record["best_train_loss_history"].append(last_few_losses)
                else:
                    data_record["best_train_model_dict"][epoch] = best_train_model_dict
                    data_record["best_train_loss"][epoch] = best_train_loss
                    data_record["best_train_loss_history"][epoch]= last_few_losses
                updated_record = True
            
            if len(last_few_losses)!=0:
                data_record["model_dict_regular"] = get_model_dict(model)
                data_record["epoch_iteration"] = str(epoch)+"_"+str(j)
                data_record["loss_regular"] = sum(last_few_losses)/len(last_few_losses)
                data_record["loss_history"] = last_few_losses 
                updated_record = True
            
            if updated_record:
                with open(filename, "wb") as f:
                    pickle.dump(data_record, f)
        del loss
        del data
        keys, values = get_keys_values(model.info, exclude=["pred"])
        record_data(train_info, values, keys)
        t_start = time.time()
    for key, item in train_info.items():
        train_info[key] = np.mean(item)
    train_loss = total_loss / count
    if epoch % args.test_interval == 0 or epoch == args.epochs - 1:
        val_loss, val_info = test(
            val_loader, model, device, args,
            density_coef=0, current_epoch=epoch, current_minibatch=0,
            **kwargs
        )
        test_loss, test_info = test(
            test_loader, model, device, args,
            density_coef=0, current_epoch=epoch, current_minibatch=0,
            **kwargs
        )
        if val_loss is None:
            val_loss = test_loss
            val_info = test_info
        to_stop = early_stopping.monitor(val_loss)

    if is_diagnose(loc="2", filename=filename):
        pdb.set_trace()
    if args.lr_scheduler_type == "rop":
        scheduler.step(val_loss)
    elif args.lr_scheduler_type == "None":
        pass
    else:
        scheduler.step()
    record_data(data_record, [epoch, train_loss], ["epoch", "train_loss"])
    record_data(data_record, list(train_info.values()), ["{}_tr".format(key) for key in train_info])
    p.print(filename)
    p.print("Epoch {:03d}:     Train: {:.4e}".format(
        epoch + 1, train_loss), end="")
    if epoch % args.test_interval == 0 or epoch == args.epochs - 1:
        record_data(data_record, [epoch, val_loss, test_loss], ["test_epoch", "val_loss", "test_loss"])
        record_data(data_record, list(val_info.values()), ["{}_val".format(key) for key in val_info])
        record_data(data_record, list(test_info.values()), ["{}_te".format(key) for key in test_info])
        p.print("       Val: {:.4e}     Test: {:.4e}".format(val_loss, test_loss), end="", is_datetime=False)
        for key, loss_ele in test_info.items():
            p.print("               {}: ({:.4e}, {})".format(key.split("loss_")[-1], loss_ele, "{:.4e}".format(test_info[key]) if key in test_info else None), end="", is_datetime=False)
    print()
    if is_diagnose(loc="3", filename=filename):
        pdb.set_trace()
    if epoch % args.save_interval == 0 and epoch >= 0:
        record_data(data_record, [epoch, get_model_dict(model)], ["save_epoch", "model_dict"])
        with open(filename, "wb") as f:
            pickle.dump(data_record, f)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        data_record["best_model_dict"] = get_model_dict(model)
        data_record["best_optimizer_dict"] = opt.state_dict()
        data_record["best_scheduler_dict"] = scheduler.state_dict()
        data_record["best_epoch"] = epoch
    if True:
        data_record["last_model_dict"] = get_model_dict(model)
        data_record["last_optimizer_dict"] = opt.state_dict()
        data_record["last_scheduler_dict"] = scheduler.state_dict()
        data_record["last_epoch"] = epoch
        with open(filename, "wb") as f:
            pickle.dump(data_record, f)
    if "to_stop" in locals() and to_stop:
        p.print("Early-stop at epoch {}.".format(epoch))
        break
    epoch += 1
record_data(data_record, [epoch, get_model_dict(model)], ["save_epoch", "model_dict"])
with open(filename, "wb") as f:
    pickle.dump(data_record, f)