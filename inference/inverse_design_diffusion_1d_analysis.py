#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

import argparse
from collections import OrderedDict
import datetime
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import gc
import pandas as pd
import pdb
import pickle
import pprint as pp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import grad
from torch_geometric.data.dataloader import DataLoader
from chaotic_ellipse_dataset import Ellipse
from utils import compute_pressForce
from tqdm import tqdm
import matplotlib.backends.backend_pdf
import pprint as pp
from IPython.display import display, HTML
pd.options.display.max_rows = 1500
pd.options.display.max_columns = 200
pd.options.display.width = 1000
pd.set_option('max_colwidth', 400)

import sys, os

from nbody_dataset import NBodyDataset
from diffusion_1d import TemporalUnet1D, GaussianDiffusion1D
from utils import p, get_item_1d, eval_simu, simulation, to_np_array, make_dir, pdump, pload
from le_pde.pytorch_net.util import filter_filename, groupby_add_keys, get_unique_keys_df, filter_df

device = torch.device("cuda:0")


# In[2]:


def plot_im(matrix, design_coef_list, consistency_coef_list, title):
    fig = plt.figure()
    # ax = plt.gca()
    pos = plt.imshow(matrix, cmap='Blues', interpolation='none')
    plt.xlabel("consistency_coef")
    plt.ylabel("design_coef")
    plt.xticks(np.arange(0, len(consistency_coef_list)), labels=consistency_coef_list)
    plt.yticks(np.arange(0, len(design_coef_list)), labels=design_coef_list)
    plt.title(title)
    fig.colorbar(pos, ax=ax)
    plt.show()


# ## Exp inv_design:

# In[ ]:


"""
standard-recurrence-10 best range: 
2-body, nt_2, noise_mean: design_coef: [0.2,0.4], consistency_coef: [0.1, 0.2], value: 0.137-0.145, 
    best: design_coef=0.4, consistency_coef=0.2, 

4-body, nt_2, noise_sum much better than mean: noise_sum: design_coef: [0.1,0.4], consistency_coef: [0.1, 0.2], value: 0.2 
    best: design_coef=0.1




"""


# In[ ]:


exp_id = "inv_design"
dirname = f"results/inverse_design_diffusion/{exp_id}/"

filename_list = sorted(filter_filename(dirname, include="record"))
df_dict_list = []
for filename in filename_list:
    df_dict = {}
    data_record = pload(dirname + filename)
    df_dict.update(data_record)
    for key in ["pred", "pred_simu", "exp_id", "date_time", "model_name"]:
        df_dict.pop(key)
    df_dict_list.append(df_dict)
df = pd.DataFrame(df_dict_list)
dff = groupby_add_keys(
    df,
    by=["compose_n_bodies", "n_composed", "design_guidance", "design_coef", "consistency_coef"],
    add_keys=[],
    other_keys=["design_obj_simu", "design_obj_simu_nonan"],
)
dff_dict = dff.to_dict()
print(len(df))


# In[ ]:


df.sort_values(by = ["design_obj_simu"])


# In[ ]:


dff


# In[ ]:


design_guidance_list = [
    "standard-recurrence-10",
    "universal-forward-pure-recurrence-10",
    "universal-forward-pure-recurrence-20",
    "universal-forward-pure-recurrence-10",
]

for design_guidance in design_guidance_list:
    design_coef_list = [0.01,0.02,0.03,0.04,0.05]
    consistency_coef_list = [0.0,0.01,0.02,0.03,0.04,0.05]
    matrix = np.zeros((5,6))
    for i, design_coef in enumerate(design_coef_list):
        for j, consistency_coef in enumerate(consistency_coef_list):
            key = (2, 0, design_guidance, design_coef, consistency_coef)
            if key in dff_dict['design_obj_simu']:
                matrix[i, j] = dff_dict['design_obj_simu'][key]
            else:
                matrix[i, j] = np.NaN
    plot_im(matrix, design_coef_list=design_coef_list, consistency_coef_list=consistency_coef_list, title=design_guidance)


# In[ ]:


design_guidance_list = [
    "standard-recurrence-10",
    # "universal-backward-pure-recurrence-10",
]

for design_guidance in design_guidance_list:
    design_coef_list = [0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.4]
    consistency_coef_list = [0.0,0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.4]
    matrix = np.zeros((len(design_coef_list),len(consistency_coef_list)))
    for i, design_coef in enumerate(design_coef_list):
        for j, consistency_coef in enumerate(consistency_coef_list):
            key = (2, 0, design_guidance, design_coef, consistency_coef)
            if key in dff_dict['design_obj_simu']:
                matrix[i, j] = dff_dict['design_obj_simu'][key]
            else:
                matrix[i, j] = np.NaN
    plot_im(matrix, design_coef_list=design_coef_list, consistency_coef_list=consistency_coef_list, title=design_guidance)


# In[ ]:


design_guidance_list = [
    # "standard-recurrence-10",
    "universal-backward-pure-recurrence-10",
]

for design_guidance in design_guidance_list:
    design_coef_list = [0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2]
    consistency_coef_list = [0.0,0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.4]
    matrix = np.zeros((len(design_coef_list),len(consistency_coef_list)))
    for i, design_coef in enumerate(design_coef_list):
        for j, consistency_coef in enumerate(consistency_coef_list):
            key = (2, 0, design_guidance, design_coef, consistency_coef)
            if key in dff_dict['design_obj_simu']:
                matrix[i, j] = dff_dict['design_obj_simu'][key]
            else:
                matrix[i, j] = np.NaN
    plot_im(matrix, design_coef_list=design_coef_list, consistency_coef_list=consistency_coef_list, title=design_guidance)


# In[ ]:


design_guidance_list = [
    "universal-forward-pure-recurrence-10",
    "universal-forward-pure-recurrence-20",
    
]

for design_guidance in design_guidance_list:
    design_coef_list = [0.001,0.002,0.005,0.01,0.02,0.03,0.04,0.05]
    consistency_coef_list = [0.0,0.001,0.002,0.005,0.01,0.02,0.03,0.04,0.05]
    matrix = np.zeros((len(design_coef_list),len(consistency_coef_list)))
    for i, design_coef in enumerate(design_coef_list):
        for j, consistency_coef in enumerate(consistency_coef_list):
            key = (2, 0, design_guidance, design_coef, consistency_coef)
            if key in dff_dict['design_obj_simu']:
                matrix[i, j] = dff_dict['design_obj_simu'][key]
            else:
                matrix[i, j] = np.NaN
    plot_im(matrix, design_coef_list=design_coef_list, consistency_coef_list=consistency_coef_list, title=design_guidance)


# In[ ]:





# In[ ]:


design_guidance_list = [
    "standard-recurrence-10",
    "universal-forward-pure-recurrence-10",
    "universal-forward-pure-recurrence-20",
    "universal-backward-pure-recurrence-10",
]

for design_guidance in design_guidance_list:
    for compose_n_bodies in [2,4]:
        for n_composed in [0,1,2]:
            design_coef_list = [0.01,0.02,0.03,0.04,0.05]
            consistency_coef_list = [0.0,0.01,0.02,0.03,0.04,0.05]
            matrix = np.zeros((5,6))
            
            for i, design_coef in enumerate(design_coef_list):
                for j, consistency_coef in enumerate(consistency_coef_list):
                    key = (compose_n_bodies, n_composed, design_guidance, design_coef, consistency_coef)
                    if key in dff_dict['design_obj_simu']:
                        matrix[i, j] = dff_dict['design_obj_simu'][key]
                    else:
                        matrix[i, j] = np.NaN
            # if compose_n_bodies == 4 and n_composed==0:
            #     pdb.set_trace()
            title = f"{compose_n_bodies}-body   comp-{n_composed}  {design_guidance}"
            print(title)
            plot_im(matrix,
                    design_coef_list=design_coef_list,
                    consistency_coef_list=consistency_coef_list,
                    title=title)


# ## Exp inv_design 9-23:

# In[15]:


exp_id = "inv_design"
date_time = "09-23"
dirname = f"results/inverse_design_diffusion/{exp_id}_{date_time}/"

filename_list = sorted(filter_filename(dirname, include="record"))
df_dict_list = []
for filename in filename_list:
    df_dict = {}
    data_record = pload(dirname + filename)
    df_dict.update(data_record)
    for key in ["pred", "pred_simu", "exp_id", "date_time", "model_name"]:
        df_dict.pop(key)
    df_dict_list.append(df_dict)
df = pd.DataFrame(df_dict_list)
dff = groupby_add_keys(
    df,
    by=["compose_n_bodies", "n_composed", "design_guidance", "design_coef", "consistency_coef", "design_fn_mode"],
    add_keys=[],
    other_keys=["design_obj_simu", "RMSE", "MAE"],
)
dff_dict = dff.to_dict()
print(len(df))
dff.style.background_gradient(cmap=plt.cm.get_cmap("PiYG_r"))


# In[9]:


design_guidance_list = [
    "standard-recurrence-10",
]

for design_guidance in design_guidance_list:
    design_coef_list = [0.01,0.02,0.05,0.1,0.2,0.4,0.7,1]
    consistency_coef_list = [0.05,0.1,0.2]
    matrix = np.zeros((len(design_coef_list),len(consistency_coef_list)))
    for i, design_coef in enumerate(design_coef_list):
        for j, consistency_coef in enumerate(consistency_coef_list):
            key = (2, 0, design_guidance, design_coef, consistency_coef, "L2")
            if key in dff_dict['design_obj_simu']:
                matrix[i, j] = dff_dict['design_obj_simu'][key]
            else:
                print(2)
                matrix[i, j] = np.NaN
    plot_im(matrix, design_coef_list=design_coef_list, consistency_coef_list=consistency_coef_list, title=design_guidance)


# ## Exp L2square 9-23:

# ### body: 2, nt: 2

# In[36]:


exp_id = "L2square"
date_time = "09-23"
dirname = f"results/inverse_design_diffusion/{exp_id}_{date_time}/"

filename_list = sorted(filter_filename(dirname, include="record"))
df_dict_list = []
for filename in filename_list:
    df_dict = {}
    data_record = pload(dirname + filename)
    df_dict.update(data_record)
    for key in ["pred", "pred_simu", "exp_id", "date_time", "model_name"]:
        df_dict.pop(key)
    df_dict_list.append(df_dict)
df = pd.DataFrame(df_dict_list)
dff = groupby_add_keys(
    df,
    by=["compose_n_bodies", "n_composed", "design_guidance", "design_coef", "design_fn_mode", "consistency_coef"],
    add_keys=[],
    other_keys=["design_obj_simu", "RMSE", "MAE"],
)
dff_dict = dff.to_dict()
print("Number of items: ", len(df))
dff.style.background_gradient(cmap=plt.cm.get_cmap("PiYG_r"))


# In[6]:


def plot_im(matrix, design_coef_list, consistency_coef_list, title):
    fig = plt.figure()
    ax = plt.gca()
    pos = plt.imshow(matrix, cmap='Blues', interpolation='none')
    plt.xlabel("consistency_coef")
    plt.ylabel("design_coef")
    plt.xticks(np.arange(0, len(consistency_coef_list)), labels=consistency_coef_list)
    plt.yticks(np.arange(0, len(design_coef_list)), labels=design_coef_list)
    plt.title(title)
    fig.colorbar(pos, ax=ax)
    plt.show()


# In[7]:


design_guidance_list = [
    "standard-recurrence-10",
]

for design_guidance in design_guidance_list:
    design_coef_list = [0.01,0.02,0.05,0.1,0.2,0.4,0.7,1]
    consistency_coef_list = [0.1,0.2]
    matrix = np.zeros((len(design_coef_list),len(consistency_coef_list)))
    for i, design_coef in enumerate(design_coef_list):
        for j, consistency_coef in enumerate(consistency_coef_list):
            key = (2, 0, design_guidance, design_coef, consistency_coef, "L2square")
            if key in dff_dict['design_obj_simu']:
                matrix[i, j] = dff_dict['design_obj_simu'][key]
            else:
                matrix[i, j] = np.NaN
    plot_im(matrix, design_coef_list=design_coef_list, consistency_coef_list=consistency_coef_list, title=design_guidance)


# ## Exp noise_sum 9-23:

# ### body: 2, nt: 2

# In[42]:


exp_id = "noise_sum"
date_time = "09-23"
dirname = f"results/inverse_design_diffusion/{exp_id}_{date_time}/"

filename_list = sorted(filter_filename(dirname, include="record"))
df_dict_list = []
for filename in filename_list:
    df_dict = {}
    data_record = pload(dirname + filename)
    df_dict.update(data_record)
    for key in ["pred", "pred_simu", "exp_id", "date_time", "model_name"]:
        df_dict.pop(key)
    df_dict_list.append(df_dict)
df = pd.DataFrame(df_dict_list)
df2 = filter_df(df, filter_dict={
    "compose_n_bodies": 2,
    "n_composed": 0,
})
dff = groupby_add_keys(
    df2,
    by=["compose_n_bodies", "n_composed", "design_guidance", "design_coef", "consistency_coef", "compose_mode"],
    add_keys=[],
    other_keys=["design_obj_simu", "RMSE", "MAE"],
)
dff_dict = dff.to_dict()
print("Number of items: ", len(df))
dff.style.background_gradient(cmap=plt.cm.get_cmap("PiYG_r"))


# ### body: 2, nt: 4

# In[43]:


exp_id = "noise_sum"
date_time = "09-23"
dirname = f"results/inverse_design_diffusion/{exp_id}_{date_time}/"

filename_list = sorted(filter_filename(dirname, include="record"))
df_dict_list = []
for filename in filename_list:
    df_dict = {}
    data_record = pload(dirname + filename)
    df_dict.update(data_record)
    for key in ["pred", "pred_simu", "exp_id", "date_time", "model_name"]:
        df_dict.pop(key)
    df_dict_list.append(df_dict)
df = pd.DataFrame(df_dict_list)
df2 = filter_df(df, filter_dict={
    "compose_n_bodies": 2,
    "n_composed": 2,
})
dff = groupby_add_keys(
    df2,
    by=["compose_n_bodies", "n_composed", "design_guidance", "design_coef", "consistency_coef", "compose_mode"],
    add_keys=[],
    other_keys=["design_obj_simu", "RMSE", "MAE"],
)
dff_dict = dff.to_dict()
print("Number of items: ", len(df))
dff.style.background_gradient(cmap=plt.cm.get_cmap("PiYG_r"))


# ### body: 4, nt: 2

# In[40]:


exp_id = "noise_sum"
date_time = "09-23"
dirname = f"results/inverse_design_diffusion/{exp_id}_{date_time}/"

filename_list = sorted(filter_filename(dirname, include="record"))
df_dict_list = []
for filename in filename_list:
    df_dict = {}
    data_record = pload(dirname + filename)
    df_dict.update(data_record)
    for key in ["pred", "pred_simu", "exp_id", "date_time", "model_name"]:
        df_dict.pop(key)
    df_dict_list.append(df_dict)
df = pd.DataFrame(df_dict_list)
# df2 = df
df2 = filter_df(df, filter_dict={
    "compose_n_bodies": 4,
    "n_composed": 0,
})
dff = groupby_add_keys(
    df2,
    by=["compose_n_bodies", "n_composed", "design_guidance", "design_coef", "consistency_coef", "compose_mode"],
    add_keys=[],
    other_keys=["design_obj_simu", "RMSE", "MAE"],
)
dff_dict = dff.to_dict()
print("Number of items: ", len(df))
dff.style.background_gradient(cmap=plt.cm.get_cmap("PiYG_r"))


# In[ ]:





# In[19]:


design_guidance_list = [
    "standard-recurrence-10",
]

for design_guidance in design_guidance_list:
    design_coef_list = [0.05,0.1,0.2,0.3,0.4]
    consistency_coef_list = [0.1,0.2]
    matrix = np.zeros((len(design_coef_list),len(consistency_coef_list)))
    for i, design_coef in enumerate(design_coef_list):
        for j, consistency_coef in enumerate(consistency_coef_list):
            key = (2, 0, design_guidance, design_coef, consistency_coef, "noise_sum")
            if key in dff_dict['design_obj_simu']:
                matrix[i, j] = dff_dict['design_obj_simu'][key]
            else:
                matrix[i, j] = np.NaN
    plot_im(matrix, design_coef_list=design_coef_list, consistency_coef_list=consistency_coef_list, title=design_guidance)


# In[ ]:




