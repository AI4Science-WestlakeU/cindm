import os
import h5py
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
import sys, os
from datetime import datetime
from time import localtime, strftime, time
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from CinDM_anonymous.le_pde.MP_Neural_PDE_Solvers.equations.PDEs import *



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


p = Printer()

def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
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


class HDF5Dataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: PDE,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False,
                 uniform_sample: int=-1,
                 is_return_super = False,
                ) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (250, 100) if base_resolution is None else base_resolution
        self.super_resolution = (250, 200) if super_resolution is None else super_resolution
        self.uniform_sample = uniform_sample
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'
        self.is_return_super = is_return_super

        if self.base_resolution[1] not in [34, 25]:
            ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
            ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        else:
            ratio_nt = 1.0
            if self.base_resolution[1] == 34:
                ratio_nx = int(200 / 33)
            elif self.base_resolution[1] == 25:
                ratio_nx = int(200 / 25)
            else:
                raise
        # assert (ratio_nt.is_integer())
        # assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        if self.base_resolution[1] not in [34, 25]:
            self.nt = self.data[self.dataset_base].attrs['nt']
            self.dt = self.data[self.dataset_base].attrs['dt']
            self.dx = self.data[self.dataset_base].attrs['dx']
            self.x = self.data[self.dataset_base].attrs['x']
            self.tmin = self.data[self.dataset_base].attrs['tmin']
            self.tmax = self.data[self.dataset_base].attrs['tmax']
        else:
            self.nt = 250
            self.dt = self.data['pde_250-50'].attrs['dt']
            if self.base_resolution[1] == 34:
                self.dx = 0.16 * 3
                self.x = self.data['pde_250-100'].attrs['x'][::3]
            elif self.base_resolution[1] == 25:
                self.dx = 0.16 * 4
                self.x = self.data['pde_250-100'].attrs['x'][::4]
            else:
                raise
            self.tmin = self.data['pde_250-50'].attrs['tmin']
            self.tmax = self.data['pde_250-50'].attrs['tmax']
        self.x_ori = self.data['pde_250-100'].attrs['x']

        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data


    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.DoubleTensor(np.concatenate((left, u_super, right), -1))
            weights = torch.DoubleTensor([[[[0.2]*5]]])
            if self.uniform_sample == -1:
                u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            else:
                u_super = F.conv1d(u_super_padded, weights, stride=(1, 2)).squeeze().numpy()
            x = self.x

            # pdb.set_trace()
            # Base resolution trajectories (numerical baseline) and equation specific parameters
            if self.uniform_sample != -1:
                u_base = self.data[f'pde_{self.base_resolution[0]}-{100}'][idx][...,::self.uniform_sample]
                u_super_core = u_super[...,::self.uniform_sample]
            else:
                u_base = self.data[self.dataset_base][idx]
                u_super_core = u_super
            variables = {}
            variables['alpha'] = self.data['alpha'][idx]
            variables['beta'] = self.data['beta'][idx]
            variables['gamma'] = self.data['gamma'][idx]

            if self.is_return_super:
                return u_base, u_super_core, u_super, x, self.x_ori, variables
            else:
                return u_base, u_super_core, x, variables

        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]

            return u_base, u_super, x, variables

        else:
            raise Exception("Wrong experiment")


class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 25,
                 t_resolution: int = 250,
                 x_resolution: int =100
                 ) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
            time_ration (int): temporal ratio between base and super resolution
            space_ration (int): spatial ratio between base and super resolution
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_res = t_resolution
        self.x_res = x_resolution

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        """
        datapoints: [16, 250, 40]
        steps: [50] * batch_size:16
        self.tw: 25
        """
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]  # dp: [250, 40], obtaining steps [25:50]
            l = dp[step:self.tw + step]  # obtaining steps [50:75]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels


    def create_graph(self,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     x: torch.Tensor,
                     variables: dict,
                     steps: list,
                     uniform_sample: int,
                    ) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor. data/labels: [B:16, tw:25, nx:40]
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor  [B:16, nx:40], repeated on B dimension
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)  # [250]

        u, x_pos, t_pos, y, batch = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):  # data_batch/label_batch:  [tw:25, nx:40]
            u = torch.cat((u, torch.transpose(torch.cat([d[None, :] for d in data_batch]), 0, 1)), )  # d: [nx:40]. u: [40, 25]
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            x_pos = torch.cat((x_pos, x[0]), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
            batch = torch.cat((batch, torch.ones(nx) * b), )

        # Calculate the edge_index
        if f'{self.pde}' == 'CE':
            dx = x[0][1] - x[0][0]
            radius = self.n * dx + 0.0001
            edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        elif f'{self.pde}' == 'WE':
            edge_index = knn_graph(x_pos, k=self.n, batch=batch.long(), loop=False)

        graph = Data(x=u, edge_index=edge_index)
        graph.y = y
        graph.pos = torch.cat((t_pos[:, None], x_pos[:, None]), 1)  # t_pos: [640,]  x_pos: 640, graph.pos: [640, 2]
        graph.batch = batch.long()

        # Equation specific parameters
        if f'{self.pde}' == 'CE':
            alpha, beta, gamma = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                alpha = torch.cat((alpha, torch.tensor([variables['alpha'][i]])[:, None]), )
                beta = torch.cat((beta, torch.tensor([variables['beta'][i]*(-1.)])[:, None]), )
                gamma = torch.cat((gamma, torch.tensor([variables['gamma'][i]])[:, None]), )

            graph.alpha = alpha
            graph.beta = beta
            graph.gamma = gamma

        elif f'{self.pde}' == 'WE':
            bc_left, bc_right, c = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                bc_left = torch.cat((bc_left, torch.tensor([variables['bc_left'][i]])[:, None]), )
                bc_right = torch.cat((bc_right, torch.tensor([variables['bc_right'][i]])[:, None]), )
                c = torch.cat((c, torch.tensor([variables['c'][i]])[:, None]), )

            graph.bc_left = bc_left
            graph.bc_right = bc_right
            graph.c = c

        return graph


    def create_next_graph(self,
                             graph: Data,
                             pred: torch.Tensor,
                             labels: torch.Tensor,
                             steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # Output is the new input
        graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]   # self.tw: 25, graph.x/pred: [640, 25]
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):  # labels: [16, 25, 40], steps: list of 16 elements
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
        graph.y = y
        graph.pos[:, 0] = t_pos

        return graph

