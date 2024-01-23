from deepsnap.hetero_graph import HeteroGraph
import numpy as np
import h5py
import pickle
import torch
from torch_geometric.data import Dataset, Data
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from CinDM_anonymous.le_pde.utils import PDE_PATH, KARMAN3D_PATH
from CinDM_anonymous.le_pde.pytorch_net.util import plot_matrices
from CinDM_anonymous.le_pde.utils import get_root_dir, to_tuple_shape
from CinDM_anonymous.le_pde.utils import read_zipped_array, write_zipped_array


class Karman3D(Dataset):
    def __init__(
        self,
        dataset="karman3d",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_y_diff=True,
        is_train=True,
        show_missing_files=False,
        data_format="pyg",
        verbose=False,
        transform=None,
        pre_transform=None,
    ):
        self.dataset = dataset
        if dataset in ["karman3d", "karman3d-small", "karman3d-large", "karman3d-large-s", "karman3d-large-d", "karman3d-large-s-d"]:
            self.root = os.path.join(KARMAN3D_PATH)
        else:
            raise
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_interval = time_interval
        self.is_y_diff = is_y_diff
        self.is_train = is_train
        self.data_format = data_format

        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1

        self.verbose = verbose
        if dataset in ["karman3d", "karman3d-small"]:
            if self.is_train:
                self.dirname = "train"
            else:
                self.dirname = "test"
        elif dataset in ["karman3d-large", "karman3d-large-s", "karman3d-large-d", "karman3d-large-s-d"]:
            self.dirname = "raw"
        else:
            raise

        self.original_shape = (256, 128, 128)
        self.dyn_dims = 3
        if self.dataset == "karman3d-large-s-d":
            self.dyn_dims = 4
        if dataset in ["karman3d", "karman3d-small"]:
            self.n_simu = 5 if self.is_train else 5
            if self.is_train:
                self.time_stamps = 80
            else:
                self.time_stamps = 20
            self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output + 1) // self.time_interval
        elif dataset in ["karman3d-large", "karman3d-large-s", "karman3d-large-d", "karman3d-large-s-d"]:
            if self.is_train:
                self.start_time = {
                    "sim_Re_90989": 2000,
                    "sim_Re_92989": 2000,
                    "sim_Re_94989": 2000,
                    "sim_Re_95489": 1501,
                    "sim_Re_95989": 2000,
                }
                self.time_stamps = {
                    "sim_Re_90989": 500,
                    "sim_Re_92989": 500,
                    "sim_Re_94989": 500,
                    "sim_Re_95489": 400,
                    "sim_Re_95989": 500,
                }
                self.n_simu = len(self.start_time)
            else:
                self.start_time = {
                    "sim_Re_93989": 2000,
                    "sim_Re_94989_2": 1501,
                }
                self.time_stamps = {
                    "sim_Re_93989": 500,
                    "sim_Re_94989_2": 500,
                }
                self.n_simu = len(self.start_time)
        else:
            raise
        self.show_missing_files = show_missing_files
        print("root: ", self.root)
        super(Karman3D, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.root + self.dirname]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dirname)

    @property
    def processed_file_names(self):
        if self.dataset in ["karman3d", "karman3d-small"]:
            return ["sim_{:06d}/dens_{:06d}.npz".format(k, i) for k in range(self.n_simu) for i in range(1000, 1000 + self.time_stamps)] + [
                    "sim_{:06d}/velo_{:06d}.npz".format(k, i) for k in range(self.n_simu) for i in range(1000, 1000 + self.time_stamps)]
        elif self.dataset in ["karman3d-large", "karman3d-large-s", "karman3d-large-d", "karman3d-large-s-d"]:
            return ["{}/dens_{:06d}.npz".format(traj_name, i) for traj_name, time_stamps in self.time_stamps.items() for i in range(self.start_time[traj_name], self.start_time[traj_name] + time_stamps)] + [
                    "{}/velo_{:06d}.npz".format(traj_name, i) for traj_name, time_stamps in self.time_stamps.items() for i in range(self.start_time[traj_name], self.start_time[traj_name] + time_stamps)]
        else:
            raise

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _process(self):
        import warnings
        from typing import Any, List
        from torch_geometric.data.makedirs import makedirs
        def _repr(obj: Any) -> str:
            if obj is None:
                return 'None'
                return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

        def files_exist(files: List[str]) -> bool:
            # NOTE: We return `False` in case `files` is empty, leading to a
            # re-processing of files on every instantiation.
            return len(files) != 0 and all([os.path.exists(f) for f in files])

        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first")

        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        if self.show_missing_files:
            missing_files = [file for file in self.processed_paths if not os.path.exists(file)]
            print("Missing files:")
            pp.pprint(sorted(missing_files))

        print('Processing...')

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_filter), path)

        print('Done!')

    def get_edge_index(self, traj_name, start_time):
        edge_index_filename = os.path.join(self.processed_dir, "edge_index.p")
        mask_valid_filename = os.path.join(self.root, self.dirname, traj_name, "mask_index.p")
        if os.path.isfile(edge_index_filename) and os.path.isfile(mask_valid_filename):
            edge_index = pickle.load(open(edge_index_filename, "rb"))
            mask_valid = pickle.load(open(mask_valid_filename, "rb"))
            return edge_index, mask_valid
        velo_array = read_zipped_array(os.path.join(self.root, self.dirname, traj_name, "velo_{:06d}.npz".format(start_time)))
        velo_invalid_mask = velo_array[:, :-1, :-1, :-1, 0] == 0
        mask_valid = torch.BoolTensor(~velo_invalid_mask).flatten()
        depth, rows, cols = self.original_shape
        cube = np.arange(depth * rows * cols).reshape(depth, rows, cols)
        edge_list = []
        for h in range(depth):
            for i in range(rows):
                for j in range(cols):
                    if h + 1 < depth:
                        edge_list.append([cube[h, i, j], cube[h+1, i, j]])
                        edge_list.append([cube[h+1, i, j], cube[h, i, j]])
                    if i + 1 < rows: #and cube[i, j] not in velo_invalid_ids and cube[i+1, j] not in velo_invalid_ids:
                        edge_list.append([cube[h, i, j], cube[h, i+1, j]])
                        edge_list.append([cube[h, i+1, j], cube[h, i, j]])
                    if j + 1 < cols: #and cube[i, j]: #not in velo_invalid_ids and cube[i, j+1] not in velo_invalid_ids:
                        edge_list.append([cube[h, i, j], cube[h, i, j+1]])
                        edge_list.append([cube[h, i, j+1], cube[h, i, j]])
        edge_index = torch.LongTensor(edge_list).T
        pickle.dump(edge_index, open(edge_index_filename, "wb"))
        pickle.dump(mask_valid, open(mask_valid_filename, "wb"))
        return edge_index, mask_valid

    def process(self):
        pass

    def len(self):
        if self.dataset in ["karman3d", "karman3d-small"]:
            return self.time_stamps_effective * self.n_simu
        elif self.dataset in ["karman3d-large", "karman3d-large-s", "karman3d-large-d", "karman3d-large-s-d"]:
            return sum([(time_stamp - self.t_cushion_input - self.t_cushion_output + 1) // self.time_interval for time_stamp in self.time_stamps.values()])

    def get(self, idx):
        """
        The idx encodes both the id and time step of the data.
        """
        effective_time_stamps = (np.array(list(self.time_stamps.values())) - self.t_cushion_input - self.t_cushion_output + 1) // self.time_interval
        cum_time_stamps = np.cumsum(effective_time_stamps)
        gt = idx - cum_time_stamps >= 0
        if gt.any():
            sim_id = np.where(gt)[0][-1] + 1
            cum_time = cum_time_stamps[sim_id - 1]
        else:
            sim_id = 0
            cum_time = 0
        time_id = idx - cum_time 

        traj_name = list(self.start_time)[sim_id]  # sim_Re_{Re}...
        traj_name_split = traj_name.split("_")
        reynolds_number = int(traj_name_split[traj_name_split.index("Re") + 1])
        nu_norm = torch.FloatTensor([1e5 / reynolds_number])
        start_time = self.start_time[traj_name]
        if self.verbose:
            print(f"sim_id: {sim_id}  traj_name: {traj_name}   start_time: {start_time}   time_id: {time_id}   input: ({start_time + time_id * self.time_interval + self.t_cushion_input -self.input_steps * self.time_interval}, {start_time + time_id * self.time_interval + self.t_cushion_input})  output: ({start_time + time_id * self.time_interval + self.t_cushion_input}, {start_time + time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval})")
        x_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, traj_name, "velo_{:06d}.npz".format(start_time + time_id * self.time_interval + self.t_cushion_input + j))) for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -2))  # [1, rows, cols, input_steps, 3]
        x_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, traj_name, "dens_{:06d}.npz".format(start_time + time_id * self.time_interval + self.t_cushion_input + j)))[...,None] for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -2))  # [1, rows, cols, input_steps, 1]
        y_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, traj_name, "velo_{:06d}.npz".format(start_time + time_id * self.time_interval + self.t_cushion_input + j))) for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -2))  # [1, rows, cols, output_steps, 3]
        y_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, traj_name, "dens_{:06d}.npz".format(start_time + time_id * self.time_interval + self.t_cushion_input + j)))[...,None] for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -2))  # [1, rows, cols, output_steps, 1]
        x_velo = x_velo[:, :-1, :-1, :-1, :, :]
        y_velo = y_velo[:, :-1, :-1, :-1, :, :]
        col_mesh, row_mesh, depth_mesh = np.meshgrid(range(self.original_shape[2]), range(self.original_shape[1]), range(self.original_shape[0]))
        x_pos = torch.FloatTensor(np.stack([depth_mesh, row_mesh, col_mesh], -1).reshape(-1, 1, 3))
        if self.dataset == "karman3d-large-s":
            nu_tensor = nu_norm.expand(x_velo[...,-1:].shape)
            x_velo = torch.cat([nu_tensor, x_velo], -1)
        elif self.dataset == "karman3d-large-d":
            x_velo = torch.cat([x_dens, x_velo], -1)
            y_velo = torch.cat([y_dens, y_velo], -1)
        elif self.dataset == "karman3d-large-s-d":
            nu_tensor = nu_norm.expand(x_velo[...,-1:].shape)
            x_velo = torch.cat([nu_tensor, x_dens, x_velo], -1)
            y_velo = torch.cat([y_dens, y_velo], -1)
        x_pos = x_pos / x_pos.max() * 2
        edge_index, mask_valid = self.get_edge_index(traj_name, start_time)
        data = Data(
            x=x_velo.reshape(-1, *x_velo.shape[-2:]).clone(), # [number_nodes: 256 * 128 * 128, input_steps, 4]
            x_pos=x_pos.clone(),  # [number_nodes: 128 * 128, 2]
            y=y_velo.reshape(-1, *y_velo.shape[-2:]).clone(),
            edge_index=edge_index,
            mask=mask_valid,
            original_shape=self.original_shape,
            dyn_dims=self.dyn_dims,
            compute_func=(0, None),
            param=nu_norm,
        )
        if self.data_format == "deepsnap":
            data = HeteroGraph(
                edge_index={("n0", "0", "n0"): data.edge_index},
                node_feature={"n0": data.x},
                node_label={"n0": data.y},
                mask={"n0": data.mask},
                directed=True,
                original_shape=(("n0", data.original_shape),),
                dyn_dims=(("n0", to_tuple_shape(data.dyn_dims)),),
                compute_func=(("n0", to_tuple_shape(data.compute_func)),),
                param={"n0": data.param[None]} if hasattr(data, "param") else {"n0": torch.ones(1,1)},
                grid_keys=("n0",),
                part_keys=(),
            )
        return data


# In[ ]:


if __name__ == "__main__":
    import matplotlib.pylab as plt
    dataset = Karman3D(
        dataset="karman3d-large-s-d",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_y_diff=True,
        is_train=False,
        show_missing_files=False,
        data_format="pyg",
        verbose=True,
        transform=None,
        pre_transform=None
    )
    data = dataset[0]
    x = data.x
    x_reshape = x[...,2].reshape(256, 128, 128)
    plt.imshow(x_reshape[:,:,60], cmap="jet")
    plt.show()
    
    # space:
    for user in range(10,200,1):
        data = dataset[0]
        x = data.x
        x_reshape = x[...,2].reshape(256, 128, 128)
        x = data.x
        x_reshape = x[...,1].reshape(256, 128, 128)
        plt.imshow(x_reshape[:,:,user], cmap="jet")
        plt.show()
    
    # time:
    for user in range(10,200,80):
        data = dataset[user]
        x = data.x
        x_reshape = x[...,1].reshape(256, 128, 128)
        plt.imshow(x_reshape[:,:,60], cmap="jet")
        plt.show()

