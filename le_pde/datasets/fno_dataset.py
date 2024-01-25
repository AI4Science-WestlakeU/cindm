import scipy.io
import numpy as np
import h5py
import pickle
import torch
from torch_geometric.data import Dataset, Data
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from cindm.le_pde.utils import PDE_PATH, FNO_PATH, INVALID_VALUE
from cindm.le_pde.utils import read_zipped_array, write_zipped_array


class FNOData(Dataset):
    def __init__(
        self,
        dataset="fno",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_y_diff=True,
        is_train=True,
        show_missing_files=False,
        n_init_smoke=1,
        is_traj=False,
        traj_len=0,
        is_y_variable_length=False,
        transform=None,
        pre_transform=None,
        # datatype="1e-4",
        verbose=False,
    ):
        self.dataset = dataset
        if "fno" in self.dataset:
            self.root = os.path.join(PDE_PATH, FNO_PATH)
        else:
            raise
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_interval = time_interval
        self.n_init_smoke = n_init_smoke
        self.is_y_diff = is_y_diff
        self.is_train = is_train
        self.verbose = verbose
        self.is_y_variable_length = is_y_variable_length

        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        if is_y_variable_length:
            self.t_cushion_output = 1
        else:
            self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1

        self.is_traj = is_traj
        self.traj_len = traj_len
        if "fno" in self.dataset:
            if self.is_train:
                self.dirname = "fno"
            else:
                self.dirname = "fno"
        else:
            raise

        self.original_shape = (64, 64)
        self.dyn_dims = 1  # density
        if self.dataset == "fno-1":
            self.n_simu = 1000 if self.is_train else 200
            self.time_stamps = 50
            print(os.path.join(self.root, 'ns_V1e-3_N5000_T50.mat'))
            f = h5py.File(os.path.join(self.root, 'ns_V1e-3_N5000_T50.mat'), 'r')
            print(f.keys())
            if self.is_train:
                self.dataset_cache =  np.array(f['u'][...,:self.n_simu])
                print("is_train=True: ", self.dataset_cache.shape)
            else:
                self.dataset_cache = np.array(f['u'][...,-self.n_simu:])
                print("is_train=False: ", self.dataset_cache.shape)
        elif self.dataset == "fno-2" or dataset == "fno":
            self.n_simu = 1000 if self.is_train else 200
            self.time_stamps = 30
            print(os.path.join(self.root, 'ns_V1e-4_N10000_T30.mat'))
            f = h5py.File(os.path.join(self.root, 'ns_V1e-4_N10000_T30.mat'), 'r')
            print(f.keys())
            if self.is_train:
                self.dataset_cache =  np.array(f['u'][:self.time_stamps, :, :, :self.n_simu])
                print("is_train=True: ", self.dataset_cache.shape)
            else:
                self.dataset_cache = np.array(f['u'][:self.time_stamps, :, :, -self.n_simu:])
                print("is_train=False: ", self.dataset_cache.shape)
        elif self.dataset == "fno-3":
            self.n_simu = 9800 if self.is_train else 200
            self.time_stamps = 30
            print(os.path.join(self.root, 'ns_V1e-4_N10000_T30.mat'))
            f = h5py.File(os.path.join(self.root, 'ns_V1e-4_N10000_T30.mat'), 'r')
            print(f.keys())
            if self.is_train:
                self.dataset_cache =  np.array(f['u'][:self.time_stamps, :, :, :self.n_simu])
                print("is_train=True: ", self.dataset_cache.shape)
            else:
                self.dataset_cache = np.array(f['u'][:self.time_stamps, :, :, -self.n_simu:])
                print("is_train=False: ", self.dataset_cache.shape)
        elif self.dataset == "fno-4":
            self.n_simu = 1000 if self.is_train else 200
            self.time_stamps = 20
            print(os.path.join(self.root, 'NavierStokes_V1e-5_N1200_T20.mat'))
            f = scipy.io.loadmat(os.path.join(self.root, 'NavierStokes_V1e-5_N1200_T20.mat'))
            print(f['u'].shape)
            if self.is_train:
                #self.dataset_cache =  np.array(f['u'][:self.time_stamps, :, :, :self.n_simu])
                self.dataset_cache =  f['u'][:self.n_simu, :, :, :]
                print("is_train=True: ", self.dataset_cache.shape)
            else:
                self.dataset_cache = f['u'][-self.n_simu:, :, :, :]
                print("is_train=False: ", self.dataset_cache.shape)
        else:
            raise

        if self.is_y_variable_length:
            if self.dataset == "fno-4":
                # dataset: (200, 64, 64, 20)
                nan_tensor = np.broadcast_to(np.array([INVALID_VALUE]), self.dataset_cache.shape[:-1] + (self.output_steps*self.time_interval-1,))
                self.dataset_cache = np.concatenate([self.dataset_cache, nan_tensor], -1)
            else:
                # dataset: (20, 64, 64, 200)
                nan_tensor = np.broadcast_to(np.array([INVALID_VALUE]), (self.output_steps*self.time_interval-1,) + self.dataset_cache.shape[1:])
                self.dataset_cache = np.concatenate([self.dataset_cache, nan_tensor], 0)
        self.show_missing_files = show_missing_files
        self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output + 1 - self.traj_len) // self.time_interval
        print("root: ", self.root)
        super(FNOData, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.root + self.dirname]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dirname)

    @property
    def processed_file_names(self):
        return ["sim_{:06d}/dens_{:06d}.npz".format(k, i) for k in range(self.n_simu) for i in range(1000, 1000 + self.time_stamps)] + [
                "sim_{:06d}/velo_{:06d}.npz".format(k, i) for k in range(self.n_simu) for i in range(1000, 1000 + self.time_stamps)]

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

    def get_edge_index(self):
        edge_index_filename = os.path.join(self.processed_dir, "edge_index.p")
        mask_valid_filename = os.path.join(self.root, self.dirname, "mask_index.p")
        if os.path.isfile(edge_index_filename) and os.path.isfile(mask_valid_filename):
            edge_index = pickle.load(open(edge_index_filename, "rb"))
            mask_valid = pickle.load(open(mask_valid_filename, "rb"))
            return edge_index, mask_valid
        #velo_array = read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/velo_{:06d}.npz".format(0, 1000)))
        velo_array = self.dataset_cache[0, :, :, 0] # read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Velocity_{:06d}.npz".format(0, 0)))
        # velo_invalid_mask = velo_array[0, :-1, :-1, 0] == 0
        velo_invalid_mask = velo_array == 0
        mask_valid = torch.BoolTensor(~velo_invalid_mask).flatten()
        #velo_invalid_ids = np.where(velo_invalid_mask.flatten())[0]
        rows, cols = self.original_shape
        cube = np.arange(rows * cols).reshape(rows, cols)
        edge_list = []
        for i in range(rows):
            for j in range(cols):
                if i + 1 < rows: #and cube[i, j] not in velo_invalid_ids and cube[i+1, j] not in velo_invalid_ids:
                    edge_list.append([cube[i, j], cube[i+1, j]])
                    edge_list.append([cube[i+1, j], cube[i, j]])
                if j + 1 < cols: #and cube[i, j]: #not in velo_invalid_ids and cube[i, j+1] not in velo_invalid_ids:
                    edge_list.append([cube[i, j], cube[i, j+1]])
                    edge_list.append([cube[i, j+1], cube[i, j]])
        edge_index = torch.LongTensor(edge_list).T
        pickle.dump(edge_index, open(edge_index_filename, "wb"))
        pickle.dump(mask_valid, open(mask_valid_filename, "wb"))
        return edge_index, mask_valid

    def process(self):
        # Does not have effect for now:
        pass

    def len(self):
        if self.is_traj:
            if self.traj_len is None:
                return self.n_simu
            else:
                time_stamps_effective = (self.time_stamps - self.t_cushion_input) // self.traj_len
                return self.n_simu * time_stamps_effective
        else:
            return self.time_stamps_effective * self.n_simu

    def get(self, idx):
        if "fno" in self.dataset:
            """
            The idx encodes both the id and time step of the data.
            """
            assert self.time_interval == 1
            dataset = self.dataset_cache
            sim_id, time_id = divmod(idx, self.time_stamps_effective)
            if self.verbose:
                print(f"sim_id: {sim_id}   time_id: {time_id}   input: ({time_id * self.time_interval + self.t_cushion_input -self.input_steps * self.time_interval}, {time_id * self.time_interval + self.t_cushion_input + self.traj_len})  output: ({time_id * self.time_interval + self.t_cushion_input}, {time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval + self.traj_len})")
            if self.dataset == "fno-4":
                # dataset: (200, 64, 64, 20)
                x_dens = torch.FloatTensor(np.stack([dataset[sim_id, :, :, time_id * self.time_interval + self.t_cushion_input + j] for j in range(-self.input_steps * self.time_interval, 0 + self.traj_len, self.time_interval)], -1))  # [1, rows, cols, input_steps, 1]
                y_dens = torch.FloatTensor(np.stack([dataset[sim_id, :, :, time_id * self.time_interval + self.t_cushion_input + j] for j in range(0, self.output_steps * self.time_interval + self.traj_len, self.time_interval)], -1))  # [1, rows, cols, output_steps, 1]
            else:
                # dataset: (20, 64, 64, 200)
                x_dens = torch.FloatTensor(np.stack([dataset[time_id * self.time_interval + self.t_cushion_input + j, :, :, sim_id] for j in range(-self.input_steps * self.time_interval, 0 + self.traj_len, self.time_interval)], -1))  # [1, rows, cols, input_steps, 1]
                y_dens = torch.FloatTensor(np.stack([dataset[time_id * self.time_interval + self.t_cushion_input + j, :, :, sim_id] for j in range(0, self.output_steps * self.time_interval + self.traj_len, self.time_interval)], -1))  # [1, rows, cols, output_steps, 1]

            col_mesh, row_mesh = np.meshgrid(range(self.original_shape[1]), range(self.original_shape[0]))
            x_pos = torch.FloatTensor(np.stack([row_mesh, col_mesh], -1).reshape(-1, 1, 2))
            x_pos = x_pos / x_pos.max() * 2
            edge_index, mask_valid = self.get_edge_index()
            data = Data(
                x=x_dens.reshape(-1, *x_dens.shape[-1:], 1).clone(),       # [number_nodes: 64 * 64, input_steps, 1]
                x_pos=x_pos.clone(),  # [number_nodes: 128 * 128, 2]
                y=y_dens.reshape(-1, *y_dens.shape[-1:], 1).clone(),       # [number_nodes: 64 * 64, input_steps, 1]
                edge_index=edge_index,
                mask=mask_valid,
                original_shape=self.original_shape,
                dyn_dims=self.dyn_dims,
                compute_func=(0, None),
            )
            return data
        elif self.is_traj:
            """
            The idx is the idx of the trajectory.

            if self.traj_len is not None, then each n_steps will be self.traj_len.
                e.g. x: [9, 10,11,12,13], [14,15,16,17,18], ...   
                     y: [10,11,12,13,14], [15,16,17,18,19], ...  

                x: starts at t_cushion_input - self.input_steps + j*self.traj_len, ends at t_cushion_input-1 + (j+1)*self.traj_len
                y: starts at t_cushion_input + j*self.traj_len, ends at t_cushion_input + (j+1)*self.traj_len
                    j goes from 0 to (self.time_stamps + 1 - t_cushion_input) // self.traj_len
            """
            if self.traj_len == 0:
                sim_id = idx
                x_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Velocity_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input-self.input_steps * self.time_interval, self.time_stamps - self.time_interval, self.time_interval)], -2))  # [1, rows, cols, input_steps, 2]
                x_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Density_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input-self.input_steps * self.time_interval, self.time_stamps - self.time_interval, self.time_interval)], -2))  # [1, rows, cols, input_steps, 1]
                y_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Velocity_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input, self.time_stamps, self.time_interval)], -2))  # [1, rows, cols, output_steps, 2]
                y_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Density_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input, self.time_stamps, self.time_interval)], -2))  # [1, rows, cols, output_steps, 1]
                x_obst = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/domain_{:06d}.npz".format(sim_id, 0))) for j in range(self.t_cushion_input-self.input_steps * self.time_interval, self.time_stamps - self.time_interval, self.time_interval)], -2))  # [1, rows, cols, input_steps, 1]
            else:
                assert self.time_interval == 1
                time_stamps_effective = (self.time_stamps - self.t_cushion_input) // self.traj_len
                sim_id, bulk_id = divmod(idx, time_stamps_effective)
                if self.dataset == "movinggas":
                    x_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Velocity_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input - self.input_steps + bulk_id*self.traj_len, self.t_cushion_input - self.input_steps + (bulk_id+1)*self.traj_len)], -2))  # [1, rows, cols, input_steps, 2]
                    x_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Density_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input - self.input_steps + bulk_id*self.traj_len, self.t_cushion_input - self.input_steps + (bulk_id+1)*self.traj_len)], -2))  # [1, rows, cols, input_steps, 1]
                    y_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Velocity_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input + bulk_id*self.traj_len, self.t_cushion_input + (bulk_id+1)*self.traj_len + self.output_steps-1)], -2))  # [1, rows, cols, output_steps, 2]
                    y_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Density_{:06d}.npz".format(sim_id, j))) for j in range(self.t_cushion_input + bulk_id*self.traj_len, self.t_cushion_input + (bulk_id+1)*self.traj_len + self.output_steps-1)], -2))  # [1, rows, cols, output_steps, 1]
                    static = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/domain_{:06d}.npz".format(sim_id, 0))) for _ in range(self.traj_len)], -2))  # [1, rows, cols, input_steps, 1]
                elif self.dataset == "karman-2d":
                    x_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/velo_{:06d}.npz".format(sim_id, j))) for j in range(1000 + self.t_cushion_input - self.input_steps + bulk_id*self.traj_len, 1000 + self.t_cushion_input - self.input_steps + (bulk_id+1)*self.traj_len)], -2))  # [1, rows, cols, input_steps, 2]
                    x_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/dens_{:06d}.npz".format(sim_id, j))) for j in range(1000 + self.t_cushion_input - self.input_steps + bulk_id*self.traj_len, 1000 + self.t_cushion_input - self.input_steps + (bulk_id+1)*self.traj_len)], -2))  # [1, rows, cols, input_steps, 1]
                    y_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/velo_{:06d}.npz".format(sim_id, j))) for j in range(1000 + self.t_cushion_input + bulk_id*self.traj_len, 1000 + self.t_cushion_input + (bulk_id+1)*self.traj_len + self.output_steps-1)], -2))  # [1, rows, cols, output_steps, 2]
                    y_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/dens_{:06d}.npz".format(sim_id, j))) for j in range(1000 + self.t_cushion_input + bulk_id*self.traj_len, 1000 + self.t_cushion_input + (bulk_id+1)*self.traj_len + self.output_steps-1)], -2))  # [1, rows, cols, output_steps, 1]
                    params = pickle.load(open(os.path.join(self.root, self.dirname, "sim_{:06d}/params.pickle".format(sim_id)), "rb"))
                    re = params["re"]
                    static = torch.tensor(1e5 / re).expand(x_velo[...,-1:].shape)
            col_mesh, row_mesh = np.meshgrid(range(self.original_shape[1]), range(self.original_shape[0]))
            x_pos = torch.FloatTensor(np.stack([row_mesh, col_mesh], -1).reshape(-1, 1, 2))
            x_pos = x_pos / x_pos.max() * 2
            edge_index, mask_valid = self.get_edge_index()
            if self.dataset == "movinggas":
                x_velo = torch.cat([static, x_dens, x_velo[:,:-1,:-1,:]], -1)
                y_velo = torch.cat([y_dens, y_velo[:,:-1,:-1,:]], -1)
                data = Data(
                    x=x_velo[0,:,:].reshape(-1, *x_velo.shape[-2:]).clone(), # [number_nodes: 128 * 128, input_steps, 2]
                    x_pos=x_pos.clone(),  # [number_nodes: 128 * 128, 2]
                    y=y_velo[0,:,:].reshape(-1, *y_velo.shape[-2:]).clone(),
                    edge_index=edge_index,
                    mask=mask_valid,
                    original_shape=self.original_shape,
                    dyn_dims=self.dyn_dims,
                    compute_func=(0, None),
                )
            elif self.dataset == "karman-2d":
                data = Data(
                    x=x_velo[0,:-1,:-1].reshape(-1, *x_velo.shape[-2:]).clone(), # [number_nodes: 128 * 256, input_steps, 2]
                    x_dens=x_dens.reshape(-1, *x_dens.shape[-2:]).clone(),       # [number_nodes: 128 * 256, input_steps, 1]
                    x_pos=x_pos.clone(),  # [number_nodes: 128 * 256, 2]
                    y=y_velo[0,:-1,:-1].reshape(-1, *y_velo.shape[-2:]).clone(),
                    y_dens=y_dens.reshape(-1, *y_dens.shape[-2:]).clone(),
                    edge_index=edge_index,
                    mask=mask_valid,
                    original_shape=self.original_shape,
                    dyn_dims=self.dyn_dims,
                    compute_func=(0, None),
                )
            else:
                raise
            return data
        else:
            """
            The idx encodes both the id and time step of the data.
            """
            sim_id, time_id = divmod(idx, self.time_stamps_effective)
            #assert 1000 + self.t_cushion_input + (time_id - self.input_steps) * self.time_interval >= 1000
            #assert 1000 + self.t_cushion_input + (time_id + self.output_steps - 1) * self.time_interval < 1000 + self.time_stamps
            x_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Velocity_{:06d}.npz".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))) for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -2))  # [1, rows, cols, input_steps, 2]
            x_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Density_{:06d}.npz".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))) for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -2))  # [1, rows, cols, input_steps, 1]
            y_velo = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Velocity_{:06d}.npz".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))) for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -2))  # [1, rows, cols, output_steps, 2]
            y_dens = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/Density_{:06d}.npz".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))) for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -2))  # [1, rows, cols, output_steps, 1]
            x_obst = torch.FloatTensor(np.stack([read_zipped_array(os.path.join(self.root, self.dirname, "sim_{:06d}/domain_{:06d}.npz".format(sim_id, 0))) for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -2))  # [1, rows, cols, input_steps, 1]
            x_velo = torch.cat([x_obst, x_dens, x_velo[:,:-1,:-1,:]], -1)
            y_velo = torch.cat([y_dens, y_velo[:,:-1,:-1,:]], -1)
            col_mesh, row_mesh = np.meshgrid(range(self.original_shape[1]), range(self.original_shape[0]))
            x_pos = torch.FloatTensor(np.stack([row_mesh, col_mesh], -1).reshape(-1, 1, 2))
            x_pos = x_pos / x_pos.max() * 2
            edge_index, mask_valid = self.get_edge_index()
            data = Data(
                #x=x_velo[0,:-1,:-1].reshape(-1, *x_velo.shape[-2:]).clone(), # [number_nodes: 128 * 128, input_steps, 2]
                x=x_velo[0,:,:].reshape(-1, *x_velo.shape[-2:]).clone(), # [number_nodes: 128 * 128, input_steps, 2]
                #x_dens=x_dens.reshape(-1, *x_dens.shape[-2:]).clone(),       # [number_nodes: 128 * 128, input_steps, 1]
                x_pos=x_pos.clone(),  # [number_nodes: 128 * 128, 2]
                #y=y_velo[0,:-1,:-1].reshape(-1, *y_velo.shape[-2:]).clone(),
                y=y_velo[0,:,:].reshape(-1, *y_velo.shape[-2:]).clone(),
                #y_dens=y_dens.reshape(-1, *y_dens.shape[-2:]).clone(),
                edge_index=edge_index,
                mask=mask_valid,
                original_shape=self.original_shape,
                dyn_dims=self.dyn_dims,
                compute_func=(0, None),
            )
            return data


# In[ ]:


if __name__ == "__main__":
    dataset = FNOData(
        dataset="fno",
        input_steps=1,
        output_steps=4,
        time_interval=1,
        is_y_diff=False,
        is_train=False,
        is_y_variable_length=True,
        verbose=True,
    )

