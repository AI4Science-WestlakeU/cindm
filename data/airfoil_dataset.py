import numpy as np
import pickle
import torch
from torch_geometric.data import Dataset, Data
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from cindm.filepath import AIRFOILS_PATH


# In[4]:


class Ellipse(Dataset):
    def __init__(
        self,
        dataset="naca_ellipse",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_y_diff=True,
        is_train=True,
        show_missing_files=False,
        # n_init_smoke=1,
        is_traj=False,
        traj_len=None,
        transform=None,
        pre_transform=None,
        is_testdata = False,
        is_y_variable_length=False,
        # is_offmask=False,
        is_bdmask=True,
    ):
        self.dataset = dataset
        #pdb.set_trace()
        if self.dataset == "chaotic_ellipse":
            self.root = "./"
        elif self.dataset.startswith('naca_ellipse'):
            self.root = AIRFOILS_PATH
        else:
            raise
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_interval = time_interval
        self.is_y_diff = is_y_diff
        self.is_train = is_train
        self.is_testdata = is_testdata
        self.is_y_variable_length = is_y_variable_length
        self.is_bdmask = is_bdmask
        # self.is_offmask = is_offmask
        
        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        if self.is_y_variable_length:
            self.t_cushion_output = 1
        else:
            self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1

        self.is_traj = is_traj
        self.traj_len = traj_len
        if (self.dataset == "chaotic_ellipse") or (self.dataset.startswith("naca_ellipse")):
            if self.is_train: 
                self.dirname="training_trajectories"
            else:
                self.dirname="test_trajectories"
        else:
            raise    

        if self.dataset == "chaotic_ellipse":
            if self.is_testdata:
                self.n_simu = 5
            else:
                self.n_simu = 400 if self.is_train else 40
            self.time_stamps = 300 - 1
            self.original_shape = (128 - 2, 128 - 2)
            self.dyn_dims = 2  # velo_X, velo_Y, density
        elif self.dataset.startswith("naca_ellipse"):
            if self.is_testdata:
                self.n_simu = 10 if self.is_train else 2
            else:
                self.n_simu = 30000 if self.is_train else 1000 # user add
            self.time_stamps = 100 # user add

            self.original_shape = (64 - 2, 64- 2)
            self.dyn_dims = 3  # velo_X, velo_Y, density
            self.x_max = None
            self.x_min = None
            self.y_max = None
            self.y_min = None
            self.p_max = None
            self.p_min = None
            self.x_bdmax = 62
            self.x_bdmin = 0
            
            # pdb.set_trace()
            normalization_filename = os.path.join(self.processed_dir, "normalization_max_min.p")
            if os.path.isfile(normalization_filename): 
                normdict = pickle.load(open(normalization_filename, "rb"))
                self.x_max = normdict["x_max"]
                self.x_min = normdict["x_min"]
                self.y_max = normdict["y_max"]
                self.y_min = normdict["y_min"]
                self.p_max = normdict["p_max"]
                self.p_min = normdict["p_min"]
            else:
                # for total_simid, total_timeid, dname in zip([1000, 200], [200, 200], ["training_trajectories", "test_trajectories"]): # by user
                for total_simid, total_timeid, dname in zip([10000, 200], [100, 100], ["training_trajectories", "test_trajectories"]): # user add 
                    for simid in range(total_simid):
                        for timeid in range(total_timeid):
                            # temp_velo = torch.FloatTensor(np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/velocity_{:06d}.npy".format(simid, timeid))), (1,2,0)))  # [rows, cols, 2]
                            temp_velo = torch.FloatTensor(np.transpose(np.load(os.path.join(self.root, dname, "sim_{:06d}/velocity_{:06d}.npy".format(simid, timeid))), (1,2,0)))  # [rows, cols, 2]
                            cand_x_max = torch.max(temp_velo[...,0])
                            if self.x_max == None:
                                self.x_max = cand_x_max
                            elif cand_x_max > self.x_max:
                                self.x_max = cand_x_max

                            cand_x_min = torch.min(temp_velo[...,0])
                            if self.x_min == None:
                                self.x_min = cand_x_min
                            elif cand_x_min < self.x_min:
                                self.x_min = cand_x_min

                            cand_y_max = torch.max(temp_velo[...,1])
                            if self.y_max == None:
                                self.y_max = cand_y_max
                            elif cand_y_max > self.y_max:
                                self.y_max = cand_y_max

                            cand_y_min = torch.min(temp_velo[...,1])
                            if self.y_min == None:
                                self.y_min = cand_y_min
                            elif cand_y_min < self.y_min:
                                self.y_min = cand_y_min

                            # temp_press = torch.FloatTensor(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/pressure_{:06d}.npy".format(simid, timeid))))
                            temp_press = torch.FloatTensor(np.load(os.path.join(self.root, dname, "sim_{:06d}/pressure_{:06d}.npy".format(simid, timeid))))              
                            cand_p_max = torch.max(temp_press)
                            if self.p_max == None:
                                self.p_max = cand_p_max
                            elif cand_p_max > self.p_max:
                                self.p_max = cand_p_max

                            cand_p_min = torch.min(temp_press)
                            if self.p_min == None:
                                self.p_min = cand_p_min
                            elif cand_p_min < self.p_min:
                                self.p_min = cand_p_min

                normalization_dict_train = {
                    "x_max": self.x_max,
                    "x_min": self.x_min,
                    "y_max": self.y_max,
                    "y_min": self.y_min,
                    "p_max": self.p_max,
                    "p_min": self.p_min
                }
                normalization_dict_test = {
                    "x_max": self.x_max,
                    "x_min": self.x_min,
                    "y_max": self.y_max,
                    "y_min": self.y_min,
                    "p_max": self.p_max,
                    "p_min": self.p_min
                }
                pickle.dump(normalization_dict_train, open(os.path.join(self.root, "training_trajectories", "normalization_max_min.p"), "wb"))
                # pickle.dump(normalization_dict_test, open(os.path.join(self.root, "test_trajectories", "normalization_max_min.p"), "wb"))
        else:
            raise
        self.show_missing_files = show_missing_files
        self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output) // self.time_interval
        super(Ellipse, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.root + self.dirname]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dirname)

    @property
    def processed_file_names(self):
        return ["sim_{:06d}/velocity_{:06d}.npz".format(k, i) for k in range(self.n_simu) for i in range(1000, 1000 + self.time_stamps)] + [
                "sim_{:06d}/pressure_{:06d}.npz".format(k, i) for k in range(self.n_simu) for i in range(1000, 1000 + self.time_stamps)]

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
        if os.path.isfile(edge_index_filename):
            edge_index = pickle.load(open(edge_index_filename, "rb"))
            return edge_index#, mask_valid
        velo_array = np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/velocity_{:06d}.npy".format(0, 0)))
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
        return edge_index#, mask_valid
    
    def discretize_boundary(self, boundary):
        r""" Compute indices of cells that include boundary nodes
        Args:
            boundary: tensor of real-valued boundary nodes, whose 
            shape is [#number of boundary nodes, 2]
        Return
            x_inds: tensor of x-indices for cells that include boundary nodes
            , whose shape is [#number of boundary nodes]
            y_inds: tensor of x-indices for cells that include boundary nodes
            , whose shape is [#number of boundary nodes]
        """
        n, m = self.original_shape
        n = n + 2
        n = m + 2
        num_bound = boundary.shape[0]
        device = boundary.device
        
        p_5 = torch.tensor([0.5], device=device).repeat(num_bound)
        x = torch.minimum(torch.maximum(boundary[:, 0], p_5), torch.tensor([n-1.5], device=device).repeat(num_bound))
        x_inds = torch.minimum(x.type(torch.int32), torch.tensor([n-2], device=device).repeat(num_bound))
        fs = x - x_inds

        y = torch.minimum(torch.maximum(boundary[:, 1], p_5), torch.tensor([m-1.5], device=device).repeat(num_bound))
        y_inds = torch.minimum(y.type(torch.int32), torch.tensor([m-2], device=device).repeat(num_bound))
        ft = y - y_inds
        
        return x_inds, y_inds
    
    def get_orientation(self, boundary):
        r""" Compute indices of cells that include boundary nodes
        Args:
            boundary: tensor of real-valued boundary nodes, whose 
            shape is [#number of boundary nodes, 2]
        Return:
            reduced indices of cells that include boundary nodes,
            , whose shape is [#number of boundary nodes, 2]
        """
        x_inds, y_inds = self.discretize_boundary(boundary)
        maxnum = 100
        indices = torch.stack((100*x_inds,y_inds), 0)
        sum_indices = indices.sum(0)
        ind_unique = torch.unique(sum_indices, sorted=True)
        x_idx = (torch.cat([(sum_indices==ind_u).nonzero()[0] for ind_u in ind_unique])).sort()[0]
        reduced_idx = x_idx
        stack_index = torch.stack((x_inds, y_inds), 0)
        
        return stack_index[:, reduced_idx].t()

    def process(self):
        # Does not have effect for now:
        if self.is_train:
            pass
            #for i in range(self.n_simu):
            #    os.system('python {}/solver_in_the_loop/karman-2d/karman.py -o karman-fdt-hires-set -r 128 -l 100 --re `echo $(( 10000 * 2**({}+4) ))` --gpu "-1" --seed 0 --thumb;'.format(get_root_dir(), i))
        else:
            pass
            #for i in range(self.n_simu):
            #    os.system('python {}/solver_in_the_loop/karman-2d/karman.py -o karman-fdt-hires-testset -r 128 -l 100 --re `echo $(( 10000 * 2**({}+3) * 3 ))` --gpu "-1" --seed 0 --thumb'.format(get_root_dir(), i))

    def len(self):
        if self.is_traj:
            if self.traj_len is None:
                return self.n_simu
            else:
                time_stamps_effective = (self.time_stamps - self.t_cushion_input) // self.traj_len
                return self.n_simu * time_stamps_effective
        else:
            return ((self.time_stamps - self.t_cushion_input - self.t_cushion_output) // self.time_interval) * self.n_simu

    def get(self, idx):
        sim_id, time_id = divmod(idx, self.time_stamps_effective)
        # Velocity for input
        #import pdb
        #pdb.set_trace()
        x_velo = torch.FloatTensor(np.stack([np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/velocity_{:06d}.npy".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))), (1,2,0))  for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -2))  # [rows, cols, input_steps, 2]
        x_velo[...,0] = (torch.clamp((x_velo[...,0] - self.x_min) / (self.x_max - self.x_min), 0, 1) - 0.5) * 2
        x_velo[...,1] = (torch.clamp((x_velo[...,1] - self.y_min) / (self.y_max - self.y_min), 0, 1) - 0.5) * 2
        x_velo[torch.isnan(x_velo)] = 0
        # Pressure for input
        x_pressure = torch.FloatTensor(np.stack([np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/pressure_{:06d}.npy".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))) for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -1))[..., None]  # [rows, cols, input_steps, 1]
        x_pressure = (torch.clamp((x_pressure - self.p_min) / (self.p_max - self.p_min), 0, 1) - 0.5) * 2
        x_pressure[torch.isnan(x_pressure)] = 0
        # Concatenate inputs
        x_feature = torch.cat((x_velo, x_pressure), -1)
        
        # Velocity for output
        y_velo = torch.FloatTensor(np.stack([np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/velocity_{:06d}.npy".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))), (1,2,0)) for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -2))  # [rows, cols, output_steps, 2]
        y_velo[...,0] = (torch.clamp((y_velo[...,0] - self.x_min) / (self.x_max - self.x_min), 0, 1) - 0.5) * 2
        y_velo[...,1] = (torch.clamp((y_velo[...,1] - self.y_min) / (self.y_max - self.y_min), 0, 1) - 0.5) * 2
        y_velo[torch.isnan(y_velo)] = 0

        # Pressure for output
        y_pressure = torch.FloatTensor(np.stack([np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/pressure_{:06d}.npy".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))) for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -1))[..., None]  # [rows, cols, input_steps, 1]
        y_pressure = (torch.clamp((y_pressure - self.p_min) / (self.p_max - self.p_min), 0, 1) - 0.5) * 2
        y_pressure[torch.isnan(y_pressure)] = 0
        
        # Concatenate outpus
        y_feature = torch.cat((y_velo, y_pressure), -1)
        
        if self.dataset == "chaotic_ellipse":
            x_bound = torch.FloatTensor(np.stack([np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/boundary_{:06d}.npy".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j))))  for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -2))
            y_bound = torch.FloatTensor(np.stack([np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/boundary_{:06d}.npy".format(sim_id, time_id * self.time_interval + self.t_cushion_input + j)))) for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -2))
        elif self.dataset.startswith("naca_ellipse"):
            x_bound = torch.FloatTensor(np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/boundary.npy".format(sim_id))))) 
            x_bound = (torch.clamp((x_bound - self.x_bdmin) / (self.x_bdmax - self.x_bdmin), 0, 1) - 0.5) * 2
            x_bound[torch.isnan(x_bound)] = 0
            y_bound = x_bound.clone()
            if self.is_bdmask:
                bdoffset = torch.FloatTensor(np.load(os.path.join(self.root, self.dirname, 'boundary_offset/sim_{:06d}.npy'.format(sim_id)))).reshape(-1,1,2)  # [rows*cols, 1, 2]
                bdmask = torch.FloatTensor(np.load(os.path.join(self.root, self.dirname, 'boundary_mask/sim_{:06d}.npy'.format(sim_id)))).reshape(-1, 1, 1)  # [rows*cols, 1, 1]
                # bdcenter = np.load(os.path.join(self.root, self.dirname, 'boundary_centerpts/np_base_pts_{:06d}.npy'.format(sim_id)))
        col_mesh, row_mesh = np.meshgrid(range(self.original_shape[1]), range(self.original_shape[0]))
        x_pos = torch.FloatTensor(np.stack([row_mesh, col_mesh], -1).reshape(-1, 1, 2))
        x_pos = x_pos / x_pos.max() * 2
        edge_index = self.get_edge_index()
        if self.dataset == "naca_ellipse":
            if self.is_bdmask:
                bdmask[torch.isnan(bdmask)] = 0
                bdoffset[torch.isnan(bdoffset)] = 0
                data = Data(
                    x=x_feature.reshape(-1, *x_feature.shape[-2:]).clone(), # [number_nodes: 62 * 62, input_steps, 3]
                    x_pos=x_pos.clone(),  # [number_nodes: 62 * 62, input_steps, 2]
                    x_bound=x_bound, #[number of nodes composing boundary, 2]
                    boundary_pos=torch.Tensor(np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/boundary.npy".format(sim_id))))), #[number of nodes in boundary, 2]
                    y=y_feature.reshape(-1, *y_feature.shape[-2:]).clone(), # [number_nodes: 62 * 62, input_steps, 3]
                    y_bound=y_bound,
                    bdmask=bdmask,
                    bdoffset=bdoffset,
                    # bdcenter=bdcenter,
                    edge_index=edge_index,
                    original_shape=self.original_shape,
                    dyn_dims=self.dyn_dims,
                    compute_func=(0, None),
                    sim_id=sim_id,
                    time_id=time_id,
                )
            else:
                data = Data(
                    x=x_feature.reshape(-1, *x_feature.shape[-2:]).clone(), # [number_nodes: 62 * 62, input_steps, 3]
                    x_pos=x_pos.clone(),  # [number_nodes: 62 * 62, input_steps, 2]
                    x_bound=x_bound, #[number of nodes composing boundary, 2]
                    boundary_pos=torch.Tensor(np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/boundary.npy".format(sim_id))))), #[number of nodes in boundary, 2]
                    y=y_feature.reshape(-1, *y_feature.shape[-2:]).clone(), # [number_nodes: 62 * 62, input_steps, 3]
                    y_bound=y_bound,
                    edge_index=edge_index,
                    original_shape=self.original_shape,
                    dyn_dims=self.dyn_dims,
                    compute_func=(0, None),
                    sim_id=sim_id,
                    time_id=time_id,
                )
        elif self.dataset == "naca_ellipse_lepde":
            if self.is_bdmask:
                static_grid = torch.cat([torch.cat([bdmask, bdoffset], -1) for _ in range(self.input_steps)], -2)
                # static_grid = torch.cat([bdmask for _ in range(self.input_steps)], -2)
                data = Data(
                    x=torch.cat([static_grid, x_feature.reshape(-1, *x_feature.shape[-2:]).clone()], -1), # [number_nodes: 62 * 62, input_steps, 4]
                    x_pos=x_pos.clone(),  # [number_nodes: 62 * 62, 1, 2]
                    x_bound=x_bound, #[number of nodes composing boundary, 2]
                    param=x_bound.clone().flatten(),
                    # param=torch.where(bdmask == 1, 0.0, 1.0),
                    boundary_pos=np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/boundary.npy".format(sim_id)))), #[number of nodes in boundary, 2]
                    # bdcenter=bdcenter,
                    y=y_feature.reshape(-1, *y_feature.shape[-2:]).clone(), # [number_nodes: 62 * 62, input_steps, 3]
                    # mask=torch.ones(62*62,  dtype=torch.bool),
                    mask = torch.where(bdmask == 1, False, True).flatten(),
                    y_bound=y_bound,
                    edge_index=edge_index,
                    original_shape=self.original_shape,
                    dyn_dims=self.dyn_dims,
                    compute_func=(0, None),
                    sim_id=sim_id,
                    time_id=time_id,
                )
            else:
                # static_grid = torch.cat([bdmask for _ in range(self.input_steps)], -2)
                data = Data(
                    x=x_feature.reshape(-1, *x_feature.shape[-2:]).clone(), # [number_nodes: 62 * 62, input_steps, 4]
                    x_pos=x_pos.clone(),  # [number_nodes: 62 * 62, 1, 2]
                    x_bound=x_bound, #[number of nodes composing boundary, 2]
                    param=x_bound.clone().flatten(),
                    # param=torch.where(bdmask == 1, 0.0, 1.0),
                    boundary_pos=np.transpose(np.load(os.path.join(self.root, self.dirname, "sim_{:06d}/boundary.npy".format(sim_id)))), #[number of nodes in boundary, 2]
                    # bdcenter=bdcenter,
                    y=y_feature.reshape(-1, *y_feature.shape[-2:]).clone(), # [number_nodes: 62 * 62, input_steps, 3]
                    # mask=torch.ones(62*62,  dtype=torch.bool),
                    # mask = torch.where(bdmask == 1, False, True).flatten(),
                    y_bound=y_bound,
                    edge_index=edge_index,
                    original_shape=self.original_shape,
                    dyn_dims=self.dyn_dims,
                    compute_func=(0, None),
                    sim_id=sim_id,
                    time_id=time_id,
                )                
        else:
            raise
            
        return data


# In[5]:


if __name__ == "__main__":
    dataset = Ellipse(
        dataset="naca_ellipse",
        input_steps=4,
        output_steps=1,
        time_interval=1,
        is_y_diff=False,
        is_train=True,
        show_missing_files=False,
        is_traj=False,
        is_testdata = True,
    )

    print(dataset[285].x.shape)

    print(dataset[285].x_bound.shape)

    print(dataset[285].edge_index.shape)

    print(dataset[285].boundary_pos)


# In[ ]:



