import numpy as np
from datetime import datetime
from termcolor import colored
import torch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from collections import deque
import pdb
from torch.utils.data import Sampler
from torch.autograd import Variable
from numbers import Number
import pickle
import math
from deepsnap.batch import Batch as deepsnap_Batch

COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine",
             "b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]

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

p = Printer(n_digits=6)

def compute_orthonormal(boundary):                          ## set the ortho-normal values based on two points
    r"""
    Precompute orthonormal vectors on boundary nodes
    Args:
        boudary: boundary, shape=[number of boundary nodes, 2]
    """
    # import pdb
    # pdb.set_trace()
    rolled_boundary = torch.roll(boundary, -1, 0)
    length = (boundary - rolled_boundary).norm(dim=1)       ## distance between two adjacent nodes
    tx = (boundary[:, 0] - rolled_boundary[:, 0])/length    ## x tangent
    ty = (boundary[:, 1] - rolled_boundary[:, 1])/length    ## y tangent
    nx = ty
    ny = -tx                                                 ## normal vector
    center = (boundary + rolled_boundary)/2
    return length, nx, ny, center


def linear_transform(pressure, center):
    r"""
    Compute magnitude of pressure on boundary nodes
    Args:
        pressure: pressure, shape=[width, height]
        center: midpoints of adjacent nodes in boundary, shape=[#number of bounday points, 2]
    """
    # flip_pressure = torch.flip(pressure, [0])
    flip_pressure = pressure
    device = flip_pressure.device
    n, m = flip_pressure.shape
    n = n #+ 2 #+ 2
    m = m #+ 2 #+ 2
    num_bound = center.shape[0]
    
    # import pdb
    # pdb.set_trace()
    p_5 = torch.tensor([0.5], device=device).repeat(num_bound)
    x = torch.minimum(torch.maximum(center[:, 0], p_5), torch.tensor([n-1.5], device=device).repeat(num_bound))
    # print("x: ", x)
    x_inds = torch.minimum(x.type(torch.int32), torch.tensor([n-2], device=device).repeat(num_bound))
    fs = x - x_inds
    
    y = torch.minimum(torch.maximum(center[:, 1], p_5), torch.tensor([m-1.5], device=device).repeat(num_bound))
    y_inds = torch.minimum(y.type(torch.int32), torch.tensor([m-2], device=device).repeat(num_bound))
    ft = y - y_inds
    
    s_mask = (fs==0)
    t_mask = (ft==0)
    bd_mask = s_mask*t_mask
    in_mask = torch.logical_not(bd_mask)
    raw_force = torch.zeros(num_bound, dtype=torch.float32, device=device)
    # raw_force[bd_mask] = flip_pressure[x_inds[bd_mask], y_inds[bd_mask]]
    raw_force[bd_mask] = flip_pressure[y_inds[bd_mask], x_inds[bd_mask]]
    
    t_weight = torch.stack([ft[in_mask], 1-ft[in_mask]], 0)
    #column
    # a_pipj = flip_pressure[x_inds[in_mask]+1, y_inds[in_mask]+1]
    a_pipj = flip_pressure[y_inds[in_mask]+1, x_inds[in_mask]+1]
    # a_pij  = flip_pressure[x_inds[in_mask]+1, y_inds[in_mask]]
    a_pij  = flip_pressure[y_inds[in_mask], x_inds[in_mask]+1]
    a_rowp1 = torch.stack([a_pipj, a_pij], 0)
    sum_a_rowp1 = torch.sum(t_weight * a_rowp1, 0)    
    #row
    # a_ipj = flip_pressure[x_inds[in_mask], y_inds[in_mask]+1]
    a_ipj = flip_pressure[y_inds[in_mask]+1, x_inds[in_mask]]
    # a_ij  = flip_pressure[x_inds[in_mask], y_inds[in_mask]]
    a_ij  = flip_pressure[y_inds[in_mask], x_inds[in_mask]]
    a_row = torch.stack([a_ipj, a_ij], 0)
    sum_a_row = torch.sum(t_weight * a_row, 0)
    
    s_weight = torch.stack([fs[in_mask], 1-fs[in_mask]], 0)
    sum_two_rows = torch.stack([sum_a_rowp1, sum_a_row], 0)
    raw_force[in_mask] = torch.sum(s_weight*sum_two_rows, 0)
        
    return raw_force
    

def compute_pressForce(pressure, boundary):
    r"""
    Compute pressure of nodes along orthonormal vectors
    Args:
        pressure: predicted pressure of model, shape=[62, 62]
        boundary: shape=[#number of boundary nodes, 2]
    """
    length, nx, ny, cen = compute_orthonormal(boundary)
    # import pdb
    # pdb.set_trace()
    pdl = linear_transform(pressure, cen)
    # print("pdl: ", pdl)
    pdl = pdl * length
    return torch.sum(pdl*nx), torch.sum(pdl*ny)


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


def get_item_1d(data, target):
    """
    Get the 1d item suitable for diffusion.

    Args:
        data: PyG data class
        target: choose from "x" and "y".

    Returns:
        x: has shape of [B, n_steps, n_bodies*feature_size], suitable for diffusion model
    """
    # pdb.set_trace()
    x = data[target]  # [B*n_bodies, n_steps, feature_size]
    batch_size = len(data.dyn_dims)
    assert x.shape[0] % batch_size == 0
    n_bodies = x.shape[0] // batch_size
    n_steps, feature_size = x.shape[1:]
    x = x.reshape(-1, n_bodies, n_steps, feature_size) / 200.  # [B, n_bodies, n_steps, feature_size]
    x = x.permute(0, 2, 1, 3)  # [B, n_steps, n_bodies, feature_size]
    x = torch.flatten(x, -2, -1)  # [B, n_steps, n_bodies*feature_size]
    return x
def get_item_1d_for_solver(data, target):
    """
    Get the 1d item suitable for diffusion.

    Args:
        data: PyG data class
        target: choose from "x" and "y".

    Returns:
        x: has shape of [B, n_steps, n_bodies*feature_size], suitable for diffusion model
    """
    # pdb.set_trace()
    x = data[target]  # [B*n_bodies, n_steps, feature_size]
    batch_size = len(data.dyn_dims)
    assert x.shape[0] % batch_size == 0
    n_bodies = x.shape[0] // batch_size
    n_steps, feature_size = x.shape[1:]
    x = x.reshape(-1, n_bodies, n_steps, feature_size)  # [B, n_bodies, n_steps, feature_size]
    x = x.permute(0, 2, 1, 3)  # [B, n_steps, n_bodies, feature_size]
    x = torch.flatten(x, -2, -1)  # [B, n_steps, n_bodies*feature_size]
    return x
def convert_softbd2hard(softbd):
    r"""
    Convert soft boundary whose value range between [0, 1] to solid boundary mask
    Args:
        softbd: grid with values ranging between 0 and 1
    Output:
        hardbd: grid with binary values, where 1 incidates existance of boundary
    """
    ones = softbd > 0.5
    zeros = softbd <= 0.5
    hardbd = torch.zeros(softbd.shape, device=softbd.device)
    hardbd[ones] = 1
    hardbd[zeros] = 0
    return hardbd

def find_isolated_points(grid):
    isolated_points = []
    rows, cols = grid.shape

    # Define possible neighboring cell offsets
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            current_cell = grid[r, c]
            if current_cell == 1:
                is_isolated = True
                if (r in [0, rows-1]) or (c in [0, cols-1]):
                    if (r in [0, rows-1]) and (c in [0, cols-1]):
                        n = 3 
                    else:
                        n = 5 
                else:
                    n = 8 

                j = 0
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc

                    # Check if the neighboring cell is inside the grid
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_cell = grid[nr, nc]
                        # Check if the neighboring cell has the same value
                        if current_cell == neighbor_cell:
                            # is_isolated = False
                            # break
                            continue
                        else:
                            j += 1
                        if n == 8 and j >= 7:
                            isolated_points.append((r, c))
                            break
                        elif n == 5 and j >= 4:
                            isolated_points.append((r, c))
                            break
                        elif n == 3 and j >= 2:
                            isolated_points.append((r, c))
                            break


                # if is_isolated and current_cell != 0:
                #     isolated_points.append((r, c))

    return isolated_points

def filter_isolated_points(hard_boundary):
    iso_points = find_isolated_points(hard_boundary)
    # print(iso_points)
    while (len(iso_points) != 0):
        index_tensor = torch.tensor(iso_points, dtype=torch.long, device=hard_boundary.device).t()
        values = torch.zeros(index_tensor.shape[1], device=hard_boundary.device)

        # Update the tensor with new values according to the indices
        hard_boundary.index_put_((index_tensor[0], index_tensor[1]), values)
        iso_points = find_isolated_points(hard_boundary)
        # print(iso_points)
        # break
    return hard_boundary

def find_clusters(hard_boundary):
    # Find the non-zero grid cells
    non_zero_cells = np.argwhere(hard_boundary.detach().cpu().numpy() != 0)

    # Create the DBSCAN clustering model
    dbscan = DBSCAN(eps=1.5, min_samples=2)

    # Fit the model to the non-zero grid cells
    labels = dbscan.fit_predict(non_zero_cells)

    clustered_grid = np.zeros_like(hard_boundary)

    for cell, label in zip(non_zero_cells, labels):
        clustered_grid[tuple(cell)] = label + 1
    # print(clustered_grid)
    return clustered_grid

def find_filtered_clusteres(hard_boundary, is_plot_clusters=False):
    clustered_grid = find_clusters(hard_boundary)
    # clustered_grid
    
    if is_plot_clusters:
        fig, ax = plt.subplots(figsize=(4,4), ncols=1)
        mappable0 = ax.imshow(clustered_grid, cmap='viridis',
                                 #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                 #interpolation="bicubic",
                                 aspect='auto'
                                 ) #,
                                 #origin='lower')
        fig.colorbar(mappable0, ax=ax)
        fig.tight_layout()    
        plt.show()

    hard_boundary = filter_isolated_points(hard_boundary)

    # print(np_clustered_grid)
    rd_clustered_grid = find_clusters(hard_boundary)
    # clustered_grid

    if is_plot_clusters:
        fig, ax = plt.subplots(figsize=(4,4), ncols=1)
        mappable0 = ax.imshow(rd_clustered_grid, cmap='viridis',
                                 #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                 #interpolation="bicubic",
                                 aspect='auto'
                                 ) #,
                                 #origin='lower')
        fig.colorbar(mappable0, ax=ax)
        fig.tight_layout()    
        plt.show()
        
    return rd_clustered_grid

from collections import deque

def find_starting_point(grid):
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 1:
                return i, j
    return None

def is_valid_move(x, y, grid):
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1

def is_boundary(x, y, grid):
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if not is_valid_move(nx, ny, grid):
            return True
    return False

def bfs_boundary_detection_without_intersection(start_x, start_y, grid):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    boundary = []
    queue = deque([(start_x, start_y)])

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while queue:
        x, y = queue.popleft()
        if visited[x][y]:
            continue

        visited[x][y] = True
        if is_boundary(x, y, grid):
            boundary.append((x, y))

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny, grid) and not visited[nx][ny]:
                queue.append((nx, ny))

    return boundary

# Identify the non-dominated solutions
def find_pareto_frontier(objective_values):
    pareto_frontier1 = []
    for i, values1 in enumerate(objective_values):
        dominated = False
        for j, values2 in enumerate(objective_values):
            if all(values1 <= values2) and any(values1 < values2):
                dominated = True
                break
        if not dominated:
            pareto_frontier1.append(values1)
        
    np_pareto_frontier1 = np.array(pareto_frontier1)
    np_pareto_frontier1 = np_pareto_frontier1[np_pareto_frontier1[:, 0].argsort()]
    # paretor_frontier1 = np_pareto_frontier1.tolist()
            
    flip_objective_values = np.copy(objective_values)
    flip_objective_values[:,0] = -flip_objective_values[:,0] 
    # print(flip_objective_values)
    pareto_frontier2 = []
    for i, values1 in enumerate(flip_objective_values):
        dominated = False
        for j, values2 in enumerate(flip_objective_values):
            if all(values2 <= values1) and any(values2 < values1):
                dominated = True
                break
        if not dominated:
            # print(values1)
            cpvalues1 = np.copy(values1)
            cpvalues1[0] = -cpvalues1[0]
            pareto_frontier2.append(cpvalues1)
    np_pareto_frontier2 = np.array(pareto_frontier2)
    np_pareto_frontier2 = np_pareto_frontier2[np.flip(np_pareto_frontier2[:, 0].argsort())]
    # paretor_frontier2 = np_pareto_frontier2.tolist()

    pareto_frontier3 = []
    for i, values1 in enumerate(objective_values):
        dominated = False
        for j, values2 in enumerate(objective_values):
            if all(values2 <= values1) and any(values2 < values1):
                dominated = True
                break
        if not dominated:
            pareto_frontier3.append(values1)
    np_pareto_frontier3 = np.array(pareto_frontier3)
    np_pareto_frontier3 = np_pareto_frontier3[np.flip(np_pareto_frontier3[:, 0].argsort())]
    # paretor_frontier3 = np_pareto_frontier3.tolist()

    flip_objective_values = np.copy(objective_values)
    flip_objective_values[:,0] = -flip_objective_values[:,0] 
    pareto_frontier4 = []
    for i, values1 in enumerate(flip_objective_values):
        dominated = False
        for j, values2 in enumerate(flip_objective_values):
            if all(values1 <= values2) and any(values1 < values2):
                dominated = True
                break
        if not dominated:
            cpvalues1 = np.copy(values1)
            cpvalues1[0] = -cpvalues1[0]
            pareto_frontier4.append(cpvalues1)
    np_pareto_frontier4 = np.array(pareto_frontier4)
    np_pareto_frontier4 = np_pareto_frontier4[np_pareto_frontier4[:, 0].argsort()]
    # paretor_frontier4 = np_pareto_frontier4.tolist()

    org_pareto_frontier = np.concatenate([np_pareto_frontier1, np_pareto_frontier2, np_pareto_frontier3, np_pareto_frontier4], axis=0)
    index_hash = 100*org_pareto_frontier[:,0] + org_pareto_frontier[:,1]
    unique_index = np.unique(index_hash, return_index=True)[1]
    pareto_frontier = [org_pareto_frontier[index] for index in sorted(unique_index)]
    # print(pareto_frontier)
    new_pareto_frontier = [pt for pt in pareto_frontier]
    put_values = []
    put_indices = []
    for i in range(len(pareto_frontier)):
        dir_x = pareto_frontier[i%len(pareto_frontier)][0] - pareto_frontier[(i+1)%len(pareto_frontier)][0]
        dir_y = pareto_frontier[i%len(pareto_frontier)][1] - pareto_frontier[(i+1)%len(pareto_frontier)][1]
        if dir_y == 0:
            if dir_x > 1:
                for k in range(1, int(dir_x)):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0] - k, pareto_frontier[i%len(pareto_frontier)][1]])
                    put_indices.append(i+1)
            elif dir_x < -1:
                # print("hey")
                for k in range(1, -int(dir_x), 1):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0] + k, pareto_frontier[i%len(pareto_frontier)][1]])
                    put_indices.append(i+1)
        elif dir_x == 0:
            if dir_y > 1:
                for k in range(1, int(dir_y)):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0], pareto_frontier[i%len(pareto_frontier)][1] - k])
                    put_indices.append(i+1)
            elif dir_y < -1:
                for k in range(1, -int(dir_y), 1):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0], pareto_frontier[i%len(pareto_frontier)][1] + k])
                    put_indices.append(i+1)
    
    if len(put_indices) > 0:
        final_pareto = np.insert(np.array(pareto_frontier), put_indices, put_values, axis=0)
        return final_pareto
    
    return np.array(pareto_frontier)

def find_cluster_boundary(np_clustered_grid, is_boundary_plot=False):
    num_cluster = int(np_clustered_grid.max())
    boundaries = []
    for i in range(num_cluster):
        cp_clustered_grid = np_clustered_grid.copy()
        single_clustered_grid = np.where(cp_clustered_grid==i+1, 1, 0).tolist()
        # print(clustered_grid)
        # plt.scatter(np.array(clustered_grid)[:,1], np.array(clustered_grid)[:,0])
        # plt.show()
        starting_point = find_starting_point(single_clustered_grid)
        # print(starting_point)
        # visited = [[False] * len(grid[0]) for _ in range(len(grid))]
        boundary = bfs_boundary_detection_without_intersection(starting_point[0], starting_point[1], single_clustered_grid)
        # print(boundary)

        np_boundary = np.array(boundary)
        if is_boundary_plot:
            plt.scatter(np_boundary[:,1], np_boundary[:,0])
            plt.show()

        objective_values = np_boundary
        pareto_frontier = find_pareto_frontier(objective_values)
        pareto_frontier = np.fliplr(pareto_frontier)[::-1]
        boundaries.append(pareto_frontier)

        # Plot the solutions and the Pareto frontier
        if is_boundary_plot:
            plt.scatter(objective_values[:, 1], objective_values[:, 0], label='Solid Points', marker='o')
            plt.scatter(pareto_frontier[:, 0], pareto_frontier[:, 1], label='Solid Boundary', marker='x', color='red')
            plt.plot(pareto_frontier[:, 0], pareto_frontier[:, 1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()
            print(pareto_frontier)
        np.unique(pareto_frontier, axis=0, return_counts=True)
        
    return boundaries


def compute_binary_pressForce(pressure, np_clustered_grid, is_boundary_plot=False):
    r"""
    Compute force on binary boundary mask
    Args:
        pressure: 2d tensor, each element represents pressure on a cell.
        np_clustered_grid: Numpy mask whose cells represent multiple solids. The shape is same as pressure.
        Values of cells are integers and cells with a same value belong to a same solid. 
        This can be either bounary mask and derived from find_filtered_clusteres function.         
    """
    # Compute the boundary of all solids in binary_boundary_mask
    # pdb.set_trace()
    final_boundaries = find_cluster_boundary(np_clustered_grid, is_boundary_plot=is_boundary_plot)
    # Compute force of boundary of each multiple solid
    pressures = {}
    for i in range(len(final_boundaries)):
        boundary = final_boundaries[i]
        # Compute force
        pressures[i+1] = compute_pressForce(pressure, torch.tensor(boundary.copy(), device=pressure.device)+0.5)
    return pressures

def reconstruct_boundary(binary_mask, bd_offset):
    r"""
    Restore boundary points from binary mask and boundary offset
    Args:
        binary_mask: 2d torch tensor, represented by binary values. Expected shape is [62, 62].
        bd_offset: 3d torch tensor, grid structure each of whose cells is represented by 2d vector.
        Expected shape is [62, 62, 2]
    Limitation:
        Multiple boundaries are not out of the scope at this moment.
        The number of points of the ouput is not consistent with the shape of original boundary (= 40)
    """    
    # Find boundary of object in boundary mask
    np_binary_mask = binary_mask.cpu().numpy()
    boundary_bd = find_cluster_boundary(np_binary_mask)[0]

    # Retrieve offset vectors on boundary_bd and boundary value
    np_offset = bd_offset.cpu().numpy()
    new_np_offset = np_offset[boundary_bd[:,1], boundary_bd[:,0], :]
    restored_boundary = (boundary_bd + 0.5) + new_np_offset

    return restored_boundary


m = 64; n = 64
maxnum = 100

def discretize_boundary(boundary):
    # import pdb
    # pdb.set_trace()
    assert boundary.shape[1] == 2
    num_bound = boundary.shape[0]
    device = boundary.device
    p_5 = torch.tensor([0.5], device=device).repeat(num_bound)
    x = torch.minimum(torch.maximum(boundary[:, 0], p_5), torch.tensor([n-1.5], device=device).repeat(num_bound))
    x_inds = torch.minimum(x.type(torch.int32), torch.tensor([n-2], device=device).repeat(num_bound))
    # fs = x - x_inds

    y = torch.minimum(torch.maximum(boundary[:, 1], p_5), torch.tensor([m-1.5], device=device).repeat(num_bound))
    y_inds = torch.minimum(y.type(torch.int32), torch.tensor([m-2], device=device).repeat(num_bound))
    # ft = y - y_inds
    return x_inds, y_inds

def find_orthogonal_line(A, B, C, x0, y0):
    m1 = torch.empty((C.shape[0],), device=C.device)
    m1[B==0] = float('inf')
    m1[B!=0] = (-A/B)[B!=0]

    m2 = torch.empty((C.shape[0],), device=C.device)
    m2[m1==float('inf')] = 0
    m2[m1!=float('inf')] = (-1 / m1)[m1!=float('inf')]

    b2 = y0 - m2 * x0  # The y-intercept of L2.

    # Return the coefficients A, B, C of the line L2 (Ax + By - C = 0)
    return m2, -1, b2

def edge_cells(polygon):
    num_vertices = len(polygon)
    edges = []
    for i in range(num_vertices):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % num_vertices]
        edge = sorted([v1, v2], key=lambda x: x[1])
        edges.append(edge)
    return edges

def find_cells_inside_curve(polygon, grid_shape):
    def horizontal_intersection(x1, y1, x2, y2, y):
        return x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    edges = edge_cells(polygon)
    grid = np.zeros(grid_shape, dtype=np.uint8)
    height, width = grid.shape
    
    for y in range(height):
        intersections = []
        for edge in edges:
            y1, y2 = edge[0][1], edge[1][1]
            if y1 < y <= y2:
                x = horizontal_intersection(*edge[0], *edge[1], y)
                intersections.append(x)
        intersections.sort()
        for i in range(0, len(intersections), 2):
            x_start, x_end = int(np.ceil(intersections[i])), int(np.floor(intersections[i + 1]))
            grid[y, x_start : x_end + 1] = 1

    return grid

def update_static_masks(torch_con_boundary):
    # import pdb
    # pdb.set_trace()
    x_inds, y_inds = discretize_boundary(torch_con_boundary)
    pointy_hash = maxnum*x_inds[20] + y_inds[20]

    indices = torch.stack((maxnum*x_inds,y_inds), 0)
    sum_indices = indices.sum(0)
    ind_unique = torch.unique(sum_indices, sorted=True) #, return_inverse=True)
    x_idx = (torch.cat([(sum_indices==ind_u).nonzero()[0] for ind_u in ind_unique])).sort()[0]
    # print(sum_indices[x_idx])
    # print(x_idx)
    repeat_sum_indices = torch.tile(sum_indices, (ind_unique.shape[0],1))
    repeat_ind_unique = torch.tile(sum_indices[x_idx].reshape(ind_unique.shape[0], 1), (1, sum_indices.shape[0]))
    org_mask = (repeat_ind_unique == repeat_sum_indices)
    fatted_mask = torch.roll(org_mask, 1, 1) + torch.roll(org_mask, -1, 1)

    relvecs = []
    base_pts = []
    base_nums = []
    for bdpt in range(sum_indices[x_idx].shape[0]):
        # i = 1
        if pointy_hash == sum_indices[x_idx][bdpt]:
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            base_pts.append(base_pt)
            relvec = torch_con_boundary[20] - base_pt
            relvecs.append(relvec)
            # base_nums.append(i)
        elif torch.sum(org_mask[bdpt]) >= 4:
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            base_pts.append(base_pt)
            relvec = torch_con_boundary[org_mask[bdpt]] - base_pt.repeat(torch_con_boundary[org_mask[bdpt]].shape[0], 1)
            ind = torch.argmin(torch.norm(relvec, dim=1))
            relvecs.append(relvec[ind])
            # base_nums.append(i)
        elif torch.sum(fatted_mask[bdpt] * torch.logical_not(org_mask[bdpt])) > 2:
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            # base_pts.append(base_pt)
            relvec = torch_con_boundary[org_mask[bdpt]] - base_pt.repeat(torch_con_boundary[org_mask[bdpt]].shape[0], 1)
            # print("disjoint: ", relvec)
            if len(relvec.shape) == 2:
                import pdb
                # pdb.set_trace()
                # for i in range(xbound[org_mask[bdpt]].shape[0]):
                    # if 2 * i + 2 == bd_points.shape[0]:
                # print("mask:", org_mask[bdpt])
                # print(xbound[org_mask[bdpt]])
                # plt.plot(xbound[org_mask[bdpt]][:, 0].numpy(), xbound[org_mask[bdpt]][:, 1].numpy())
                #     # else:
                #     #    plt.plot(bd_points[2*i:2*i+2, 0].numpy(), bd_points[2*i:2*i+2, 1].numpy())
                # plt.scatter(base_pt[0].numpy(), base_pt[1].numpy())
                # plt.show()
                # for i in range(relvec.shape[0]):
                #     row_relvec = relvec[i]
                #     relvecs.append(row_relvec)
                #     base_pts.append(base_pt)
                #     i += 1
                relvecs.append(relvec[-1])
                base_pts.append(base_pt)
                # base_nums.append(i)
            else:
                relvecs.append(relvec)
                base_pts.append(base_pt)
                # base_nums.append(i)
        elif torch.sum(org_mask[bdpt]) == 1:
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            base_pts.append(base_pt)
            relvec = torch_con_boundary[org_mask[bdpt]] - base_pt
            # print("rel: ", relvec[0])
            relvecs.append(relvec[0])
            # base_nums.append(i)
        else:    
            if fatted_mask[bdpt][0] and fatted_mask[bdpt][-1]:
                rollnum = 1
                for _ in range(0, 100):
                    temprole = torch.roll(fatted_mask[bdpt], rollnum, 0)
                    if temprole[0] and temprole[-1]:
                        rollnum += 1    
                    else:
                        break
                #import pdb
                #pdb.set_trace()
                x_pts = torch.roll(torch_con_boundary[fatted_mask[bdpt]], rollnum, 0)            
            else:
                x_pts = torch_con_boundary[fatted_mask[bdpt]]

            bd_points = torch.cat([x_pts[0:1], x_pts[1:-1].repeat(1, 2).reshape(-1,2), x_pts[-1:]], dim=0)
            dire_vec = bd_points[0::2] - bd_points[1::2]
            const = bd_points[0::2, 1] - bd_points[0::2, 0] * dire_vec[:,1]/dire_vec[:,0]

            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            base_pts.append(base_pt)
            # base_nums.append(i)
            base_points = base_pt.repeat(const.shape[0], 1)
            slope = dire_vec[:,1]/dire_vec[:,0]
            # import pdb
            # pdb.set_trace()
            ax, by, con = find_orthogonal_line(slope, -torch.ones((const.shape[0],), device=torch_con_boundary.device), const, base_points[:,0], base_points[:,1])

            # for i in range(int(bd_points.shape[0]/2)):
            #     if 2 * i + 2 == bd_points.shape[0]:
            #         plt.plot(bd_points[2*i:, 0].numpy(), bd_points[2*i:, 1].numpy())
            #     else:
            #         plt.plot(bd_points[2*i:2*i+2, 0].numpy(), bd_points[2*i:2*i+2, 1].numpy())
            # plt.scatter(base_pt[0].numpy(), base_pt[1].numpy())
            # plt.show()

            al = -ax/by
            bl = con
            cl = dire_vec[:,1]/dire_vec[:,0]
            dl = const

            # print(org_mask[bdpt,:])
            intersection = torch.stack([(dl - bl)/(al - cl), (al*dl - bl*cl)/(al - cl)]).t()
            # print(intersection)
            # print(base_pt)

            relvec = intersection - torch.tile(base_pt, (intersection.shape[0], 1))
            #print(relvec.sum(0)/relvec.shape[0])
            relvecs.append(relvec.sum(0)/relvec.shape[0])

    ### Check number of offset vectors is same as that of boundary cells of solid
    # print(len(base_pts), sum_indices[x_idx].shape[0])
    assert len(base_pts) == sum_indices[x_idx].shape[0]
    
    # import pdb
    # pdb.set_trace()    
    
    bd_offset = torch.stack(relvecs)
    offset_grid_bound = torch.zeros((62, 62, 2), device=torch_con_boundary.device)
    offset_grid_bound[x_inds, y_inds] = torch.tensor([1, 1], dtype=torch.float32, device=torch_con_boundary.device)
    offset_grid_bound = offset_grid_bound.transpose(1,0)

    # offset_grid = find_cells_inside_curve(torch.stack((x_inds, y_inds), -1).tolist(), grid_bound.shape)
    offset_grid = find_cells_inside_curve(torch.stack((x_inds, y_inds), -1).detach().cpu().tolist(), (62, 62))
    # fig, ax = plt.subplots(figsize=(4,4), ncols=1)
    # ax.imshow(offset_grid, cmap='viridis',
    #          aspect='auto',
    #          origin='lower')
    # plt.show()
    inner_solid_mask = np.copy(offset_grid)
    offset_grid = offset_grid.reshape(62, 62, 1)
    offset_grid = np.concatenate([offset_grid, offset_grid], -1)

    offset_union = offset_grid_bound + torch.tensor(offset_grid, device=torch_con_boundary.device)
    offset_union[(offset_union.sum(-1) > 2),:] = torch.tensor([1, 1], dtype=torch.float32, device=torch_con_boundary.device)
    offset_union.index_put_((y_inds[x_idx], x_inds[x_idx]), bd_offset)    

    # np_offset_union = offset_union.detach().cpu().numpy()

    # fig, ax = plt.subplots(figsize=(4,4), ncols=1)
    # ax.imshow(np_offset_union[...,1], cmap='viridis',
    #          aspect='auto',
    #          origin='lower')
    # plt.show()

    # gtmask = data.node_feature["n0"][:,0,2].reshape(62,62).detach().cpu().numpy()
    # fig, ax = plt.subplots(figsize=(4,4), ncols=1)
    # ax.imshow(gtmask, cmap='viridis',
    #          aspect='auto',
    #          origin='lower')
    # plt.show()

    # print((data.node_feature["n0"][:,-1,1].reshape(62,62).detach().cpu().numpy() - np_offset_union[...,0]).sum())
    # print((data.node_feature["n0"][:,-1,2].reshape(62,62).detach().cpu().numpy() - np_offset_union[...,1]).sum())
    
    # updated_offset_mask = np_offset_union

    grid_bound = torch.zeros((62, 62), device=torch_con_boundary.device)
    grid_bound[x_inds, y_inds] = 1
    # union = grid_bound.transpose(1,0).detach().cpu().numpy() + inner_solid_mask
    union = grid_bound.transpose(1,0) + torch.tensor(inner_solid_mask, device=torch_con_boundary.device)
    union[union == 2] = 1

    # fig, ax = plt.subplots(figsize=(8,4), ncols=2)
    # ax[0].imshow(union.flatten().reshape(62,62), cmap='viridis',
    #          aspect='auto',
    #          origin='lower')
    # ax[1].imshow(data.node_feature["n0"][:,-1,0].reshape(62,62).detach().cpu().numpy(), cmap='viridis',
    #          aspect='auto',
    #          origin='lower')
    # plt.show()

    # print((data.node_feature["n0"][:,-1,0].reshape(62,62).detach().cpu().numpy() - union).sum())
    updated_solid_mask = union
    
    return updated_solid_mask, offset_union #updated_offset_mask

def update_bdfeature(reconstructed_boundary):
    upd_solid_mask, upd_solid_offset = update_static_masks(reconstructed_boundary)
    torch_batch_mask = torch.where(upd_solid_mask==1, False, True).clone().flatten()
    upd_solid_mask = upd_solid_mask[...,None]
    static_feature = torch.cat((upd_solid_mask, upd_solid_offset), -1)
    multi_static_feat = torch.stack([static_feature for _ in range(4)], -2).reshape(-1,4,3)
    return multi_static_feat, torch_batch_mask

def update_data(reconstructed_bound, optimdata, original_data, constant_var, opt_var):
    mul_static_feat, tor_batch_mask = update_bdfeature(reconstructed_bound) 

    a = deepsnap_Batch
    batch, _ = a._init_batch_fields(optimdata.keys, [])
    batch.batch = optimdata.batch.clone()
    batch.compute_func = optimdata.compute_func
    batch.directed = optimdata.directed.detach().clone()
    batch.dyn_dims = optimdata.dyn_dims
    batch.edge_attr = optimdata.edge_attr
    batch.edge_index = {('n0','0','n0'): optimdata.edge_index[('n0','0','n0')].detach().clone()}
    batch.edge_label_index = {('n0','0','n0'): optimdata.edge_label_index[('n0','0','n0')].detach().clone()}
    batch.grid_keys = optimdata.grid_keys
    batch.mask = {"n0": tor_batch_mask.detach()}
    batch.node_feature = {"n0": torch.cat((mul_static_feat, original_data.node_feature["n0"][...,3:].detach()), -1)}
    batch.node_label = {"n0": optimdata.node_label["n0"].detach().clone()}
    batch.node_label_index = {"n0": optimdata.node_label_index["n0"].detach().clone()}
    batch.node_pos = {"n0": optimdata.node_pos["n0"].detach().clone()}
    batch.original_shape = optimdata.original_shape
    batch.param = {"n0": torch.cat((constant_var, opt_var), 0).transpose(1,0).flatten()[None,:]}
    batch.params = optimdata.params
    batch.part_keys = optimdata.part_keys
    batch.task = optimdata.task    
    optimdata = batch    
    
    return optimdata


import random
class CustomSampler(Sampler):
    def __init__(self, data,batch_size,noncollision_hold_probability,distance_threshold):
        self.data = data
        self.batch_size=batch_size
        self.noncollision_hold_probability=noncollision_hold_probability
        self.distance_threshold=distance_threshold

    def __iter__(self):
        indices = []
        p_list=[]
        j_list=[]
        dis_list=[]
        Flag=0
        num1=0
        num2=0
        distance=0.
        distance2=0.
        seed = 42
        random.seed(seed)
        # pdb.set_trace()
        #preprocess data to get sampleing index for more collisions
        # for i in range(self.data.len()):
        #     Flag=0
        #     data=self.data[i]
        #     data=torch.cat([data.x,data.y],dim=1)
        #     for j in range(data.shape[1]):
        #         distance=torch.sqrt((data[0,j,0]-data[1,j,0])**2+(data[0,j,1]-data[1,j,1])**2)
        #         distance_body1_wall_min=min([data[0,j,0],200-data[0,j,0],data[0,j,1],200-data[0,j,1]])
        #         distance_body2_wall_min=min([data[1,j,0],200-data[1,j,0],data[1,j,1],200-data[1,j,1]])
        #         if distance<=self.distance_threshold: # the collision between body1 and body2
        #             Flag=1
        #             num1=num1+1
        #             break
        #         if (distance_body1_wall_min<self.distance_threshold/2.0) or (distance_body2_wall_min<self.distance_threshold/2.0): # the collision between body and wall
        #             Flag=1
        #             num2=num2+1
        #             break
        #     p=random.uniform(0, 1)
        #     p_list.append(p)
        #     if p<self.noncollision_hold_probability:
        #         Flag=1
        #     if Flag==1:
        #         indices.append(i)
        # indices_np=np.load("/user/project/inverse_design/dataset/nbody_dataset/nbody-2/customerSampler_indices.npy")
        
        indices_np=torch.arange(0, 300000, 1, dtype=torch.int)
        indices=indices_np
        random.shuffle(indices)
        indices = torch.tensor(indices)
        self.indices=indices
        # pdb.set_trace()
        # indices_np=np.array(indices)# around 1/3 datas can be selected of raw dataset ,so in order to have the same size dataset finally，I just make the larger raw dataset by doing 6000 simulations(3 times of 2000 before)
        # np.save("/user/project/inverse_design/dataset/nbody_dataset/nbody-4/speed-100/customerSampler_indices.npy",indices_np)
        return iter(indices)
    def __len__(self):
        return len(self.indices)
import torch.nn as nn
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target):
        # pdb.set_trace()
        loss = torch.abs(predicted - target)
        predicted_reshape=predicted.reshape(predicted.shape[0],predicted.shape[1],int(predicted.shape[2]/4),4)
        target_reshape=target.reshape(target.shape[0],target.shape[1],int(target.shape[2]/4),4)
        loss2=torch.abs((predicted_reshape[:,:,:,2])**2+(predicted_reshape[:,:,:,3])**2-((target_reshape[:,:,:,2])**2+(target_reshape[:,:,:,3])**2)).reshape(loss.shape[0],loss.shape[1],2,1)
        # loss3=torch.abs(torch.atan(predicted_reshape[:,:,:,2]/(predicted_reshape[:,:,:,3]))-torch.atan(target_reshape[:,:,:,2]/target_reshape[:,:,:,3])).reshape(loss.shape[0],loss.shape[1],2,1)/(3.1415926*2)
        return torch.cat([loss.reshape(loss.shape[0],loss.shape[1],2,int(loss.shape[2]/2)),loss2],dim=3)
import matplotlib
def visulization(filename,cond,pred,n_bodies,conditioned_steps,rollout_steps,num_features):
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    fontsize = 16
    for i in range(1):
        i=i*1
        fig = plt.figure(figsize=(18,15))
        if conditioned_steps!=0:
            cond_reshape = cond.reshape(cond.shape[0], conditioned_steps, n_bodies,num_features).to('cpu')
        pred_reshape = pred.reshape(cond.shape[0], rollout_steps, n_bodies,num_features).to('cpu')
        # y_gt_reshape = y_gt.reshape(cond.shape[0], rollout_steps, n_bodies,num_features)
        for j in range(n_bodies):
            # cond:
            if conditioned_steps!=0:
                marker_size_cond = np.linspace(1, 2, conditioned_steps) * 100
                plt.plot(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                plt.scatter(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], marker="+", linestyle="--", s=marker_size_cond)
            # # y_gt:
            # marker_size_y_gt = np.linspace(2, 3, rollout_steps) * 100
            # plt.plot(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-.")
            # plt.scatter(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], marker=".", linestyle="-.", s=marker_size_y_gt)
            # pred:
            marker_size_pred = np.linspace(2, 3, rollout_steps) * 100
            plt.plot(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
            plt.scatter(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], marker="v", linestyle="-", s=marker_size_pred)
            plt.xlim([0,1])
            plt.ylim([0,1])
        # loss_item = (pred[i] - y_gt[i]).abs().mean().item()
        plt.title(f"reverse", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        pdf.savefig(fig)
        i=i/1
    pdf.close()

import argparse
import numpy as np
import pprint as pp
import pymunk
import pygame
import random
from tqdm.auto import tqdm
# from utils import make_dir,get_item_1d
import imageio
import pdb
from torch_geometric.data.dataloader import DataLoader

def add_body(space,mass,radius,x,y,vx,vy):
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    body.position = x, y

    # Assign a random initial velocity
    body.velocity = vx, vy

    shape = pymunk.Circle(body, radius)
    shape.elasticity = 1.0
    shape.friction = 0.0
    space.add(body, shape)
    return body

def add_walls(space,width, height):
    walls = [
        pymunk.Segment(space.static_body, (0, 0), (0, height), 1),
        pymunk.Segment(space.static_body, (0, height), (width, height), 1),
        pymunk.Segment(space.static_body, (width, height), (width, 0), 1),
        pymunk.Segment(space.static_body, (width, 0), (0, 0), 1)
    ]
    for wall in walls:
        wall.elasticity = 1.0
        wall.friction = 0.0
        space.add(wall)

def draw_bodies(screen, bodies,radius):
    for body in bodies:
        x, y = body.position
        color = pygame.Color(body.color)
        pygame.draw.circle(screen, color, (int(x), int(y)), radius)

def run_simulation(space, bodies, n_steps, clock,n_simulation ,screen=None,filename=None,n_bodies=2,radius=20):
    if filename!=None:
        videofile=filename+f"/trajectory_balls_{n_bodies}_simu_{n_simulation+1}_steps_{n_steps}.gif"
    position_data = np.zeros((n_steps, len(bodies), 4))
    frames=[]
    for i in range(n_steps):
        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

        for j, body in enumerate(bodies):
            position_data[i, j, :2] = body.position
            position_data[i, j, 2:] = body.velocity

        if screen is not None:
            screen.fill((255, 255, 255))
            draw_bodies(screen, bodies,radius=radius)  # Draw the bodies with colors

        space.step(1/60.0)
        if screen is not None:
            pygame.display.flip()
            screen_data = pygame.surfarray.array3d(screen)
            frames.append(screen_data)
        clock.tick(60)
    # if screen is not None and filename!=None:
    #     imageio.mimsave(videofile, frames)
    return position_data

##this is the groundtruth forward solver
def simulation(features,n_steps,filename=None,width=200,height=200,radius=20,mass=1):
    '''
    features :torch.tensor [batch_size,n_bodies,num_features],features including x,y,vx,vy
    n_steps: int ,simulation steps
    filename: the filename for saving results visulization
    '''
    pygame.init()
    if filename!=None:
        screen = pygame.display.set_mode((width, height))
    else:
        screen = None
    frames=[]
    clock = pygame.time.Clock()

    all_position_data = []
    batch_size=features.shape[0]
    n_bodies=features.shape[1]
    num_features=features.shape[2]
    for sim in tqdm(range(batch_size)):
        if sim % 50 == 0 and sim > 0:
            print(f"simu: {sim}")
        space = pymunk.Space()
        space.gravity = (0, 0)

        add_walls(space,width=width,height=height)  # Add the walls

        #add bodies
        bodies=[]
        # pdb.set_trace()
        for i in range(n_bodies):
            body=add_body(space=space,mass=mass,radius=radius,x=features[sim,i,0],y=features[sim,i,1],vx=features[sim,i,2],vy=features[sim,i,3])
            bodies.append(body)

        # Assign a random color to each body
        for body in bodies:
            body.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        position_data = run_simulation(space, bodies, n_steps, clock,sim,screen=screen,filename=filename,n_bodies=n_bodies,radius=radius)
        if position_data is not None:
            all_position_data.append(position_data)
        else:
            break
        # Save the position data to an .npy file
        all_position_data_save = np.array(all_position_data)
    if filename!=None:
        filename2=filename+f"/trajectory_balls_{n_bodies}_simu_{batch_size}_steps_{n_steps}.npy"
        whether_saved=np.save(filename2, all_position_data_save) ##[n_simulations, n_steps, n_bodies, 4 ] 4 inculde (x,y,vx,vy)：
        if whether_saved:    
            print("Successfully saved result in "+filename)
        else:
            print("Failed to save data")

    pygame.quit()
        
    return torch.tensor(all_position_data)

def eval_simu(cond_design, design_fn, n_bodies, rollout_steps, time_interval=4):
    '''
    input:
        cond_design:[batch_size,conditioned_steps,n_bodies*num_features]
        design_fn :function
        n_bodies : int
        rollout_steps: int
    output:
        pred_simu: tensor [batch_size,rollout_steps,n_bodies*num_features]
        design_obj_simu: tensor [1]
    '''
    assert cond_design.shape[-1]//4 == n_bodies
    cond_simu = cond_design[:,-1,:]*200.
    cond_simu = cond_simu.reshape(cond_simu.shape[0], n_bodies, -1)
    # pdb.set_trace()
    pred_simu = simulation(features=cond_simu, n_steps=rollout_steps*time_interval)
    pred_simu = pred_simu.reshape(pred_simu.shape[0], pred_simu.shape[1], -1)
    pred_simu = pred_simu[:, time_interval-1::time_interval]
    pred_simu = pred_simu.to(cond_design.device) / 200.
    design_obj_simu = design_fn(pred_simu)

    return pred_simu, design_obj_simu

#tool function for GNS
def time_diff(input_seq):
    return input_seq[:, 1:] - input_seq[:, :-1]

def get_random_walk_noise(pos_seq,noise_std):
    if pos_seq.shape[1]==1:
        noise_shape = (pos_seq.shape[0], pos_seq.shape[1], pos_seq.shape[2])
    else:
        noise_shape = (pos_seq.shape[0], pos_seq.shape[1]-1, pos_seq.shape[2])
    n_step_vel = noise_shape[1]
    acc_noise = np.random.normal(0, noise_std / n_step_vel ** 0.5, size=noise_shape).astype(np.float32)
    vel_noise = np.cumsum(acc_noise, axis=1)
    pos_noise = np.cumsum(vel_noise, axis=1)
    if pos_seq.shape[1]!=1:
        pos_noise = np.concatenate([np.zeros_like(pos_noise[:, :1]),
                                pos_noise], axis=1)

    return pos_noise


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


def pdump(file, filename):
    """Dump a file via pickle."""
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def pload(filename):
    """Load a filename saved as pickle."""
    with open(filename, "rb") as f:
        file = pickle.load(f)
    return file
import scipy.stats as stats
def caculate_confidence_interval(data):
    ''''
    input example: abs(pred_design-pred_simu)
    '''
    list_dim=range(data.dim())
    if data.dim()>1:
        MAE_batch_size = torch.mean(data, dim=tuple(list_dim[1:]))
    else:
        MAE_batch_size =data
    mean = torch.mean(MAE_batch_size)
    
    std_dev = torch.std(MAE_batch_size)
    min_value=min(MAE_batch_size)
    confidence_level = 0.95
    # pdb.set_trace()
    n = len(data)
    # kk = stats.t.ppf((1 + confidence_level) / 2, n - 1) * (std_dev / (n ** 0.5))
    margin_of_error= std_dev* 1.96/ torch.sqrt(torch.tensor(n,dtype=float))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    print("mean:", mean.item())
    print("std:", std_dev.item())
    print(f"margin_of_error:", margin_of_error)

    return mean,std_dev,margin_of_error,min_value
def caculate_num_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("???????????????????????????Number of parameter: %.2fM" % (total/1e6))
    # pdb.set_trace()
    return
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    