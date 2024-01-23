import re
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_boundary(x, y, save_path):
    plt.xlim([0, 62])
    plt.ylim([0, 62])
    for i in range(len(x)):
        plt.plot(x[i] + [x[i][0]], y[i] + [y[i][0]])
    plt.savefig(save_path)
    plt.close()
    # plt.show()

def load_one_boundary(f):
    data = f.read()
    data_into_list = re.split("\n|[|[ |, | ]|]", data)
    a = []
    for w in data_into_list:
        if w:
            a.append(float(w))

    b = []
    for v in a:
        if v != 0.:
            b.append(v)

    npb = np.array(b)
    
    return npb[0::2].tolist(), npb[1::2].tolist()

def plot(root_dir):
    if not os.path.exists(os.path.join(root_dir, "plots")):
        os.mkdir(os.path.join(root_dir, "plots"))
    all_res = []
    filelist = os.listdir(os.path.join(root_dir, "boundaries"))
    filelist.sort()
    for simnum, s_f in enumerate(filelist):   
        # print("simnum: ", simnum, filelist, s_f)
        if s_f.startswith("."):
            continue
        x, y = [], []
        for b_f in os.listdir(os.path.join(root_dir, "boundaries", s_f)):
            # f = open("./designed_boundaries/sim_{0}/boundary_{1}.txt".format(simnum, i))
            f = open(os.path.join(root_dir, "boundaries", s_f, b_f))
            x_b, y_b = load_one_boundary(f)
            x.append(x_b)
            y.append(y_b)
        
        save_path = os.path.join(root_dir, "plots/sim_{:06d}.png".format(simnum))
        plot_boundary(x, y, save_path)
            # np.save("./plots/sim_{:06d}/boundary_{:06d}".format(simnum, i), np.stack([npb[0::2], npb[1::2]]))
        
def metric(lift, drag, lam=1, use_frac=False):
    if use_frac: # maximize is better
        return abs(lift/ drag) 
    else: # minimize is better
        return -abs(lift) + lam * abs(drag)

def metric_batch(forces_array_batch, lam=1):
    # forces_array_batch: (batch_size, 100, num_boundaries, 3) # 100 is the time steps, 3 is x, y, z forces
    forces_array_batch = forces_array_batch[:,:,:,0:2] # ignore zero z force, output forces_array_batch size: (batch_size, 100, num_boundaries, 2)
    drag = np.sum(forces_array_batch[:,:,:,0], axis=2) # drag force sum over boundaries, output drag size: (batch_size, 100)
    lift = np.sum(forces_array_batch[:,:,:,1], axis=2) # lift force sum over boundaries, output lift size: (batch_size, 100)
    drag_mean = np.mean(drag, axis=1) # average over time steps, output drag size: (batch_size)
    lift_mean = np.mean(lift, axis=1) # average over time steps, output lift size: (batch_size)
    
    drag_min = np.min(abs(drag_mean))
    lift_max = np.max(abs(lift_mean))  
    obj = metric(lift, drag, lam, use_frac=False) 
    lift_over_drag = metric(lift, drag, lam, use_frac=True) 
    obj_mean = np.mean(obj, axis=1) # average over time steps, output obj_mean size: (batch_size)
    lift_over_drag_mean = np.mean(lift_over_drag, axis=1) # average over time steps, output lift_over_drag_mean size: (batch_size)
    obj_min = np.min(abs(obj_mean))
    lift_over_drag_max = np.max(abs(lift_over_drag_mean))
    print("drag, lift", drag_min, lift_max, obj_min, lift_over_drag_max)

    return drag_min, lift_max, obj_min, lift_over_drag_max

def metric_over_batches(batches_dict, lam=1):
    drag_list = []
    lift_list = []
    obj_list = []
    lift_over_drag_list = []
    for batch_id in batches_dict:
        # print("batch_id: ", batches_dict[batch_id].shape)
        drag_min, lift_max, obj_min, lift_over_drag_max = metric_batch(batches_dict[batch_id], lam=lam)
        drag_list.append(drag_min)
        lift_list.append(lift_max)
        obj_list.append(obj_min)
        lift_over_drag_list.append(lift_over_drag_max)  
    
    # compute mean and std over batches
    drag_mean = np.mean(drag_list)
    drag_std = np.std(drag_list)
    lift_mean = np.mean(lift_list)
    lift_std = np.std(lift_list)
    obj_mean = np.mean(obj_list)
    obj_std = np.std(obj_list)
    lift_over_drag_mean = np.mean(lift_over_drag_list)
    lift_over_drag_std = np.std(lift_over_drag_list)
    
    return drag_mean, drag_std, lift_mean, lift_std, obj_mean, obj_std, lift_over_drag_mean, lift_over_drag_std
        
        

# conver force txt file to npy array
def read_forces_from_txt(fname):
    f = open(fname)
    lines = f.readlines()
    forces_array = []
    for t in range(len(lines)):
        lines[t] = lines[t].replace("[", "").replace("]", "").replace("\n", "")
        data = lines[t].split(":")[1].split(",")
        force_bd1 = [float(x) for x in data[:3]]
        if len(data) < 6:
            forces_array.append([force_bd1])
        elif 6 <= len(data) < 9:
            force_bd2 = [float(y) for y in data[3:6]]
            forces_array.append([force_bd1, force_bd2])
        elif 9 <= len(data) < 12:
            force_bd3 = [float(z) for z in data[6:9]]
            forces_array.append([force_bd1, force_bd2, force_bd3])
        else:
            raise
    forces_array = np.array(forces_array) # (time_step, n_boundaries, 3)
    return forces_array

def evaluation(force_dir, output_array_folder, lam=1, use_frac=False):
    # save npy force for all boundaries in each simuation of 100 time steps
    if not os.path.exists(output_array_folder):
        os.mkdir(output_array_folder)
        
    all_res = []
    filelist = os.listdir(force_dir)
    filelist.sort()
    for sim, file in enumerate(filelist):
        fname= os.path.join(force_dir, file)
        if not fname.endswith(".txt"):
            continue
        forces_array = read_forces_from_txt(fname) # (time_step, n_boundaries, 3)
        res = mean_metric(forces_array, lam, use_frac)
        all_res.append(res)
        np.save(os.path.join(output_array_folder, "sim_{:06d}.npy".format(sim)), forces_array)
    # all_res.append(res)
    final_res = np.min(np.array(all_res)) # min over all simulations is the optimal design
    print("minimal objective: ", final_res)

def parse_index_map(index_map_file):
    index_map = open(index_map_file).readlines()
    index_map_dict = {}
    for line in index_map:
        line = line.replace("\n", "")
        line = line.split(", ")
        index_map_dict[line[1]] = line[0]
    return index_map_dict

def evaluation_batches(root_dir, lam=1):
    force_dir = root_dir + "/forces"
    index_map_dict = parse_index_map(os.path.join(root_dir, "index_map.txt"))
    force_files = os.listdir(force_dir)
    force_files.sort()
    batches_dict = {}
    for file in force_files:
        if file.startswith("."):
            continue
        origion_filename = index_map_dict[file[:-4]]
        batch_id = origion_filename.split("_")[1]
        fname = os.path.join(force_dir, file)
        forces_array = read_forces_from_txt(fname) # (time_step, n_boundaries, 3)
        print("forces_array: ", forces_array.shape)
        if batch_id not in batches_dict:
            batches_dict[batch_id] = [forces_array]
        else:
            batches_dict[batch_id].append(forces_array)
    for batch_id in batches_dict:
        batches_dict[batch_id] = np.array(batches_dict[batch_id])

    res = metric_over_batches(batches_dict, lam=lam)
    print("drag_mean: ", res[0])
    print("drag_std: ", res[1])
    print("lift_mean: ", res[2])
    print("lift_std: ", res[3])
    print("obj_mean: ", res[4])
    print("obj_std: ", res[5])
    print("lift_over_drag_mean: ", res[6])
    print("lift_over_drag_std: ", res[7])
    return res

if __name__ == "__main__":
    # root_dir = "double"
    root_dir = "single"
    use_plot = False # boundaries folder and boundaries files in it are needed if set True
    lam = 1

    # if use_plot:
    #     plot(root_dir)
    res = evaluation_batches(root_dir, lam=lam)