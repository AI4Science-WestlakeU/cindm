import os
import torch
import random
from random import sample
from torch.utils.data import Dataset
import cindm.utils as utils
import pickle
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from cindm.GNS_model.config import _C as C
import torch.nn.functional as F
import pdb
from torch_geometric.data.dataloader import DataLoader
def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())
class nbody_gns_dataset(Dataset):
    def __init__(self, data_dir,
                  phase='train',
                  rollout_steps=240,
                  n_his=4,
                  time_interval=4,
                  verbose=1,
                  input_steps=4,
                  output_steps=1,
                  n_bodies=2,
                  is_train=True,
                  device="cuda"
                  ):
        self.data_dir = data_dir
        self.phase = phase
        self.metadata = _read_metadata(self.data_dir)
        self.rollout_steps=rollout_steps
        self.n_his=n_his
        self.time_interval=time_interval
        self.time_stamps=980
        self.verbose=verbose
        self.input_steps=input_steps
        self.output_steps=output_steps
        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output) // self.time_interval
        self.data=None
        self.n_bodies=n_bodies
        self.is_train=is_train
        self.pred_steps=output_steps
        self.device=device
        self.total_n_simu=2000
        if is_train:
            self.n_simu=1800
        else:
            self.n_simu=200
        for key in self.metadata:
            self.metadata[key] = torch.from_numpy(np.array(self.metadata[key]).astype(np.float32))
        if self.n_bodies==4:
            self.data = torch.FloatTensor(np.load("/user/project/inverse_design/dataset/nbody_dataset/train/trajectory_balls_4_simu_2000_steps_1000.npy"))
        elif self.n_bodies==2:
            self.data = torch.FloatTensor(np.load("/user/project/inverse_design/dataset/nbody_dataset/train/trajectory_balls_2_simu_2000_steps_1000.npy"))
        elif self.n_bodies==3:
            self.data = torch.FloatTensor(np.load("/user/project/inverse_design/dataset2/nbody_dataset/nbody-3/speed-100/trajectory_balls_3_simu_1000_steps_1000.npy"))

    def __len__(self):
        return 200*self.n_simu

    def __getitem__(self, idx):
        
        
        sim_id, time_id = divmod(idx, self.time_stamps_effective)
        if not self.is_train:
            sim_id += self.total_n_simu - self.n_simu
        if self.verbose >= 1:
            print("sim_id, time_id:", (sim_id, time_id))
            print("start:", time_id * self.time_interval + self.t_cushion_input - self.input_steps * self.time_interval)
            print("mid:", time_id * self.time_interval + self.t_cushion_input)
            print("end:", time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval)
        # self.data shape: [n_simu, n_steps:1000, n_bodies, n_features:2]
        # x: [input_steps, n_bodies, n_features

        x = self.data[sim_id, time_id * self.time_interval + self.t_cushion_input - self.input_steps * self.time_interval: time_id * self.time_interval + self.t_cushion_input: self.time_interval].transpose(1,0)
        y = self.data[sim_id, time_id * self.time_interval + self.t_cushion_input: time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval: self.time_interval].transpose(1,0)
        


        particle_type=torch.ones(self.n_bodies).long()
        poss = x[:,:,:2]
        tgt_poss =y[:,:,:2]

        nonk_mask = get_non_kinematic_mask(particle_type)

        # Inject random walk noise
        if self.phase == 'train':
            sampled_noise = utils.get_random_walk_noise(poss,C.NET.NOISE) #[1688 6 2]
            sampled_noise = torch.tensor(sampled_noise) * nonk_mask[:, None, None]
            poss = poss + sampled_noise

            tgt_poss = tgt_poss + sampled_noise[:, -1:]
        
        tgt_vels = torch.tensor(utils.time_diff(np.concatenate([poss, tgt_poss], axis=1)))
        tgt_accs = torch.tensor(utils.time_diff(tgt_vels))
        tgt_vels = tgt_vels[:, -self.pred_steps:]
        tgt_accs = tgt_accs[:, -self.pred_steps:]
        poss=poss
        tgt_poss=tgt_poss
        poss =poss/200.
        tgt_vels = tgt_vels/200.
        tgt_accs =tgt_accs/200.
        particle_type = particle_type
        nonk_mask = nonk_mask
        tgt_poss = tgt_poss/200.

        # poss2=data['position'][:,:]
        # data_test=poss[:104,0,:]
        # data2=poss[104:,0,:]
        # import matplotlib.pylab as plt
        # for i in range(160):
        #     i=i*3
        #     data_test=poss2[:104,i,:]
        #     data2=poss2[104:,i,:]
        #     figure=plt.figure(figsize=(18,15))
        #     data = data_test
        #     plt.scatter(data[:,0], data[:,1],color="red")
        #     plt.scatter(data2[:,0], data2[:,1],color="green")
        #     plt.xlim([0,1])
        #     plt.ylim([0,1])
        #     plt.show()
        #     plt.savefig(f"/user/project/test/GNS-PyTorch/data/test_data/{i}.png")
        return poss, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss

def get_non_kinematic_mask(particle_types):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  return particle_types ==1


class nbody_gns_dataset_cond_one(Dataset):
    def __init__(self, data_dir,
                  phase='train',
                  rollout_steps=240,
                  n_his=1,
                  time_interval=4,
                  verbose=1,
                  input_steps=1,
                  output_steps=1,
                  n_bodies=2,
                  is_train=True,
                  device="cuda"
                  ):
        self.data_dir = data_dir
        self.phase = phase
        self.metadata = _read_metadata(self.data_dir)
        self.rollout_steps=rollout_steps
        self.n_his=n_his
        self.time_interval=time_interval
        self.time_stamps=980
        self.verbose=verbose
        self.input_steps=input_steps
        self.output_steps=output_steps
        self.t_cushion_input = (self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1)
        self.t_cushion_output=self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1
        self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output) // self.time_interval
        self.data=None
        self.n_bodies=n_bodies
        self.is_train=is_train
        self.pred_steps=output_steps
        self.device=device
        self.total_n_simu=2000
        self.delta_t=self.time_interval*(1./60.)
        if is_train:
            self.n_simu=1800
        else:
            self.n_simu=200
        for key in self.metadata:
            self.metadata[key] = torch.from_numpy(np.array(self.metadata[key]).astype(np.float32))
        if self.n_bodies==4:
            self.data = torch.FloatTensor(np.load(self.data_dir+"/train/trajectory_balls_4_simu_2000_steps_1000.npy"))
        elif self.n_bodies==2:
            self.data = torch.FloatTensor(np.load(self.data_dir+"/train/trajectory_balls_2_simu_2000_steps_1000.npy"))
        elif self.n_bodies==3:
            self.data = torch.FloatTensor(np.load("/user/project/inverse_design/dataset2/nbody_dataset/nbody-3/speed-100/trajectory_balls_3_simu_1000_steps_1000.npy"))
        elif self.n_bodies==8:
            # self.total_n_simu=200
            # if is_train:
            #     self.n_simu=180
            # else:
            #     self.n_simu=20
            self.data = torch.FloatTensor(np.load(self.data_dir+"/train/trajectory_balls_8_simu_200_steps_1000.npy"))

    def __len__(self):
        return self.time_stamps_effective * self.n_simu

    def __getitem__(self, idx):
        
        
        sim_id, time_id = divmod(idx, self.time_stamps_effective)
        if not self.is_train:
            sim_id += self.total_n_simu - self.n_simu
        if self.verbose >= 1:
            print("sim_id, time_id:", (sim_id, time_id))
            print("start:", time_id * self.time_interval + self.t_cushion_input - self.input_steps * self.time_interval)
            print("mid:", time_id * self.time_interval + self.t_cushion_input)
            print("end:", time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval)
        # self.data shape: [n_simu, n_steps:1000, n_bodies, n_features:4]
        # x: [input_steps, n_bodies, n_features
        if self.output_steps%2!=-0:
            self.output_steps=self.output_steps+1
        x = self.data[sim_id, time_id * self.time_interval + self.t_cushion_input - self.input_steps * self.time_interval: time_id * self.time_interval + self.t_cushion_input: self.time_interval].transpose(1,0)
        y = self.data[sim_id, time_id * self.time_interval + self.t_cushion_input: time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval: self.time_interval].transpose(1,0)
        # y_next = self.data[sim_id,time_id * self.time_interval + self.t_cushion_input: time_id * self.time_interval + self.t_cushion_input + (self.output_steps+1) * self.time_interval: self.time_interval].transpose(1,0)
        # y_next=y_next[:,1:self.output_steps+1,:]
        # Hack：
        y_next =y[:,1:,:]
        y=y[:,:-1,:]
        if y.shape[1]<self.output_steps or y_next.shape[1]<self.output_steps:
            print("sim_id, time_id:", (sim_id, time_id))
            print("start:", time_id * self.time_interval + self.t_cushion_input - self.input_steps * self.time_interval)
            print("mid:", time_id * self.time_interval + self.t_cushion_input)
            print("end:", time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval)
            print("EEEEEEEEEEEEE????????????????????????")
        # y_last=torch.cat([x,y[:,1:self.output_steps,:]],dim=1)
        # x [n_bodies,3,n_bodies*n_features]
        #y [n_bodies,num_timesteps,n_features]
        particle_type=torch.ones(self.n_bodies).long()
        # pdb.set_trace()
        poss = x[:,-1:,:2]
        vel=x[:,-1:,2:]*self.delta_t
        tgt_poss =y[:,:,:2]

        nonk_mask = get_non_kinematic_mask(particle_type)

        # Inject random walk noise
        if self.phase == 'train':
            sampled_noise = utils.get_random_walk_noise(poss,C.NET.NOISE) #[1688 6 2]
            sampled_noise = torch.tensor(sampled_noise) * nonk_mask[:, None, None]
            poss = poss + sampled_noise

            tgt_poss = tgt_poss + sampled_noise[:, -1:]
        try:
            tgt_vels = torch.tensor(utils.time_diff(np.concatenate([poss, tgt_poss], axis=1)))
            tgt_accs =(y_next[:,:,:2]-tgt_poss)-tgt_vels #a(k)=v(k+1)-v(k)
            # print("poss shape",poss.shape)
            # print("tgt_poss shape",tgt_poss.shape)
            # print("y_next shape",y_next.shape)
            # print("tgt_vels shape",tgt_vels.shape)
        except:
            print("poss shape",poss.shape)
            print("tgt_poss shape",tgt_poss.shape)
            print("y_next shape",y_next.shape)
            print("tgt_vels shape",tgt_vels.shape)


        # tgt_vels = y[:,:,2:]
        # tgt_accs =(tgt_vels-y_last[:,:,2:])/self.delta_t #a(k)=(v(k+1)-v(k))/delta_t pay attention
        tgt_vels = tgt_vels[:, -self.pred_steps:]
        tgt_accs = tgt_accs[:, -self.pred_steps:]
        poss=poss
        tgt_poss=tgt_poss
        poss =poss/200.
        vel=vel/200.
        tgt_vels = tgt_vels/200.
        tgt_accs =tgt_accs/200.
        particle_type = particle_type
        nonk_mask = nonk_mask
        tgt_poss = tgt_poss/200.

        # poss2=data['position'][:,:]
        # data_test=poss[:104,0,:]
        # data2=poss[104:,0,:]
        # import matplotlib.pylab as plt
        # for i in range(160):
        #     i=i*3
        #     data_test=poss2[:104,i,:]
        #     data2=poss2[104:,i,:]
        #     figure=plt.figure(figsize=(18,15))
        #     data = data_test
        #     plt.scatter(data[:,0], data[:,1],color="red")
        #     plt.scatter(data2[:,0], data2[:,1],color="green")
        #     plt.xlim([0,1])
        #     plt.ylim([0,1])
        #     plt.show()
        #     plt.savefig(f"/user/project/test/GNS-PyTorch/data/test_data/{i}.png")
        # pdb.set_trace()
        return poss, vel,tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss

if __name__=="__main__":
    test_set=nbody_gns_dataset_cond_one(
        data_dir="/zhangtao/cindm/dataset/nbody_dataset",
        phase='test',
        time_interval=4,
        verbose=0,
        output_steps=23,
        n_bodies=2,
        is_train=False,
        device="cuda"
        )
    # pdb.set_trace()
    dataloader_GNS=DataLoader(test_set, batch_size=500, shuffle=False, pin_memory=True, num_workers=6)
    for data_GNS in dataloader_GNS:
        break
    pdb.set_trace()
