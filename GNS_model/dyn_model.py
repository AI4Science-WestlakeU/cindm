import torch
import torch.nn as nn
import torch.nn.functional as F
from cindm.GNS_model.config import _C as C
from cindm.GNS_model.layers.GNN_dmwater import GraphNet
from scipy import spatial
import numpy as np
import cindm.GNS_model.utils as utils
import pdb

def poss_offset(poss,n_bodies):
    batch_size=int(poss.shape[0]/n_bodies)
    tensor = torch.arange(0, 2*batch_size, step=2)
    offset = tensor.repeat_interleave(poss.shape[-1]*n_bodies)
    offset=offset.reshape(-1,poss.shape[-1])
    poss_offset=poss.clone()+offset.to(poss.device)

    return poss_offset
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.node_dim_in = C.NET.NODE_FEAT_DIM_IN
        self.edge_dim_in = C.NET.EDGE_FEAT_DIM_IN

        self.hidden_size = C.NET.HIDDEN_SIZE
        self.out_size = C.NET.OUT_SIZE
        num_layers = C.NET.GNN_LAYER
        
        self.particle_emb = nn.Embedding(C.NUM_PARTICLE_TYPES, C.NET.PARTICLE_EMB_SIZE)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        self.graph = GraphNet(layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size),
        )

    def _construct_graph_nodes(self, poss, particle_type, metadata):
        '''
        poss: [num_particles,time_step,feature_dim] [2,4,2]
        particle_type :[2]
        '''
        vels = utils.time_diff(poss) #[2,3,2]
        # vels = (vels - metadata['vel_mean'])/metadata['vel_std']
        n_vel, d_vel = vels.shape[1], vels.shape[2]
        assert n_vel == C.N_HIS - 1
        vels = vels.reshape([-1, n_vel*d_vel]) #[2,6]

        pos_last = poss[:, -1] #[2,1,2]
        dist_to_walls = torch.cat(
                        [pos_last - metadata['bounds'][:, 0],
                        -pos_last + metadata['bounds'][:, 1]], 1) #[2,4]
        dist_to_walls = torch.clip(dist_to_walls/C.NET.RADIUS, -1, 1) 

        type_emb = self.particle_emb(particle_type) #[2,2]
        # pdb.set_trace()
        node_attr = torch.cat([vels,
                               dist_to_walls,
                               type_emb], axis=1)
        return node_attr #[2,12]

    def _construct_graph_edges(self, pos):
        # pdb.set_trace()
        '''
        pos :[2,2]
        '''
        device = pos.device
        collapsed = False

        n_particles = pos.shape[0]
        # Calculate undirected edge list using KDTree
        poss_array=np.array(pos.detach().to("cpu").numpy())
        point_tree = spatial.cKDTree(poss_array)
        undirected_pairs = np.array(list(point_tree.query_pairs(C.NET.RADIUS, p=2))).T
        undirected_pairs = torch.from_numpy(undirected_pairs).to(device)
        pairs = torch.cat([undirected_pairs, torch.flip(undirected_pairs, dims=(0,))], dim=1).long()
        if C.NET.SELF_EDGE:
            self_pairs = torch.stack([torch.arange(n_particles, device=device), 
                                    torch.arange(n_particles, device=device)])
            pairs = torch.cat([pairs, self_pairs], dim=1)

        # check if prediction collapsed in long term unrolling
        # if pairs.shape[1] > C.NET.MAX_EDGE_PER_PARTICLE * n_particles:
        #     collapsed = True
        if pairs.shape!=torch.Size([0]):
            #pairs: [2,2]
            senders = pairs[0] #[num_particles]
            receivers = pairs[1]

            # Calculate corresponding relative edge attributes (distance vector + magnitude)
            dist_vec = (pos[senders] - pos[receivers]) #[2,2]
            dist_vec = dist_vec / C.NET.RADIUS ################################ What is this parameter? 
            dist = torch.linalg.norm(dist_vec, dim=1, keepdims=True) #[2,1]
            edges = torch.cat([dist_vec, dist], dim=1) #[num_edges,3]

            return edges, senders, receivers, collapsed
        else:
            return pairs,pairs,pairs,collapsed
        

    def forward(self, poss, particle_type, metadata, nonk_mask, tgt_poss, num_rollouts=10, phase='train'):
        # pdb.set_trace()
        pred_accns = []
        pred_poss = []
        for j in range(int(num_rollouts/(self.out_size/2))):
            # pdb.set_trace()
            nodes = self._construct_graph_nodes(poss, particle_type, metadata)
            
            edges, senders, receivers, collapsed = self._construct_graph_edges(poss[:, -1])

            nodes = self.node_encoder(nodes) #[2,10] --> [2,64]
            if edges.shape !=torch.Size([0]):
                edges = self.edge_encoder(edges) #[num_edges,num_features] [2,3] --> [2,64]
                nodes, edges = self.graph(nodes, edges, senders, receivers)

            pred_accn = self.decoder(nodes) #[2,_C.NET.OUT_SIZE]

            
            
            if pred_accn.shape[1]==2:
                pred_acc = pred_accn #[2,_C.NET.OUT_SIZE]
                pred_accns.append(pred_accn)
                pred_vel = poss[:, -1] - poss[:, -2] #[2,2]
                pred_pos = poss[:, -1] + pred_vel + pred_acc   #   ##x(k+1)=x(k)+v(k)+a(k+1)=x(k)+v(k+1)
                #poss [2,4,2]
                #pred_vel [2,2]
                #pred_acc [2,8]


                # replace kinematic nodes
                pred_pos = torch.where(nonk_mask[:, None].bool(), pred_pos, tgt_poss[:, 0])
                poss = torch.cat([poss[:, 1:], pred_pos[:, None]], dim=1)
                pred_poss.append(pred_pos)
            elif pred_accn.shape[1]!=2:
                pred_acc=pred_accn.reshape(pred_accn.shape[0],int(pred_accn.shape[1]/2),2) #[num_particles,num_timesteps,num_features]
                pred_vel=torch.zeros_like(pred_acc) 
                pred_pos=torch.zeros_like(pred_acc)
                v_last=poss[:,-1:,:]-poss[:,-2:-1,:]
                for i in range(pred_acc.shape[1]):
                    if i==0:
                        pred_vel[:,i,:]=v_last[:,0,:]+pred_acc[:,i,:]
                    else:
                        pred_vel[:,i,:]=pred_vel[:,i-1,:]+pred_acc[:,i,:]
                for i in range(pred_acc.shape[1]):
                    if i==0:
                        pred_pos[:,i,:]=poss[:,-1,:]+pred_vel[:,i,:]
                    else:
                        pred_pos[:,i,:]=pred_pos[:,i-1,:]+pred_vel[:,i,:]
                if j==0:
                    pred_poss=pred_pos
                    pred_accns=pred_acc
                else:
                    pred_poss=torch.cat([pred_poss,pred_pos],dim=1)
                    pred_accns=torch.cat([pred_poss,pred_poss],dim=1)
                #update poss
                if poss.shape[1]>pred_pos.shape[1]:
                    poss=torch.cat([poss[:,pred_pos.shape[1]:],pred_pos],dim=1)
                elif poss.shape[1]<pred_pos.shape[1]:
                    poss=pred_pos[:,-poss.shape[1]:,:]
                else:
                    poss=pred_pos
            if collapsed:
                break

        # pdb.set_trace()
        if self.out_size==2:
            pred_accns = torch.stack(pred_accns).permute(1, 0, 2)
            pred_poss = torch.stack(pred_poss).permute(1, 0, 2)

        outputs = {
            'pred_accns': pred_accns,
            'pred_poss': pred_poss,
            'pred_collaposed': collapsed
        }
        
        return outputs












class Net_cond_one(nn.Module):
    def __init__(self,output_size,n_bodies=None):
        super(Net_cond_one, self).__init__()

        self.node_dim_in = C.NET.NODE_FEAT_DIM_IN
        self.edge_dim_in = C.NET.EDGE_FEAT_DIM_IN

        self.hidden_size = C.NET.HIDDEN_SIZE
        self.out_size = output_size
        num_layers = C.NET.GNN_LAYER
        self.delta_t=4.*(1./60.)
        self.n_bodies=n_bodies
        
        self.particle_emb = nn.Embedding(C.NUM_PARTICLE_TYPES, C.NET.PARTICLE_EMB_SIZE)

        #input encoder shape [n_bodeis,n_features]
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        self.graph = GraphNet(layers=num_layers)
        # pdb.set_trace()
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size),
        )

    def _construct_graph_nodes(self, poss,vel, particle_type, metadata):
        '''
        poss: [num_particles,time_step,feature_dim] [2,4,2]
        particle_type :[2]
        '''
        vels =vel  #[2,1,2]
        # vels = (vels - metadata['vel_mean'])/metadata['vel_std']
        n_vel, d_vel = vels.shape[1], vels.shape[2]
        vels = vels.reshape([-1, n_vel*d_vel]) #[2,2]

        pos_last = poss[:, -1] #[2,1,2]
        dist_to_walls = torch.cat(
                        [pos_last - metadata['bounds'][:, 0],
                        -pos_last + metadata['bounds'][:, 1]], 1) #[2,4]
        dist_to_walls = torch.clip(dist_to_walls/C.NET.RADIUS, -1, 1) 

        type_emb = self.particle_emb(particle_type) #[2,2]
        # pdb.set_trace()
        node_attr = torch.cat([vels,
                               dist_to_walls,
                               type_emb], axis=1)
        return node_attr #[2,8]

    def _construct_graph_edges(self, pos):
        # pdb.set_trace()
        '''
        pos :[2,2] [n_bodies,2]
        '''
        device = pos.device
        collapsed = False

        n_particles = pos.shape[0] 
        ##offset the poss among different batchs
        if n_particles>self.n_bodies:
            pos_offset=poss_offset(pos,self.n_bodies)
        else:
            pos_offset=pos
        # Calculate undirected edge list using KDTree
        poss_array=np.array(pos_offset.detach().to("cpu").numpy())
        point_tree = spatial.cKDTree(poss_array)
        undirected_pairs = np.array(list(point_tree.query_pairs(C.NET.RADIUS, p=2))).T
        undirected_pairs = torch.from_numpy(undirected_pairs).to(device)
        pairs = torch.cat([undirected_pairs, torch.flip(undirected_pairs, dims=(0,))], dim=1).long()
        if C.NET.SELF_EDGE:
            self_pairs = torch.stack([torch.arange(n_particles, device=device), 
                                    torch.arange(n_particles, device=device)])
            pairs = torch.cat([pairs, self_pairs], dim=1)

        # check if prediction collapsed in long term unrolling
        # if pairs.shape[1] > C.NET.MAX_EDGE_PER_PARTICLE * n_particles:
        #     collapsed = True
        if pairs.shape!=torch.Size([0]):
            #pairs: [2,2]
            senders = pairs[0] #[num_particles]
            receivers = pairs[1]

            # Calculate corresponding relative edge attributes (distance vector + magnitude)
            dist_vec = (pos[senders] - pos[receivers]) #[2,2]
            dist_vec = dist_vec / C.NET.RADIUS ################################ What is this parameter? 
            dist = torch.linalg.norm(dist_vec, dim=1, keepdims=True) #[2,1]
            edges = torch.cat([dist_vec, dist], dim=1) #[num_edges,3]

            return edges, senders, receivers, collapsed
        else:
            return pairs,pairs,pairs,collapsed
        

    def forward(self, poss, vel,particle_type, metadata, nonk_mask, tgt_poss, num_rollouts=10, phase='train'):
        # pdb.set_trace()
        # pdb.set_trace()
        pred_accns = [] #[2,1,2]
        pred_poss = []
        for j in range(int(num_rollouts/(self.out_size/2))):
            # pdb.set_trace()
            nodes = self._construct_graph_nodes(poss,vel, particle_type, metadata) #[2,8]
            # pdb.set_trace()
            
            edges, senders, receivers, collapsed = self._construct_graph_edges(poss[:, -1])
            if nodes.shape[1]==10:
                pdb.set_trace()
            
            nodes = self.node_encoder(nodes) #[2,10] --> [2,64]
            if edges.shape !=torch.Size([0]):
                edges = self.edge_encoder(edges) #[num_edges,num_features] [2,3] --> [2,64]
                nodes, edges = self.graph(nodes, edges, senders, receivers)

            pred_accn = self.decoder(nodes) #[2,_C.NET.OUT_SIZE]

            
            
            if pred_accn.shape[1]==2:
                pred_acc = pred_accn #[2,_C.NET.OUT_SIZE]
                pred_accns.append(pred_accn)
                prev_vel = vel[:,-1,:] #[2,2]
                pred_pos = poss[:, -1] + prev_vel + pred_acc   #   ##x(k+1)=x(k)+v(k)+a(k+1)=x(k)+v(k+1)
                #poss [2,4,2]
                #pred_vel [2,2]
                #pred_acc [2,8]


                # replace kinematic nodes
                pred_pos = torch.where(nonk_mask[:, None].bool(), pred_pos, tgt_poss[:, 0])
                vel=pred_pos-poss[:,-1,:]
                vel=vel.reshape(vel.shape[0],-1,vel.shape[1])
                poss = torch.cat([poss[:, 1:], pred_pos[:, None]], dim=1)
                pred_poss.append(pred_pos)

                # pred_acc = pred_accn #[2,_C.NET.OUT_SIZE]
                # pred_accns.append(pred_accn)
                # pred_accn=pred_accn.reshape(pred_accn.shape[0],-1,2)
                # pred_vel = vel+pred_accn[:,0:1,:]*self.delta_t #[2,2] v(k+1)=v(k)+a(k)*delta_t
                # pred_pos = poss[:, -1,:] + vel[:,-1,:]*self.delta_t  #   ##x(k+1)=x(k)+v(k)+a(k+1)=x(k)+v(k+1)
                # #poss [2,4,2]
                # #pred_vel [2,2]
                # #pred_acc [2,8]


                # # replace kinematic nodes
                # # pdb.set_trace()
                # pred_pos = torch.where(nonk_mask[:, None].bool(), pred_pos, tgt_poss[:, 0])
                # pred_poss.append(pred_pos)
                # poss = pred_pos.reshape(pred_pos.shape[0],-1,pred_pos.shape[1]) #update poss using new pred_poss to rollout new steps
                # vel=pred_vel
            elif pred_accn.shape[1]!=2:
                pred_acc=pred_accn.reshape(pred_accn.shape[0],int(pred_accn.shape[1]/2),2) #[num_particles,num_timesteps,num_features]
                pred_vel=torch.zeros_like(pred_acc) 
                pred_pos=torch.zeros_like(pred_acc)
                v_last=vel[:,-1:,:]
                for i in range(pred_acc.shape[1]):
                    if i==0:
                        pred_vel[:,i,:]=v_last[:,0,:]+pred_acc[:,i,:]
                    else:
                        pred_vel[:,i,:]=pred_vel[:,i-1,:]+pred_acc[:,i,:]
                for i in range(pred_acc.shape[1]):
                    if i==0:
                        pred_pos[:,i,:]=poss[:,-1,:]+pred_vel[:,i,:]
                    else:
                        pred_pos[:,i,:]=pred_pos[:,i-1,:]+pred_vel[:,i,:]
                if j==0:
                    pred_poss=pred_pos
                    pred_accns=pred_acc
                else:
                    pred_poss=torch.cat([pred_poss,pred_pos],dim=1)
                    pred_accns=torch.cat([pred_poss,pred_poss],dim=1)
                #update poss
                if poss.shape[1]>pred_pos.shape[1]:
                    poss=torch.cat([poss[:,pred_pos.shape[1]:],pred_pos],dim=1)
                elif poss.shape[1]<pred_pos.shape[1]:
                    poss=pred_pos[:,-poss.shape[1]:,:]
                else:
                    poss=pred_pos   
                #update vel
                vel=utils.time_diff(pred_pos)

                # pred_acc=pred_accn.reshape(pred_accn.shape[0],int(pred_accn.shape[1]/2),2) #[num_particles,num_timesteps,num_features]
                # pred_vel=torch.zeros_like(pred_acc) 
                # pred_pos=torch.zeros_like(pred_acc)
                # v_last=vel
                # for i in range(pred_acc.shape[1]):
                #     if i==0:
                #         pred_vel[:,i,:]=v_last[:,0,:]+pred_acc[:,i,:]*self.delta_t
                #     else:
                #         pred_vel[:,i,:]=pred_vel[:,i-1,:]+pred_acc[:,i,:]*self.delta_t
                # for i in range(pred_acc.shape[1]):
                #     if i==0:
                #         pred_pos[:,i,:]=poss[:,-1,:]+v_last[:,-1,:]*self.delta_t
                #     else:
                #         pred_pos[:,i,:]=pred_pos[:,i-1,:]+pred_vel[:,i-1,:]*self.delta_t
                # if j==0:
                #     pred_poss=pred_pos
                #     pred_accns=pred_acc
                # else:
                #     pred_poss=torch.cat([pred_poss,pred_pos],dim=1)
                #     pred_accns=torch.cat([pred_poss,pred_poss],dim=1)
                # #update poss &&vel
                # if poss.shape[1]>pred_pos.shape[1]:
                #     poss=torch.cat([poss[:,pred_pos.shape[1]:],pred_pos],dim=1)
                # elif poss.shape[1]<pred_pos.shape[1]:
                #     poss=pred_pos[:,-poss.shape[1]:,:]
                #     vel=pred_vel[:,-vel.shape[1]:,:]
                # else:
                #     poss=pred_pos
                #     vel=pred_vel
            if collapsed:
                break

        # pdb.set_trace()
        if self.out_size==2:
            pred_accns = torch.stack(pred_accns).permute(1, 0, 2)
            pred_poss = torch.stack(pred_poss).permute(1, 0, 2)

        outputs = {
            'pred_accns': pred_accns,
            'pred_poss': pred_poss,
            'pred_collaposed': collapsed
        }
        
        return outputs
    
def GNS_inference(data_GNS,cond,gns_model,metadata,device,rollout_steps=23,is_batch_for_GNS=False):
    # pdb.set_trace()
    metadata={key: value.to(device) for key, value in metadata.items()}
    if is_batch_for_GNS:
        cond=cond.reshape(cond.shape[0]*cond.shape[1],cond.shape[2],-1)
        poss_cond=cond[:,:,:2]
        vel=cond[:,:,2:]
        #all became [batch_size*n_bodies,...] <- [batch_size,n_bodies,n_steps,n_features]
        _,__, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS #[batch_size,n_bodies,n_steps,n_features]
        n_bodies=tgt_vels.shape[1]
        # pdb.set_trace()
        y_gt=torch.cat([tgt_poss,tgt_vels],dim=3)
        tgt_accs=tgt_accs.reshape(tgt_accs.shape[0]*tgt_accs.shape[1],tgt_accs.shape[2],tgt_accs.shape[3])
        tgt_vels=tgt_vels.reshape(tgt_vels.shape[0]*tgt_vels.shape[1],tgt_vels.shape[2],tgt_vels.shape[3])
        tgt_poss=tgt_poss.reshape(tgt_poss.shape[0]*tgt_poss.shape[1],tgt_poss.shape[2],tgt_poss.shape[3])
        particle_type=particle_type.reshape(particle_type.shape[0]*particle_type.shape[1])
        nonk_mask=nonk_mask.reshape(nonk_mask.shape[0]*nonk_mask.shape[1])
        if cond.shape[0]!=tgt_accs.shape[0]:
            n_batched=int(cond.shape[0]/tgt_accs.shape[0])
            tgt_accs=torch.cat([tgt_accs]*n_batched,dim=0)
            tgt_vels=torch.cat([tgt_vels]*n_batched,dim=0)
            tgt_poss=torch.cat([tgt_poss]*n_batched,dim=0)
            particle_type=torch.cat([particle_type]*n_batched,dim=0)
            nonk_mask=torch.cat([nonk_mask]*n_batched,dim=0)

        
        # pdb.set_trace()
        num_rollouts=rollout_steps
        outputs = gns_model(poss_cond.to(device),vel.to(device) ,particle_type.to(device), metadata, nonk_mask.to(device), tgt_poss.to(device), num_rollouts=num_rollouts, phase='train')

        pred=outputs["pred_poss"]
        # pred_vel=torch.tensor(utils.time_diff(np.concatenate([poss_cond.to("cpu").detach(), pred.to("cpu").to("cpu").detach()], axis=1)))
        poss_cat=torch.cat((poss_cond,pred),dim=1)
        pred_vel=poss_cat[:,1:,:]-poss_cat[:,0:-1,:]

        pred=torch.cat([pred,pred_vel.to(pred.device)],dim=2) #[batch_size*n_bodies,n_steps,n_features]
        pred=pred.reshape(-1,n_bodies,pred.shape[1],pred.shape[2]) #[batch_size,n_bodies,n_steps,n_features]
        pred=pred.permute(0,2,1,3)
        pred=pred.reshape(pred.shape[0],pred.shape[1],-1)
        y_gt=y_gt.permute(0,2,1,3)#->[batch_size,n_steps,n_bodies,n_features]
        y_gt=y_gt.reshape(y_gt.shape[0],y_gt.shape[1],-1)

        cond=cond.reshape(-1,n_bodies,cond.shape[1],cond.shape[2])

        return y_gt,pred,cond
    else:
        _,__, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
        # rollout_steps=tgt_poss.shape[2]
        for i in range(cond.shape[0]):
            # if i>0:
            # pdb.set_trace()
            j=i%tgt_accs.shape[0]
            _,__, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss =data_GNS
            poss_cond=cond[:,:,:,:2]
            ##now the model just support batch_size=1
            # poss_cond=poss_cond[i]
            vel=cond[:,:,:,2:]
            tgt_accs=tgt_accs[j,:,:rollout_steps]
            tgt_vels=tgt_vels[j,:,:rollout_steps]
            particle_type=particle_type[j]
            nonk_mask=nonk_mask[j]
            tgt_poss=tgt_poss[j,:,:rollout_steps]

            num_rollouts=rollout_steps
            outputs = gns_model(poss_cond[i,:,:,:].to(device),vel[i,:,:,:].to(device) ,particle_type.to(device), metadata, nonk_mask.to(device), tgt_poss.to(device), num_rollouts=num_rollouts, phase='train')
            torch.cuda.empty_cache()



            y_gt_i=tgt_poss
            pred_i=outputs["pred_poss"] #[n_bodies,n_steps,2]
            # pdb.set_trace()
            pred_vel=torch.tensor(utils.time_diff(np.concatenate([poss_cond[i,:,:,:].to("cpu").detach(), pred_i.to("cpu").to("cpu").detach()], axis=1)))
            y_gt_i=torch.cat([y_gt_i,tgt_vels],dim=2).permute(1,0,2)
            y_gt_i=y_gt_i.reshape(-1,y_gt_i.shape[0],y_gt_i.shape[1]*y_gt_i.shape[2])
            pred_i=torch.cat([pred_i,pred_vel.to(pred_i.device)],dim=2).permute(1,0,2)
            pred_i=pred_i.reshape(-1,pred_i.shape[0],pred_i.shape[1]*pred_i.shape[2])
            if i==0:
                y_gt=y_gt_i
                pred=pred_i
            else:
                y_gt=torch.cat([y_gt,y_gt_i],dim=0)
                pred=torch.cat([pred,pred_i],dim=0)
        # pdb.set_trace()
        return y_gt,pred,cond #pred.shape [batch_size,rollout_steps,n_bodies*n_features]