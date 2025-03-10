import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian

import numpy as np
import pickle
from pathlib import Path


class DefineDataset:
    def __init__(self, datafile:Path, number_trajectories:int = 110, number_ts:int = 10, var_noise:float = 0.000, num_eig:int = 5):
        self.filename = datafile
        self.number_trajectories = number_trajectories
        self.number_ts = number_ts
        self.var_noise = var_noise
        self.num_eig = num_eig

        self._spt_data_list = []
        self._spc_data_list = []


    def assemble_dataset(self):
        with open(self.filename,'rb') as file:
            data = pickle.load(file)

            for i, trajectory in enumerate(data.keys()):
                if(i==self.number_trajectories):
                    break
                print("Trajectory: ",i)

                #We iterate over all the time steps to produce an example graph except
                #for the last one, which does not have a following time step to produce
                #node output values
                for ts in range(len(data[trajectory]['current_node_pos'])-1):

                    if(ts==self.number_ts):
                        break

                    #Get node features
                    ref_node_pos = torch.from_numpy(data[trajectory]['ref_node_pos'][ts])
                    current_node_pos = torch.from_numpy(data[trajectory]['current_node_pos'][ts])
                    del_ext_force = torch.from_numpy(data[trajectory]["del_ext_force"][ts])
                    node_type_idx = data[trajectory]['node_types'][ts].astype(int)
                    # convert to one hot tensor form
                    node_type = torch.zeros(node_type_idx.shape[0], np.unique(node_type_idx).shape[0])
                    for j in range(node_type_idx.shape[0]):
                        node_type[j, node_type_idx[j, 0]] = 1
                    
                    stress = torch.from_numpy(data[trajectory]['stress'][ts])
                    stress = stress + (self.var_noise**0.5)*torch.randn(stress.shape)
                    # x = torch.cat((ref_node_pos, current_node_pos, del_ext_force, node_type), dim=-1).type(torch.float)
                    x = torch.cat((stress, node_type), dim=-1).type(torch.float)
                    # x = node_type.type(torch.float)
                    n_nodes = torch.tensor([[x.shape[0]]])

                    edge_index = torch.from_numpy(data[trajectory]['edge_index'][ts]).type(torch.long)

                    #Get edge features (mesh space as well as current configuration)
                    uR_i = torch.from_numpy(data[trajectory]['ref_node_pos'][ts])[edge_index[0]]
                    uR_j = torch.from_numpy(data[trajectory]['ref_node_pos'][ts])[edge_index[1]]
                    uR_ij = uR_i-uR_j
                    uR_ij = uR_ij + (self.var_noise**0.5)*torch.randn(uR_ij.shape)
                    uR_ij_norm = torch.norm(uR_ij, p=2, dim=1, keepdim=True)
                    uC_i = torch.from_numpy(data[trajectory]['current_node_pos'][ts])[edge_index[0]]
                    uC_j = torch.from_numpy(data[trajectory]['current_node_pos'][ts])[edge_index[1]]
                    uC_ij = uC_i - uC_j
                    uC_ij = uC_ij + (self.var_noise**0.5)*torch.randn(uC_ij.shape)
                    uC_ij_norm = torch.norm(uC_ij, p=2, dim=1, keepdim=True)

                    edge_attr = torch.cat((uR_ij, uR_ij_norm), dim=-1).type(torch.float)  #, uC_ij, uC_ij_norm

                    del_ext_force = del_ext_force #+ (var_noise**0.5)*torch.randn(del_ext_force.shape)
                    del_ext_force = del_ext_force.type(torch.float)

                    # Node outputs, for training
                    del_displacement = torch.from_numpy(data[trajectory]['current_node_pos'][ts+1]-data[trajectory]['current_node_pos'][ts])
                    y = del_displacement.type(torch.float)

                    #Node outputs, for testing
                    strain_energy = torch.from_numpy(data[trajectory]['strain_energy'][ts])

                    #Data needed for visualization code
                    current_node_pos = torch.from_numpy(data[trajectory]['current_node_pos'][ts])

                    # spectral graph information
                    ed_idx, ed_w = get_laplacian(edge_index)
                    lap = torch.sparse_coo_tensor(ed_idx, ed_w, (x.shape[0], x.shape[0]))
                    lap = lap.to_dense()
                    E, V = torch.linalg.eigh(lap)
                    idx_st = 0
                    E = E[idx_st:idx_st+self.num_eig]
                    V = V[:, idx_st:idx_st+self.num_eig]
                    V = V/torch.sqrt(torch.sum(V**2, axis=0))

                    spx = E.reshape(-1, 1)
                    sp_edge_index_init = torch.ones(self.num_eig, self.num_eig) - torch.eye(self.num_eig)
                    sp_edge_index_init = sp_edge_index_init.to_sparse()
                    sp_edge_index = sp_edge_index_init.indices()
                    sp_edge_attr = torch.ones((sp_edge_index.shape[1], 1), dtype = torch.float)

                    self._spt_data_list.append(Data(x = x, edge_index = edge_index, edge_attr = edge_attr, del_ext_force = del_ext_force, y = y,
                                        stress = stress, strain_energy = strain_energy, current_node_pos = current_node_pos))

                    self._spc_data_list.append(Data(x = spx, edge_index = sp_edge_index, edge_attr = sp_edge_attr, V = V, n_nodes = n_nodes))

    

        data_list = {"spt_data_list": self._spt_data_list, 
                    	"spc_data_list": self._spc_data_list}
        
        return data_list
    