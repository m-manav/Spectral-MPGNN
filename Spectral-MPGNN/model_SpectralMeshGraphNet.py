import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from ProcessorLayer import ProcessorLayer
from InputNormalize import normalize

class SpectralMeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, sp_input_dim_node, sp_input_dim_edge, 
                    hidden_dim, dim_ext_force, output_dim, args, emb=False):
        super(SpectralMeshGraphNet, self).__init__()
        """
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """

        self.num_layers = args.num_layers
        self.device = args.device

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear(input_dim_edge , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim)
                              )

        self.spnode_encoder = Sequential(Linear(sp_input_dim_node , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim))

        self.spedge_encoder = Sequential(Linear(sp_input_dim_edge , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim)
                              )                 


        self.processor = nn.ModuleList()
        self.sp_processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, dim_ext_force, hidden_dim))
            self.sp_processor.append(processor_layer(hidden_dim, 0, hidden_dim))


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear( 2*hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )


    def build_processor_model(self):
        return ProcessorLayer


    def eigenpooling(self, V, x, idx_lim_Gnodes):
        spx_pooled = torch.empty((0, x.shape[1])).to(self.device)

        for ii in range(idx_lim_Gnodes.shape[0]):
            idx1 = idx_lim_Gnodes[ii, 0]
            idx2 = idx_lim_Gnodes[ii, 1]
            spx_pooled_part = torch.matmul(torch.t(V[idx1:idx2, :]), x[idx1:idx2, :])
            spx_pooled = torch.cat((spx_pooled, spx_pooled_part), dim = 0)
        
        return spx_pooled


    def eigenbroadcasting(self, V, spx, idx_lim_Gnodes):
        x_broadcasted = torch.empty((0, spx.shape[1])).to(self.device)
        n_eigmodes  = V.shape[1]

        for ii in range(idx_lim_Gnodes.shape[0]):
            idx1 = idx_lim_Gnodes[ii, 0]
            idx2 = idx_lim_Gnodes[ii, 1]
            x_broadcasted_part = torch.matmul(V[idx1:idx2, :], spx[n_eigmodes*ii:n_eigmodes*(ii+1), :])
            x_broadcasted = torch.cat((x_broadcasted, x_broadcasted_part), dim = 0)
        
        return x_broadcasted


    def forward(self,spt_data,spc_data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr, del_ext_force = spt_data.x, spt_data.edge_index, spt_data.edge_attr, spt_data.del_ext_force

        spx, sp_edge_index,  sp_edge_attr, V, n_nodes = spc_data.x, spc_data.edge_index, spc_data.edge_attr, spc_data.V, spc_data.n_nodes

        n_nodes_cumsum = torch.cumsum(n_nodes.reshape(-1, ), dim=0)
        idx_lim_Gnodes = torch.zeros((n_nodes.shape[0], 2), dtype = torch.int64)
        idx_lim_Gnodes[1:, 0] = n_nodes_cumsum[:-1]
        idx_lim_Gnodes[:, 1] = n_nodes_cumsum

        x = normalize(x,mean_vec_x,std_vec_x)
        edge_attr = normalize(edge_attr,mean_vec_edge,std_vec_edge)
        del_ext_force = normalize(del_ext_force,mean_vec_force,std_vec_force)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension
        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        spx = self.spnode_encoder(spx)
        sp_edge_attr = self.spedge_encoder(sp_edge_attr)

        x_new = torch.cat((x, self.eigenbroadcasting(V, spx, idx_lim_Gnodes)), dim = 1)
        spx = torch.cat((spx, self.eigenpooling(V, x, idx_lim_Gnodes)), dim = 1)
        x = x_new

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x_i, edge_attr = self.processor[i](x,edge_index,edge_attr,del_ext_force)
            spx_i, sp_edge_attr = self.sp_processor[i](spx,sp_edge_index,sp_edge_attr,del_ext_force = None)

            x = x[:, 0:x_i.shape[1]]
            spx = spx[:, 0:x_i.shape[1]]
            x_new = torch.cat((x_i, self.eigenbroadcasting(V, spx, idx_lim_Gnodes)), dim = 1)
            spx = torch.cat((spx_i, self.eigenpooling(V, x, idx_lim_Gnodes)), dim = 1)
            x = x_new

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)

    def loss(self, pred, inputs, loss_mask, mean_vec_y, std_vec_y):
        #Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)

        #Find sum of square errors
        error=torch.sum((labels-pred)**2,axis=1)

        #Root and mean the errors for the nodes we calculate loss for
        if loss_mask == torch.tensor([1], device=self.device):
            loss = torch.sqrt(torch.mean(error))            
        else:
            loss = torch.sqrt(torch.mean(error[loss_mask]))
        
        return loss
    