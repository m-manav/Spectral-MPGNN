import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing


class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Spectral_MP_GNN(nn.Module):
    def __init__(self, model_config):
        super(Spectral_MP_GNN, self).__init__()
        """
        Spectral Message Passing Graph Neural Network (Spectral_MP_GNN)
        
        This model operates on spatial and spectral graphs using message passing layers.
        The spatial graph contains physical or geometric information, while the spectral
        graph uses graph spectral properties (eigenmodes).

        Components:
        - Encoder: Encodes node and edge features for both graphs.
        - Processor: Performs multiple message passing layers for information propagation.
        - Decoder: Generates final predictions from the node embeddings.
        - Eigenpooling and Eigenbroadcasting: Connect spatial and spectral graphs.

        Args:
        - model_config: Configuration object containing input dimensions, hidden dimensions,
          output dimensions, number of message passing layers, and external forces.
        """

        # Initialize input dimensions for nodes and edges (spatial and spectral)
        self.input_dim_node = model_config.input_dim_node
        self.input_dim_edge = model_config.input_dim_edge
        self.spc_input_dim_node = model_config.spc_input_dim_node
        self.spc_input_dim_edge = model_config.spc_input_dim_edge

        # Hidden layer size, external force dimension, and output size
        self.hidden_dim = model_config.hidden_dim
        self.dim_ext_force = model_config.dim_ext_force
        self.output_dim = model_config.output_dim
        self.num_MP_layers = model_config.num_MP_layers

        # Encoders to transform node and edge features to hidden dimension space
        self.node_encoder = encoder(self.input_dim_node, self.hidden_dim)
        self.edge_encoder = encoder(self.input_dim_edge, self.hidden_dim)
        self.spc_node_encoder = encoder(self.spc_input_dim_node, self.hidden_dim)
        self.spc_edge_encoder = encoder(self.spc_input_dim_edge, self.hidden_dim)

        # Initialize processor layers for message passing on spatial and spectral graphs
        self.processor = nn.ModuleList()
        self.spc_processor = nn.ModuleList()
        assert self.num_MP_layers >= 1, "Number of message passing layers is not >=1"

        for _ in range(self.num_MP_layers):
            self.processor.append(
                ProcessorLayer(self.hidden_dim, self.hidden_dim, self.dim_ext_force)
            )
            self.spc_processor.append(
                ProcessorLayer(self.hidden_dim, self.hidden_dim, 0)
            )

        # Decoder to predict the output from the node embeddings
        self.node_decoder = decoder(self.hidden_dim, self.output_dim)

    def forward(
        self,
        spa_data,
        spc_data,
    ):
        """
        Forward pass of the model.
        Args:
        - spa_data: Spatial graph data containing node features, edge index, edge attributes, and external forces.
        - spc_data: Spectral graph data containing node features, edge index, edge attributes, and eigenmodes.
        """

        # Extract spatial graph data
        x, edge_index, edge_attr, del_ext_force = (
            spa_data.x,
            spa_data.edge_index,
            spa_data.edge_attr,
            spa_data.del_ext_force,
        )

        # Extract spectral graph data
        spc_x, spc_edge_index, spc_edge_attr, V, n_nodes = (
            spc_data.x,
            spc_data.edge_index,
            spc_data.edge_attr,
            spc_data.V,
            spc_data.n_nodes,
        )

        # Calculate cumulative node indices for separating graph structures in batch processing
        n_nodes_cumsum = torch.cumsum(
            n_nodes.reshape(
                -1,
            ),
            dim=0,
        )
        idx_lim_Gnodes = torch.zeros(
            (n_nodes.shape[0], 2), dtype=torch.int64, device=x.device
        )
        idx_lim_Gnodes[1:, 0] = n_nodes_cumsum[:-1]
        idx_lim_Gnodes[:, 1] = n_nodes_cumsum

        # Encode node and edge features using the respective encoders
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        spc_x = self.spc_node_encoder(spc_x)
        spc_edge_attr = self.spc_edge_encoder(spc_edge_attr)

        # Perform eigenbroadcasting and eigenpooling to exchange information between spatial and spectral domains
        x_new = torch.cat((x, self.eigenbroadcasting(V, spc_x, idx_lim_Gnodes)), dim=1)
        spc_x = torch.cat((spc_x, self.eigenpooling(V, x, idx_lim_Gnodes)), dim=1)
        x = x_new

        # Perform message passing through multiple processor layers
        for i in range(self.num_MP_layers):
            x_i, edge_attr = self.processor[i](x, edge_index, edge_attr, del_ext_force)
            spc_x_i, spc_edge_attr = self.spc_processor[i](
                spc_x, spc_edge_index, spc_edge_attr, del_ext_force=None
            )

            # Concatenate processed results
            x = x[:, 0 : x_i.shape[1]]
            spc_x = spc_x[:, 0 : x_i.shape[1]]
            x_new = torch.cat(
                (x_i, self.eigenbroadcasting(V, spc_x, idx_lim_Gnodes)), dim=1
            )
            spc_x = torch.cat((spc_x_i, self.eigenpooling(V, x, idx_lim_Gnodes)), dim=1)
            x = x_new

        # Decode the final node representations to produce outputs
        return self.node_decoder(x)

    def eigenpooling(self, V, x, idx_lim_Gnodes):
        """
        Perform eigenpooling to aggregate spatial nodal features to spectral nodal features.
        Each set of spatial features is transformed using the transposed eigenvector matrix.
        """
        spx_pooled = torch.empty((0, x.shape[1])).to(x.device)

        for ii in range(idx_lim_Gnodes.shape[0]):
            idx1 = idx_lim_Gnodes[ii, 0]
            idx2 = idx_lim_Gnodes[ii, 1]
            spx_pooled_part = torch.matmul(torch.t(V[idx1:idx2, :]), x[idx1:idx2, :])
            spx_pooled = torch.cat((spx_pooled, spx_pooled_part), dim=0)

        return spx_pooled

    def eigenbroadcasting(self, V, spx, idx_lim_Gnodes):
        """
        Perform eigenbroadcasting to transfer spectral nodal features back to spatial nodes.
        The operation uses the eigenvector matrix to project spectral features.
        """
        x_broadcasted = torch.empty((0, spx.shape[1])).to(spx.device)
        n_eigmodes = V.shape[1]

        for ii in range(idx_lim_Gnodes.shape[0]):
            idx1 = idx_lim_Gnodes[ii, 0]
            idx2 = idx_lim_Gnodes[ii, 1]
            x_broadcasted_part = torch.matmul(
                V[idx1:idx2, :], spx[n_eigmodes * ii : n_eigmodes * (ii + 1), :]
            )
            x_broadcasted = torch.cat((x_broadcasted, x_broadcasted_part), dim=0)

        return x_broadcasted

    def loss(self, inp, pred):
        error = torch.sum((inp.y - pred) ** 2, axis=1)
        loss = torch.mean(error)

        return loss


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels=128, out_channels=128, dim_ext_force=0):
        super(ProcessorLayer, self).__init__()
        """
        ProcessorLayer is a custom message-passing layer used in Graph Neural Networks (GNNs).
        It updates both node and edge features through a message-passing mechanism, with optional
        external force data used for enhanced predictions.
        
        Args:
        - in_channels (int): hidden dimension.
        - out_channels (int): hidden dimension.
        - dim_ext_force (int): Dimension of external force data, if applicable.
        """

        self.dim_ext_force = dim_ext_force
        self.out_channels = out_channels

        self.edge_mlp = encoder(5 * in_channels, out_channels)
        self.node_mlp = encoder(3 * in_channels + dim_ext_force, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset MLP weights to ensure model parameters are initialized properly.
        """

        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.edge_mlp.apply(weight_reset)
        self.node_mlp.apply(weight_reset)

    def forward(self, x, edge_index, edge_attr, del_ext_force, size=None):
        """
        Forward pass for message passing and node feature update.
        """

        # Perform message passing and obtain node updates (out) and updated edge features
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )

        if self.dim_ext_force > 0:
            updated_nodes = torch.cat([x, out, del_ext_force], dim=1)
        else:
            updated_nodes = torch.cat([x, out], dim=1)

        updated_nodes = self.node_mlp(updated_nodes)

        # Apply residual connection for stable gradient flow (skip connection)
        updated_nodes = x[:, 0 : updated_nodes.shape[1]] + updated_nodes

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        Compute the messages to be passed along the edges.
        """
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges) + edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        Aggregate incoming messages using sum reduction.
        """

        num_nodes = torch.max(edge_index[0, :]).item() + 1
        index = edge_index[0, :].unsqueeze(0)
        index = torch.permute(index, (1, 0))
        index = torch.broadcast_to(index, updated_edges.shape)

        out = torch.zeros(
            (num_nodes, self.out_channels),
            dtype=updated_edges.dtype,
            device=updated_edges.device,
        )
        out.scatter_reduce_(0, index, updated_edges, reduce="sum", include_self=False)

        return out, updated_edges
