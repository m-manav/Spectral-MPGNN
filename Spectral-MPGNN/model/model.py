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
    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        spc_input_dim_node,
        spc_input_dim_edge,
        hidden_dim,
        dim_ext_force,
        output_dim,
        num_MP_layers,
    ):
        super(Spectral_MP_GNN, self).__init__()
        """
        model: 
        (1) Encoder (for nodes and edges) 
        (2) Processor
        (3) Decoder (for nodes only)

        """

        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.spc_input_dim_node = spc_input_dim_node
        self.spc_input_dim_edge = spc_input_dim_edge
        self.hidden_dim = hidden_dim
        self.dim_ext_force = dim_ext_force
        self.output_dim = output_dim
        self.num_MP_layers = num_MP_layers

        self.node_encoder = encoder(self.input_dim_node, self.hidden_dim)
        self.edge_encoder = encoder(self.input_dim_edge, self.hidden_dim)
        self.spc_node_encoder = encoder(self.spc_input_dim_node, self.hidden_dim)
        self.spc_edge_encoder = encoder(self.spc_input_dim_edge, self.hidden_dim)

        self.processor = nn.ModuleList()
        self.spc_processor = nn.ModuleList()
        assert self.num_MP_layers >= 1, "Number of message passing layers is not >=1"

        for _ in range(self.num_layers):
            self.processor.append(ProcessorLayer(hidden_dim, dim_ext_force, hidden_dim))
            self.spc_processor.append(ProcessorLayer(hidden_dim, 0, hidden_dim))

        # decoder: only for nodes
        self.node_decoder = decoder(hidden_dim, output_dim)

    def forward(
        self,
        spa_data,
        spc_data,
    ):
        x, edge_index, edge_attr, del_ext_force = (
            spa_data.x,
            spa_data.edge_index,
            spa_data.edge_attr,
            spa_data.del_ext_force,
        )

        spc_x, spc_edge_index, spc_edge_attr, V, n_nodes = (
            spc_data.x,
            spc_data.edge_index,
            spc_data.edge_attr,
            spc_data.V,
            spc_data.n_nodes,
        )

        n_nodes_cumsum = torch.cumsum(
            n_nodes.reshape(
                -1,
            ),
            dim=0,
        )
        idx_lim_Gnodes = torch.zeros((n_nodes.shape[0], 2), dtype=torch.int64)
        idx_lim_Gnodes[1:, 0] = n_nodes_cumsum[:-1]
        idx_lim_Gnodes[:, 1] = n_nodes_cumsum

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        spc_x = self.spc_node_encoder(spc_x)
        spc_edge_attr = self.spc_edge_encoder(spc_edge_attr)

        x_new = torch.cat((x, self.eigenbroadcasting(V, spc_x, idx_lim_Gnodes)), dim=1)
        spc_x = torch.cat((spc_x, self.eigenpooling(V, x, idx_lim_Gnodes)), dim=1)
        x = x_new

        # message passing
        for i in range(self.num_MP_layers):
            x_i, edge_attr = self.processor[i](x, edge_index, edge_attr, del_ext_force)
            spc_x_i, spc_edge_attr = self.spc_processor[i](
                spc_x, spc_edge_index, spc_edge_attr, del_ext_force=None
            )

            x = x[:, 0 : x_i.shape[1]]
            spc_x = spc_x[:, 0 : x_i.shape[1]]
            x_new = torch.cat(
                (x_i, self.eigenbroadcasting(V, spc_x, idx_lim_Gnodes)), dim=1
            )
            spc_x = torch.cat((spc_x_i, self.eigenpooling(V, x, idx_lim_Gnodes)), dim=1)
            x = x_new

        return self.node_decoder(x)

    def eigenpooling(self, V, x, idx_lim_Gnodes):
        spx_pooled = torch.empty((0, x.shape[1])).to(x.device)

        for ii in range(idx_lim_Gnodes.shape[0]):
            idx1 = idx_lim_Gnodes[ii, 0]
            idx2 = idx_lim_Gnodes[ii, 1]
            spx_pooled_part = torch.matmul(torch.t(V[idx1:idx2, :]), x[idx1:idx2, :])
            spx_pooled = torch.cat((spx_pooled, spx_pooled_part), dim=0)

        return spx_pooled

    def eigenbroadcasting(self, V, spx, idx_lim_Gnodes):
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


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, dim_ext_force, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        """
        in_channels: dim of node embeddings, out_channels: dim of edge embeddings

        """
        self.edge_mlp = encoder(5 * in_channels, out_channels)
        self.node_mlp = encoder(3 * in_channels + dim_ext_force, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, del_ext_force, size=None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """

        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )  # out has the shape of [E, out_channels]

        if del_ext_force == None:
            updated_nodes = torch.cat([x, out], dim=1)
        else:
            updated_nodes = torch.cat(
                [x, out, del_ext_force], dim=1
            )  # Complete the aggregation through self-aggregation

        updated_nodes = self.node_mlp(updated_nodes)  # residual connection
        updated_nodes = (
            x[:, 0 : updated_nodes.shape[1]] + updated_nodes
        )  # residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        updated_edges = torch.cat(
            [x_i, x_j, edge_attr], dim=1
        )  # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges = self.edge_mlp(updated_edges) + edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0

        # out = scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce="sum")

        out = torch.zeros(2, dtype=updated_edges.dtype)
        out.scatter_reduce_(
            node_dim, edge_index[0, :], updated_edges, reduce="sum", include_self=False
        )

        return out, updated_edges
