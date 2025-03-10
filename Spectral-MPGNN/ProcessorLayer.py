import torch
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing

# from torch_scatter import scatter


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, dim_ext_force, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(
            Linear(5 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        self.node_mlp = Sequential(
            Linear(3 * in_channels + dim_ext_force, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

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
