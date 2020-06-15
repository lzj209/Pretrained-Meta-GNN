import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch import optim


class empty_GINConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(empty_GINConv, self).__init__()
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, param):
        #params=['mlp.0.weight', 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias', 'edge_embedding1.weight', 'edge_embedding2.weight']
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = F.embedding(edge_attr[:,0], param[-2]) + F.embedding(edge_attr[:,1], param[-1])
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, param=param)

    def message(self, x_j, edge_attr, param):
        return x_j + edge_attr

    def update(self, aggr_out,param):
        aggr_out = F.linear(aggr_out, param[0], param[1])
        aggr_out = F.relu(aggr_out)
        aggr_out = F.linear(aggr_out, param[2], param[3])
        return aggr_out



class empty_GNN(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(empty_GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(empty_GINConv(emb_dim, aggr = "add"))

    def forward(self, x, edge_index, edge_attr, param, bn_param):

        #param include[x_embedding1.weight', 'x_embedding2.weight', ['gnns.0], ['gnns.1'] ... batch_norms.weight, batch_norms.bias]
        #bn_param include 'batch_norms.0.running_mean', 'batch_norms.0.running_var'...]

        x = F.embedding(x[:,0], param[0][1]) + F.embedding(x[:,1], param[1][1])
        h_list = [x]
        for layer in range(self.num_layer):
            layer_name = "gnns."+str(layer)+"."
            layer_param = [y for x,y in param if layer_name in x]
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, layer_param)
            h = F.batch_norm(h, bn_param[2*layer][1], bn_param[2*layer+1][1], 
                    param[-2*(self.num_layer-layer)][1], param[-2*(self.num_layer-layer)+1][1], training=self.training)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class empty_GNN_graphpred(torch.nn.Module):
    
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(empty_GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = empty_GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool

    def forward(self, data, params, bn_params):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(x, edge_index, edge_attr, params[:-2], bn_params)

        return F.linear(self.pool(node_representation, batch), params[-2][1], params[-1][1])
