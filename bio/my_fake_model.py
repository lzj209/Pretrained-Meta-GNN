import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch import optim

class empty_GINConv(MessagePassing):

    def __init__(self, input_layer, aggr):
        super(empty_GINConv, self).__init__()
        # multi-layer perceptron
        #self.mlp = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))

        ### Mapping 0/1 edge features to embedding
        #self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        #if self.input_layer:
        #    self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
        #    torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, param, bn_param):

        #param include [mlp.0.weight', 'mlp.0.bias', 'mlp.1.weight', 'mlp.1.bias', 
        #              mlp.3.weight', 'mlp.3.bias', 'edge_encoder.weight', 'edge_encoder.bias', 
        #              'input_node_embedding.weight'(if input layer)]
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:,7] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = F.linear(edge_attr, param[6], param[7])
        #edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = F.embedding(x.to(torch.int64).view(-1,), param[8])
            #x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, param=param, bn_param=bn_param)

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim = 1)

    def update(self, aggr_out, param, bn_param):
        aggr_out = F.linear(aggr_out, param[0], param[1])
        aggr_out = F.batch_norm(aggr_out,bn_param[0],bn_param[1],param[2],param[3],self.training)
        aggr_out = F.relu(aggr_out)
        aggr_out = F.linear(aggr_out, param[4], param[5])
        return aggr_out

class empty_GNN(torch.nn.Module):

    def __init__(self, num_layer, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(empty_GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False
            self.gnns.append(empty_GINConv(aggr = "add", input_layer = input_layer))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr, param, bn_param):
        h_list = [x]
        for layer in range(self.num_layer):
            layer_name = "gnns."+str(layer)+"."
            layer_param = [y for x,y in param if layer_name in x]
            layer_bn_param = [y for x,y in bn_param if layer_name in x]
         
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, layer_param, layer_bn_param)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]

        return node_representation


class empty_GNN_graphpred(torch.nn.Module):
    def __init__(self, num_layer, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(empty_GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = empty_GNN(num_layer, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        #elif graph_pooling == "attention":
        #    self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        #self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)

    def forward(self, data, params, bn_params):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(x, edge_index, edge_attr, params[:-2], bn_params)

        pooled = self.pool(node_representation, batch)

        center_node_rep = node_representation[data.center_node_idx]

        graph_rep = torch.cat([pooled, center_node_rep], dim = 1)


        #return self.graph_pred_linear(graph_rep)
        return F.linear(graph_rep, params[-2][1], params[-1][1])
