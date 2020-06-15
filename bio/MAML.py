from model import GNN_graphpred
import torch
import torch.nn as nn
import torch.optim as optim
import copy
class MAML(torch.nn.Module):
    def __init__(self, meta_lr, maml_lr, n_way, update_step, args):
        super(MAML, self).__init__()
        self.meta_lr = meta_lr
        self.maml_lr = maml_lr
        self.n_way = n_way
        self.update_step = update_step

        self.net = GNN_graphpred(args.num_layer, args.emb_dim, n_way, args.JK, args.drop_ratio, args.graph_pooling, args.gnn_type)
        self.net.from_pretrained(args.model_file)

    def forward(self, task_num, x_spt, x_qry, label_spt, label_qry_vec, label_qry_idx):
        creterion = nn.BCEWithLogitsLoss()
        grad = None
        right = [0 for _ in range(self.update_step)]
        for i in range(task_num):
            cpnet = copy.deepcopy(self.net)    
            optimizer = optim.Adam([{'params' : cpnet.gnn.parameters(), 'lr' : 0},
                                    {'params' : cpnet.graph_pred_linear.parameters(), 'lr' : self.meta_lr}], lr=self.meta_lr)
            #optimizer = optim.Adam(cpnet.parameters(), lr = self.meta_lr)

            for j in range(self.update_step):
                cpnet.train()
                pred = cpnet(x_spt[i])    
                y = label_spt[i].view(pred.shape).to(torch.float64)
                optimizer.zero_grad()
                loss = creterion(pred.double(), y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    cpnet.eval()
                    pred  = cpnet(x_qry[i]).argmax(dim=1)
                    right[j]+=((pred.long() == label_qry_idx[i].long()).sum())
            
            cpnet.train()
            pred = cpnet(x_qry[i])
            y = label_qry_vec[i].view(pred.shape).to(torch.float64)
            optimizer.zero_grad()
            loss = creterion(pred.double(), y)
            midgrad = torch.autograd.grad(loss, cpnet.parameters())
            if grad==None:
                grad = midgrad
            else:
                grad += midgrad
    
        idx = 0
        for key in self.net.state_dict():
            if("running_mean" in key or "running_var" in key or "num_batches_tracked" in key):
                continue
            if(key == "graph_pred_linear.weight" or key == "graph_pred_linear.bias"):
                continue
            else:
                self.net.state_dict()[key] -= grad[idx]/task_num * self.maml_lr
            idx+=1
        return right
            
    def test(self, x_spt, x_qry, label_spt, label_qry):
        cpnet = copy.deepcopy(self.net)    
        optimizer = optim.Adam([{'params' : cpnet.gnn.parameters(), 'lr' : 0},
                                {'params' : cpnet.graph_pred_linear.parameters(), 'lr' : self.meta_lr}], lr=self.meta_lr)
        #optimizer = optim.Adam(cpnet.parameters(), lr = self.meta_lr)
        creterion = nn.BCEWithLogitsLoss()
        right = [0 for _ in range(self.update_step)]
        for j in range(self.update_step):
            cpnet.train()
            pred = cpnet(x_spt)    
            y = label_spt.view(pred.shape).to(torch.float64)
            optimizer.zero_grad()
            loss = creterion(pred.double(), y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cpnet.eval()
                pred  = cpnet(x_qry).argmax(dim=1)
                right[j]+=((pred.long() == label_qry.long()).sum())
            
        return right

