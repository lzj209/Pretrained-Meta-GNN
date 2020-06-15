from model import GNN_graphpred
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from my_fake_model import empty_GNN_graphpred
class MAML(torch.nn.Module):
    def __init__(self, meta_lr, maml_lr, n_way, update_step, args):
        super(MAML, self).__init__()
        self.meta_lr = meta_lr
        self.maml_lr = maml_lr
        self.n_way = n_way
        self.update_step = update_step

        self.net = GNN_graphpred(args.num_layer, args.emb_dim, n_way, args.JK, args.drop_ratio, args.graph_pooling, args.gnn_type)
        self.net.from_pretrained(args.model_file)
        self.params = [[key, self.net.state_dict()[key]] for key in self.net.state_dict() if ('num_batches_tracked' not in key and "running_mean" not in key and "running_var" not in key)]
        self.bn_params = [[key, self.net.state_dict()[key]] for key in self.net.state_dict() if("running_mean" in key or "running_var" in key)]

        for x,y in self.params:
            if("running_mean" in x or "running_var" in x):
                continue
            y.requires_grad_(True)

        self.empty_net = empty_GNN_graphpred(args.num_layer, args.JK, args.drop_ratio, args.graph_pooling, args.gnn_type)
        self.optimizer = optim.Adam([y for x,y in self.params], lr=self.maml_lr)

    def forward(self, task_num, x_spt, x_qry, label_spt, label_qry_vec, label_qry_idx):
        creterion = nn.BCEWithLogitsLoss()
        right = [0 for _ in range(self.update_step-1)]
        total_loss = torch.zeros(1).double().cuda()

        for i in range(task_num):
            
            #cpnet = copy.deepcopy(self.net)    
            #optimizer = optim.Adam([{'params' : cpnet.gnn.parameters(), 'lr' : self.meta_lr},
            #                        {'params' : cpnet.graph_pred_linear.parameters(), 'lr' : self.meta_lr*5}], lr=self.meta_lr)
            #optimizer = optim.Adam(cpnet.parameters(), lr = self.meta_lr)
            self.empty_net.train()
            pred = self.empty_net(x_spt[i], self.params, self.bn_params)
            y = label_spt[i].view(pred.shape).to(torch.float64)
            loss = creterion(pred.double(), y)
            grad = torch.autograd.grad(loss, [y for x,y in self.params])
            fast_param = [[x[0],x[1]-y*self.meta_lr if 'graph_pred' in x[0] else x[1]-y*self.meta_lr*0.1] for x,y in zip(self.params, grad)]
            #fast_param = [[x[0],x[1]-y*self.meta_lr] for x,y in zip(self.params, grad)]
            for j in range(self.update_step -1):
                #cpnet.train()
                self.empty_net.train()
                pred = self.empty_net(x_spt[i], fast_param, self.bn_params)    
                y = label_spt[i].view(pred.shape).to(torch.float64)
                loss = creterion(pred.double(), y)
                grad = torch.autograd.grad(loss, [y for x,y in fast_param])
                fast_param = [[x[0],x[1]-y*self.meta_lr if 'graph_pred' in x[0] else x[1]-y*self.meta_lr*0.1] for x,y in zip(fast_param, grad)]

                with torch.no_grad():
                    self.empty_net.eval()
                    pred = self.empty_net(x_qry[i], fast_param, self.bn_params).argmax(dim=1)
                    right[j]+=((pred.long() == label_qry_idx[i].long()).sum())
            
            self.empty_net.train()
            pred = self.empty_net(x_qry[i], fast_param, self.bn_params)
            y = label_qry_vec[i].view(pred.shape).to(torch.float64)
            loss = creterion(pred.double(), y)
            total_loss += loss

        self.optimizer.zero_grad()
        total_loss /= task_num
        print(total_loss.item())
        total_loss.backward()
        self.optimizer.step()

        return right
            
    def test(self, x_spt, x_qry, label_spt, label_qry):
        #cpnet = copy.deepcopy(self.net)    
        #optimizer = optim.Adam([{'params' : cpnet.gnn.parameters(), 'lr' : self.meta_lr},
        #                        {'params' : cpnet.graph_pred_linear.parameters(), 'lr' : self.meta_lr*5}], lr=self.meta_lr)
        #optimizer = optim.Adam(cpnet.parameters(), lr = self.meta_lr)
        creterion = nn.BCEWithLogitsLoss()
        right = [0 for _ in range(self.update_step-1)]
        self.empty_net.train()
        pred = self.empty_net(x_spt, self.params, self.bn_params)
        y = label_spt.view(pred.shape).to(torch.float64)
        loss = creterion(pred.double(), y)
        grad = torch.autograd.grad(loss, [y for x,y in self.params]
)
        fast_param = [[x[0],x[1]-y*self.meta_lr if 'graph_pred' in x[0] else x[1]-y*self.meta_lr*0.1] for x,y in zip(self.params, grad)]
        #fast_param = [[x[0],x[1]-y*self.meta_lr] for x,y in zip(self.params, grad)]
        for j in range(self.update_step -1):
            self.empty_net.train()
            pred = self.empty_net(x_spt, fast_param, self.bn_params)    
            y = label_spt.view(pred.shape).to(torch.float64)
            loss = creterion(pred.double(), y)
            grad = torch.autograd.grad(loss, [y for x,y in fast_param])
            fast_param = [[x[0],x[1]-y*self.meta_lr if 'graph_pred' in x[0] else x[1]-y*self.meta_lr*0.1] for x,y in zip(fast_param, grad)]
            #fast_param = [[x[0],x[1]-y*self.meta_lr] for x,y in zip(fast_param, grad)]
            with torch.no_grad():
                self.empty_net.eval()
                pred = self.empty_net(x_qry, fast_param, self.bn_params).argmax(dim=1)
                right[j]+=((pred.long() == label_qry.long()).sum())
    
        #for j in range(self.update_step):
        #    cpnet.train()
        #    pred = cpnet(x_spt)    
        #    y = label_spt.view(pred.shape).to(torch.float64)
        #    optimizer.zero_grad()
        #    loss = creterion(pred.double(), y)
        #    loss.backward()
        #    optimizer.step()
        #    with torch.no_grad():
        #        cpnet.eval()
        #        pred  = cpnet(x_qry).argmax(dim=1)
        #        right[j]+=((pred.long() == label_qry.long()).sum())
            
        return right

