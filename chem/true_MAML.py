from model import GNN_graphpred
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from my_fake_model import empty_GNN_graphpred
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np 
from sklearn.metrics import roc_auc_score


goal_task = [7, 0, 0, 501, 9, 1, 0]
dataset_task_num = [12, 1, 1, 617, 27, 2, 1]
class MAML(torch.nn.Module):
    def __init__(self, meta_lr, maml_lr, update_step, args, device):
        super(MAML, self).__init__()
        self.meta_lr = meta_lr
        self.maml_lr = maml_lr
        self.update_step = update_step
        self.task_num = args.task_num
        self.k_shots = args.k_shots
        self.k_query = args.k_query

        self.net = GNN_graphpred(args.num_layer, args.emb_dim, 1, args.JK, args.drop_ratio, args.graph_pooling, args.gnn_type)
        self.net.from_pretrained(args.model_file)
        self.params = [[key, self.net.state_dict()[key]] for key in self.net.state_dict() if ('num_batches_tracked' not in key and "running_mean" not in key and "running_var" not in key)]
        self.bn_params = [[key, self.net.state_dict()[key]] for key in self.net.state_dict() if("running_mean" in key or "running_var" in key)]
        for x,y in self.params:
            y.requires_grad_(True)
            y.to(device)
        self.empty_net = empty_GNN_graphpred(args.num_layer, args.emb_dim, 1, args.JK, args.drop_ratio, args.graph_pooling, args.gnn_type)
        self.optimizer = optim.Adam([y for x,y in self.params], lr=self.maml_lr)

    def forward(self, x_spt, x_qry, task_ids):
        criterion = nn.BCEWithLogitsLoss()
        total_loss = torch.zeros(1).double().cuda()
        self.empty_net.train()
        for i in range(self.task_num):
            pred = self.empty_net(x_spt[i], self.params, self.bn_params)
            spt_y_idx = [j*dataset_task_num[task_ids[i]] + goal_task[task_ids[i]] for j in range(self.k_shots*2)]
            qry_y_idx = [j*dataset_task_num[task_ids[i]] + goal_task[task_ids[i]] for j in range(self.k_query)]
            y = x_spt[i].y[torch.tensor(spt_y_idx)].view(pred.shape).to(torch.float64)
            #Whether y is non-null or not.
            is_valid = y**2 > 1e-5
            #Loss matrix
            loss_mat = criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)

            grad = torch.autograd.grad(loss, [y for x,y in self.params])
            fast_param = [[x[0],x[1]-y*self.meta_lr] for x,y in zip(self.params, grad)]

            for j in range(self.update_step -1):
                #cpnet.train()
                pred = self.empty_net(x_spt[i], fast_param, self.bn_params) 
                y = x_spt[i].y[torch.tensor(spt_y_idx)].view(pred.shape).to(torch.float64)
                #Whether y is non-null or not.
                is_valid = y**2 > 1e-5
                #Loss matrix
                loss_mat = criterion(pred.double(), (y+1)/2)
                #loss matrix after removing null target
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat)/torch.sum(is_valid)
                grad = torch.autograd.grad(loss, [y for x,y in fast_param])
                fast_param = [[x[0],x[1]-y*self.meta_lr] for x,y in zip(fast_param, grad)]
            pred = self.empty_net(x_qry[i], fast_param, self.bn_params)

            y = x_qry[i].y[torch.tensor(qry_y_idx)].view(pred.shape).to(torch.float64)
            is_valid = y**2 > 1e-5
            loss_mat = criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)

            total_loss += loss

        self.optimizer.zero_grad()
        total_loss/=self.task_num
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
            
    def test(self, x_spt, dataloader, device):
        self.empty_net.train()
        criterion = nn.BCEWithLogitsLoss()
        pred = self.empty_net(x_spt, self.params, self.bn_params)
        y = x_spt.y.view(pred.shape).to(torch.float64)
        is_valid = y**2 > 1e-5
        loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        grad = torch.autograd.grad(loss, [y for x,y in self.params])
        fast_param = [[x[0],x[1]-y*self.meta_lr] for x,y in zip(self.params, grad)]
        for j in range(self.update_step -1):
            pred = self.empty_net(x_spt, fast_param, self.bn_params) 
            y = x_spt.y.view(pred.shape).to(torch.float64)
            #Whether y is non-null or not.
            is_valid = y**2 > 1e-5
            #Loss matrix
            loss_mat = criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            grad = torch.autograd.grad(loss, [y for x,y in fast_param])
            fast_param = [[x[0],x[1]-y*self.meta_lr] for x,y in zip(fast_param, grad)]

        self.empty_net.eval()
        y_true = []
        y_scores = []
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                pred = self.empty_net(batch, fast_param, self.bn_params)
            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

        return sum(roc_list)/len(roc_list) #y_true.shape[1]

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

