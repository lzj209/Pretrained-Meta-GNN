import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

import random

from true_MAML import MAML

datasets = [] 
dataset_name = ["tox21", "bace", "bbbp", "toxcast", "sider", "clintox", "hiv"]
goal_task = [7, 0, 0, 501, 9, 1, 0]
criterion = nn.BCEWithLogitsLoss(reduction = "none")

def get_datasets():
    for i in range(7):
        dataset = dataset_name[i]
        dataset = MoleculeDataset("dataset/" + dataset, dataset=dataset)
        datasets.append(dataset)

train_datasets = []
valid_datasets = []

final_test_dataset = None
def split_dataset(frac_train = 0.1, frac_test = 0.9):
    for i in range(6):
        dataset = datasets[i]
        smiles_list = pd.read_csv('dataset/' + dataset_name[i] + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, task_idx=goal_task[i], null_value=0, frac_train=frac_train,frac_valid=frac_test, frac_test=0)
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
    smiles_list = pd.read_csv('dataset/' + dataset_name[6] + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, _test_dataset = scaffold_split(datasets[6], smiles_list, task_idx=goal_task[6], null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    
    train_datasets.append(train_dataset)
    global final_test_dataset
    final_test_dataset = _test_dataset 

positive_ids = []
negitive_ids = []

def get_positive_and_negitive():
    for i in range(7):
        positive_id = []
        negitive_id = []
        for idx,data in enumerate(train_datasets[i]):
            if(data.y[goal_task[i]]==1):
                positive_id.append(idx)
            elif(data.y[goal_task[i]]==-1):
                negitive_id.append(idx)
        positive_ids.append(positive_id)
        negitive_ids.append(negitive_id)
        
def get_train_data(num_task, k_shot, k_query, device):
    selected_task = [random.randint(0,5) for _ in range(num_task)]
    x_spt = []
    x_qry = []
    for task in selected_task:
        spt_data = random.sample(positive_ids[task], k_shot) + random.sample(negitive_ids[task], k_shot)
        random.shuffle(spt_data)
        train_data = train_datasets[task][torch.tensor(spt_data)]
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        for batch in train_loader:
            batch=batch.to(device)
            x_spt.append(batch)
        qry_data = [random.randint(0,len(valid_datasets[task])-1) for _ in range(k_query)]
        query_data = valid_datasets[task][torch.tensor(qry_data)]
        query_loader = DataLoader(query_data, batch_size=len(query_data), shuffle=False)
        for batch in query_loader:
            batch=batch.to(device)
            x_qry.append(batch)
    return x_spt, x_qry, selected_task

def train(model, args, device, epoches):
    total_loss = 0
    for epoch in tqdm(range(epoches)):
        x_spt, x_qry, task_ids= get_train_data(args.task_num, args.k_shots, args.k_query, device)
        loss = model(x_spt, x_qry, task_ids)
        total_loss += loss
    total_loss/= epoches
    return total_loss

def eval(model, args, device):
    spt_data = [positive_ids[6][i] for i in range(args.k_shots)] + [negitive_ids[6][i] for i in range(args.k_shots)]
    random.shuffle(spt_data)
    train_data = train_datasets[6][torch.tensor(spt_data)]
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    x_spt = None
    for batch in train_loader:
        batch = batch.to(device)
        x_spt = batch 
    dataloader = DataLoader(final_test_dataset, batch_size=32, shuffle=False, num_workers = 4)
    acc = model.test(x_spt, dataloader, device)
    return acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    
    parser.add_argument('--k_shots', type=int, default=1,
                        help="samples for each catergory in MAML")
    parser.add_argument('--k_query', type=int, default=12,
                        help="samples for query in each meta-dataset")
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--maml_lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--update_step', type=int, default=6,
                        help="optimize step for each meta-dataset")
    parser.add_argument('--task_num', type=int, default=6,
                        help="task_num for each maml step")

    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--model_file', type=str, default = 'model_gin/supervised_contextpred.pth', help='filename to read the model (if there is any)')
    #parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    random.seed(209)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    model = MAML(args.meta_lr, args.maml_lr, args.update_step, args, device)
    #set up model
    model.to(device)
    model.net.to(device)
    model.params = [[x,y.to(device)] for x,y in model.params]
    model.bn_params = [[x,y.to(device)] for x,y in model.bn_params]
    model.empty_net.to(device)
    get_datasets()
    split_dataset()
    get_positive_and_negitive()

    fname = 'runs/MAML' + str(args.runseed) + '/' + 'MAML_result'
    if os.path.exists(fname):
        shutil.rmtree(fname)
        print("removed the existing file.")
    writer = SummaryWriter(fname)

    for i in range(args.epochs):
        print(i)
        loss = train(model, args, device, 100)
        print('train_loss:'+str(loss))
        acc = eval(model, args, device)
        print('test_auc:'+str(acc))
        writer.add_scalar('data/train loss', loss, i)
        writer.add_scalar('data/test auc', acc, i)
    
    writer.close()
    
'''


    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()
'''
if __name__ == "__main__":
    main()
