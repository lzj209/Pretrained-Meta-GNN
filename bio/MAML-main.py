import torch
import torch.nn as nn
from loader import BioDataset
import argparse
import numpy as np
from true_MAML import MAML
import random
from torch_geometric.data import DataLoader
import tqdm
import pickle

function_dataset = []

def split_dataset_by_function(dataset):
    for i in range(40):
        idx = []    
        for j in range(len(dataset)):
            if dataset[j].go_target_downstream[i]:
                idx.append(j)
        function_dataset.append(dataset[torch.tensor(idx)])

function_train_dataset = []
function_query_dataset = []

def split_function_dataset(frac_train = 0.6, frac_query = 0.4):
    for i in range(40):
        num_mols = len(function_dataset[i])
        all_idx = list(range(num_mols))
        random.shuffle(all_idx)

        train_idx = all_idx[:int(frac_train * num_mols)]
        query_idx = all_idx[int(frac_train * num_mols):]

        function_train_dataset.append(function_dataset[i][torch.tensor(train_idx).long()])
        function_query_dataset.append(function_dataset[i][torch.tensor(query_idx).long()])

def get_data_and_label(data_list, n_ways):
    data = []
    label_vec = []
    label_idx = [] 
    for idx, graph in data_list:
        data.append(graph)
        midlabel = [0 for _ in range(n_ways)]
        midlabel[idx] = 1
        label_vec.extend(midlabel)
        label_idx.append(idx)

    mid_dataset = BioDataset('dataset/supervised', data_type='supervised', empty = True)
    mid_dataset.data, mid_dataset.slices = mid_dataset.collate(data)

    mid_loader = DataLoader(mid_dataset, batch_size = len(mid_dataset), shuffle=False)
    data = [x for x in mid_loader]
    data = data[0]
    return data, torch.Tensor(label_vec), torch.Tensor(label_idx)

def generate_train_sample(n_ways, k_shot, k_query):
    #selected_function = random.sample(range(0,20-n_ways), n_ways)
    selected_function = list(range(0,n_ways))   

    data_list = []
    for idx,num in enumerate(selected_function):
        mid_list = random.sample(list(range(len(function_train_dataset[num]))), k_shot)
        data_list.extend([[idx, function_train_dataset[num][mid]] for mid in mid_list])
    random.shuffle(data_list)
    
    train_data, train_label_vec, train_label_idx = get_data_and_label(data_list, n_ways)

    data_list = []
    for idx,num in enumerate(selected_function):
        mid_list = random.sample(list(range(len(function_query_dataset[num]))), k_query)
        data_list.extend([[idx, function_query_dataset[num][mid]] for mid in mid_list])
    random.shuffle(data_list)
    data_list = data_list[:k_query]
    query_data, query_label_vec, query_label_idx = get_data_and_label(data_list, n_ways)

    return [train_data, train_label_vec, query_data, query_label_vec, query_label_idx]

def generate_test_sample(n_ways, k_shot, k_query):
    selected_function = range(20-n_ways,20)
    data_list = []
    for idx,num in enumerate(selected_function):
        mid_list = random.sample(list(range(len(function_train_dataset[num]))), k_shot)
        data_list.extend([[idx, function_train_dataset[num][mid]] for mid in mid_list])
    random.shuffle(data_list)
    
    train_data, train_label_vec, train_label_idx = get_data_and_label(data_list, n_ways)

    data_list = []
    for idx,num in enumerate(selected_function):
        mid_list = random.sample(list(range(len(function_query_dataset[num]))), k_query)
        data_list.extend([[idx, function_query_dataset[num][mid]] for mid in mid_list])
    random.shuffle(data_list)
    data_list = data_list[:k_query]
    query_data, query_label_vec, query_label_idx = get_data_and_label(data_list, n_ways)

    return [train_data, train_label_vec, query_data, query_label_vec, query_label_idx]

def train(model, args, device):
    for epoch in range(args.epochs):
        print(str(epoch) +":")
        x_spt = []
        x_qry = []
        label_spt = []
        label_qry_vec = []
        label_qry_idx = []
        for j in range(args.task_num):
            templist = generate_train_sample(args.n_ways, args.k_shots, args.k_query)
            templist = [x.to(device) for x in templist]
            x_spt.append(templist[0])
            label_spt.append(templist[1])
            x_qry.append(templist[2])
            label_qry_vec.append(templist[3])
            label_qry_idx.append(templist[4])
        rights = model(args.task_num, x_spt, x_qry, label_spt, label_qry_vec, label_qry_idx)
        print([right.item() / (args.k_query * args.task_num) for right in rights])


def test(model, args, device):
    right = [0 for _ in range(args.update_step)]
    test_sum = 0
    for epoch in range(args.testepochs):
        templist = generate_test_sample(args.n_ways, args.k_shots, args.k_query)
        templist = [x.to(device) for x in templist]
        x_spt = templist[0]
        label_spt = templist[1]
        x_qry = templist[2]
        label_qry_vec = templist[3]
        label_qry_idx = templist[4]
        rights = model.test(x_spt, x_qry, label_spt, label_qry_idx)
        right = [x+y.item() for x,y in zip(right,rights)]
        test_sum += args.k_query

    print([righ / test_sum for righ in right]) 


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of META-pre-training-GNN of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # MAML settings
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs to train (default: 5000)')

    parser.add_argument('--testepochs', type=int, default=50, 
                        help='number of epochs to test (default: 500)')
    
    parser.add_argument('--meta_lr', type=float, default=0.05,
                        help='MAML inner learning rate (default: 0.01)')
    parser.add_argument('--maml_lr', type=float, default=5e-5,
                        help='outer learning rate (default: 0.001)')
    parser.add_argument('--n_ways', type=int, default=2,
                        help='ways to distinguish in MAML')
    parser.add_argument('--k_shots', type=int, default=2,
                        help="samples for each catergory in MAML")
    parser.add_argument('--k_query', type=int, default=6,
                        help="samples for query in each meta-dataset")
    parser.add_argument('--update_step', type=int, default=8,
                        help="optimize step for each meta-dataset")
    parser.add_argument('--task_num', type=int, default=5,
                        help="task_num for each maml step")
    
    #Model settings
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
    parser.add_argument('--model_file', type=str, default = 'model_gin/supervised_contextpred.pth', help='filename to read the pretrain model')
    parser.add_argument('--model_save_dir', type=str, default = 'model_MAML/', help='output file dir')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=209, help = "Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for running experiments.")

    args = parser.parse_args()
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    root_supervised = 'dataset/supervised'
    dataset = BioDataset(root_supervised, data_type='supervised')

    random.seed(args.seed)
    #split_dataset_by_function(dataset)

    global function_dataset

    indata_file = open("splitted_dataset.pkl", 'rb')
    function_dataset = pickle.load(indata_file)

    split_function_dataset()
    
    model = MAML(args.meta_lr, args.maml_lr, args.n_ways, args.update_step, args)
    model.to(device)
    model.params = [[x,y.to(device)] for x,y in model.params]
    model.bn_params = [[x,y.to(device)] for x,y in model.bn_params]
    model.empty_net.to(device) 
    train(model, args, device)

if __name__ == "__main__":
    main()
