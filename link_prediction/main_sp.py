import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy as sp

import datetime, time
import os, sys
import collections, re
import argparse
sys.path.append('..')

import link_data_process
from utils import *
from layers import Discriminator
from models import TailGCN_SP, TailGAT_SP, TailSAGE_SP
from sklearn.metrics import average_precision_score, ndcg_score


#Get parse argument
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='cs-citation', help='dataset')
parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
parser.add_argument("--g_sigma", type=float, default=0.1, help='G deviation')
parser.add_argument("--eta", type=float, default=0.01, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
parser.add_argument("--k", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')

parser.add_argument("--arch", type=int, default=1, help='1: gcn, 2: gat, 3: graphsage')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--seed", type=int, default=0, help='Random seed')
parser.add_argument("--epochs", type=int, default=1000, help='Epochs')
parser.add_argument("--patience", type=int, default=200, help='Patience')
parser.add_argument("--id", type=int, default=1, help='gpu ids')
parser.add_argument("--ablation", type=int, default=0, help='ablation mode')

args = parser.parse_args()

cuda = torch.cuda.is_available()
criterion = nn.BCELoss()
neg_num = 9

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.id)

device = 'cuda' if cuda else 'cpu' 
dataset = args.dataset

save_path = 'saved_model/' + dataset 
if not os.path.exists(save_path):
    os.mkdir(save_path)

cur_time = datetime.datetime.now()
cur_time = cur_time.strftime("%d-%m-%Y_%H:%M:%S")

save_path = os.path.join(save_path, cur_time)  
if not os.path.exists(save_path):
    os.mkdir(save_path)


print(str(args))


def train_disc(epoch):
    disc.train()
    optimizer_D.zero_grad()

    embed_h, norm1, norm2 = embed_model(features, adj, True, adj_self, norm_adj_self)
    embed_t, _ , _ = embed_model(features, tail_adj, False, tail_adj_self, norm_tail_adj_self)

    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    # loss
    errorD = criterion(prob_h[idx_train], h_labels)
    errorG = criterion(prob_t[idx_train], t_labels)
    L_d = (errorD + errorG)/2 

    L_d.backward()
    optimizer_D.step()
    return L_d


def train_embed(epoch):
   
    embed_model.train()
    optimizer.zero_grad()
    
    embed_h, norm1, norm2 = embed_model(features, adj, True, adj_self, norm_adj_self)
    embed_t, _ , _ = embed_model(features, tail_adj, False, tail_adj_self, norm_tail_adj_self)

    base_embed_h = embed_h[train_base_nodes]
    pos_embed_h = embed_h[train_pos_nodes]
    neg_embed_h = embed_h[train_neg_nodes]

    dot_pos_h = torch.sum(base_embed_h * pos_embed_h, dim=1)
    dot_neg_h = torch.sum(base_embed_h * neg_embed_h, dim=1)

    base_embed_t = embed_t[train_base_nodes]
    pos_embed_t = embed_t[train_pos_nodes]
    neg_embed_t = embed_t[train_neg_nodes]

    dot_pos_t = torch.sum(base_embed_t * pos_embed_t, dim=1)
    dot_neg_t = torch.sum(base_embed_t * neg_embed_t, dim=1)


    loss_h = -(torch.mean(F.logsigmoid(dot_pos_h - dot_neg_h), dim=0, keepdim=True))
    loss_t = -(torch.mean(F.logsigmoid(dot_pos_t - dot_neg_t), dim=0, keepdim=True))
    loss_link = (loss_h + loss_t)/2


    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    errorD = criterion(prob_h[idx_train], h_labels)
    errorG = criterion(prob_t[idx_train], t_labels)
    L_d = errorG
  
    norm = torch.mean(norm1[idx_train]) + torch.mean(norm2[idx_train])
    L_all = loss_link - (args.eta * L_d) + args.mu * norm

    L_all.backward()
    optimizer.step()

    #Validate
    embed_model.eval()
    embed_val, _, _ = embed_model(features, adj, False, adj_self, norm_adj_self)
    base_embed = embed_val[val_base_nodes]
    
    pred_list = torch.zeros(val_base_nodes.shape[0], 0, device=device)
    for i in range(val_rank_nodes.shape[0]):
        rank_embed = embed_val[val_rank_nodes[i]]
        dot = torch.sum(base_embed * rank_embed, dim=1)
        score = torch.sigmoid(dot).view(-1,1)
        pred_list = torch.cat((pred_list, score), dim=1)
    pred_list = pred_list.cpu().detach().numpy()

    sum_map = 0
    sum_ndcg = 0
    for i in range(idx_val.shape[0]):
        AP = average_precision_score(labels, pred_list[i])
        NDCG = ndcg_score([labels], [pred_list[i]])
        sum_map += AP
        sum_ndcg += NDCG

    MAP = sum_map / idx_val.shape[0]
    NDCG = sum_ndcg / idx_val.shape[0]

    return (L_all, loss_link, L_d), MAP, NDCG


def test():
    embed_model.eval()
    embed_test, _, _ = embed_model(features, adj, False, adj_self, norm_adj_self)
    base_embed = embed_test[test_base_nodes]
    
    pred_list = torch.zeros(test_base_nodes.shape[0], 0, device=device)
    for i in range(rank_nodes.shape[0]):
        rank_embed = embed_test[rank_nodes[i]]
        dot = torch.sum(base_embed * rank_embed, dim=1)
        score = torch.sigmoid(dot).view(-1,1)
        pred_list = torch.cat((pred_list, score), dim=1)
    pred_list = pred_list.cpu().detach().numpy()

    sum_map = 0
    sum_ndcg = 0
    for i in range(idx_test.shape[0]):
        AP = average_precision_score(labels, pred_list[i])
        NDCG = ndcg_score([labels], [pred_list[i]])
        sum_map += AP
        sum_ndcg += NDCG

    MAP = sum_map / idx_test.shape[0]
    NDCG = sum_ndcg / idx_test.shape[0]

    log =   "MAP={:.4f} ".format(MAP) + \
            "NDCG={:.4f}".format(NDCG)
    print(log) 




features, adj, tail_adj, gt, idx, _ = link_data_process.load_dataset(dataset, k=args.k)

features = torch.FloatTensor(features)
gt = torch.LongTensor(gt)


adj_self = adj + sp.sparse.eye(adj.shape[0])
tail_adj_self = tail_adj + sp.sparse.eye(adj.shape[0])

new_adj = torch.FloatTensor(adj.todense())

idx_train = torch.LongTensor(idx[0])
idx_val = torch.LongTensor(idx[1])
idx_test = torch.LongTensor(idx[2])

# Training Sampling  
train_base_nodes = gt[idx_train, 0]
train_pos_nodes = gt[idx_train, 1]

neg_nodes = []
for node in train_base_nodes:
    neighbor = np.where(new_adj[node] == 0)
    neg = np.random.choice(neighbor[0], 1) 
    neg_nodes.append(neg)   
train_neg_nodes = np.asarray(neg_nodes).reshape(-1)


# Validate Sampling
val_base_nodes = gt[idx_val, 0]
val_pos_nodes = gt[idx_val, 1]

val_rank_nodes = []
for i in range(val_base_nodes.shape[0]):
    neighbor = np.where(new_adj[val_base_nodes[i]] == 0) #.nonzero(as_tuple=False)
    neg = np.random.choice(neighbor[0], neg_num, replace=False)
    nodes = np.insert(neg, 0, val_pos_nodes[i])
    val_rank_nodes.append(nodes)   
val_rank_nodes = np.transpose(np.asarray(val_rank_nodes))


# Testing Sampling
test_base_nodes = gt[idx_test, 0]
test_pos_nodes = gt[idx_test, 1]

rank_nodes = []
for i in range(test_base_nodes.shape[0]):
    neighbor = np.where(new_adj[test_base_nodes[i]] == 0) #.nonzero(as_tuple=False)
    neg = np.random.choice(neighbor[0], neg_num, replace=False)
    nodes = np.insert(neg, 0, test_pos_nodes[i])
    rank_nodes.append(nodes)   
rank_nodes = np.transpose(np.asarray(rank_nodes))


adj = data_process.normalize(adj)
tail_adj = data_process.normalize(tail_adj)
adj = data_process.convert_sparse_tensor(adj) #torch.FloatTensor(adj.todense())
tail_adj = data_process.convert_sparse_tensor(tail_adj) #torch.FloatTensor(tail_adj.todense()) 

if args.arch == 2:
    adj_self = torch.FloatTensor(adj_self.todense())
    tail_adj_self = torch.FloatTensor(tail_adj_self.todense())
    adj_self = adj_self.nonzero(as_tuple=False).t()
    tail_adj_self = tail_adj_self.nonzero(as_tuple=False).t()
    norm_adj_self = None
    norm_tail_adj_self = None

else:
    adj_self = data_process.convert_sparse_tensor(adj_self)
    tail_adj_self = data_process.convert_sparse_tensor(tail_adj_self)

    norm_adj_self = torch.unsqueeze(torch.sparse.sum(adj_self, dim=1).to_dense(), 1)
    norm_tail_adj_self = torch.unsqueeze(torch.sparse.sum(tail_adj_self, dim=1).to_dense(), 1)

#print(idx_train.shape, idx_val.shape, idx_test.shape)
print("Data Processing done!")


r_ver = 1
if features.shape[1] > 1000:
    r_ver = 2
nclass = 16


# Model and optimizer
if args.arch == 1:
    embed_model = TailGCN_SP(nfeat=features.shape[1],
            nclass=nclass,
            params=args,
            device=device,
            ver=r_ver,
            ablation=args.ablation)

elif args.arch == 2:
    embed_model = TailGAT_SP(nfeat=features.shape[1],
            nclass=nclass,
            params=args,
            device=device,
            ver=r_ver,
            ablation=args.ablation)

else:
    embed_model = TailSAGE_SP(nfeat=features.shape[1],
            nclass=nclass,
            params=args,
            device=device,
            ver=r_ver,
            ablation=args.ablation)

    
optimizer = optim.Adam(embed_model.parameters(),
                    lr=args.lr, weight_decay=args.lamda)

disc = Discriminator(nclass)
optimizer_D = optim.Adam(disc.parameters(),
                    lr=args.lr, weight_decay=args.lamda)


if cuda:
    #embed_model.cuda()
    disc.cuda()
    #features = features.cuda()
    #labels = labels.cuda()


labels = np.zeros(neg_num+1)
labels[0] = 1

h_labels = torch.full((len(idx_train), 1), 1.0, device=device) 
t_labels = torch.full((len(idx_train), 1), 0.0, device=device) 

best_map = 0.0
map_early_stop = 0.0
best_loss = 10000.0
loss_early_stop = 0.0
epoch_early_stop = 0
cur_step = 0

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    t = time.time()

    L_d = train_disc(epoch)
    L_d = train_disc(epoch)

    Loss, map_val, ndcg_val = train_embed(epoch)

    
    log =   'Epoch: {:d} '.format(epoch+1) + \
            'loss_link: {:.4f} '.format(Loss[1].item()) + \
            'loss_d: {:.4f} '.format(L_d) + \
            'loss_all: {:.4f} '.format(Loss[0].item()) + \
            'MAP = {:.4f} '.format(map_val) + \
            'NDCG = {:.4f} '.format(ndcg_val)
            #'time: {:.4f}s'.format(time.time() - t))
    print(log) 

    #save best model 
    if map_val >= best_map:  
        map_early_stop = map_val
        epoch_early_stop = epoch

        torch.save(embed_model,os.path.join(save_path, 'model.pt'))
        print('Model saved!') 
        #test()

        best_map = np.max((map_val, best_map))
        cur_step = 0
    else:
        cur_step += 1
        if cur_step == args.patience:
            early_stop= 'Early Stopping at epoch {:d}'.format(epoch) + \
                        'acc {:.4f}'.format(map_early_stop)         

            print(early_stop)
            break


print("Training Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
print('Testing ...')
embed_model = torch.load(os.path.join(save_path,'model.pt'))
test()
