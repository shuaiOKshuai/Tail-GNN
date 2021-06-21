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
from utils import metrics
from layers import Discriminator
from models import TailGNN
from sklearn.metrics import average_precision_score, ndcg_score


#Get parse argument
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='actor', help='dataset')
parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
parser.add_argument("--eta", type=float, default=1, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--k", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')


parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
parser.add_argument("--seed", type=int, default=83, help='Random seed')
parser.add_argument("--epochs", type=int, default=1000, help='Epochs')
parser.add_argument("--patience", type=int, default=300, help='Patience')
parser.add_argument("--id", type=int, default=0, help='gpu ids')
parser.add_argument("--arch", type=int, default=1, help='1: gcn, 2: gat, 3: graphsage')
parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')

args = parser.parse_args()

cuda = torch.cuda.is_available()
criterion = nn.BCELoss()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.id)

device = 'cuda' if cuda else 'cpu'
dataset = args.dataset

neg_num = 9

save_path = 'saved_model/' + dataset 
if not os.path.exists(save_path):
    os.mkdir(save_path)

cur_time = datetime.datetime.now()
cur_time = cur_time.strftime("%d-%m-%Y_%H:%M:%S")

save_path = os.path.join(save_path, cur_time)  
if not os.path.exists(save_path):
    os.mkdir(save_path)   

print(str(args))


def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))

    return sum_m 


def train_disc(epoch, batch):
    disc.train()
    optimizer_D.zero_grad()

    embed_h, _, _ = embed_model(features, adj, True)
    embed_t, _, _ = embed_model(features, tail_adj, False)

    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    # loss
    errorD = criterion(prob_h[batch], h_labels)
    errorG = criterion(prob_t[batch], t_labels)
    L_d = (errorD + errorG)/2 

    L_d.backward()
    optimizer_D.step()
    return L_d


features, adj, tail_adj, gt, idx, labels = link_data_process.load_dataset(dataset, k=args.k)

idx_train = torch.LongTensor(idx[0])
idx_val = torch.LongTensor(idx[1])
idx_test = torch.LongTensor(idx[2])
features = torch.FloatTensor(features)
gt = torch.LongTensor(gt)

labels = np.argmax(labels,1)
nclass = 16 


new_adj = adj.copy().todense()
adj = torch.FloatTensor(adj.todense())
tail_adj = torch.FloatTensor(tail_adj.todense())


# Training Sampling  
train_base_nodes = gt[idx_train, 0]
train_pos_nodes = gt[idx_train, 1]


neg_nodes = []
for node in train_base_nodes:
    neighbor = np.where(adj[node] == 0)
    neg = np.random.choice(neighbor[0], 1) 
    neg_nodes.append(neg)   
train_neg_nodes = np.asarray(neg_nodes).reshape(-1)


# Validate Sampling
val_base_nodes = gt[idx_val, 0]
val_pos_nodes = gt[idx_val, 1]
val_rank_nodes = []
for i in range(val_base_nodes.shape[0]):
    neighbor = np.where(adj[val_base_nodes[i]] == 0) #.nonzero(as_tuple=False)
    neg = np.random.choice(neighbor[0], neg_num, replace=False)
    nodes = np.insert(neg, 0, val_pos_nodes[i])
    val_rank_nodes.append(nodes)   
val_rank_nodes = np.transpose(np.asarray(val_rank_nodes))


# Testing Sampling
test_base_nodes = gt[idx_test, 0]
test_pos_nodes = gt[idx_test, 1]
rank_nodes = []
for i in range(test_base_nodes.shape[0]):
    neighbor = np.where(adj[test_base_nodes[i]] == 0) #.nonzero(as_tuple=False)
    neg = np.random.choice(neighbor[0], neg_num, replace=False)
    nodes = np.insert(neg, 0, test_pos_nodes[i])
    rank_nodes.append(nodes)   
rank_nodes = np.transpose(np.asarray(rank_nodes))

print("Data Processing done!")
r_ver = 2
'''
if features.shape[1] > 900:
    r_ver = 2
'''



# Model and optimizer
embed_model = TailGNN(nfeat=features.shape[1],
        nclass=nclass,
        params=args,
        device=device,
        ver=r_ver)
    
optimizer = optim.Adam(embed_model.parameters(),
                    lr=args.lr, weight_decay=args.lamda)

feat_disc = nclass
disc = Discriminator(feat_disc)
optimizer_D = optim.Adam(disc.parameters(),
                    lr=args.lr, weight_decay=args.lamda)


if cuda:
    embed_model = embed_model.cuda()
    disc = disc.cuda()
    features = features.cuda()
    adj = adj.cuda()
    tail_adj = tail_adj.cuda()


labels = np.zeros(neg_num+1)
labels[0] = 1
h_labels = torch.full((len(train_base_nodes), 1), 1.0, device=device) 
t_labels = torch.full((len(train_base_nodes), 1), 0.0, device=device) 

best_map = 0.0
map_early_stop = 0.0
best_loss = 10000.0
loss_early_stop = 0.0
epoch_early_stop = 0
cur_step = 0


def train_embed(epoch, batch):
    embed_model.train()
    optimizer.zero_grad()
    
    embed_h, output_h, support_h  = embed_model(features, adj, True)
    embed_t, output_t, support_t  = embed_model(features, tail_adj, False)

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

    #weight regularizer
    m_h = normalize_output(support_h, batch)
    m_t = normalize_output(support_t, batch)

    
    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    errorG = criterion(prob_t[batch], t_labels)
    L_d = errorG
    L_all = loss_link - (args.eta * L_d) + args.mu * m_h  

    L_all.backward()
    optimizer.step()
    
    #Validate
    embed_model.eval()
    embed_val, _, _ = embed_model(features, adj, False)
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
    sum_mrr = 0
    sum_hit1 = 0

    for i in range(idx_val.shape[0]):
        AP = average_precision_score(labels, pred_list[i])
        NDCG = ndcg_score([labels], [pred_list[i]])

        true = pred_list[i, 0]
        sort_list = np.sort(pred_list[i])[::-1]
        rank = int(np.where(sort_list == true)[0][0]) + 1
        sum_mrr += (1/rank)

        if pred_list[i, 0] == np.max(pred_list[i]):
            sum_hit1 += 1

        sum_map += AP
        sum_ndcg += NDCG

    H1 = sum_hit1 / idx_val.shape[0]
    MRR = sum_mrr / idx_val.shape[0]
    MAP = sum_map / idx_val.shape[0]
    NDCG = sum_ndcg / idx_val.shape[0]

    return (L_all, loss_link, L_d), MAP, NDCG, MRR, H1


def test():
    embed_model.eval()
    embed_test, _, _ = embed_model(features, adj, False)
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
    sum_mrr = 0
    sum_hit1 = 0
    for i in range(idx_test.shape[0]):
        AP = average_precision_score(labels, pred_list[i])
        NDCG = ndcg_score([labels], [pred_list[i]])

        true = pred_list[i, 0]
        sort_list = np.sort(pred_list[i])[::-1]
        rank = int(np.where(sort_list == true)[0][0]) + 1
        sum_mrr += (1/rank)

        if pred_list[i, 0] == np.max(pred_list[i]):
            sum_hit1 += 1

        sum_map += AP
        sum_ndcg += NDCG

    H1 = sum_hit1 / idx_test.shape[0]
    MRR = sum_mrr / idx_test.shape[0]
    MAP = sum_map / idx_test.shape[0]
    NDCG = sum_ndcg / idx_test.shape[0]

    log =   "MAP={:.4f} ".format(MAP) + \
            "NDCG={:.4f} ".format(NDCG) 
            
    print(log) 



# Train model
t_total = time.time()
for epoch in range(args.epochs):
    t = time.time()

    L_d = train_disc(epoch, train_base_nodes)
    L_d = train_disc(epoch, train_base_nodes)

    Loss, map_val, ndcg_val, mrr_val, h1_val = train_embed(epoch, train_base_nodes)

 
    log =   'Epoch: {:d} '.format(epoch+1) + \
            'loss_all: {:.4f} '.format(Loss[0].item()) + \
            'MAP = {:.4f} '.format(map_val) + \
            'NDCG = {:.4f} '.format(ndcg_val)

    
    print(log) 
    #save best model 
    if map_val >= best_map:  
        map_early_stop = map_val
        epoch_early_stop = epoch

        torch.save(embed_model,os.path.join(save_path, 'model.pt'))
        print('Model saved') 
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
print('Test ..')
embed_model = torch.load(os.path.join(save_path,'model.pt'))
test()


