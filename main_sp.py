import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy as sp
import datetime, time
import collections, re
import os, argparse

from utils import *
from layers import Discriminator
from models import TailGCN_SP, TailGAT_SP, TailSAGE_SP


#Get parse argument
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='cs-citation', help='dataset')
parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
parser.add_argument("--eta", type=float, default=0.01, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--k", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')

parser.add_argument("--arch", type=int, default=1, help='1: gcn, 2: gat, 3: graphsage')
parser.add_argument("--seed", type=int, default=0, help='Random seed')
parser.add_argument("--epochs", type=int, default=1000, help='Epochs')
parser.add_argument("--patience", type=int, default=200, help='Patience')
parser.add_argument("--id", type=int, default=1, help='gpu ids')
parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
parser.add_argument("--g_sigma", type=float, default=0.1, help='G deviation')

args = parser.parse_args()

cuda = torch.cuda.is_available()
criterion = nn.BCELoss()

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
    #L_cls = F.nll_loss(F.softmax(embed_h[idx_train], dim=1), labels[idx_train]) + F.nll_loss(F.softmax(embed_t[idx_train], dim=1), labels[idx_train])
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


    # loss
    L_cls_h = F.nll_loss(F.log_softmax(embed_h[idx_train], dim=1), labels[idx_train])
    L_cls_t = F.nll_loss(F.log_softmax(embed_t[idx_train], dim=1), labels[idx_train])
    L_cls = (L_cls_h + L_cls_t)/2
    

    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    errorD = criterion(prob_h[idx_train], h_labels)
    errorG = criterion(prob_t[idx_train], t_labels)
    L_d = errorG
    

    norm = torch.mean(norm1[idx_train]) + torch.mean(norm2[idx_train])

    L_all = L_cls - (args.eta * L_d) + args.mu * norm
    #L_all = L_cls + (lambda_d * L_g) + mu * norm

    L_all.backward()
    optimizer.step()
    acc_train = metrics.accuracy(embed_h[idx_train], labels[idx_train])

    # validate:
    embed_model.eval()
    embed_val, _, _ = embed_model(features, adj, False, adj_self, norm_adj_self)

    loss_val = F.nll_loss(F.log_softmax(embed_val[idx_val], dim=1), labels[idx_val])
    acc_val = metrics.accuracy(embed_val[idx_val], labels[idx_val])

    return (L_all, L_cls, L_d), acc_train, loss_val, acc_val


def test():
    embed_model.eval()
    embed_test, _, _ = embed_model(features, adj, False, adj_self, norm_adj_self)
    
    loss_test = F.nll_loss(F.log_softmax(embed_test[idx_test], dim=1), labels[idx_test])
    acc_test = metrics.accuracy(embed_test[idx_test], labels[idx_test])
    f1_test = metrics.micro_f1(embed_test.cpu(), labels.cpu(), idx_test)

    log = "Test set results:" + \
            "loss= {:.4f} ".format(loss_test.item()) + \
            "accuracy={:.4f} ".format(acc_test.item()) + \
            "f1={:.4f}".format(f1_test.item()) 
    print(log)
    '''
    with open('our' + dataset + '_embed.npy' , 'wb') as f:
        np.save(f, embed_test)
    '''
    return




features, adj, labels, idx = data_process.load_dataset(dataset, k=args.k)
features = torch.FloatTensor(features)

path = 'dataset/' + args.dataset + '/'
if os.path.exists(path + 'tail_adj.npz'):
    tail_adj = sp.sparse.load_npz(path + 'tail_adj.npz')
    tail_adj = tail_adj.tolil()
else:
    tail_adj = data_process.link_dropout(adj, idx[0])
    sp.sparse.save_npz(path + 'tail_adj', tail_adj.tocoo())


labels = np.argmax(labels, 1)
labels = torch.LongTensor(labels)

adj_self = adj + sp.sparse.eye(adj.shape[0])
tail_adj_self = tail_adj + sp.sparse.eye(adj.shape[0])

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

idx_train = torch.LongTensor(idx[0])
idx_val = torch.LongTensor(idx[1])
idx_test = torch.LongTensor(idx[2])
#print(idx_train.shape, idx_val.shape, idx_test.shape)

print("Data Processing done!")


r_ver = 1
if features.shape[1] > 1000:
    r_ver = 2
nclass = labels.max().item() + 1


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
    disc.cuda()
    labels = labels.cuda()


h_labels = torch.full((len(idx_train), 1), 1.0, device=device) 
t_labels = torch.full((len(idx_train), 1), 0.0, device=device) 

#writer.add_graph(embed_model, (features, adj, tail_adj))

best_acc = 0.0
best_loss = 10000.0
acc_early_stop = 0.0
loss_early_stop = 0.0
epoch_early_stop = 0
cur_step = 0


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    t = time.time()

    if args.ablation == 4:
        L_d = 0
    else:
        L_d = train_disc(epoch)
        L_d = train_disc(epoch)

    Loss, acc_train, loss_val, acc_val = train_embed(epoch)

    
    log =   'Epoch: {:d} '.format(epoch+1) + \
            'loss_cls: {:.4f} '.format(Loss[1].item()) + \
            'loss_all: {:.4f} '.format(Loss[0].item()) + \
            'loss_D: {:.4f} '.format(L_d) + \
            'train: {:.4f} '.format(acc_train) + \
            'val: {:.4f} '.format(acc_val)
    print(log) 

    #save best model 
    if acc_val >= best_acc: 
        acc_early_stop = acc_val
        loss_early_stop = loss_val
        epoch_early_stop = epoch
        cur_step = 0 
        torch.save(embed_model,os.path.join(save_path,'model.pt'))
        best_acc = np.max((acc_val, best_acc))
        curr_step = 0
        print('Model saved!')
        #test()
    else:
        cur_step += 1
        if cur_step == args.patience:
            early_stop= 'Early Stopping at epoch {:d}'.format(epoch) + \
                        'loss {:.4f}'.format(loss_early_stop) + \
                        'acc {:.4f}'.format(acc_early_stop)                    
            print(early_stop)   
            break


print("Training Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
print('Test ...')
embed_model = torch.load(os.path.join(save_path,'model.pt'))
test()


