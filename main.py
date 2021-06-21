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
from models import TailGNN

#Get parse argument
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='actor', help='dataset')
parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
parser.add_argument("--eta", type=float, default=0.1, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--k", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')

parser.add_argument("--arch", type=int, default=1, help='1: gcn, 2: gat, 3: graphsage')
parser.add_argument("--seed", type=int, default=0, help='Random seed')
parser.add_argument("--epochs", type=int, default=1000, help='Epochs')
parser.add_argument("--patience", type=int, default=300, help='Patience')
parser.add_argument("--id", type=int, default=0, help='gpu ids')
parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')
parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
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


def train_embed(epoch, batch):
    embed_model.train()
    optimizer.zero_grad()
    
    embed_h, output_h, support_h  = embed_model(features, adj, True)
    embed_t, output_t, support_t  = embed_model(features, tail_adj, False)

    # loss    
    L_cls_h = F.nll_loss(output_h[batch], labels[batch])
    L_cls_t = F.nll_loss(output_t[batch], labels[batch])
    L_cls = (L_cls_h + L_cls_t)/2

    #weight regularizer
    m_h = normalize_output(support_h, batch)
    m_t = normalize_output(support_t, batch)

    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    errorG = criterion(prob_t[batch], t_labels)
    L_d = errorG
    L_all = L_cls - (args.eta * L_d) + args.mu * m_h 

    L_all.backward()
    optimizer.step()
    acc_train = metrics.accuracy(embed_h[batch], labels[batch])

    # validate:
    embed_model.eval()
    _, embed_val, _ = embed_model(features, adj, False)
    loss_val = F.nll_loss(embed_val[idx_val], labels[idx_val])
    acc_val = metrics.accuracy(embed_val[idx_val], labels[idx_val])

    return (L_all, L_cls, L_d), acc_train, loss_val, acc_val


def test():
    embed_model.eval()
    _, embed_test, _ = embed_model(features, adj, False)
    loss_test = F.nll_loss(embed_test[idx_test], labels[idx_test])
    
    acc_test = metrics.accuracy(embed_test[idx_test], labels[idx_test])
    f1_test = metrics.micro_f1(embed_test.cpu(), labels.cpu(), idx_test)

    log =   "Test set results: " + \
            "loss={:.4f} ".format(loss_test.item()) + \
            "accuracy={:.4f} ".format(acc_test.item()) + \
            "f1={:.4f}".format(f1_test.item())
            
    print(log) 
    return


features, adj, labels, idx = data_process.load_dataset(dataset, k=args.k)
features = torch.FloatTensor(features)
labels = np.argmax(labels,1)
labels = torch.LongTensor(labels)

tail_adj = data_process.link_dropout(adj, idx[0])
adj = torch.FloatTensor(adj.todense())
tail_adj = torch.FloatTensor(tail_adj.todense())

idx_train = torch.LongTensor(idx[0])
idx_val = torch.LongTensor(idx[1])
idx_test = torch.LongTensor(idx[2])

if args.dataset == 'email':
    idx_train = np.genfromtxt('dataset/' + args.dataset + '/train.csv')
    idx_test = np.genfromtxt('dataset/' + args.dataset + '/test.csv')
    idx_train = torch.LongTensor(idx_train-1)
    idx_test = torch.LongTensor(idx_test-1)

print("Data Processing done!")


r_ver = 1
nclass = labels.max().item() + 1

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
    labels = labels.cuda()
    adj = adj.cuda()
    tail_adj = tail_adj.cuda()


h_labels = torch.full((len(idx_train), 1), 1.0, device=device) 
t_labels = torch.full((len(idx_train), 1), 0.0, device=device) 


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

    L_d = train_disc(epoch, idx_train)
    L_d = train_disc(epoch, idx_train)

    Loss, acc_train, loss_val, acc_val = train_embed(epoch, idx_train)

    log =   'Epoch: {:d} '.format(epoch+1) + \
            'loss_train: {:.4f} '.format(Loss[0].item()) + \
            'loss_val: {:.4f} '.format(loss_val) + \
            'acc_train: {:.4f} '.format(acc_train) + \
            'acc_val: {:.4f} '.format(acc_val)
    print(log)

    #save best model 
    if acc_val >= best_acc: 
        acc_early_stop = acc_val
        loss_early_stop = loss_val
        epoch_early_stop = epoch

        torch.save(embed_model,os.path.join(save_path,'model.pt'))
        best_loss = np.min((loss_val, best_loss))
        print('Model saved!')
            
        best_acc = np.max((acc_val, best_acc))
        cur_step = 0
    else:
        cur_step += 1
        if cur_step == args.patience:
            early_stop= 'Early Stopping at epoch {:d} '.format(epoch) + \
                        'loss {:.4f} '.format(loss_early_stop) + \
                        'acc {:.4f}'.format(acc_early_stop)         
            print(early_stop)
            break


print("Training Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
print('Test ...')
embed_model = torch.load(os.path.join(save_path,'model.pt'))
test()


