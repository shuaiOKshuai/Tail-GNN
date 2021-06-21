import numpy as np
import scipy.sparse as sp
import torch
import collections
import sys, os
import pickle as pkl
import networkx as nx
from multiprocessing import Pool
from itertools import product
#from torch_geometric.datasets import CitationFull, Planetoid, Coauthor

folder = 'dataset/'

def normalize_adj(adj, norm_type=1, iden=False):
    # 1: mean norm, 2: spectral norm
    # add the diag into adj, namely, the self-connection. then normalization
    if iden:
        adj = adj + np.eye(adj.shape[0])       # self-loop
    if norm_type==1:
        D = np.sum(adj, axis=1)
        adjNor = adj / D
        adjNor[np.isinf(adjNor)] = 0.
    else:
        adj[adj > 0.0] = 1.0
        D_ = np.diag(np.power(np.sum(adj, axis=1), -0.5)) 
        adjNor = np.dot(np.dot(D_, adj), D_)
    
    return adjNor, adj 


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = np.where(rowsum==0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)    
    mx = r_mat_inv.dot(mx)

    return mx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = np.where(rowsum==0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    
    if sp.issparse(features):
        return features.todense()
    else:
        return features


def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def convert_to_torch_tensor(features, adj, tail_adj, labels, idx_train, idx_val, idx_test):
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = convert_sparse_tensor(adj)             #+ sp.eye(adj.shape[0]))
    tail_adj = convert_sparse_tensor(tail_adj)   #+ sp.eye(tail_adj.shape[0])    
    iden = sp.eye(adj.shape[0])
    iden = convert_sparse_tensor(iden)

    return features, adj, tail_adj, iden, labels, idx_train, idx_val, idx_test


def link_dropout(adj, idx, k=5):
    
    tail_adj = adj.copy()
    num_links = np.random.randint(k, size=idx.shape[0]) 
    num_links += 1

    for i in range(idx.shape[0]):
        index = tail_adj[idx[i]].nonzero()[1]
        new_idx = np.random.choice(index, num_links[i], replace=False)
        tail_adj[idx[i]] = 0.0
        for j in new_idx:
            tail_adj[idx[i], j] = 1.0
    return tail_adj


# split head vs tail nodes
def split_nodes(adj, k=5):

    num_links = np.sum(adj, axis=1)
    idx_train = np.where(num_links > k)[0]
        
    idx_valtest = np.where(num_links <= k)[0]
    np.random.shuffle(idx_valtest)

    p = int(idx_valtest.shape[0] / 3)
    idx_val = idx_valtest[:p]
    idx_test = idx_valtest[p:]

    return idx_train, idx_val, idx_test
   


def process_email(path, k=5):

    load_features = np.genfromtxt("{}graph.embeddings".format(path), skip_header=1, dtype=np.float32)
    idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
    features = idx[:,1:]

    load_labels = np.genfromtxt("{}labels.txt".format(path), dtype=np.int32)
    labels = encode_onehot(load_labels[:,1])

    edges = np.genfromtxt("{}edges.txt".format(path), dtype=np.int32)    
    #print(edges.shape[0])
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()    
    
    # remove self connection
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    # label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj, k=k)

    return features, adj, labels, (idx_train, idx_val, idx_test)



def process_squirrel(path, k=5):
    with open("{}node_feature_label.txt".format(path), 'rb') as f:
        clean_lines = (line.replace(b'\t',b',') for line in f)
        load_features = np.genfromtxt(clean_lines, skip_header=1, dtype=np.float32, delimiter=',')
    
    idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
    labels = encode_onehot(idx[:,-1])
    features = idx[:,1:-1]

    edges = np.genfromtxt("{}graph_edges.txt".format(path), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()
    
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj, k=k)
    features = preprocess_features(features)
    return features, adj, labels, (idx_train, idx_val, idx_test)


def process_actor(path, k=5):
    
    num_feat = 931
    def to_array(feat):
        new_feat = np.zeros(num_feat, dtype=float)
        for i in feat:
            new_feat[int(i)-1] = 1.
        return new_feat

    features = []
    labels = []

    with open("{}node_feature_label.txt".format(path), 'r') as f:
        f.readline()
        for line in f:
            idx, feat, label = line.strip().split('\t')
            feat = [n for n in feat.split(',')]
            
            labels.append(label)
            feat = to_array(feat)
            features.append(feat)        

    features = np.asarray(features)
    labels = encode_onehot(labels)
    labels = np.asarray(labels, dtype=int)

    edges = np.genfromtxt("{}graph_edges.txt".format(path), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    adj = adj.tolil()
    ind = np.where(adj.todense() > 1.0)
    for i in range(ind[0].shape[0]):
        adj[ind[0][i], ind[1][i]] = 1.

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()

    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    #label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj, k=k)
    print(idx_train.shape, idx_val.shape, idx_test.shape)

    features = preprocess_features(features)
    return features, adj, labels, (idx_train, idx_val, idx_test)


def process_cs(path, k=5):
    # citation edge file is symmetric already, preprocess for redundant edges

    if os.path.exists(path + 'feat.npy') and os.path.exists(path + 'adj.npz'):
        with open(path + 'feat.npy', 'rb') as f:
            idx = np.load(f)
            features = np.load(f)
            labels = np.load(f)
        adj = sp.load_npz(path + 'adj.npz')
    else:
        load_features = np.genfromtxt("{}graph.node".format(path), skip_header=1, dtype=np.float32)
        idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
        labels = encode_onehot(idx[:,1])
        features = idx[:,2:]

        #write to npy file
        with open(path + 'feat.npy', 'wb') as f:
            np.save(f, idx)
            np.save(f, features)
            np.save(f, labels)
        
        edges = np.genfromtxt("{}graph.edge".format(path), dtype=np.int32)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
        sp.save_npz(path + 'adj', adj)
    
    print('Done loading')

    adj = adj.tolil()
    #preprocess
    ind = np.where(adj.todense() > 1.0)
    for i in range(ind[0].shape[0]):
        adj[ind[0][i], ind[1][i]] = 1.
    
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    # label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj, k=k)
    features = preprocess_features(features)

    return features, adj, labels, (idx_train, idx_val, idx_test)


def process_amazon(path, k=5):
    # Load data files
    # Use a portion of 1M nodes

    if os.path.exists(path + 'feat.npy') and os.path.exists(path + 'adj.npz'):
        with open(path + 'feat.npy', 'rb') as f:
            features = np.load(f)
            labels = np.load(f)
        adj = sp.load_npz(path + 'adj.npz')
    else:
        feats = np.load(tf.io.gfile.GFile('{}/amazon2M-feats.npy'.format(path), 'rb')).astype(np.float32)
        G = json_graph.node_link_graph(json.load(tf.io.gfile.GFile('{}/amazon2M-G.json'.format(path))))
        
        id_map = json.load(tf.io.gfile.GFile('{}/amazon2M-id_map.json'.format(path)))
        is_digit = list(id_map.keys())[0].isdigit()
        id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
        
        class_map = json.load(tf.io.gfile.GFile('{}/amazon2M-class_map.json'.format(path)))
        is_instance = isinstance(list(class_map.values())[0], list)
        class_map = {(int(k) if is_digit else k): (v if is_instance else int(v)) for k, v in class_map.items()}
        
        print('Done loading')

        # Generate edge list
        edges = []
        for edge in G.edges():
            #if edge[0] < _nodes and edge[1] < _nodes:
            edges.append((id_map[edge[0]], id_map[edge[1]]))
        
        # Total Number of Nodes in the Graph
        _nodes = len(id_map)

        # All Edges in the Graph
        _edges = np.array(edges, dtype=np.int32)
        #print(_edges.shape[0])

        # Generate Labels
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            labels = np.zeros((_nodes, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], :] = np.array(class_map[k])
        else:
            num_classes = len(set(class_map.values()))
            labels = np.zeros((_nodes, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], class_map[k]] = 1
        

        def construct_adj(e, shape):
            adj = sp.csr_matrix((np.ones((e.shape[0]), dtype=np.float32), (e[:, 0], e[:, 1])), shape=shape)
            #adj += adj.transpose()
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            return adj.tolil()

        adj = construct_adj(_edges, (_nodes, _nodes))  
    
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(feats)
        features = scaler.transform(feats)
        
        with open(path + 'feat.npy', 'wb') as f:
            np.save(f, features)
            np.save(f, labels)
        sp.save_npz(path + 'adj', adj)
    

    print('create graph...')
    num = 1000000 
    adj = adj[:num, :num]
    features = features[:num]
    labels = labels[:num]

    graph = nx.from_scipy_sparse_matrix(adj)
    graph_cc = max(nx.connected_component_subgraphs(graph), key=len)
    
    print('remove nodes..')
    rm_node = np.setdiff1d(graph.nodes(), graph_cc.nodes())
    features = np.delete(features, rm_node)
    labels = np.delete(labels, rm_node)

    print('done')
    adj = nx.adjacency_matrix(graph_cc)

    with open(path + 'node_id.txt', 'w') as f:
        nodes = sorted(graph_cc.nodes())
        for node in nodes:
            f.write(str(node) + '\n')


    # label head tail for train/test
    idx_train, idx_val, idx_test = split_nodes(adj, k=k)
    print(idx_train.shape[0], idx_val.shape[0], idx_test.shape[0])
    return features, adj, labels, (idx_train, idx_val, idx_test)



def load_dataset(dataset, path=None, k=5):
    np.random.seed(0)
    if path == None:
        path = folder + dataset + '/'
    else:
        path = path + dataset + '/'

    DATASET = {
        'email': process_email,
        'cs-citation': process_cs,
        'squirrel': process_squirrel,
        'actor': process_actor,
        'amazon': process_amazon
    }   

    if dataset not in DATASET:
        return ValueError('Dataset not available')
    else:
        print('Preprocessing data ...')
        return DATASET[dataset](path=path, k=k)



if __name__ == "__main__":
    _, adj, _, _ = load_dataset('actor', path='../dataset/', k=5)
        