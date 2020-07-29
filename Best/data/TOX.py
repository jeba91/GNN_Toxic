import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
import csv

import dgl


import random
random.seed(42)

from sklearn.model_selection import StratifiedKFold, train_test_split


# *note
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']

def format_data(dataset):
    """
        Utility function to recover data,
        INTO-> dgl/pytorch compatible format
    """
    graphs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    for graph in graphs:
        graph.ndata['feat'] = torch.Tensor(graph.ndata['feat'].float()) # dgl 4.0
        graph.edata['feat'] = torch.Tensor(graph.edata['feat'].float()) # dgl 4.0

    return TOXFormDataset(graphs, labels)

def get_all_split_idx(dataset, k_splits = 3):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 10 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 10 fold have unique test set.
    """
    root_idx_dir = './data/TOX/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + dataset.name + '_train.index')):
        print("[!] Splitting the data into train/val/test ...")

        # Using 10-fold cross val to compare with benchmark papers
        # k_splits = 10

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)
        k_data_splits = []

        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)

        for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
            remain_index, test_index = indexes[0], indexes[1]

            remain_set = format_data([dataset[index] for index in remain_index])

            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                    range(len(remain_set.graph_lists)),
                                                    test_size=0.111,
                                                    stratify=remain_set.graph_labels,
                                                    random_state = 42)

            train, val = format_data(train), format_data(val)
            test = format_data([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train = open(root_idx_dir + dataset.name + '_train.index', 'a+')
            f_val = open(root_idx_dir + dataset.name + '_val.index', 'a+')
            f_test = open(root_idx_dir + dataset.name + '_test.index', 'a+')

            f_train_w = csv.writer(f_train)
            f_val_w = csv.writer(f_val)
            f_test_w = csv.writer(f_test)

            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")

        f_train.close()
        f_val.close()
        f_test.close()



    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + dataset.name + '_' + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx

class TOXFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


class TOXLoad(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_name, num_graphs):
        self.data_dir = data_dir
        self.num_graphs = num_graphs
        self.name = data_name

        with open(data_dir + "/%s.pickle" % data_name,"rb") as f:
            self.data = pickle.load(f)

        print(num_graphs)
        assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"

        """
        data is a list of Molecule dict objects with following attributes

          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, 'ALL'))

        for molecule in self.data:
            node_features = torch.tensor(molecule['atom_type'], dtype=torch.long)

            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(1,-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()

            g.add_nodes(molecule['num_atom'])
            if self.name in ["PBT_Rep2", "PBT_Repn2", "CMR_Rep2", "CM_Rep2", "R_Rep2"]:
                edges_dimension = 8
            else:
                edges_dimension = 4

            e_feats = torch.zeros(edge_features.shape[1], node_features.shape[1]+edges_dimension)
            e_feats[range(e_feats.shape[0]), node_features.shape[1]+edge_features-1] = 1

            g.ndata['feat'] = torch.cat([node_features, torch.zeros(node_features.shape[0],edges_dimension).long()],1).half()



            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())

            g.edata['feat'] = e_feats.half()

            self.graph_lists.append(g)
            self.graph_labels.append(int(molecule['label']))

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g

class TOXDataset(torch.utils.data.Dataset):
    def __init__(self, name, ksplits=10):
        t0 = time.time()
        self.name = name

        pbt = ['PBT_Rep1', 'PBT_Rep2', 'PBT_Rep3', 'PBT_Rep4']
        pbtn = ['PBT_Repn1', 'PBT_Repn2', 'PBT_Repn3', 'PBT_Repn4']
        cmr = ['CMR_Rep1', 'CMR_Rep2', 'CMR_Rep3', 'CMR_Rep4']

        data_dir = './data/TOX'

        if name in pbt:
            dataset = TOXLoad(data_dir, self.name, num_graphs=494)
        elif name in pbtn:
            dataset = TOXLoad(data_dir, self.name, num_graphs=971)
        elif name in cmr:
            dataset = TOXLoad(data_dir, self.name, num_graphs=652)

        print("[!] Dataset: ", self.name)

        # this function splits data into train/val/test and returns the indices
        self.all_idx = get_all_split_idx(dataset, ksplits)

        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in range(ksplits)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in range(ksplits)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in range(ksplits)]

        print("Time taken: {:.4f}s".format(time.time()-t0))

    def format_dataset(self, dataset):
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        for graph in graphs:
            graph.ndata['feat'] = torch.Tensor(graph.ndata['feat'].float()) # dgl 4.0
            graph.edata['feat'] = torch.Tensor(graph.edata['feat'].float()) # dgl 4.0

        return TOXFormDataset(graphs, labels)


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e


    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(10):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]

        for split_num in range(10):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)
