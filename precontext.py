import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import os
import math
import random
import torch
import torch.optim as optim
import torch_geometric.data as DATA
from torch_geometric.data import Batch, InMemoryDataset
from tools.model import *
from drugmodels.ginconv import GINConvNet
from sklearn.metrics import roc_auc_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def graph_data_obj_to_nx_simple(data):
    G = nx.Graph()
    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx)
        pass

    edge_index = data.edge_index.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)
    return G

def nx_to_graph_data_obj_simple(G):
    """
    NX->Pyg
    """
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        # atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_feature = node['atom_num_idx']
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        # edge_features_list = []
        for i, j, edge in G.edges(data=True):
            # edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            # edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            # edge_features_list.append(edge_feature)
 
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # edge_attr = torch.tensor(np.array(edge_features_list),
        #                          dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    # data = DATA.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = DATA.Data(x=x, edge_index=edge_index)
    return data

def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        self.k = k
        self.l1 = l1
        self.l2 = l2
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1
 
    def __call__(self, data, root_idx=None):
        data.x_context = None
        num_atoms = data.x.size()[0]
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]
        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_index_context = context_data.edge_index
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            data.overlap_context_substruct_idx = torch.tensor(context_substruct_overlap_idxes_reorder)
 
        return data
 
    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


class BatchSubstructContext(DATA.Data):
    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch 
 
    @staticmethod
    def from_data_list(data_list):
        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_index_context", "x_context"]
        for key in keys:
            batch[key] = []
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []
        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0
        i = 0
        for data in data_list:
            if hasattr(data, "x_context") & data.x_context!=None:
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))
                for key in ["center_substruct_idx",  "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)
                for key in ["overlap_context_substruct_idx",  "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)
                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1
        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)
        return batch.contiguous()
    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0
    def cumsum(self, key, item):
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]
    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1

class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)

def pretrain(model, context_model, loader, optimizer_substruct, optimizer_context, criterion):
    model.train()
    context_model.train()
    balanced_loss_accum = 0
    acc_accum = 0
    auc_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        subdata = DATA.Data().to(device)
        subdata.x,subdata.edge_index = batch.x_substruct.float(), batch.edge_index_substruct
        substruct_rep = model(subdata)
        substruct_rep = substruct_rep[batch.center_substruct_idx]
        contextdata = DATA.Data().to(device)
        contextdata.x, contextdata.edge_index = batch.x_context.float(), batch.edge_index_context
        overlapped_node_rep = context_model(contextdata)
        overlapped_node_rep = overlapped_node_rep[batch.overlap_context_substruct_idx]
        expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) 
                                    for i in range(len(substruct_rep))], dim = 0)
        pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim = 1)
        shifted_expanded_substruct_rep = []        
        neg_samples = 1
        for i in range(neg_samples):
            shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), i+1)]
            shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat(
                    (batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))
        shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
        pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((neg_samples, 1)), dim = 1)
        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()
        loss = loss_pos + neg_samples*loss_neg
        loss.backward()
        optimizer_substruct.step()
        optimizer_context.step()
        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        # acc_accum += 0.5* (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) 
        #                     + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))
        auc_accum += roc_auc_score(torch.cat((torch.ones(len(pred_pos)).detach().cpu(), torch.zeros(len(pred_neg)).detach().cpu()),dim=0).numpy(),
                                   torch.cat((pred_pos.detach().cpu(), pred_neg.detach().cpu()),dim=0).numpy())

    return balanced_loss_accum/step, auc_accum/step


def drug_pretrain(datalist):
    model = GINConvNet(input_dim=datalist[0].x.shape[1], pretrain_flag=True).to(device)
    context_model = GINConvNet(input_dim=datalist[0].x.shape[1], pretrain_flag=True).to(device)
    optimizer_substruct = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer_context = optim.Adam(context_model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    num_epochs = 100
    log_loss = []
    log_acc = []
    transform = ExtractSubstructureContextPair(2,1,7)
    transformdatalist = []
    for g in datalist:
        tempdata = transform(g)
        transformdatalist.append(tempdata)
    data_list_filtered = [data for data in transformdatalist if 'x_context' in data]
    dataset = Batch.from_data_list(data_list_filtered)
    loader = DataLoaderSubstructContext(dataset=dataset)
    tolerance = 0
    max_tolerance = 10
    min_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_acc = pretrain(model,context_model,loader,optimizer_substruct,optimizer_context,criterion)
        if train_loss < min_loss:
            min_loss = train_loss
            tolerance = 0
        else:
            tolerance +=1
        if tolerance>=max_tolerance:
            print("Early stopping triggered. Training stopped.")
            break
    return model.state_dict()

def main_precontext():
    drug_encoder_dict_pth = os.path.join('result', 'drug_encoder.pth')
    file_path = os.path.join('data', 'smile_inchi326.csv')
    if os.path.exists(drug_encoder_dict_pth):
        print("pretrain done")
    else:
        drug_smiles_df = pd.read_csv(file_path, index_col=0)
        smiles = drug_smiles_df['smiles'].tolist()
        drug_pyg_list = []
        for smile in smiles:
            _, x, edge_index = smile_to_graph(smile)
            x = torch.tensor(np.array(x), device=device).float()
            edge_index = torch.tensor(edge_index, device=device).t()
            temp_pyg = DATA.Data(
                x=x,
                edge_index=edge_index
            )
            drug_pyg_list.append(temp_pyg)
        drug_encoder_dict = drug_pretrain(drug_pyg_list)
        torch.save(drug_encoder_dict, drug_encoder_dict_pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('drug pretrain')
    parser.add_argument('--dataset', dest='dataset', type=str, default='./data/smile_inchi326.csv', help='choose the .csv file to train drug model')
    parser.add_argument('--out', dest='out', type=str, default='./result/drug_encoder.pth', help='filename to save drug encoder pth')
    args = parser.parse_args()
    drug_encoder_dict_pth = args.out
    file_path = args.dataset
    if os.path.exists(drug_encoder_dict_pth):
        print("pretrain done")
    else:
        drug_smiles_df = pd.read_csv(file_path, index_col=0)
        smiles = drug_smiles_df['smiles'].tolist()
        drug_pyg_list = []
        for smile in smiles:
            _, x, edge_index = smile_to_graph(smile)
            x = torch.tensor(np.array(x), device=device).float()
            edge_index = torch.tensor(edge_index, device=device).t()
            temp_pyg = DATA.Data(
                x=x,
                edge_index=edge_index
            )
            drug_pyg_list.append(temp_pyg)
        drug_encoder_dict = drug_pretrain(drug_pyg_list)
        torch.save(drug_encoder_dict, drug_encoder_dict_pth)
