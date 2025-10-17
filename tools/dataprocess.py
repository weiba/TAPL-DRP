import csv
import numpy as np
import pandas as pd
import torch
import io
from rdkit import Chem
import requests
import networkx as nx
import pickle


def exp_similarity(tensor: torch.Tensor, sigma: torch.Tensor, normalize=True):
    """
    :param tensor: an torch tensor
    :param sigma: scale parameter
    :param normalize: normalize or not
    :return: exponential similarity
    """
    if normalize:
        tensor = torch_z_normalized(tensor, dim=1)
    tensor_dist = torch_euclidean_dist(tensor, dim=0)
    exp_dist = torch.exp(-tensor_dist/(2*torch.pow(sigma, 2)))
    return exp_dist


# 计算样本欧式距离
def torch_euclidean_dist(tensor: torch.Tensor, dim=0):
    """
    :param tensor: a 2D torch tensor
    :param dim:
        0 : represent row
        1 : represent col
    :return: return euclidean distance
    """
    if dim:
        tensor_mul_tensor = torch.mm(torch.t(tensor), tensor)
    else:
        tensor_mul_tensor = torch.mm(tensor, torch.t(tensor))
    diag = torch.diag(tensor_mul_tensor)
    n_diag = diag.size()[0]
    tensor_diag = diag.repeat([n_diag, 1])
    diag = diag.view([n_diag, -1])
    dist = torch.sub(torch.add(tensor_diag, diag), torch.mul(tensor_mul_tensor, 2))
    # dist = torch.clamp(dist, min=0)  # 将负数值替换为0 zijijiade
    dist = torch.sqrt(dist)
    # dist = torch.masked_fill(dist, torch.isinf(dist) | torch.isnan(dist), 0)  # 将无效的距离值替换为0
    return dist


# ctl normalized
def torch_z_normalized(tensor: torch.Tensor, dim=0):
    """
    :param tensor: an 2D torch tensor
    :param dim:
        0 : normalize row data
        1 : normalize col data
    :return: Gaussian normalized tensor
    """
    mean = torch.mean(tensor, dim=1-dim)
    std = torch.std(tensor, dim=1-dim)
    if dim:
        tensor_sub_mean = torch.sub(tensor, mean)
        tensor_normalized = torch.div(tensor_sub_mean, std)
    else:
        size = mean.size()[0]
        tensor_sub_mean = torch.sub(tensor, mean.view([size, -1]))
        tensor_normalized = torch.div(tensor_sub_mean, std.view([size, -1]))
    return tensor_normalized



# 取师兄数据里的行索引得到735维的突变特征
def save_cell_mut_matrix(cellindex):
    f = open("PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    cell_dict = {}
    mut_dict = {}
    matrix_list = []
    for i in range(len(cellindex)):
        cell_dict[str(cellindex[i])] = i
    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col
        if cell_id in cell_dict:
            row = cell_dict[cell_id]
            if is_mutated == 1:
                matrix_list.append((row, col))
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))
    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1
    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    return cell_dict, cell_feature


def get_smiles(cid):
    """
    根据cids获取药物smiles
    Args:
        cids: 药物cids列表

    Returns:
        cids的smiles dict
    """
    drugsmiles = None
    api = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'
    url = api + ','.join(cid) + '/property/CanonicalSMILES/CSV'
    res = requests.get(url)
    tempf = io.StringIO(res.text)
    reader = csv.reader(tempf)
    next(reader)
    for row in reader:
        tempcid = row[0]
        smiles = row[1]
        smiles = smiles.strip('"')
        drugsmiles = smiles
    return drugsmiles


# 拼完整的药物指纹
def cat_tensor_with_drug(x: torch.Tensor, drug: torch.Tensor):
    drug = drug.repeat(x.shape[0], 1)
    return torch.cat((x, drug), dim=1)


#########
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def append_file(file_name, data):
    with open(file_name, 'a') as f:
        f.write(str(data)+'\n')
    f.close()

def get_tcga735matrix():
    tcga = pd.read_csv("tcga_mut.csv", index_col=0, header=0)
    tcgat = tcga.T
    # 索引里删除后三位 -01 -06
    tcgat = tcgat.rename(index=lambda x: x[:-3])

    tcga_pa_drug_bi = pd.read_csv("../data/TCGA/patient_drug_binary.csv", index_col=0, header=0)  # 402
    keeprows = tcga_pa_drug_bi.index.tolist()  #
    tcgat = tcgat[tcgat.index.isin(keeprows)]  # 372

    f = pd.read_csv("PANCANCER_Genetic_feature.csv", index_col=0, header=0)
    mutlist = list(set(f['genetic_feature'].tolist()))
    ######## tcga这里还没弄完
    keepcols = [item.rstrip('_mut') for item in mutlist]  # 这里没有算拷贝数变异的那些
    filtered_tcgat = tcgat.filter(items=keepcols)


def drugdataprocess(cellindex:list, drugcid):
    # file test
    ic50file = pd.read_csv("../data/GDSC/cell_drug.csv")  # cell-drug ic50
    datasanyuanzu = [] # drug, cell, ic50
    drugsmile = get_smiles(drugcid)  # 药物smile
    smilegraph = smile_to_graph(drugsmile)  # smilegraph
    cellnum, cellfeature = save_cell_mut_matrix(cellindex=cellindex)  # dict, list
    for i in cellindex:
        ic50 = ic50file.loc[i, drugcid]
        datasanyuanzu.append((drugcid, i, ic50))
    xd = []
    xc = []
    y = []
    for data in datasanyuanzu:
        drug, cell, ic50 = data
