
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import json
import torch
from collections import OrderedDict
from rdkit import Chem
import numpy as np
import networkx as nx
import os
from torch_geometric.data import Data
from Bio import SeqIO


def dic_normalize(dic):
    min_val, max_val = min(dic.values()), max(dic.values())
    range_val = max_val - min_val
    normalized_dict = {k: (v - min_val) / range_val for k, v in dic.items()}
    normalized_dict['X'] = (max_val + min_val) / 2.0
    return normalized_dict


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                     1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    # print(smile)
    mol = Chem.MolFromSmiles(smile)
    # print('mol: ', mol)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    # print(features)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)

    # x = torch.tensor(features, dtype=torch.float)
    x = torch.tensor(np.array(features), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
    # return c_size, features, edge_index


# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    # k = float(len(pro_res_table))
    # pwm_mat = np.log2(ppm_mat / (1.0 / k))
    # pssm_mat = pwm_mat
    # print(pssm_mat)
    return pssm_mat

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    # print('target_feature')
    # print(pssm.shape)
    # print(other_feature.shape)

    # print(other_feature.shape)
    # return other_feature
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    # if 'X' in target_sequence:
    #     print(target_key)
    feature = target_feature(aln_file, target_sequence)
    return feature

# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_size = len(target_sequence)
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file, allow_pickle=True)
    # print(contact_map.shape)
    # print(type(contact_map))
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    x = torch.tensor(target_feature, dtype=torch.float)
    edge_index = torch.tensor(target_edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
    # return target_size, target_feature, target_edge_index


def read_drug_smiles(file_path):
    id2smile = {}
    drug2g = {}
    drug = []
    # Open the file and read line by line
    id = 0
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split by colon
            parts = line.strip().split(': ')
            # print(parts)
            if len(parts) == 2:
                drug_id = parts[0].strip()
                smile = parts[1].strip()
                # print(smile)
                # print(type(smile))
                # print(id)
                id2smile[id] = smile
                g = smile_to_graph(smile)
                drug2g[id] = g
                drug.append(g)
                id += 1
    
    return id2smile, drug2g, drug

# Usage example
# file_path = 'data/drug_smiles.txt'
# drug2id, id2g = read_drug_smiles(file_path)
# print(drug2id)
# print(id2g)

def read_protein_seq(path):
    seq_path = os.path.join(path,'seq')
    contact_path = os.path.join(path, 'pconsc4')
    msa_path = os.path.join(path, 'aln')
    protein2id = {}
    id2seq = {}
    id = 0
    for file in os.listdir(seq_path):
        # print(os.path.join(seq_path, file))
        for record in SeqIO.parse(os.path.join(seq_path,file), "fasta"):
            seq = record.seq
        # print(seq)
        protein_name = file[0:6]
        # print(protein_name)
        protein2id[protein_name] = id
        g = target_to_graph(protein_name, seq, contact_path, msa_path)
        id2seq[id] = g
        id += 1
    return protein2id, id2seq

# path = '/home/natalie/project_2Dgraph_HGL/DGraphDTA/data/samples'
# protein2id, id2seq = read_protein_seq(path)
# print(protein2id)
# print(id2seq)

def prepare_protein(file_path):
    contact_path = "data/new_data/pconsc4"
    msa_path = "data/new_data/aln"
    id2seq = {}
    id2protein = {}
    protein2g = {}
    protein_g = []
    with open(file_path, 'r') as file:
        id = 0
        for line in file:
            # Strip whitespace and split by colon
            parts = line.strip().split(': ')
            # print(parts)
            if len(parts) == 2:
                protein = parts[0].strip()
                seq = parts[1].strip()
                # print(protein, seq)
                # break
                id2seq[id] = seq
                id2protein[id] = protein
                g = target_to_graph(protein, seq, contact_path, msa_path)
                protein2g[id] = g
                protein_g.append(g)
                id +=1
    return id2seq, id2protein, protein2g, protein_g

# path = 'data/protein_sequences.txt'
# p2g = prepare_protein(path)
# print(p2g)

import torch
from torch_geometric.data import Data
 
def create_interaction_graph(drug_embeddings, target_embeddings, edge_index, edge_labels):
    num_drugs = drug_embeddings.size(0)
    num_targets = target_embeddings.size(0)
    total_nodes = num_drugs + num_targets
    print(total_nodes)

    # Initialize node features tensor
    x = torch.zeros((total_nodes, drug_embeddings.size(1)))  # Assuming both embeddings have the same dimension
    # Assign embeddings to the correct node indices
    x[:num_drugs] = drug_embeddings
    x[num_drugs:num_drugs+num_targets] = target_embeddings

    # Convert edge_index and edge_labels to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)
 
    # Create the graph data object with node features and edge list
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_labels)
    return data

import torch
from torch_geometric.data import Data

def create_drug_interaction_graph(drug_embeddings, interaction_matrix):
    """
    创建药物-药物交互图，不考虑蛋白信息。
    
    参数:
        drug_embeddings (Tensor): 药物嵌入，形状为 (num_drugs, embedding_dim)
        interaction_matrix (array-like or Tensor): 药物-药物交互矩阵，形状为 (num_drugs, num_drugs)
            非零值（例如1）表示存在交互。
            
    返回:
        Data: 一个 PyG 的 Data 对象，其中包含节点特征 x 和边索引 edge_index。
    """
    num_drugs = drug_embeddings.size(0)
    # 使用药物嵌入作为节点特征
    x = drug_embeddings
    edge_index = []
    
    # 如果交互矩阵是对称的（无向图），可以只遍历上三角部分，
    # 对于每个交互添加双向边。
    for i in range(num_drugs):
        for j in range(i + 1, num_drugs):
            # 如果 interaction_matrix[i, j] 为 True 或非零，则认为存在交互
            if interaction_matrix[i, j]:
                edge_index.append([i, j])
                edge_index.append([j, i])
    
    # 转换为 tensor 并调整为 PyG 要求的形状 [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index)
    return data


import torch
from torch_geometric.data import Data

def create_interaction_graph_20240911(drug_embeddings, target_embeddings, interaction_matrix):
    num_drugs = drug_embeddings.size(0)
    num_targets = target_embeddings.size(0)
    total_nodes = num_drugs + num_targets
    # print(total_nodes)
    # Initialize node features tensor
    x = torch.zeros((total_nodes, drug_embeddings.size(1)))  # Assuming both embeddings have the same dimension

    # Assign embeddings to the correct node indices
    x[:num_drugs] = drug_embeddings
    x[num_drugs:num_drugs+num_targets] = target_embeddings

    # Create edge indices
    edge_index = []
    for i in range(num_drugs):
        for j in range(num_targets):
            if interaction_matrix[i, j]:  # Assuming 1 denotes an interaction
                drug_idx = i
                target_idx = num_drugs + j  # Offset by number of drugs
                edge_index.append([drug_idx, target_idx])
                edge_index.append([target_idx, drug_idx])  # For undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the graph data object with node features and edge list
    data = Data(x=x, edge_index=edge_index)
    return data

import torch
from torch_geometric.data import Data

def create_interaction_graph_20250209(drug_embeddings, target_embeddings, interaction_matrix):
    """
    Build a bipartite drug-target graph for link prediction, including edge_label (0 or 1).
    
    Args:
        drug_embeddings (Tensor): shape [num_drugs, emb_dim]
        target_embeddings (Tensor): shape [num_targets, emb_dim]
        interaction_matrix (Tensor): shape [num_drugs, num_targets], with 0/1 entries
    
    Returns:
        data (Data): PyG Data object containing:
            - x (Tensor): Node features of shape [num_drugs + num_targets, emb_dim]
            - edge_index (LongTensor): shape [2, E], containing both positive and negative edges
            - edge_label (FloatTensor): shape [E], each entry is 0 or 1
    """
    num_drugs     = drug_embeddings.size(0)
    num_targets   = target_embeddings.size(0)
    emb_dim       = drug_embeddings.size(1)
    total_nodes   = num_drugs + num_targets

    # 1) Prepare node feature matrix x
    x = torch.zeros((total_nodes, emb_dim), dtype=torch.float32)
    x[:num_drugs]             = drug_embeddings
    x[num_drugs:num_drugs+num_targets] = target_embeddings

    # 2) Build edge_index and edge_label for ALL drug-target pairs
    edge_list   = []
    label_list  = []
    
    for i in range(num_drugs):
        for j in range(num_targets):
            label = interaction_matrix[i, j].item()  # 0 or 1
            # Undirected => we add two edges: (drug->target) and (target->drug)
            drug_idx   = i
            target_idx = num_drugs + j
            
            edge_list.append([drug_idx, target_idx])
            edge_list.append([target_idx, drug_idx])
            
            # The label is the same in both directions
            label_list.append(label)
            label_list.append(label)

    # Convert to Tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape [2, E]
    edge_label = torch.tensor(label_list, dtype=torch.float32)               # shape [E]

    # 3) Create PyG Data
    data = Data(
        x=x,
        edge_index=edge_index
    )
    data.edge_label = edge_label  # Store the labels in the data object
    
    return data

def create_interaction_graph_positives_only(drug_embeddings, target_embeddings, interaction_matrix):
    """
    Build a bipartite graph for link prediction, storing ONLY positive edges in edge_index.
    """
    num_drugs   = drug_embeddings.size(0)
    num_targets = target_embeddings.size(0)
    emb_dim     = drug_embeddings.size(1)
    total_nodes = num_drugs + num_targets

    # 1) Node features
    x = torch.zeros((total_nodes, emb_dim), dtype=torch.float32)
    x[:num_drugs] = drug_embeddings
    x[num_drugs:] = target_embeddings

    # 2) Build edge_index from positives only
    edge_list  = []
    for i in range(num_drugs):
        for j in range(num_targets):
            if interaction_matrix[i, j] == 1:  # only take the positive edges
                drug_idx   = i
                target_idx = num_drugs + j
                # For undirected, add both directions
                edge_list.append([drug_idx, target_idx])
                edge_list.append([target_idx, drug_idx])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 3) Create PyG Data
    data = Data(x=x, edge_index=edge_index)
    return data


def create_interaction_graph_202205(drug_embeddings, target_embeddings, interaction_matrix):
    num_drugs = drug_embeddings.size(0)
    num_targets = target_embeddings.size(0)
    total_nodes = num_drugs + num_targets
    # print(total_nodes)
    # Initialize node features tensor
    x = torch.zeros((total_nodes, drug_embeddings.size(1)))  # Assuming both embeddings have the same dimension

    # Assign embeddings to the correct node indices
    x[:num_drugs] = drug_embeddings
    x[num_drugs:num_drugs+num_targets] = target_embeddings

    # Create edge indices
    edge_index = []
    for i in range(num_drugs):
        for j in range(num_targets):
            if interaction_matrix[i, j]:  # Assuming 1 denotes an interaction
                drug_idx = i
                target_idx = num_drugs + j  # Offset by number of drugs
                edge_index.append([drug_idx, target_idx])
                edge_index.append([target_idx, drug_idx])  # For undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the graph data object with node features and edge list
    data = Data(x=x, edge_index=edge_index)
    return data


def create_interaction_graph_20250203(drug_embeddings, target_embeddings, interaction_matrix, neg_ratio=1.0):
    """
    构造大图，其中 drug_embeddings: [num_drugs, d]，target_embeddings: [num_targets, d]
    interaction_matrix: NumPy 数组或 tensor，形状为 [num_drugs, num_targets]，1 表示正例，0 表示无交互。
    
    neg_ratio: 负采样比例，例如 1.0 表示负样本数量与正样本数量相同。
    """
    if isinstance(interaction_matrix, np.ndarray):
        interaction_matrix = torch.tensor(interaction_matrix, dtype=torch.bool)
    else:
        interaction_matrix = interaction_matrix.bool()
        
    num_drugs = drug_embeddings.size(0)
    num_targets = target_embeddings.size(0)
    total_nodes = num_drugs + num_targets

    # 初始化节点特征 tensor
    x = torch.zeros((total_nodes, drug_embeddings.size(1)))
    x[:num_drugs] = drug_embeddings
    x[num_drugs: num_drugs+num_targets] = target_embeddings

    pos_edges = []
    neg_edges = []

    # 遍历交互矩阵构造正例边
    for i in range(num_drugs):
        for j in range(num_targets):
            if interaction_matrix[i, j]:
                drug_idx = i
                target_idx = num_drugs + j
                # 添加正例边（无向图，两方向）
                pos_edges.append([drug_idx, target_idx])
                pos_edges.append([target_idx, drug_idx])
    
    # 将正例边转换为 tensor
    if len(pos_edges) > 0:
        pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
    else:
        pos_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 负采样：随机选取 interaction_matrix==0 的边
    # 注意：负边数量可能很大，因此我们只随机采样一部分
    all_neg_edges = []
    for i in range(num_drugs):
        for j in range(num_targets):
            if not interaction_matrix[i, j]:
                drug_idx = i
                target_idx = num_drugs + j
                all_neg_edges.append([drug_idx, target_idx])
                all_neg_edges.append([target_idx, drug_idx])
    # 负样本数量采样
    num_pos = pos_edge_index.size(1)
    num_neg_to_sample = int(num_pos * neg_ratio)
    if len(all_neg_edges) > 0:
        all_neg_edges = np.array(all_neg_edges)
        # 随机采样
        perm = np.random.permutation(len(all_neg_edges))
        selected = all_neg_edges[perm[:num_neg_to_sample]]
        neg_edge_index = torch.tensor(selected, dtype=torch.long).t().contiguous()
    else:
        neg_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 合并正负边
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    # 构造 edge_label
    pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.float)
    neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float)
    edge_label = torch.cat([pos_labels, neg_labels], dim=0)
    
    # 构造 Data 对象，并将链接预测属性加入
    data = Data(x=x, edge_index=edge_index)
    data.edge_label_index = edge_index.clone()
    data.edge_label = edge_label
    return data


def create_ddi_graph_20240923(drug_embeddings, interaction_matrix):
    num_drugs = drug_embeddings.size(0)

    # Use drug embeddings as node features
    x = drug_embeddings

    # Create edge indices for undirected graph
    edge_index = []
    for i in range(num_drugs):
        for j in range(i + 1, num_drugs):  # Only consider upper triangle for undirected
            if interaction_matrix[i, j]:  # Assuming 1 denotes an interaction
                edge_index.append([i, j])
                edge_index.append([j, i])  # Ensure undirected connectivity

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the graph data object with node features and edge list
    data = Data(x=x, edge_index=edge_index)
    return data

def create_ppi_graph_20240923(target_embeddings, interaction_matrix):
    num_target = target_embeddings.size(0)

    # Use drug embeddings as node features
    x = target_embeddings

    # Create edge indices for undirected graph
    edge_index = []
    for i in range(num_target):
        for j in range(i + 1, num_target):  # Only consider upper triangle for undirected
            if interaction_matrix[i, j]:  # Assuming 1 denotes an interaction
                edge_index.append([i, j])
                edge_index.append([j, i])  # Ensure undirected connectivity

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the graph data object with node features and edge list
    data = Data(x=x, edge_index=edge_index)
    return data


def create_interaction_graph_20240912(interaction_matrix):
    num_drugs = interaction_matrix.shape[0]
    num_targets = interaction_matrix.shape[1]
    total_nodes = num_drugs + num_targets
    feature_dim = 128
    # print(total_nodes)
    # Initialize node features tensor
    x = torch.randn((total_nodes, feature_dim))

    # Create edge indices
    edge_index = []
    for i in range(num_drugs):
        for j in range(num_targets):
            if interaction_matrix[i, j]:  # Assuming 1 denotes an interaction
                drug_idx = i
                target_idx = num_drugs + j  # Offset by number of drugs
                edge_index.append([drug_idx, target_idx])
                edge_index.append([target_idx, drug_idx])  # For undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the graph data object with node features and edge list
    data = Data(x=x, edge_index=edge_index)
    return data


def load_drug_dict():
    dict_d = {}
    d_dict = {}
    drug_dict_path = 'data/drug_dict.txt'
    # drug_dict = np.loadtxt(drug_dict_path)
    # Open the file and read line by line
    with open(drug_dict_path, 'r') as file:
        for idx, line in enumerate(file):
            # Strip any leading/trailing whitespace characters including '\n'
            drug_name = line.strip()
            # Store in dictionary with line number as key and drug name as value
            dict_d[idx] = drug_name
            d_dict[drug_name] = idx
    return dict_d, d_dict

# dict_d = load_drug_dict()
# print(dict_d)

def load_protein_dict():
    dict_p = {}
    p_dict = {}
    protein_dict_path = 'data/protein_dict.txt'
    # drug_dict = np.loadtxt(drug_dict_path)
    # Open the file and read line by line
    with open(protein_dict_path, 'r') as file:
        for idx, line in enumerate(file):
            # Strip any leading/trailing whitespace characters including '\n'
            protein_name = line.strip()
            # Store in dictionary with line number as key and drug name as value
            # if protein_name == 'A6NG28':
            #     print(idx)
            # dict_p[idx] = protein_name
            # p_dict[protein_name] = idx
            if protein_name in p_dict:
                p_dict[protein_name].append(idx)
            else:
                p_dict[protein_name] = [idx]
    return dict_p, p_dict

# dict_p = load_protein_dict()
# print(dict_p)


def load_drug_protein_interaction():
    network_path = 'data/'
    interaction_matrix = np.loadtxt(network_path + 'mat_drug_protein.txt')
    threshold = 0.5  # Define a threshold if needed
    interaction_matrix = (interaction_matrix >= threshold).astype(int)
    return interaction_matrix

def load_drug_drug_interaction():
    network_path = 'data/'
    interaction_matrix = np.loadtxt(network_path + 'mat_drug_drug.txt')
    threshold = 0.5  # Define a threshold if needed
    ddi_matrix = (interaction_matrix >= threshold).astype(int)
    return ddi_matrix

# drug_name = 'DB00176'
# protein_name = 'O00204'

# dict_d, d_dict = load_drug_dict()
# # print(dict_d)
# dict_p, p_dict = load_protein_dict()
# # print(p_dict)

# interaction_matrix = load_drug_protein_interaction()
# # DB00050: 671, 1322, 88, 1465, 758
# d_id = d_dict[drug_name]
# p_id = p_dict[protein_name]
# print(d_id, p_id)
# edge = interaction_matrix[0][1332]
# print(edge)

# positions = np.where(interaction_matrix == 1)
# # Zip the row and column indices together
# position_list = list(zip(positions[0], positions[1]))
# print("Positions of 1s:", position_list)


# def get_drug_protein():
#     file_path = 'data/drug_smiles.txt'
#     id2s, d2g = read_drug_smiles(file_path)
#     # print(drug2id)
#     # print(id2g)

#     path = 'data/protein_sequences.txt'
#     id2seq, id2protein, p2g = prepare_protein(path)
#     # print(p2g)
#     # print(id2seq)

#     return id2s, d2g, id2seq, id2protein, p2g

# def save_to_txt(data, filename):
#     with open(filename, 'w') as file:
#         for key, value in data.items():
#             # Assuming the value is already in a string format suitable for saving
#             file.write(f"{key},{value}\n")


# id2s, d2g, id2seq, id2protein, p2g = get_drug_protein()
# save_to_txt(id2s, 'data/id2smile.txt')
# save_to_txt(id2seq, 'data/id2seq.txt')
# save_to_txt(id2protein, 'data/id2protein.txt')


def load_drug_drug_interaction():
    network_path = 'data/'
    # interaction_matrix = np.loadtxt(network_path + 'mat_drug_drug.txt')
    interaction_matrix = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
    threshold = 0.5  # Define a threshold if needed
    ddi_matrix = (interaction_matrix >= threshold).astype(int)
    return ddi_matrix


def load_protein_protein_interaction():
    network_path = 'data/'
    interaction_matrix = np.loadtxt(network_path + 'mat_protein_protein.txt')
    threshold = 0.5  # Define a threshold if needed
    ppi_matrix = (interaction_matrix >= threshold).astype(int)
    return ppi_matrix









