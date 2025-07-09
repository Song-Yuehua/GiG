import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gep, global_sort_pool
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep, global_sort_pool
from torch_geometric.nn.conv.sage_conv import SAGEConv
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import global_mean_pool

class FusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(FusionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class GatedFusion(nn.Module):
    def __init__(self, local_dim, global_dim):
        super().__init__()
        self.gate_linear = nn.Linear(local_dim + global_dim, 1)

    def forward(self, local_x, global_x):
        # local_x.shape = [B, local_dim]
        # global_x.shape = [B, global_dim]
        combined = torch.cat([local_x, global_x], dim=-1)  # [B, local_dim+global_dim]
        gate = torch.sigmoid(self.gate_linear(combined))   # [B, 1]

        # Weighted sum: alpha * global + (1-alpha)*local
        out = gate * global_x + (1 - gate) * local_x
        return out


# GCN based model
class EmbeddingGNNAddGlobal(torch.nn.Module):
    def __init__(self, num_features_mol=78, output_dim=128, dropout=0.2):
        super(EmbeddingGNNAddGlobal, self).__init__()

        print('GNNNet Loaded')
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1) Project 128 -> 216, so global has same dim as local
        self.proj_global = nn.Linear(128, 312)

        # 2) Gated fusion on 216 + 216 => final shape 216
        self.gated_fusion = GatedFusion(local_dim=312, global_dim=312)

        # self.fusion_layer = FusionLayer(input_dim=312 + 128, 
        #                                 output_dim=312, 
        #                                 dropout=dropout)

        # combined layers
        # self.fc1 = nn.Linear(2 * output_dim, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, global_emb=None):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch


        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_batch.size(), 'batch', target_batch.size())

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # print("x size: ", x.shape)
        # print("global embedding: ", global_emb.shape)

        if global_emb is not None:
            # Project global => [batch_size, 216]
            global_emb = self.proj_global(global_emb)
            # Gated fusion => [batch_size, 216]
            x = self.gated_fusion(x, global_emb)
            # # You need to map them to the same dimension, then combine
            # # For example, let's just do a concatenation:
            # x = torch.cat([x, global_emb], dim=-1)  
            # # print("x size after fusion: ", x.shape)
            # # Then maybe reduce dimension:
            # x = self.fusion_layer(x)  # e.g. a Linear(...)

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        return x


# GCN based model
class EmbeddingGNNAddGlobal_p(torch.nn.Module):
    def __init__(self, num_features_pro=54, output_dim=128, dropout=0.2):
        super(EmbeddingGNNAddGlobal_p, self).__init__()

        print('GNNNet Loaded')
        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1) Project 128 -> 216, so global has same dim as local
        self.proj_global = nn.Linear(128, 216)

        # 2) Gated fusion on 216 + 216 => final shape 216
        self.gated_fusion = GatedFusion(local_dim=216, global_dim=216)

        # self.fusion_layer = FusionLayer(input_dim=216 + 128, 
        #                                 output_dim=216, 
        #                                 dropout=dropout)

    def forward(self, data_pro, global_emb=None):
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_batch.size(), 'batch', target_batch.size())

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # print("x size: ", xt.shape)
        # print("global embedding: ", global_emb.shape)

        if global_emb is not None:
            # Project global => [batch_size, 216]
            global_emb = self.proj_global(global_emb)
            # Gated fusion => [batch_size, 216]
            xt = self.gated_fusion(xt, global_emb)
            # # You need to map them to the same dimension, then combine
            # # For example, let's just do a concatenation:
            # xt = torch.cat([xt, global_emb], dim=-1)  
            # # print("x size after fusion: ", xt.shape)
            # # Then maybe reduce dimension:
            # xt = self.fusion_layer(xt)  # e.g. a Linear(...)

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        return xt
    
# ---------------------------------------------------------------------------------------------------------



# GCN based model
class EmbeddingGNN(torch.nn.Module):
    def __init__(self, num_features_mol=78, output_dim=128, dropout=0.2):
        super(EmbeddingGNN, self).__init__()

        print('GNNNet Loaded')
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        # self.fc1 = nn.Linear(2 * output_dim, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch


        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_batch.size(), 'batch', target_batch.size())

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        return x


# GCN based model
class EmbeddingGNN_p(torch.nn.Module):
    def __init__(self, num_features_pro=54, output_dim=128, dropout=0.2):
        super(EmbeddingGNN_p, self).__init__()

        print('GNNNet Loaded')
        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data_pro):
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_batch.size(), 'batch', target_batch.size())

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        return xt

# class InteractionGNN(torch.nn.Module):
#     def __init__(self, hidden_dim):
#         super(InteractionGNN, self).__init__()
#         self.conv1 = GCNConv(hidden_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)

#     def forward(self, x, edge_index):
#         x = torch.relu(self.conv1(x, edge_index))
#         x = global_mean_pool(torch.relu(self.conv2(x, edge_index)))
#         return x


import torch
from torch_geometric.nn import GCNConv

class InteractionGNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(InteractionGNN, self).__init__()
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    # def forward(self, data):
    #     x, edge_index = data.x, data.edge_index
    
    def forward(self, x, edge_index):
        
        # Apply GCN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # Prepare to classify edges
        # Retrieve node embeddings for both the source and target of each edge
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)

        # Classify edges
        edge_probabilities = self.edge_classifier(edge_features).squeeze()

        return edge_probabilities

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels) # cached only for transductive learning
 
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LinkPredictionModel(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class LinkPredictionModel_GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(LinkPredictionModel_GAT, self).__init__()
        # Adjust the number of heads as needed, here using 8 heads as an example
        self.conv1 = GATConv(num_features, hidden_dim, heads=8, dropout=0.2)
        # For subsequent layers, you might want to reduce the number of features per head to manage the dimensionality
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, concat=False, dropout=0.2)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    






