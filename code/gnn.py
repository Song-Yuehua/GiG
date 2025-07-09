import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gep, global_sort_pool
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep, global_sort_pool
from torch_geometric.nn.conv.sage_conv import SAGEConv
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import global_mean_pool

# class InteractionGNN(torch.nn.Module):
#     def __init__(self, hidden_dim):
#         super(InteractionGNN, self).__init__()
#         self.conv1 = GCNConv(hidden_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)

#     def forward(self, x, edge_index):
#         x = torch.relu(self.conv1(x, edge_index))
#         x = global_mean_pool(torch.relu(self.conv2(x, edge_index)))
#         return x

# === 1. MLP Classifier ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single output for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Probability output (0 to 1)
        return x
    
class MLPClassifierLogits(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifierLogits, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  # No sigmoid here, return raw logits
        return logits  # shape [batch_size, 1]
    
class MLPClassifier_b(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier_b, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 单输出，二分类
        self.shift = nn.Parameter(torch.zeros(1))  # 可学习的 shift 参数

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = nn.LeakyReLU()(self.fc1(x))
        # 在 fc2 的输出上加上 shift 后再经过 sigmoid 得到概率
        x = torch.sigmoid(self.fc2(x) + self.shift)
        print("shift: ", self.shift)
        return x
    
class MLPClassifier_b_T(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier_b_T, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 单输出，二分类
        # self.shift = nn.Parameter(torch.zeros(1))  # 可学习的 shift 参数

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = nn.LeakyReLU()(self.fc1(x))
        # 在 fc2 的输出上加上 shift 后再经过 sigmoid 得到概率
        # x = torch.sigmoid(self.fc2(x) + self.shift)
        x = self.fc2(x)
        # print("shift: ", self.shift)
        return x

class MLPClassifier_ab(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier_ab, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 单输出
        # 初始化 shift 参数 b，初始为 0
        self.b = nn.Parameter(torch.zeros(1))
        # 初始化 a 参数，建议初始值为1，使得激活函数初始状态与标准 sigmoid 相似
        self.a = nn.Parameter(torch.ones(1))
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # 得到 logits 后进行线性变换，再加上 a 和 b 调整后进入 sigmoid
        logits = self.fc2(x)
        out = torch.sigmoid(self.a * logits + self.b)
        # 输出为 [0,1] 的概率
        print("a: ", self.a)
        print("b: ", self.b)
        return out
    
class MLPClassifier_ab_t(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier_ab_t, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 单输出
        # 初始化 shift 参数 b，初始为 0
        self.b = nn.Parameter(torch.zeros(1))
        # 初始化 a 参数，建议初始值为1，使得激活函数初始状态与标准 sigmoid 相似
        self.a = nn.Parameter(torch.ones(1))
        self.log_temperature = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # 得到 logits 后进行线性变换，再加上 a 和 b 调整后进入 sigmoid
        logits = self.fc2(x)
        temperature = torch.exp(self.log_temperature)
        out = torch.sigmoid((self.a * logits + self.b)/temperature)
        # 输出为 [0,1] 的概率
        print("a: ", self.a)
        print("b: ", self.b)
        print('t: ', temperature)
        return out

class MLPClassifier_local(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier_local, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 生成 logits
        # 定义一个小网络，用来为每个样本生成局部偏置
        self.shift_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        logits = self.fc2(h)
        local_shift = self.shift_net(h)  # 为每个样本生成局部偏置
        # 最终输出：sigmoid(a * logits + local_shift)，这里 a 可视情况加入
        out = torch.sigmoid(logits + local_shift)
        return out

    
### Add a learnable parameter tau into my mlp ###

from Gumbel_Sigmoid import gumbel_sigmoid, gumbel_sigmoid_st
import torch.nn.functional as F

class MLPClassifierLogits_gumbel_sigmoid(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_tau=0.5):
        super(MLPClassifierLogits_gumbel_sigmoid, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 可学习的 log_tau，用以确保 tau 始终 > 0
        # 例如初始化时令 log_tau = log(init_tau)
        self.log_tau = nn.Parameter(torch.log(torch.tensor([init_tau], dtype=torch.float32)))

    def forward(self, x, use_gumbel=False, straight_through=True):
        """
        如果 use_gumbel=True，则在 forward 里直接返回 gumbel_sigmoid
        如果 use_gumbel=False，则返回原始 logits
        """
        # 1) 计算 logits
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x).squeeze(dim=-1)  # shape [batch]

        if not use_gumbel:
            # 只返回原始 logits（留给外部调用 .sigmoid/.bce等）
            return logits
        else:
            # 2) 计算当前 tau
            # 这里用 softplus 或 exp 都行，只要能保证 >0
            tau = F.softplus(self.log_tau)  # or torch.exp(self.log_tau)

            # 3) 调用 Gumbel-Sigmoid
            if straight_through:
                return gumbel_sigmoid_st(logits, tau)
            else:
                return gumbel_sigmoid(logits, tau)

import torch
import torch.nn as nn
import torch.nn.functional as F

from Gumbel_Sigmoid import gumbel_sigmoid, gumbel_sigmoid_st

GUMBEL_MEAN = 0.5772156649

class MLPClassifierLogits_gumbel_sigmoid_20250314(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_tau=0.5):
        super(MLPClassifierLogits_gumbel_sigmoid_20250314, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Trainable log_tau ensures tau > 0
        self.log_tau = nn.Parameter(torch.log(torch.tensor([init_tau], dtype=torch.float32)))

    def forward(self, x, 
                use_gumbel=False, 
                straight_through=True, 
                approx_mean=False):
        """
        Args:
            x: input tensor of shape [batch_size, input_dim]
            use_gumbel (bool): if True, uses Gumbel-Sigmoid for sampling
            straight_through (bool): if using Gumbel, whether to use ST 
            approx_mean (bool): if True, and use_gumbel=False, 
                                return sigmoid((logits + GUMBEL_MEAN)/tau) 
                                to emulate the average Gumbel effect.

        Returns:
            Either:
              - raw logits (float32) if (use_gumbel=False and approx_mean=False)
              - a probability in [0,1] from Gumbel-Sigmoid if use_gumbel=True
              - a deterministic approximation using GUMBEL_MEAN and learned tau 
                if use_gumbel=False and approx_mean=True
        """

        # 1) Compute raw logits
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x).squeeze(dim=-1)  # shape [batch]

        # 2) Figure out how we produce outputs
        if use_gumbel:
            # Gumbel path
            tau = F.softplus(self.log_tau)  # ensure tau > 0
            if straight_through:
                return gumbel_sigmoid_st(logits, tau)
            else:
                return gumbel_sigmoid(logits, tau)

        else:
            # No Gumbel sampling
            if approx_mean:
                # 3) Return approximate Gumbel shift 
                #    prob = sigmoid((logits + GUMBEL_MEAN)/tau)
                tau = F.softplus(self.log_tau)
                return torch.sigmoid((logits + GUMBEL_MEAN) / tau)
            else:
                # Just return raw logits for plain BCE usage
                return logits

class MLPClassifierLogits_gumbel_sigmoid_tau_fixed_20250317(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifierLogits_gumbel_sigmoid_tau_fixed_20250317, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # Fixed tau value of 0.5 (no trainable parameter)
        self.tau = 0.5
        
    def forward(self, x, 
                use_gumbel=False, 
                straight_through=True,
                approx_mean=False):
        """
        Args:
            x: input tensor of shape [batch_size, input_dim]
            use_gumbel (bool): if True, uses Gumbel-Sigmoid for sampling
            straight_through (bool): if using Gumbel, whether to use ST
            approx_mean (bool): if True, and use_gumbel=False, 
                               return sigmoid((logits + GUMBEL_MEAN)/tau)
                               to emulate the average Gumbel effect.
        Returns:
            Either:
            - raw logits (float32) if (use_gumbel=False and approx_mean=False)
            - a probability in [0,1] from Gumbel-Sigmoid if use_gumbel=True
            - a deterministic approximation using GUMBEL_MEAN and fixed tau
              if use_gumbel=False and approx_mean=True
        """
        # 1) Compute raw logits
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x).squeeze(dim=-1)  # shape [batch]
        
        # 2) Figure out how we produce outputs
        if use_gumbel:
            # Gumbel path with fixed tau
            if straight_through:
                return gumbel_sigmoid_st(logits, self.tau)
            else:
                return gumbel_sigmoid(logits, self.tau)
        else:
            # No Gumbel sampling
            if approx_mean:
                # 3) Return approximate Gumbel shift
                # prob = sigmoid((logits + GUMBEL_MEAN)/tau)
                return torch.sigmoid((logits + GUMBEL_MEAN) / self.tau)
            else:
                # Just return raw logits for plain BCE usage
                return logits

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
    

class LinkPredictionModel_SAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(LinkPredictionModel_SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


from torch_geometric.nn import GINConv 
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential

class LinkPredictionModel_GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(LinkPredictionModel_GIN, self).__init__()
        nn1 = Sequential(Linear(num_features, hidden_dim), 
                          ReLU())
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        nn2 = Sequential(Linear(hidden_dim, hidden_dim), 
                          ReLU())
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x






