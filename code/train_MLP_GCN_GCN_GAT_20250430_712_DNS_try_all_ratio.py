import random
import numpy as np
import torch
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, accuracy_score,
    roc_curve
)
import torch.nn.functional as F
# ====== Import your existing methods =====
from data_process import (
    read_drug_smiles,       
    prepare_protein,        
    load_drug_protein_interaction,    
    create_interaction_graph_positives_only 
)

# ====== Import your GNN models =====
from gnn_20250209 import (
    EmbeddingGNN,      
    EmbeddingGNN_p,    
    # LinkPredictionModel,
    MLPClassifierLogits_gumbel_sigmoid_20250314
)
from gnn import LinkPredictionModel_GAT
from Gumbel_Sigmoid import gumbel_sigmoid_st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GUMBEL_MEAN = 0.5772156649 

# ====== Custom Edge Splitting =====
import math
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)


###################################################################################
# 在“Final Test”之后，加入对 MLP 输入对的可视化
###################################################################################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_pair_embeddings(pair_emb, color_values, out_path, title):
    """
    使用 t-SNE 将药物-蛋白对的 MLP 输入向量降到 2D，并根据 color_values 着色。
    color_values 可以是真实标签 (0/1) 或预测概率 (0~1)。
    """
    tsne = TSNE(n_components=2, random_state=0)
    emb_2d = tsne.fit_transform(pair_emb.cpu().numpy())  # [N_pairs, 2]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=color_values,
        cmap="viridis",   # 或者 'coolwarm' 等其它 Colormap
        alpha=0.7
    )
    plt.colorbar(sc)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def split_edges_by_pair(edge_index, num_drugs,
                        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    - 将无向 drug‑protein 交互视为 *一条* edge_pair。
    - 先对 edge_pair 做随机拆分，然后再一次性把正向+反向
      两条边都加入对应 split，保证统计时两个脚本一致。
    """
    # 1) 先抽取 “唯一交互对” (drug, protein)
    mask_forward = (edge_index[0] < num_drugs) & (edge_index[1] >= num_drugs)
    drugs  = edge_index[0, mask_forward]
    pros   = edge_index[1, mask_forward] - num_drugs          # 0‑based protein idx
    pairs  = torch.stack([drugs, pros], dim=1)                # [N_pairs, 2]

    # 2) 随机打乱并按比例切块
    perm   = torch.randperm(pairs.size(0))
    n_train = int(train_ratio * len(perm))
    n_val   = int(val_ratio   * len(perm))

    idx_tr  = perm[:n_train]
    idx_va  = perm[n_train:n_train+n_val]
    idx_te  = perm[n_train+n_val:]

    split_dict = {"train": pairs[idx_tr],
                  "val"  : pairs[idx_va],
                  "test" : pairs[idx_te]}

    def _to_edge_index(pairs_tensor):
        """把 (d, p) → [[d, p+offset] , [p+offset, d]]"""
        rows = []
        cols = []
        for d, p in pairs_tensor.tolist():
            p_off = p + num_drugs
            rows.extend([d, p_off])
            cols.extend([p_off, d])
        return torch.tensor([rows, cols], dtype=torch.long)

    train_ei = _to_edge_index(split_dict["train"])
    val_ei   = _to_edge_index(split_dict["val"])
    test_ei  = _to_edge_index(split_dict["test"])
    return train_ei, val_ei, test_ei

def train_and_evaluate(
    neg_ratio,
    device='cpu',
    EPOCHS=500,
    LR=1e-4,
    BATCH_SIZE=64,
    EMBEDDING_DIM=128,
    HIDDEN_DIM=64
):
    """
    Train the model from scratch using a specified negative ratio 
    (negatives : positives in each training step).
    Return final performance metrics on the test set.
    """
    # ---------------------------------------------------------------------
    # 1) Load your data the same way you already do
    drug_smiles_path = 'data/drug_smiles.txt'
    id2smile, drug2g, drug_list = read_drug_smiles(drug_smiles_path)
    protein_path = 'data/protein_sequences.txt'
    _, _, _, protein_list = prepare_protein(protein_path)
    interaction_matrix = load_drug_protein_interaction()

    num_drugs, num_proteins = interaction_matrix.shape

    # ---------------------------------------------------------------------
    # 2) Build subgraph DataLoaders
    drug_dataset    = DrugDataset(drug_list)
    protein_dataset = ProteinDataset(protein_list)

    drug_loader    = DataLoader(drug_dataset,    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    protein_loader = DataLoader(protein_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ---------------------------------------------------------------------
    # 3) Initialize Models
    model_d      = EmbeddingGNN().to(device)
    model_p      = EmbeddingGNN_p().to(device)
    model_link   = LinkPredictionModel_GAT(num_features=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(device)
    mlp_classifier = MLPClassifierLogits_gumbel_sigmoid_20250314(input_dim=EMBEDDING_DIM, hidden_dim=64).to(device)

    optimizer = optim.Adam(
        list(model_d.parameters()) +
        list(model_p.parameters()) +
        list(model_link.parameters()) +
        list(mlp_classifier.parameters()), 
        lr=LR
    )
    criterion = nn.BCELoss()

    # ---------------------------------------------------------------------
    # 4) Construct the full bipartite graph
    with torch.no_grad():
        init_drug_emb = torch.randn((num_drugs, EMBEDDING_DIM), device=device)
        init_prot_emb = torch.randn((num_proteins, EMBEDDING_DIM), device=device)

    full_data = create_interaction_graph_positives_only(init_drug_emb, init_prot_emb, interaction_matrix)
    train_edges, val_edges, test_edges = split_edges_by_pair(
        full_data.edge_index, num_drugs,
        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

    print(f'train_edges: {train_edges}, val_edges; {val_edges}, test_edges: {test_edges}')

    train_data = Data(x=full_data.x, edge_index=train_edges).to(device)
    val_data   = Data(x=full_data.x, edge_index=val_edges).to(device)
    test_data  = Data(x=full_data.x, edge_index=test_edges).to(device)

    # Build adjacency_set = known positives from train+val+test
    adjacency_set = set()
    for edge_idx in [train_data.edge_index, val_data.edge_index, test_data.edge_index]:
        E = edge_idx.size(1)
        for i in range(E):
            src = edge_idx[0, i].item()
            dst = edge_idx[1, i].item()
            # bipartite logic
            if src < num_drugs and dst >= num_drugs:
                adjacency_set.add((src, dst - num_drugs))
            elif dst < num_drugs and src >= num_drugs:
                adjacency_set.add((dst, src - num_drugs))

    # ---------------------------------------------------------------------
    # 5) Training loop
    best_val_auc = 0.0
    best_val_pr = 0.0
    best_val_loss = float('inf')
    early_stopping_patience = 100
    early_stopping_counter = 0

    train_losses = []
    val_losses   = []
    val_aucs     = []
    val_auprs    = []

    for epoch in range(EPOCHS):
        # -- Put subgraph GATs in train mode
        model_d.train()
        model_p.train()
        model_link.train()
        mlp_classifier.train()

        # A) Get new drug/protein embeddings from the subgraph GATs
        drug_emb = get_all_drug_embeddings_train(model_d, drug_loader, EMBEDDING_DIM)
        prot_emb = get_all_protein_embeddings_train(model_p, protein_loader, EMBEDDING_DIM)

        # B) Combine into train_data.x
        x_combined = torch.cat([drug_emb, prot_emb], dim=0)
        train_data.x = x_combined.requires_grad_()  # in-place or out-of-place is fine

        # C) LinkPredictionModel forward
        optimizer.zero_grad()
        z = model_link(train_data.x, train_data.edge_index)

        pos_src, pos_dst = get_positive_drug_target_edges(train_data.edge_index, num_drugs)
        pos_src = pos_src.to(device)
        pos_dst = pos_dst.to(device)
        num_pos = pos_src.size(0)

        num_pos = pos_src.size(0)
        print(f"ratio: {neg_ratio}, num_pos: {num_pos}")
        num_neg = int(num_pos * neg_ratio)
        print(f"negtive samples: {num_neg}")

        # Dynamic negative sampling
        neg_src, neg_dst = dynamic_negative_sampling(
            mlp_classifier=mlp_classifier,
            embeddings=z,
            num_drugs=num_drugs,
            num_proteins=num_proteins,
            num_samples=num_neg,
            adjacency_set=adjacency_set,
            device=device
        )

        # (Optional) if you fail to get enough "hard negatives", fill with random
        # if len(neg_src) < num_neg:
        #     extra_needed = num_neg - len(neg_src)
        #     extra_neg_src, extra_neg_dst = sample_negative_edges(
        #         num_drugs, num_proteins, adjacency_set, extra_needed
        #     )
        #     neg_src = torch.cat([neg_src, extra_neg_src.to(device)], dim=0)
        #     neg_dst = torch.cat([neg_dst, extra_neg_dst.to(device)], dim=0)

        # Concatenate positives & negatives
        all_src = torch.cat([pos_src, neg_src], dim=0)
        all_dst = torch.cat([pos_dst, neg_dst], dim=0)

        labels_pos = torch.ones(num_pos, device=device)
        labels_neg = torch.zeros(num_neg, device=device)
        all_labels = torch.cat([labels_pos, labels_neg], dim=0)

        # MLP forward
        src_emb = z[all_src]
        dst_emb = z[all_dst]
        mlp_input = torch.cat([src_emb, dst_emb], dim=1)

        # Gumbel or normal Sigmoid
        # (You can keep your existing gumbel logic. For brevity, I’ll do a simple sigmoid here.)
        # raw_logits = mlp_classifier(mlp_input, use_gumbel=False)   # shape [num_edges,]
        # preds      = torch.sigmoid(raw_logits)

        if epoch < 600:
            # Just do normal Sigmoid
            raw_logits = mlp_classifier(mlp_input, use_gumbel=False)
            preds = torch.sigmoid(raw_logits)
        else:
            # Let MLP do gumbel
            preds = mlp_classifier(mlp_input, use_gumbel=True, straight_through=True)

        loss = criterion(preds, all_labels)
        loss.backward()
        optimizer.step()

        current_tau = F.softplus(mlp_classifier.log_tau).item()
        print(f"Epoch {epoch}, loss={loss.item():.4f}, tau={current_tau:.4f}")

        train_losses.append(loss.item())

        # (D) Validation
        with torch.no_grad():
            val_data.x = x_combined  # re-use same embeddings for val
            val_results = evaluate_model_negative_sampling(
                model_link,
                mlp_classifier,
                val_data,
                adjacency_set,
                criterion,
                num_drugs,
                num_proteins,
                epoch=epoch,
                device=device
            )
            val_loss = val_results['loss']
            val_auc  = val_results['auc_roc']
            val_aupr = val_results['auc_pr']

        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_auprs.append(val_aupr)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_d_state_dict': model_d.state_dict(),
                'model_p_state_dict': model_p.state_dict(),
                'model_link_state_dict': model_link.state_dict(),
                'mlp_classifier_state_dict': mlp_classifier.state_dict()
            }, f'code/20250416/20250420_GCNGCNGAT_712_gs/MLP_GCNGCNGAT_613_20250407_gumbel_sogmoid_mean_learn_tau_DNS_neg_{neg_ratio}_best_model_by_val_auc.pth')
            print(f"  --> Saved best model by val_auc: {val_auc:.4f}")
            # early_stopping_counter = 0
        # elif val_auc < best_val_auc - 0.03:
        #     early_stopping_counter += 1
        #     print(f"  --> Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_d_state_dict': model_d.state_dict(),
                'model_p_state_dict': model_p.state_dict(),
                'model_link_state_dict': model_link.state_dict(),
                'mlp_classifier_state_dict': mlp_classifier.state_dict()
            }, f'code/20250416/20250420_GCNGCNGAT_712_gs/MLP_GCNGCNGAT_613_20250407_gumbel_sogmoid_mean_learn_tau_DNS_neg_{neg_ratio}_best_model_by_val_loss.pth')
            print(f"  --> Saved best model by val_loss: {val_loss:.4f}")

        # if early_stopping_counter >= early_stopping_patience:
        #     print(f"[neg_ratio={neg_ratio}] Early stopping at epoch {epoch}")
        #     break

        if val_aupr > best_val_pr:
            best_val_pr = val_aupr
            torch.save({
                'model_d_state_dict': model_d.state_dict(),
                'model_p_state_dict': model_p.state_dict(),
                'model_link_state_dict': model_link.state_dict(),
                'mlp_classifier_state_dict': mlp_classifier.state_dict()
            }, f'code/20250416/20250420_GCNGCNGAT_712_gs/MLP_GCNGCNGAT_613_20250407_gumbel_sogmoid_mean_learn_tau_DNS_neg_{neg_ratio}_best_model_by_val_aucpr.pth')
            print(f"  --> Saved best model by val_aucpr: {val_aupr:.4f}")


        if epoch % 20 == 0:
            print(f"[Epoch {epoch:03d} | neg_ratio={neg_ratio}]  "
                  f"Train Loss={loss.item():.4f},  Val Loss={val_loss:.4f},  Val AUC={val_auc:.4f},  Val AUPR={val_aupr:.4f}")

    # ---------------------------------------------------------------------
    # 6) Final Test

    checkpoint_auc = torch.load(f'code/20250416/20250420_GCNGCNGAT_712_gs/MLP_GCNGCNGAT_613_20250407_gumbel_sogmoid_mean_learn_tau_DNS_neg_{neg_ratio}_best_model_by_val_aucpr.pth', map_location=device)
    model_d.load_state_dict(checkpoint_auc['model_d_state_dict'])
    model_p.load_state_dict(checkpoint_auc['model_p_state_dict'])
    model_link.load_state_dict(checkpoint_auc['model_link_state_dict'])
    mlp_classifier.load_state_dict(checkpoint_auc['mlp_classifier_state_dict'])

    model_d.eval()
    model_p.eval()
    model_link.eval()
    mlp_classifier.eval()

    with torch.no_grad():
        final_drug_emb = get_all_drug_embeddings(model_d, drug_loader, EMBEDDING_DIM)
        final_prot_emb = get_all_protein_embeddings(model_p, protein_loader, EMBEDDING_DIM)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        def visualize_embeddings(embeddings, labels, title="Embeddings Visualization"):
            # embeddings: [N, D] tensor, labels: [N] array-like (e.g., class indices)
            tsne = TSNE(n_components=2, random_state=0)
            emb_2d = tsne.fit_transform(embeddings.cpu().numpy())
            
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", alpha=0.7)
            plt.title(title)
            # plt.colorbar(scatter)
            plt.colorbar(scatter, ticks=[0, 1], label="Node Type (0: Drug, 1: Protein)")
            plt.savefig(f"code/20250416/20250420_GCNGCNGAT_712_gs/MLP_GCNGCNGAT_613_20250407_gumbel_sogmoid_mean_learn_tau_DNS_neg_{neg_ratio}_features_analysis.png")
            plt.close()

        # For example, if you have separate embeddings for drugs and proteins,
        # you could label them accordingly (e.g., 0 for drugs, 1 for proteins)
        combined_emb = torch.cat([final_drug_emb, final_prot_emb], dim=0)
        labels = [0]*drug_emb.size(0) + [1]*prot_emb.size(0)
        visualize_embeddings(combined_emb, labels, title="Drug vs. Protein Embeddings")


    test_data.x = torch.cat([final_drug_emb, final_prot_emb], dim=0)
    test_results = evaluate_test_negative_sampling(
        neg_ratio,
        model_link,
        mlp_classifier,
        test_data,
        adjacency_set,
        num_drugs,
        num_proteins,
        criterion,
        device=device
    )
    # Example metrics you might collect:
    final_test_auc   = test_results['auc_roc']
    final_test_aupr  = test_results['auc_pr']
    final_test_acc   = test_results['acc']
    final_test_f1    = test_results['f1']
    final_test_mcc   = test_results['mcc']
    final_test_precision = test_results['precision']
    final_test_recall = test_results['recall']
    final_test_loss  = test_results['loss']
    final_probs = test_results['probs']
    final_fpr = test_results['fpr']
    final_tpr =  test_results['tpr']
    final_thresholds = test_results['thresholds']

    for idx, th in enumerate(final_thresholds):
        # 计算在该阈值下，被判定为正类的样本数
        print('########################')
        print('neg_ratio: ', neg_ratio)
        print('########################')
        count_predicted_positive = np.sum(final_probs >= th)
        
        print(
            f"Threshold[{idx}] = {th:.4f} | "
            f"FPR={final_fpr[idx]:.4f} | "
            f"TPR={final_tpr[idx]:.4f} | "
            f"PredictedPositive={count_predicted_positive}"
        )


    plt.figure()
    plt.plot(final_fpr, final_tpr, color="blue", label=f"ROC curve (AUC = {final_test_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on the Test Set")
    plt.legend(loc="lower right")
    plt.grid(True)

    # 4) 保存图像
    plt.savefig(f"code/20250416/20250420_GCNGCNGAT_712_gs/test_roc_curve_{neg_ratio}.png")
    plt.close()

    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Plot AUC and AUPR
    plt.subplot(1, 2, 2)
    plt.plot(range(len(val_aucs)), val_aucs, label="Val AUC (ROC)")
    plt.plot(range(len(val_auprs)), val_auprs, label="Val AUPR")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation AUC-ROC & AUPR")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"code/20250416/20250420_GCNGCNGAT_712_gs/MLP_GCNGCNGAT_613_20250407_gumbel_sogmoid_mean_learn_tau_DNS_neg_{neg_ratio}_training_metrics.png")
    plt.close()

    return {
        'current_tau': current_tau,
        'test_auc':  final_test_auc,
        'test_aupr': final_test_aupr,
        'acc':       final_test_acc,
        'precision': final_test_precision,
        'recall': final_test_recall,
        'f1':        final_test_f1,
        'mcc':       final_test_mcc,
        'loss':      final_test_loss
    }

###################################################################################
###################################################################################

def evaluate_test_negative_sampling(
    neg_ratio,
    model_link,
    mlp_classifier,
    data,            # e.g. test_data with positive edges
    adjacency_set,   # set of (drug_idx, target_idx) for *all known* positives
    num_drugs,
    num_proteins,
    criterion,
    device='cpu'
):
    """
    Perform negative-sampling-based test evaluation 
    on a balanced set of pos/neg edges from data.edge_index.
    Returns AUC, AUPR, plus precision, recall, F1, MCC, etc.
    """
    model_link.eval()
    mlp_classifier.eval()

    with torch.no_grad():
        # 1) GNN forward pass to get node embeddings
        z = model_link(data.x, data.edge_index)

        # 2) Collect positive edges in test_data
        pos_src, pos_dst = get_positive_drug_target_edges(data.edge_index, num_drugs)
        pos_src = pos_src.to(device)
        pos_dst = pos_dst.to(device)
        num_pos = pos_src.size(0)

        # 3) Sample the same number of negative edges
        neg_src_list = []
        neg_dst_list = []
        tries = 0
        max_tries = 50 * num_pos
        while len(neg_src_list) < num_pos and tries < max_tries:
            tries += 1
            d = random.randint(0, num_drugs - 1)
            t = random.randint(0, num_proteins - 1)
            if (d, t) not in adjacency_set:
                neg_src_list.append(d)
                neg_dst_list.append(num_drugs + t)

        neg_src = torch.tensor(neg_src_list, dtype=torch.long, device=device)
        neg_dst = torch.tensor(neg_dst_list, dtype=torch.long, device=device)

        # 4) Build combined edge list (pos + neg)
        all_src = torch.cat([pos_src, neg_src], dim=0)
        all_dst = torch.cat([pos_dst, neg_dst], dim=0)

        labels_pos = torch.ones(num_pos, device=device)
        labels_neg = torch.zeros(neg_src.size(0), device=device)
        all_labels = torch.cat([labels_pos, labels_neg], dim=0)

        # 5) MLP forward pass
        src_emb = z[all_src]
        dst_emb = z[all_dst]
        mlp_input = torch.cat([src_emb, dst_emb], dim=1)
        # preds = mlp_classifier(mlp_input).squeeze()
        probs_test = mlp_classifier(mlp_input, use_gumbel=False, approx_mean=True)

        # 6) BCE loss
        loss = criterion(probs_test, all_labels).item()

        # 7) Predictions & metrics
        probs = probs_test.cpu().numpy()  # probabilities in [0,1]
        labels_np = all_labels.cpu().numpy()
        binary_pred = (probs >= 0.5).astype(int)

        # 4) 做 t-SNE 可视化（可选地画两张图：一个按真实标签着色，一个按预测概率着色）
        # 按真实标签着色
        visualize_pair_embeddings(
            pair_emb=mlp_input,
            color_values=labels_np,
            out_path=f"code/20250416/20250420_GCNGCNGAT_712_gs/mlp_input_tsne_by_label_{neg_ratio}.png",
            title="MLP Input TSNE (Color by True Label)"
        )

        # 按预测概率着色
        visualize_pair_embeddings(
            pair_emb=mlp_input,
            color_values=probs,
            out_path=f"code/20250416/20250420_GCNGCNGAT_712_gs/mlp_input_tsne_by_prob_{neg_ratio}.png",
            title="MLP Input TSNE (Color by Predicted Probability)"
        )

        # AUC & AUPR
        fpr, tpr, thresholds = roc_curve(labels_np, probs)
        auc_roc = roc_auc_score(labels_np, probs)
        auc_pr  = average_precision_score(labels_np, probs)

        # Accuracy, precision, recall, F1, MCC
        acc       = accuracy_score(labels_np, binary_pred)
        precision = precision_score(labels_np, binary_pred, zero_division=0)
        recall    = recall_score(labels_np, binary_pred, zero_division=0)
        f1        = f1_score(labels_np, binary_pred, zero_division=0)
        mcc       = matthews_corrcoef(labels_np, binary_pred)

    return {
        'loss': loss,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'probs': probs,
        'labels_np': labels_np,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def evaluate_model_negative_sampling(
    model_link, 
    mlp_classifier, 
    data,                 # e.g. val_data or test_data
    adjacency_set,        # set of (drug_idx, prot_idx) that are known positives (train+val+test or just train)
    criterion, 
    num_drugs, 
    num_proteins,
    epoch,
    device='cpu'
):
    """
    Approximate evaluation of link prediction metrics by sampling negatives
    equal to the number of positive edges in 'data.edge_index'.
    
    Args:
        model_link: your GNN LinkPredictionModel
        mlp_classifier: the MLP decoder
        data (Data): PyG Data with x (node embeddings), edge_index (positive edges)
        adjacency_set (set): set of (drug_idx, target_idx) that are known positives
                             in the entire dataset or at least in this split
        criterion: BCE loss (e.g. nn.BCEWithLogitsLoss())
        num_drugs, num_proteins: integer counts
        device: 'cpu' or 'cuda'

    Returns:
        loss (float),
        auc (float),
        aupr (float)
    """
    model_link.eval()
    mlp_classifier.eval()

    with torch.no_grad():
        # 1) Get GNN embeddings
        z = model_link(data.x, data.edge_index)

        # 2) Gather positive edges from data.edge_index
        pos_src, pos_dst = get_positive_drug_target_edges(data.edge_index, num_drugs)
        pos_src = pos_src.to(device)
        pos_dst = pos_dst.to(device)
        num_pos = pos_src.size(0)

        # 3) Sample the same number of negative edges
        #    e.g. if num_pos=100, we get 100 negative edges
        neg_src_list = []
        neg_dst_list = []
        tries = 0
        max_tries = 50 * num_pos  # avoid infinite loops
        while len(neg_src_list) < num_pos and tries < max_tries:
            tries += 1
            d = random.randint(0, num_drugs-1)
            t = random.randint(0, num_proteins-1)
            if (d, t) not in adjacency_set:
                neg_src_list.append(d)
                neg_dst_list.append(num_drugs + t)
        # Convert to tensors
        neg_src = torch.tensor(neg_src_list, dtype=torch.long).to(device)
        neg_dst = torch.tensor(neg_dst_list, dtype=torch.long).to(device)

        # 4) Build MLP input for pos and neg
        all_src = torch.cat([pos_src, neg_src], dim=0)
        all_dst = torch.cat([pos_dst, neg_dst], dim=0)

        # label 1 for positive, 0 for negative
        labels_pos = torch.ones(num_pos, device=device)
        labels_neg = torch.zeros(len(neg_src_list), device=device)
        all_labels = torch.cat([labels_pos, labels_neg], dim=0)

        # 5) Pass to MLP
        src_emb = z[all_src]  # [pos_count + neg_count, hidden_dim]
        dst_emb = z[all_dst]  # same shape
        mlp_input = torch.cat([src_emb, dst_emb], dim=1)
        probs_val = mlp_classifier(mlp_input, use_gumbel=False, approx_mean=True)

        # 6) Compute BCE loss
        loss = criterion(probs_val, all_labels).item()

        # 7) Compute approximate AUC & AUPR
        probabilities = probs_val.cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        auc  = roc_auc_score(labels_np, probabilities)
        aupr = average_precision_score(labels_np, probabilities)

    # return loss, auc, aupr 
    return {
        'loss': loss,
        'auc_roc': auc,
        'auc_pr': aupr,
        'probs': probabilities,      # <<< NEW
        'labels_np': labels_np       # <<< NEW
    }





def manual_link_split(edge_index, edge_label, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    
    E = edge_index.size(1)
    perm = torch.randperm(E)  # shuffle E edges
    
    train_end = math.floor(train_ratio * E)
    val_end   = train_end + math.floor(val_ratio * E)
    
    train_idx = perm[:train_end]
    val_idx   = perm[train_end:val_end]
    test_idx  = perm[val_end:]
    
    train_edge_index = edge_index[:, train_idx]
    train_edge_label = edge_label[train_idx]
    
    val_edge_index = edge_index[:, val_idx]
    val_edge_label = edge_label[val_idx]
    
    test_edge_index = edge_index[:, test_idx]
    test_edge_label = edge_label[test_idx]
    
    return (train_edge_index, train_edge_label,
            val_edge_index,   val_edge_label,
            test_edge_index,  test_edge_label)

# def create_data_split(full_graph, edge_index, edge_label):
#     data_split = Data()
#     data_split.x = full_graph.x
#     data_split.num_nodes = full_graph.num_nodes
#     data_split.edge_index = edge_index
#     data_split.edge_label = edge_label
#     return data_split

def get_positive_drug_target_edges(edge_index, num_drugs):
    """
    edge_index: shape [2, E], containing only positive edges (undirected).
    Returns pos_src, pos_dst (each shape = [E/2]) for (drug->protein) edges.
    Note: Because your graph is undirected (two directions),
          you'll see each edge twice (e.g. drug->target and target->drug).
          We'll keep only the drug->target direction to avoid duplication in BCE.
    """
    pos_src = []
    pos_dst = []
    E = edge_index.size(1)
    for i in range(E):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        # Keep only (drug, target) with drug < num_drugs < target
        if src < num_drugs and dst >= num_drugs:
            pos_src.append(src)
            pos_dst.append(dst)

    return torch.tensor(pos_src, dtype=torch.long), torch.tensor(pos_dst, dtype=torch.long)


# =============== 1) Custom Datasets ===============
class DrugDataset(Dataset):
    def __init__(self, drug_list):
        super().__init__()
        self.drug_list = drug_list  # list of Data (PyG)

    def __len__(self):
        return len(self.drug_list)

    def __getitem__(self, idx):
        return self.drug_list[idx], idx

class ProteinDataset(Dataset):
    def __init__(self, protein_list):
        super().__init__()
        self.protein_list = protein_list

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        return self.protein_list[idx], idx

def collate_fn(batch):
    graphs = [b[0] for b in batch]
    idxs   = [b[1] for b in batch]
    return graphs, torch.tensor(idxs, dtype=torch.long)

# ========== create_interaction_matrix, prepare_mlp_features_and_labels, etc. ==========
# def create_interaction_matrix(data, num_drugs, num_proteins):
#     interaction_matrix = torch.zeros((num_drugs, num_proteins), device=data.edge_index.device)
#     for i in range(data.edge_index.size(1)):
#         src, dest = data.edge_index[:, i]
#         if 0 <= src < num_drugs and num_drugs <= dest < num_drugs + num_proteins:
#             interaction_matrix[src, dest - num_drugs] = data.edge_label[i]
#         else:
#             print(f"Warning: Edge ({src}, {dest}) is outside the drug-protein range.")
#     return interaction_matrix

# def hard_negative_mining(embeddings, num_drugs, num_proteins, num_samples, adjacency_set, threshold=0.4):
#     neg_src_list = []
#     neg_dst_list = []
#     for _ in range(num_samples * 50):
#         d = random.randint(0, num_drugs - 1)
#         t = random.randint(0, num_proteins - 1)
#         if (d, t) not in adjacency_set:
#             # Calculate similarity score
#             normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
#             similarity_score = (normalized_embeddings[d] * normalized_embeddings[num_drugs + t]).sum().item()
#             # similarity_score = (embeddings[d] * embeddings[num_drugs + t]).sum().item()
#             # print(f"Similarity score for (drug {d}, protein {t}): {similarity_score}")
#             if similarity_score < threshold:
#                 neg_src_list.append(d)
#                 neg_dst_list.append(num_drugs + t)
#             if len(neg_src_list) >= num_samples:
#                 break
#     return torch.tensor(neg_src_list), torch.tensor(neg_dst_list), similarity_score

def dynamic_negative_sampling(mlp_classifier, embeddings, num_drugs, num_proteins, 
                              num_samples, adjacency_set, device='cpu'):
    neg_src_list = []
    neg_dst_list = []

    tries = 0
    max_tries = num_samples * 10

    while len(neg_src_list) < num_samples and tries < max_tries:
        tries += 1

        # Random sampling of drug and target
        d = random.randint(0, num_drugs - 1)
        t = random.randint(0, num_proteins - 1)
        
        # Check if it's a known positive
        if (d, t) in adjacency_set:
            continue

        neg_src_list.append(d)
        neg_dst_list.append(num_drugs + t)

    # Make sure to move tensors to the specified device
    neg_src_tensor = torch.tensor(neg_src_list, device=device)
    neg_dst_tensor = torch.tensor(neg_dst_list, device=device)
    
    if len(neg_src_list) == 0:
        return neg_src_tensor, neg_dst_tensor
    
    # Calculate prediction scores
    src_emb = embeddings[neg_src_tensor]
    dst_emb = embeddings[neg_dst_tensor]
    
    # Construct MLP input
    mlp_input = torch.cat([src_emb, dst_emb], dim=1)
    with torch.no_grad():
        preds = mlp_classifier(mlp_input, use_gumbel=False, approx_mean=True)
    print("The length of preds: ", len(preds))
    
    # Select hard negatives (highest scoring)
    topk_indices = torch.topk(preds, k=min(num_samples, len(preds))).indices
    hard_neg_src = neg_src_tensor[topk_indices]
    hard_neg_dst = neg_dst_tensor[topk_indices]

    return hard_neg_src, hard_neg_dst


def create_interaction_matrix(data, num_drugs, num_proteins):
    """
    Build a (num_drugs x num_proteins) matrix from an undirected bipartite graph.
    For each edge, if (src, dest) is (drug, protein) or (protein, drug),
    we fill the matrix with the edge_label. 
    """
    matrix = torch.zeros((num_drugs, num_proteins), device=data.edge_index.device)
    E = data.edge_index.size(1)
    
    for i in range(E):
        src, dest = data.edge_index[:, i]
        label = data.edge_label[i]
        
        # Case 1: src is drug, dest is protein
        if 0 <= src < num_drugs and num_drugs <= dest < num_drugs + num_proteins:
            matrix[src, dest - num_drugs] = label

        # Case 2: dest is drug, src is protein (the reversed edge)
        elif 0 <= dest < num_drugs and num_drugs <= src < num_drugs + num_proteins:
            matrix[dest, src - num_drugs] = label
        
        else:
            print(f"Warning: Edge ({src}, {dest}) is outside the drug-protein range.")

    return matrix

# def sample_negative_edges(num_drugs, num_targets, 
#                           adjacency_set, num_samples):
#     """
#     adjacency_set: a Python set of (drug_idx, target_idx) that are positive
#     num_samples: how many negative edges to sample

#     Returns: (neg_src, neg_dst) as Tensors of length = num_samples
#     """
#     neg_src_list = []
#     neg_dst_list = []
#     max_tries = 10 * num_samples  # just for safety

#     tries = 0
#     count = 0
#     while count < num_samples and tries < max_tries:
#         tries += 1
#         d = random.randint(0, num_drugs-1)
#         t = random.randint(0, num_targets-1)
#         if (d, t) not in adjacency_set:
#             neg_src_list.append(d)
#             # for bipartite, the target node in overall index is offset by num_drugs
#             neg_dst_list.append(num_drugs + t)
#             count += 1

#     neg_src = torch.tensor(neg_src_list, dtype=torch.long)
#     neg_dst = torch.tensor(neg_dst_list, dtype=torch.long)
#     return neg_src, neg_dst

def sample_negative_edges(num_drugs, num_targets, adjacency_set, num_samples):
    """
    Sample random negatives as a fallback when hard negatives are insufficient.
    """
    neg_src_list = []
    neg_dst_list = []
    max_tries = 10 * num_samples
    sampled_pairs = set()
    tries = 0

    while len(neg_src_list) < num_samples and tries < max_tries:
        tries += 1
        d = random.randint(0, num_drugs-1)
        t = random.randint(0, num_targets-1)
        if (d, t) not in adjacency_set and (d, t) not in sampled_pairs:
            neg_src_list.append(d)
            neg_dst_list.append(num_drugs + t)
            sampled_pairs.add((d, t))
    
    return torch.tensor(neg_src_list), torch.tensor(neg_dst_list)

def prepare_mlp_features_and_labels(embeddings, interaction_matrix):
    num_drugs, num_proteins = interaction_matrix.shape
    feature_vectors = []
    labels = []
    for i in range(num_drugs):
        for j in range(num_proteins):
            combined_features = torch.cat((embeddings[i], embeddings[num_drugs + j]), dim=0)
            feature_vectors.append(combined_features)
            labels.append(interaction_matrix[i, j].item())
    return torch.stack(feature_vectors), torch.tensor(labels, dtype=torch.float32)

# ========== Get all drug/protein embeddings ========== 
def get_all_drug_embeddings_train(model_d, drug_loader, embedding_dim):
    model_d.train()
    num_drugs = len(drug_loader.dataset)
    all_drug_embeddings = torch.zeros((num_drugs, embedding_dim), device=device)
    for graphs, idxs in drug_loader:
        # print("idx: ", idxs)
        batch = graphs.to(device)
        emb_batch = model_d(batch)  
        for b_i, real_idx in enumerate(idxs):
            all_drug_embeddings[real_idx] = emb_batch[b_i]
    return all_drug_embeddings

def get_all_protein_embeddings_train(model_p, protein_loader, embedding_dim):
    model_p.train()
    num_proteins = len(protein_loader.dataset)
    all_protein_embeddings = torch.zeros((num_proteins, embedding_dim), device=device)

    for graphs, idxs in protein_loader:
        batch = graphs.to(device)
        emb_batch = model_p(batch)  
        for b_j, real_idx in enumerate(idxs):
            all_protein_embeddings[real_idx] = emb_batch[b_j]
    return all_protein_embeddings


# ========================================================
def get_all_drug_embeddings(model_d, drug_loader, embedding_dim):
    model_d.eval()
    num_drugs = len(drug_loader.dataset)
    all_drug_embeddings = torch.zeros((num_drugs, embedding_dim), device=device)

    with torch.no_grad():
        for graphs, idxs in drug_loader:
            batch = graphs.to(device)
            emb_batch = model_d(batch)  
            for b_i, real_idx in enumerate(idxs):
                all_drug_embeddings[real_idx] = emb_batch[b_i]
    return all_drug_embeddings


def get_all_protein_embeddings(model_p, protein_loader, embedding_dim):
    model_p.eval()
    num_proteins = len(protein_loader.dataset)
    all_protein_embeddings = torch.zeros((num_proteins, embedding_dim), device=device)

    with torch.no_grad():
        for graphs, idxs in protein_loader:
            batch = graphs.to(device)
            emb_batch = model_p(batch)  
            for b_j, real_idx in enumerate(idxs):
                all_protein_embeddings[real_idx] = emb_batch[b_j]
    return all_protein_embeddings

# ========== Evaluate Model ========== 
def evaluate_model(model_link, mlp_classifier, data, interaction_matrix, criterion):
    model_link.eval()
    mlp_classifier.eval()
    with torch.no_grad():
        final_embeddings = model_link(data.x, data.edge_index)
        mlp_features, mlp_labels = prepare_mlp_features_and_labels(final_embeddings, interaction_matrix)
        mlp_features = mlp_features.to(device)
        mlp_labels   = mlp_labels.to(device)

        predictions = mlp_classifier(mlp_features).squeeze()
        loss = criterion(predictions, mlp_labels).item()

        prob = predictions.sigmoid().cpu().numpy()
        labels = mlp_labels.cpu().numpy()

        auc  = roc_auc_score(labels, prob)
        aupr = average_precision_score(labels, prob)

    return loss, auc, aupr

# ========== Main ========== 
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neg_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 
                  0.6, 0.7, 0.8, 0.9, 1.0]
    tau = []

    results_dict = {
        'neg_ratio': [],
        'AUC': [],
        'AUPR': [],
        'Acc': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'MCC': []
    }

    for ratio in neg_ratios:
        print("\n==============================")
        print(f" Running experiment with neg_ratio = {ratio}")
        print("==============================\n")
        
        metrics = train_and_evaluate(
            neg_ratio=ratio,
            device=device,
            EPOCHS=1000  # you can shorten epochs to speed up
        )
        print("Under neg_ratio: ", ratio)
        print("current tau: ", metrics['current_tau'])
        tau.append(metrics['current_tau'])
        # Store the final test metrics
        results_dict['neg_ratio'].append(ratio)
        results_dict['AUC'].append(metrics['test_auc'])
        results_dict['AUPR'].append(metrics['test_aupr'])
        results_dict['Acc'].append(metrics['acc'])
        results_dict['Precision'].append(metrics['precision'])
        results_dict['Recall'].append(metrics['recall'])
        results_dict['F1'].append(metrics['f1'])
        results_dict['MCC'].append(metrics['mcc'])

    # After finishing all experiments, plot or print the table
    import matplotlib.pyplot as plt

    # Plot AUC vs. neg_ratio
    plt.figure()
    plt.plot(results_dict['neg_ratio'], results_dict['AUC'], marker='o', label='AUC')
    plt.plot(results_dict['neg_ratio'], results_dict['AUPR'], marker='x', label='AUPR')
    plt.xlabel("Negative Ratio (negatives : positives)")
    plt.ylabel("Metric Value")
    plt.title("Test AUC and AUPR vs. Negative Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig("code/20250416/20250420_GCNGCNGAT_712_gs/negative_ratio_auc_aupr.png")
    plt.close()

    plt.figure()
    plt.plot(results_dict['neg_ratio'], results_dict['Acc'], marker='o', label='Accuracy')
    plt.plot(results_dict['neg_ratio'], results_dict['Precision'], marker='x', label='precision')
    plt.plot(results_dict['neg_ratio'], results_dict['Recall'], marker='*', label='recall')
    plt.xlabel("Negative Ratio (negatives : positives)")
    plt.ylabel("Metric Value")
    plt.title("Test Acc, Precision and Recall vs. Negative Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig("code/20250416/20250420_GCNGCNGAT_712_gs/negative_ratio_acc_precision_recall.png")
    plt.close()

    # (Optional) Similarly, plot F1, MCC vs. neg_ratio
    plt.figure()
    plt.plot(results_dict['neg_ratio'], results_dict['F1'], marker='o', label='F1')
    plt.plot(results_dict['neg_ratio'], results_dict['MCC'], marker='x', label='MCC')
    plt.xlabel("Negative Ratio")
    plt.ylabel("Metric Value")
    plt.title("Test F1 and MCC vs. Negative Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig("code/20250416/20250420_GCNGCNGAT_712_gs/negative_ratio_f1_mcc.png")
    plt.close()

    # Optionally print the raw results
    print("\nFinal summary of test metrics:")
    for i, ratio in enumerate(results_dict['neg_ratio']):
        print(f"neg_ratio={ratio}, "
              f"AUC={results_dict['AUC'][i]:.4f}, "
              f"AUPR={results_dict['AUPR'][i]:.4f}, "
              f"Acc={results_dict['Acc'][i]:.4f}, "
              f"Precision={results_dict['Precision'][i]:.4f}, "
              f"Recall={results_dict['Recall'][i]:.4f}, "
              f"F1={results_dict['F1'][i]:.4f}, "
              f"MCC={results_dict['MCC'][i]:.4f}")
    print(tau)


if __name__ == "__main__":
    main()
