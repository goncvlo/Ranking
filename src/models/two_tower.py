import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BPRDataset(Dataset):
    def __init__(self, df_train, features, num_negatives: int = 4):
        self.df = df_train[df_train["rating"] > 0].reset_index(drop=True)
        self.user_idx = self.df["user_id_encoded"].to_numpy()
        self.item_idx = self.df["item_id_encoded"].to_numpy()

        self.user_feats = (
            features["user"]
            .drop(columns=["user_id", "user_id_encoded"])
            .to_numpy(dtype=np.float32)
            )
        self.item_feats = (
            features["item"]
            .drop(columns=["item_id", "item_id_encoded"])
            .to_numpy(dtype=np.float32)
            )
        
        # for negative sampling
        self.user_pos_items = (
            self.df
            .groupby("user_id_encoded")["item_id_encoded"]
            .apply(set).to_dict()
            )
        self.num_items = features["item"].shape[0]
        self.num_negatives = num_negatives

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_idx = row["user_id_encoded"]
        pos_idx = row["item_id_encoded"]

        # negative sampling
        neg_indices = []
        while len(neg_indices) < self.num_negatives:
            neg_idx = np.random.randint(0, self.num_items)
            if neg_idx not in self.user_pos_items[user_idx]:
                neg_indices.append(neg_idx)

        # convert to tensors
        user_vec = torch.from_numpy(self.user_feats[user_idx])
        pos_vec = torch.from_numpy(self.item_feats[pos_idx])
        neg_vec = torch.from_numpy(self.item_feats[neg_indices])
        
        return user_vec, pos_vec, neg_vec


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers=[128], dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        # normalize output embeddings here
        return torch.nn.functional.normalize(self.mlp(x), p=2, dim=-1)
    

class TwoTowerModel(torch.nn.Module):
    def __init__(self, user_dim, item_dim, embedding_dim=64, 
                 user_layers=[128], item_layers=[128], dropout=0.0):
        super().__init__()
        self.user_mlp = MLP(user_dim, embedding_dim, user_layers, dropout)
        self.item_mlp = MLP(item_dim, embedding_dim, item_layers, dropout)

    def forward(self, user_vec, item_vec):
        user_emb = self.user_mlp(user_vec)
        item_emb = self.item_mlp(item_vec)
        return user_emb, item_emb

    def dot_score(self, user_vec, item_vec):
        user_emb, item_emb = self.forward(user_vec, item_vec)
        return (user_emb * item_emb).sum(dim=1)
    

def bpr_loss_multi(user_vec, pos_vec, neg_vecs, model):
    batch_size, num_neg, _ = neg_vecs.size()
    
    user_emb, pos_emb = model(user_vec, pos_vec)      # [B, D]
    user_expanded = user_emb.unsqueeze(1).expand(-1, num_neg, -1)  # [B, K, D]

    neg_embs = model.item_mlp(neg_vecs.view(-1, neg_vecs.size(-1))).view(batch_size, num_neg, -1)  # [B, K, D]

    # scores (cosine similarity since embeddings normalized)
    pos_score = (user_emb * pos_emb).sum(dim=1, keepdim=True)      # [B, 1]
    neg_score = (user_expanded * neg_embs).sum(dim=2)              # [B, K]

    diff = pos_score - neg_score                                   # [B, K]
    return -torch.mean(torch.nn.functional.logsigmoid(diff))

