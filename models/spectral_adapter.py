import torch
import torch.nn as nn
from timm.models.layers import DropPath


def get_laplacian(adj_matrix, normalize=True):
    """
    Compute the graph Laplacian matrix.

    Args:
        adj_matrix (torch.Tensor): The adjacency matrix (batch_size, vertices, vertices).
        normalize (bool): Whether to compute the normalized Laplacian.

    Returns:
        torch.Tensor: The Laplacian matrix (batch_size, vertices, vertices).
    """
    if normalize:
        # Degree matrix: sum of rows
        D = torch.sum(adj_matrix, dim=-1)  # (batch_size, vertices)
        # Avoid division by zero by adding epsilon to D
        D_inv_sqrt = torch.rsqrt(D + 1e-6)  # Inverse square root
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)  # Batch-wise diagonal matrices
        # Normalized Laplacian
        L = torch.eye(adj_matrix.size(-1), device=adj_matrix.device) - \
            D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    else:
        # Degree matrix
        D = torch.sum(adj_matrix, dim=-1)  # (batch_size, vertices)
        D = torch.diag_embed(D)  # Batch-wise diagonal matrices
        # Unnormalized Laplacian
        L = D - adj_matrix
    return L

def get_basis(center):
    L = torch.cdist(center, center)
    L = 1 / (L / torch.min(L[L > 0], dim=-1, keepdim=True).values + torch.eye(L.size(-1), device=L.device).unsqueeze(0))
    L = get_laplacian(L)
    _, U = torch.linalg.eigh(L)
    return U # This should be "U.transpose(-2, -1)", we keep it for reproducing our results in paper.

class SpectralAdapter(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.rank = cfg.rank

        num_heads = 6
        attn_dropout = 0.0
        self.norm_ly1 = nn.LayerNorm(self.rank)

        self.act = nn.SiLU()
        self.down = nn.Linear(dim, self.rank)
        self.up = nn.Linear(self.rank, dim)

        self.scale=1.

        self.attr_kv = nn.Linear(dim, self.rank, bias=False)  # 把 attr 映射到 rank 维
        self.xf_q    = nn.Linear(self.rank, self.rank, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.rank, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )

        self.cross_ln = nn.LayerNorm(self.rank)
        self.cross_drop = nn.Dropout(attn_dropout)

        self.adapt1 = nn.Linear(self.rank, self.rank)
        nn.init.zeros_(self.adapt1.weight)
        nn.init.zeros_(self.adapt1.bias)
        
        self.drop_adapt1 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, input, attr, U):
        h = self.down(input)


        x = h[:, 1:, :]
        h = self.act(h)
        
        x_f = U @ x

        attr_f = U @ attr
        h_f = x_f
        q = self.xf_q(self.cross_ln(x_f))         # [B, K, rank]
        kv = self.attr_kv(attr_f)                 # [B, K, rank]
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x_f = h_f + self.cross_drop(attn_out)

        h_f = x_f
        x_f = self.norm_ly1(x_f)
        x_f = h_f + self.drop_adapt1(self.act(self.drop_out(self.adapt1(x_f))))
        x = U.transpose(-2, -1) @ x_f
        
        h[:, 1:, :] = x + h[:, 1:, :]
        h = self.up(h)
        return h*self.scale
