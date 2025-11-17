# -----------------------------------------------------------
# graph_ntu.py — Official ST-GCN NTU RGB+D Graph (25 joints)
# -----------------------------------------------------------

import numpy as np
import torch

num_node = 25
self_links = [(i, i) for i in range(num_node)]

# Official ST-GCN "inward" list (1-based)
inward = [
    # spine & head
    (1,2), (2,21), (21,3), (3,4),

    # left arm
    (21,9), (9,10), (10,11), (11,12), (12,24), (12,25),

    # right arm
    (21,5), (5,6), (6,7), (7,8), (8,22), (8,23),

    # left leg
    (1,17), (17,18), (18,19), (19,20),

    # right leg
    (1,13), (13,14), (14,15), (15,16)
]

# convert 1-based → 0-based
inward = [(i-1, j-1) for (i, j) in inward]
outward = [(j, i) for (i, j) in inward]


# ---------- Build adjacency matrices ----------
def edge2mat(edges, num_node=25):
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[j, i] = 1
    return A


A_self = edge2mat(self_links, num_node)
A_in   = edge2mat(inward, num_node)
A_out  = edge2mat(outward, num_node)


# ---------- Normalize D^-1/2 * A * D^-1/2 ----------
def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    Dn = np.diag(np.power(Dl, -0.5))
    Dn[np.isinf(Dn)] = 0
    return Dn @ A @ Dn


A_norm = np.stack([
    normalize_digraph(A_self),
    normalize_digraph(A_in),
    normalize_digraph(A_out)
], axis=0)

# adjacency matrix ready for ST-GCN
A_torch = torch.tensor(A_norm, dtype=torch.float32)

if __name__ == "__main__":
    print("NTU A shape:", A_torch.shape)
