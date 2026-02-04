import torch
import torch.nn.functional as F
import math


# --- Helper Functions ---
def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise cosine similarity between two sets of vectors.
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))


# --- Scan Function (Modified logic) ---

def cas_scan(x: torch.Tensor, num_clusters: int = 16, reverse: bool = False):
    """
    Content-Aware Scan (CAS) based on feature similarity.
    This function processes a tensor of shape [B, L, C].

    Sorting Logic:
    - Inter-cluster: Sorted by the L2 norm of cluster centers in descending order.
    - Intra-cluster:
        - Odd-ranked clusters (1st, 3rd, 5th...): Sorted by cosine similarity (Descending).
        - Even-ranked clusters (2nd, 4th, 6th...): Sorted by cosine similarity (Ascending).

    Inputs:
        x: Tensor of shape [B, L, C].
        num_clusters: Number of clusters to form (must be a perfect square).
        reverse: If True, performs a global reverse of the final scanned sequence.

    Returns:
        A tuple (scanned_x, sort_indices):
        - scanned_x: Reordered data in [B, C, L] format.
        - sort_indices: Indices required for the `cas_reverse` function.
    """
    B, L, C = x.shape
    H = W = int(math.sqrt(L))
    if H * W != L:
        raise ValueError("Input sequence length L must be a perfect square.")

    k = int(math.sqrt(num_clusters))
    if k * k != num_clusters:
        raise ValueError(f"num_clusters must be a perfect square, but got {num_clusters}")

    # 1. Propose cluster centers using adaptive average pooling
    x_2d = x.transpose(1, 2).reshape(B, C, H, W)
    centers_2d = F.adaptive_avg_pool2d(x_2d, (k, k))
    centers = centers_2d.flatten(2).transpose(1, 2)

    # 2. Assign cluster IDs and compute similarity
    sim_matrix = pairwise_cos_sim(x, centers)
    cluster_ids = torch.argmax(sim_matrix, dim=2)

    # 3. Determine inter-cluster order (based on L2 norm)
    center_norms = torch.norm(centers, p=2, dim=-1)
    _, cluster_order = torch.sort(center_norms, dim=1, descending=True)

    # 4. Determine intra-cluster order and create final sorting keys
    # cluster_rank: The rank of each cluster in the scanning sequence (0, 1, 2, ...)
    cluster_rank = torch.empty_like(cluster_order)
    cluster_rank.scatter_(1, cluster_order,
                          torch.arange(num_clusters, device=x.device, dtype=cluster_order.dtype).expand(B, -1))

    # token_cluster_rank: The scan rank of the cluster each token belongs to
    token_cluster_rank = torch.gather(cluster_rank, 1, cluster_ids)

    # intra_cluster_sim: Similarity of each token to its assigned cluster center
    intra_cluster_sim = torch.gather(sim_matrix, 2, cluster_ids.unsqueeze(-1)).squeeze(-1)

    # --- Core Sorting Logic ---
    # Odd ranks (0, 2...): Descending (large sim first) -> key = rank + sim
    # Even ranks (1, 3...): Ascending (small sim first) -> key = rank - sim

    alternating_modifier = torch.ones_like(token_cluster_rank, dtype=torch.float32)

    # Identify tokens belonging to clusters with odd index in 1-based counting (rank 1, 3, 5...)
    is_even_rank = (token_cluster_rank % 2) != 0
    alternating_modifier[is_even_rank] = -1.0

    # Build the sorting key
    sorting_key = token_cluster_rank.float() + alternating_modifier * intra_cluster_sim.float()

    # 5. Execute final sort
    # We use descending=True globally:
    # - For rank 0: key = 0 + sim. Higher sim -> higher key -> comes first.
    # - For rank 1: key = 1 - sim. Lower sim -> higher key -> comes first.
    _, final_sort_indices = torch.sort(sorting_key, dim=1, descending=(not reverse))
    sorted_x = torch.gather(x, 1, final_sort_indices.unsqueeze(-1).expand(-1, -1, C))

    # Return in [B, C, L] format
    return sorted_x.transpose(-1, -2), final_sort_indices


# --- Wrapper Functions ---

def cas_scan_bchw(x: torch.Tensor, num_clusters: int = 16, reverse: bool = False):
    """
    Convenience wrapper for CAS on [B, C, H, W] tensors.
    """
    B, C, H, W = x.shape
    x_bld = x.flatten(2).transpose(-1, -2)
    return cas_scan(x_bld, num_clusters=num_clusters, reverse=reverse)


def cas_reverse(x: torch.Tensor, sort_indices: torch.Tensor):
    """
    Reverses the Content-Aware Scan using the provided indices.
    """
    processed_sorted_x = x.transpose(-1, -2)
    B, L, C = processed_sorted_x.shape
    reversed_x_bld = torch.zeros_like(processed_sorted_x)
    index_for_scatter = sort_indices.unsqueeze(-1).expand(-1, -1, C)
    reversed_x_bld.scatter_(1, index_for_scatter, processed_sorted_x)
    return reversed_x_bld.transpose(-1, -2)


# --- Example Usage ---
if __name__ == '__main__':
    B, C, H, W = 2, 16, 8, 8
    L = H * W
    num_clusters = 16

    # Input in BLC format
    x_blc = torch.randn(B, L, C)

    # Input in BCHW format
    x_bchw = torch.randn(B, C, H, W)

    print(f"Input Shape (BLC): {x_blc.shape}")
    print(f"Input Shape (BCHW): {x_bchw.shape}")
    print("-" * 30)

    # 1. Test BLC Input
    scanned_blc, indices_blc = cas_scan(x_blc, num_clusters=num_clusters)
    reversed_blc = cas_reverse(scanned_blc, indices_blc)

    print("Testing BLC Input:")
    print(f"Scanned Shape: {scanned_blc.shape}")
    print(f"Reconstruction Match (BLC): {torch.allclose(reversed_blc.transpose(-1, -2), x_blc, atol=1e-6)}")
    print("-" * 30)

    # 2. Test BCHW Input
    scanned_bchw, indices_bchw = cas_scan_bchw(x_bchw, num_clusters=num_clusters)
    reversed_bchw = cas_reverse(scanned_bchw, indices_bchw)

    reversed_bchw_reshaped = reversed_bchw.reshape(B, C, H, W)
    print("Testing BCHW Input:")
    print(f"Scanned Shape: {scanned_bchw.shape}")
    print(f"Reconstruction Match (BCHW): {torch.allclose(reversed_bchw_reshaped, x_bchw, atol=1e-6)}")