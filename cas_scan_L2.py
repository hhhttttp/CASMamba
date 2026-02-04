import torch
import torch.nn.functional as F
import math


# --- 辅助函数 (无变化) ---
def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    计算两组向量之间的成对余弦相似度。
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))


# --- 扫描函数 (已修改为您需要的逻辑) ---

def cas_scan(x: torch.Tensor, num_clusters: int = 16, reverse: bool = False):
    """
    基于特征相似度的内容感知扫描（Content-Aware Scan, CAS）。
    此函数接收一个 [B, L, C] 形状的张量。

    修改后的逻辑：
    - 簇间排序: 依然按照聚类中心的L2范数降序排列。
    - 簇内排序:
        - 奇数顺序的簇 (第1, 3, 5...个): 按余弦相似度从大到小 (降序)。
        - 偶数顺序的簇 (第2, 4, 6...个): 按余弦相似度从小到大 (升序)。

    输入:
        x: [B, L, C] 形状的张量。
        num_clusters: 要形成的簇的数量（必须是平方数）。
        reverse: (此参数在此修改版中作用被固定逻辑覆盖，但为保持接口一致性而保留)
                 当为 True 时，执行全局反向扫描。
    返回:
        一个元组 (scanned_x, sort_indices):
        - scanned_x: 重排后的数据，格式为 [B, C, L]。
        - sort_indices: `cas_reverse` 函数恢复顺序所需的索引。
    """
    B, L, C = x.shape
    H = W = int(math.sqrt(L))
    if H * W != L:
        raise ValueError("输入序列长度L必须是平方数。")

    k = int(math.sqrt(num_clusters))
    if k * k != num_clusters:
        raise ValueError(f"num_clusters 必须是平方数, 但得到的是 {num_clusters}")

    # 1. 使用函数式的自适应平均池化来提案聚类中心
    x_2d = x.transpose(1, 2).reshape(B, C, H, W)
    centers_2d = F.adaptive_avg_pool2d(x_2d, (k, k))
    centers = centers_2d.flatten(2).transpose(1, 2)

    # 2. 分配簇ID并获取相似度
    sim_matrix = pairwise_cos_sim(x, centers)
    cluster_ids = torch.argmax(sim_matrix, dim=2)

    # 3. 确定簇间顺序（按L2范数），这部分逻辑不变
    center_norms = torch.norm(centers, p=2, dim=-1)
    _, cluster_order = torch.sort(center_norms, dim=1, descending=True)

    # 4. 确定簇内顺序（按余弦相似度）并创建最终排序键
    # cluster_rank 表示每个簇在扫描顺序中的排名 (0, 1, 2, ...)
    cluster_rank = torch.empty_like(cluster_order)
    cluster_rank.scatter_(1, cluster_order,
                          torch.arange(num_clusters, device=x.device, dtype=cluster_order.dtype).expand(B, -1))

    # token_cluster_rank 表示每个token所属簇的扫描排名
    token_cluster_rank = torch.gather(cluster_rank, 1, cluster_ids)

    # intra_cluster_sim 表示每个token与其簇中心的相似度
    intra_cluster_sim = torch.gather(sim_matrix, 2, cluster_ids.unsqueeze(-1)).squeeze(-1)

    # ########################### 核心修改点 ###########################
    #
    # 用户要求：
    # - 奇数簇 (rank 0, 2...): 降序 (从大到小)
    # - 偶数簇 (rank 1, 3...): 升序 (从小到大)
    #
    # 我们将使用 torch.sort(..., descending=True)
    #
    # 1. 创建一个调整项 (modifier)，默认值为 +1.0
    #    (这将用于奇数簇: key = rank + sim, 降序排 = sim 降序)
    alternating_modifier = torch.ones_like(token_cluster_rank, dtype=torch.float32)

    # 2. 找到所有属于偶数扫描顺序 (第2, 4, 6...个) 的token
    #    这些簇的排名 (rank) 是 1, 3, 5...
    is_even_scan_order = (token_cluster_rank % 2) != 0

    # 3. 将这些 (偶数簇) token 的调整项设为 -1.0
    #    (这将用于偶数簇: key = rank - sim, 降序排 = sim 升序)
    alternating_modifier[is_even_scan_order] = -1.0

    # 4. 应用调整项，构建新的排序键
    sorting_key = token_cluster_rank.float() + alternating_modifier * intra_cluster_sim.float()
    #
    # ##################################################################

    # 5. 获取最终排序索引并应用
    # 统一使用 descending=True 进行排序。
    # - 奇数簇 (rank 0, 2...): key = rank + sim。key越大 -> sim越大 -> 实现降序
    # - 偶数簇 (rank 1, 3...): key = rank - sim。key越大 -> sim越小 -> 实现升序
    # reverse 参数现在可以用于全局反转这个交替顺序
    _, final_sort_indices = torch.sort(sorting_key, dim=1, descending=(not reverse))
    sorted_x = torch.gather(x, 1, final_sort_indices.unsqueeze(-1).expand(-1, -1, C))

    # 返回 [B, C, L] 格式的扫描后数据和索引
    return sorted_x.transpose(-1, -2), final_sort_indices


# --- 封装函数 (无变化) ---

def cas_scan_bchw(x: torch.Tensor, num_clusters: int = 16, reverse: bool = False):
    """
    针对 [B, C, H, W] 张量的内容感知扫描（CAS）。
    这是一个对 `cas_scan` 的便捷封装。
    """
    B, C, H, W = x.shape
    x_bld = x.flatten(2).transpose(-1, -2)
    # 此函数无需修改，直接调用新的 cas_scan 即可
    return cas_scan(x_bld, num_clusters=num_clusters, reverse=reverse)


# --- 恢复函数 (无变化) ---

def cas_reverse(x: torch.Tensor, sort_indices: torch.Tensor):
    """
    使用 `cas_scan` 提供的索引来反转内容感知扫描。
    此函数无需修改，因为它仅依赖索引来恢复顺序，与扫描方向无关。
    """
    processed_sorted_x = x.transpose(-1, -2)
    B, L, C = processed_sorted_x.shape
    reversed_x_bld = torch.zeros_like(processed_sorted_x)
    index_for_scatter = sort_indices.unsqueeze(-1).expand(-1, -1, C)
    reversed_x_bld.scatter_(1, index_for_scatter, processed_sorted_x)
    return reversed_x_bld.transpose(-1, -2)


# --- 使用示例 ---
if __name__ == '__main__':
    # 创建一些随机输入数据
    B, C, H, W = 2, 16, 8, 8  # L = H * W = 64
    L = H * W
    num_clusters = 16  # 簇数量

    # BLC 格式的输入
    x_blc = torch.randn(B, L, C)

    # BCHW 格式的输入
    x_bchw = torch.randn(B, C, H, W)

    print(f"输入形状 (BLC): {x_blc.shape}")
    print(f"输入形状 (BCHW): {x_bchw.shape}")
    print("-" * 30)

    # 1. 测试 BLC 输入的新扫描函数
    scanned_blc, indices_blc = cas_scan(x_blc, num_clusters=num_clusters)
    reversed_blc = cas_reverse(scanned_blc, indices_blc)

    print("测试 BLC 输入:")
    print(f"扫描后形状: {scanned_blc.shape}")
    print(f"恢复后形状: {reversed_blc.transpose(-1, -2).shape}")
    # 验证恢复是否成功
    print(f"恢复是否成功 (BLC): {torch.allclose(reversed_blc.transpose(-1, -2), x_blc, atol=1e-6)}")
    print("-" * 30)

    # 2. 测试 BCHW 输入的封装函数
    scanned_bchw, indices_bchw = cas_scan_bchw(x_bchw, num_clusters=num_clusters)
    reversed_bchw = cas_reverse(scanned_bchw, indices_bchw)

    print("测试 BCHW 输入:")
    print(f"扫描后形状: {scanned_bchw.shape}")
    # 恢复后的形状是 [B, C, L]，需要 reshape 回 [B, C, H, W]
    reversed_bchw_reshaped = reversed_bchw.reshape(B, C, H, W)
    print(f"恢复后形状: {reversed_bchw_reshaped.shape}")
    # 验证恢复是否成功
    print(f"恢复是否成功 (BCHW): {torch.allclose(reversed_bchw_reshaped, x_bchw, atol=1e-6)}")