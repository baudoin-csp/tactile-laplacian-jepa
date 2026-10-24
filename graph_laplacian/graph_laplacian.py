"""
Graph Laplacian of the ViT patch grid.

Sparsh turns a (B, 6, 320, 240) input into (B, 300, 768) tokens. Token i is the 16x16 patch at
(row, col) = (i // W, i % W) of the 20 x 15 grid (row-major, as produced by the patch embedding):

    node 0   = top-left      node W-1     = top-right       (W = 15 -> node 14)
    node N-W = bottom-left   node N-1     = bottom-right    (N = 300 -> nodes 285, 299)

Nodes are the patches, edges link neighbouring patches. Only the placement on the grid matters,
not the token values, so without masking the Laplacian is the same for every sample.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def grid_adjacency(grid_size: Tuple[int, int], connectivity: int = 4) -> torch.Tensor:
    """(N, N) binary adjacency of an H x W grid, N = H * W, row-major node order.
    connectivity=4: patches sharing a side; connectivity=8: also diagonal patches."""
    assert connectivity in (4, 8), "connectivity must be 4 or 8"
    H, W = grid_size
    rows, cols = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    rows, cols = rows.flatten(), cols.flatten()  # coordinates of node i = row * W + col
    d_row = (rows[:, None] - rows[None, :]).abs()
    d_col = (cols[:, None] - cols[None, :]).abs()
    if connectivity == 4:
        adjacency = d_row + d_col == 1
    else:
        adjacency = torch.maximum(d_row, d_col) == 1
    return adjacency.float()


def laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """L = D - A for (..., N, N) adjacency matrices, with D the diagonal degree matrix."""
    return torch.diag_embed(adjacency.sum(dim=-1)) - adjacency


class GridLaplacian(nn.Module):
    """Batched graph Laplacian L = D - A of the patch grid.

    Args:
        grid_size: (H, W) patch grid, e.g. (20, 15) for a 320 x 240 input with 16 x 16 patches.
        connectivity: 4 (default, strict up/down/left/right) or 8 (adds diagonals).

    forward(batch_size, keep_idx=None):
        keep_idx=None:    (B, N, N), the full-grid Laplacian for every sample (a broadcast view, no copy).
        keep_idx (B, K):  (B, K, K), the Laplacian of the sub-graph of kept patches (e.g. JEPA context).
                          Row/column k corresponds to token keep_idx[b, k], the same order as apply_masks.
                          Degrees are recomputed on the sub-graph, i.e. this is not L[keep][:, keep].
    """

    def __init__(self, grid_size: Tuple[int, int], connectivity: int = 4):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.num_nodes = self.grid_size[0] * self.grid_size[1]
        self.connectivity = connectivity
        self.register_buffer(
            "adjacency", grid_adjacency(self.grid_size, connectivity), persistent=False
        )
        self.register_buffer("laplacian", laplacian(self.adjacency), persistent=False)

    def forward(
        self, batch_size: int = 1, keep_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if keep_idx is None:
            # Same L for every sample, returned as a (B, N, N) view of one matrix: no memory is copied.
            # Do not modify it in place (it would change the stored L); call .clone() first if needed.
            return self.laplacian.expand(batch_size, -1, -1)
        sub_adjacency = self.adjacency[
            keep_idx[:, :, None], keep_idx[:, None, :]
        ]  # (B, K, K)
        return laplacian(sub_adjacency)
