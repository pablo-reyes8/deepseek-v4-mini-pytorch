# ============================================================
# CSA LIGHTNING INDEXER
# ============================================================

from src.csa_components import * 

class CSALightningIndexer(nn.Module):
    """
    Lightning-style CSA indexer.

    Inputs:
        index_q: [B, T, H_i, I]
        index_weights: [B, T, H_i]
        I_comp: [B, S, I]
        comp_valid_mask: [B, S]

    Outputs:
        topk_indices: [B, T, K]
        topk_scores: [B, T, K]
        topk_mask: [B, T, K]
        index_scores: optional [B, T, S]
    """

    def __init__(
        self,
        compression_factor: int,
        top_k: int):

        super().__init__()

        if compression_factor <= 0:
            raise ValueError(
                f"compression_factor must be > 0, got {compression_factor}"
            )

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")

        self.compression_factor = compression_factor
        self.top_k = top_k

    def _build_global_allowed_mask(
        self,
        T: int,
        S: int,
        device: torch.device) -> torch.Tensor:

        """
        Conservative causal rule:
            allowed[t, s] = s < floor(t / compression_factor)

        Only fully completed compressed blocks are selectable. The current
        block is represented through the exact local sliding-window branch.
        """
        token_pos = torch.arange(T, device=device)
        query_block = token_pos // self.compression_factor
        block_pos = torch.arange(S, device=device)

        return block_pos[None, :] < query_block[:, None]

    def forward(
        self,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
        I_comp: torch.Tensor,
        comp_valid_mask: torch.Tensor,
        need_scores: bool = False):

        if index_q.dim() != 4:
            raise ValueError(
                f"index_q must have shape [B,T,H_i,I], got {tuple(index_q.shape)}"
            )

        if I_comp.dim() != 3:
            raise ValueError(
                f"I_comp must have shape [B,S,I], got {tuple(I_comp.shape)}"
            )

        B, T, H_i, I = index_q.shape
        B2, S, I2 = I_comp.shape

        if B2 != B:
            raise ValueError(f"Batch mismatch: index_q B={B}, I_comp B={B2}")

        if I2 != I:
            raise ValueError(f"Indexer dim mismatch: index_q I={I}, I_comp I={I2}")

        if index_weights.shape != (B, T, H_i):
            raise ValueError(
                f"index_weights must have shape {(B, T, H_i)}, "
                f"got {tuple(index_weights.shape)}"
            )

        if comp_valid_mask.shape != (B, S):
            raise ValueError(
                f"comp_valid_mask must have shape {(B, S)}, "
                f"got {tuple(comp_valid_mask.shape)}"
            )

        raw = torch.einsum(
            "bthi,bsi->bths",
            index_q,
            I_comp,
        )
        # [B,T,H_i,S]

        raw = F.relu(raw)

        index_scores = (index_weights[..., None] * raw).sum(dim=2)
        # [B,T,S]

        causal_allowed = self._build_global_allowed_mask(
            T=T,
            S=S,
            device=index_q.device,
        )
        # [T,S]

        allowed = causal_allowed[None, :, :] & comp_valid_mask[:, None, :].bool()
        # [B,T,S]

        mask_value = torch.finfo(index_scores.dtype).min
        masked_scores = index_scores.masked_fill(~allowed, mask_value)

        K = min(self.top_k, S)

        topk_scores, topk_indices = torch.topk(
            masked_scores,
            k=K,
            dim=-1,
        )

        topk_mask = torch.gather(
            allowed,
            dim=-1,
            index=topk_indices,
        )

        topk_scores = torch.where(
            topk_mask,
            topk_scores,
            torch.zeros_like(topk_scores),
        )

        if need_scores:
            return topk_indices, topk_scores, topk_mask, index_scores

        return topk_indices, topk_scores, topk_mask