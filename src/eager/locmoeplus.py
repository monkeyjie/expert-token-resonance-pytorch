import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Unofficial implementation of https://arxiv.org/abs/2406.00023 

class GrAPLayer(nn.Module):
    """Grouped Average Pooling as described in LocMoE paper."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        assert (
            hidden_dim % num_experts == 0
        ), "Hidden dimension must be divisible by number of experts"
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.group_size = hidden_dim // num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_length, hidden_dim]
        # Reshape to group dimensions
        x = x.view(*x.shape[:-1], self.num_experts, self.group_size)
        # Average pool over group dimension
        return x.mean(dim=-1)


class LocMoEPlusLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        expert_dim: int,
        min_capacity: int = 4,
        expert_dropout: float = 0.1,
        locality_weight: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.min_capacity = min_capacity
        self.locality_weight = locality_weight

        # GrAP layer for token feature extraction
        self.grap = GrAPLayer(input_dim, num_experts)

        # Initialize experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(expert_dropout),
                    nn.Linear(expert_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

        # Initialize router weights with orthogonal initialization
        self.router = nn.Linear(num_experts, num_experts, bias=False)
        nn.init.orthogonal_(self.router.weight)

        # Affinity threshold for adaptive capacity
        self.affinity_threshold = nn.Parameter(torch.tensor(0.5))

    def compute_affinity_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between tokens and router weights."""
        # Apply GrAP for feature extraction
        x_pooled = self.grap(x)  # shape: [batch_size, seq_length, num_experts]

        # Normalize features and weights
        x_norm = F.normalize(x_pooled, dim=-1)
        w_norm = F.normalize(self.router.weight, dim=1)

        # Compute affinities
        affinities = torch.matmul(x_norm, w_norm)
        return affinities

    def compute_adaptive_capacity(
        self, affinity_scores: torch.Tensor, sequence_length: int
    ) -> int:
        """Compute adaptive capacity based on affinity scores."""
        mean_affinity = affinity_scores.mean()
        adaptive_capacity = max(
            self.min_capacity,
            int(
                sequence_length * torch.sigmoid(mean_affinity - self.affinity_threshold)
            ),
        )
        return adaptive_capacity

    def route_tokens(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        local_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hybrid routing combining TCR and ECR strategies with adaptive capacity."""
        batch_size, sequence_length, _ = inputs.shape

        # Compute affinity scores and routing probabilities
        affinity_scores = self.compute_affinity_scores(inputs)
        router_probs = F.softmax(affinity_scores, dim=-1)

        # Compute adaptive capacity for this batch
        adaptive_capacity = self.compute_adaptive_capacity(
            affinity_scores, sequence_length
        )

        # Create dispatch masks combining TCR and ECR
        # First apply TCR - each token picks its top expert
        tcr_mask = torch.zeros(
            batch_size, sequence_length, self.num_experts, device=inputs.device
        )
        top_expert_idx = router_probs.argmax(dim=-1)

        batch_idx = torch.arange(batch_size, device=inputs.device).unsqueeze(1)
        seq_idx = torch.arange(sequence_length, device=inputs.device).unsqueeze(0)
        tcr_mask[batch_idx, seq_idx, top_expert_idx] = 1.0

        # Then apply ECR - each expert picks its top-k tokens based on adaptive capacity
        ecr_mask = torch.zeros_like(tcr_mask)
        for expert_idx in range(self.num_experts):
            expert_affinity = affinity_scores[..., expert_idx]
            top_k = min(adaptive_capacity, sequence_length)
            top_tokens = torch.topk(expert_affinity, k=top_k, dim=-1).indices

            batch_idx = torch.arange(batch_size, device=inputs.device).unsqueeze(1)
            ecr_mask[batch_idx, top_tokens, expert_idx] = 1.0

        # Combine TCR and ECR masks
        dispatch_mask = tcr_mask * ecr_mask

        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, self.num_experts)
            dispatch_mask = dispatch_mask * expanded_mask

        return router_probs, affinity_scores, dispatch_mask

    def compute_locality_loss(
        self, router_probs: torch.Tensor, local_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence between current and localized distributions."""
        batch_size, seq_length, num_experts = router_probs.shape

        if local_indices is None:
            local_dist = torch.ones_like(router_probs) / num_experts
        else:
            local_dist = torch.zeros_like(router_probs)
            local_dist.scatter_(2, local_indices.unsqueeze(-1), 1.0)
            local_dist = local_dist / local_dist.sum(dim=-1, keepdim=True).clamp(
                min=1e-6
            )

        return F.kl_div(router_probs.log(), local_dist, reduction="batchmean")

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        local_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass combining routed expert outputs."""
        router_probs, affinity_scores, dispatch_mask = self.route_tokens(
            inputs, mask, local_indices
        )

        # Dispatch tokens to experts and combine outputs
        final_output = torch.zeros_like(inputs)
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = dispatch_mask[..., expert_idx].unsqueeze(-1)
            if expert_mask.sum() > 0:
                expert_input = inputs * expert_mask
                expert_output = expert(expert_input)
                final_output = final_output + expert_output

        # Compute auxiliary losses if training
        if self.training:
            # Router entropy loss
            router_entropy = (
                -(router_probs * torch.log(router_probs + 1e-9)).sum(-1).mean()
            )

            # Affinity loss
            affinity_loss = F.mse_loss(
                affinity_scores,
                torch.ones_like(affinity_scores) * self.affinity_threshold,
            )

            # Locality loss
            locality_loss = self.compute_locality_loss(router_probs, local_indices)

            # Combine losses
            self.aux_loss = (
                router_entropy + affinity_loss + self.locality_weight * locality_loss
            )

        return final_output


def test_locmoe_plus():
    batch_size = 2
    sequence_length = 128
    input_dim = 768
    num_experts = 4
    expert_dim = 2048

    model = LocMoEPlusLayer(
        input_dim=input_dim,
        num_experts=num_experts,
        expert_dim=expert_dim,
    ).cuda()

    inputs = torch.randn(batch_size, sequence_length, input_dim).cuda()
    mask = torch.ones(batch_size, sequence_length).cuda()
    local_indices = torch.randint(0, num_experts, (batch_size, sequence_length)).cuda()

    outputs = model(inputs, mask, local_indices)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    if hasattr(model, "aux_loss"):
        print(f"Auxiliary loss: {model.aux_loss.item()}")


def test_locmoe_plus2():
    batch_size = 2
    sequence_length = 128
    input_dim = 768
    num_experts = 4
    expert_dim = 2048

    model = LocMoEPlusLayer(
        input_dim=input_dim,
        num_experts=num_experts,
        expert_dim=expert_dim,
    ).cuda()

    # Create sample inputs
    inputs = torch.randn(batch_size, sequence_length, input_dim).cuda()
    mask = torch.ones(batch_size, sequence_length).cuda()

    # We don't need local_indices for basic testing
    outputs = model(inputs, mask)

    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    if hasattr(model, "aux_loss"):
        print(f"Auxiliary loss: {model.aux_loss.item()}")


if __name__ == "__main__":
    test_locmoe_plus()
    test_locmoe_plus2()
