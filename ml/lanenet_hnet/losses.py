from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def discriminative_embedding_loss(
    embeddings: torch.Tensor,
    instance_masks: torch.Tensor,
    delta_var: float = 0.5,
    delta_dist: float = 1.5,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    batch_size = embeddings.shape[0]

    for batch_idx in range(batch_size):
        emb = embeddings[batch_idx].permute(1, 2, 0)
        mask = instance_masks[batch_idx]
        instance_ids = torch.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]
        if len(instance_ids) == 0:
            losses.append(emb.sum() * 0.0)
            continue

        means: list[torch.Tensor] = []
        var_terms: list[torch.Tensor] = []
        for instance_id in instance_ids:
            pixels = emb[mask == instance_id]
            mean = pixels.mean(dim=0)
            means.append(mean)
            distance = torch.norm(pixels - mean, dim=1)
            var_terms.append(torch.mean(F.relu(distance - delta_var) ** 2))

        var_loss = torch.stack(var_terms).mean()
        if len(means) > 1:
            means_tensor = torch.stack(means)
            distances = torch.cdist(means_tensor, means_tensor)
            eye = torch.eye(len(means), device=embeddings.device, dtype=torch.bool)
            distances = distances[~eye]
            dist_loss = torch.mean(F.relu(2 * delta_dist - distances) ** 2)
        else:
            dist_loss = var_loss * 0.0
        reg_loss = torch.mean(torch.norm(torch.stack(means), dim=1))
        losses.append(var_loss + dist_loss + 0.001 * reg_loss)

    return torch.stack(losses).mean()


class LaneNetLoss(nn.Module):
    def __init__(self, embedding_weight: float = 0.1, hnet_weight: float = 0.001) -> None:
        super().__init__()
        self.embedding_weight = embedding_weight
        self.hnet_weight = hnet_weight
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.register_buffer("identity_h", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(
        self,
        binary_logits: torch.Tensor,
        embeddings: torch.Tensor,
        h_params: torch.Tensor,
        binary_masks: torch.Tensor,
        instance_masks: torch.Tensor,
    ) -> torch.Tensor:
        binary = self.binary_loss(binary_logits, binary_masks)
        embedding = discriminative_embedding_loss(embeddings, instance_masks)
        identity = self.identity_h.to(h_params.device).expand_as(h_params)
        hnet = F.mse_loss(h_params, identity)
        return binary + self.embedding_weight * embedding + self.hnet_weight * hnet

