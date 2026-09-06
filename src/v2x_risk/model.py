from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import GraphBatch


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if batch.numel() == 0:
        return x.new_empty((0, x.shape[-1]))
    graph_count = int(batch.max().item()) + 1
    pooled = x.new_zeros((graph_count, x.shape[-1]))
    pooled.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=graph_count).to(x.dtype).clamp_min(1)
    return pooled / counts.unsqueeze(1)


class GraphAttentionLayer(nn.Module):
    """Sparse multi-head graph attention implemented with native PyTorch operations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, out_features))
        self.attention_target = nn.Parameter(torch.empty(heads, out_features))
        self.edge_projection = nn.Linear(1, heads, bias=False)
        output_width = heads * out_features if concat else out_features
        self.bias = nn.Parameter(torch.zeros(output_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = x.shape[0]
        projected = self.linear(x).view(node_count, self.heads, self.out_features)

        self_nodes = torch.arange(node_count, device=x.device, dtype=torch.long)
        self_edges = torch.stack([self_nodes, self_nodes])
        edge_index = torch.cat([edge_index, self_edges], dim=1)
        self_weights = edge_attr.new_ones((node_count, 1))
        edge_attr = torch.cat([edge_attr, self_weights], dim=0)

        source, target = edge_index[0], edge_index[1]
        scores = (projected[source] * self.attention_source).sum(dim=-1)
        scores = scores + (projected[target] * self.attention_target).sum(dim=-1)
        scores = scores + self.edge_projection(torch.log1p(edge_attr.clamp_min(0)))
        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)

        target_index = target[:, None].expand(-1, self.heads)
        maximum = scores.new_full((node_count, self.heads), -torch.inf)
        maximum.scatter_reduce_(0, target_index, scores, reduce="amax", include_self=True)
        exponent = torch.exp(scores - maximum[target])
        denominator = scores.new_zeros((node_count, self.heads))
        denominator.index_add_(0, target, exponent)
        attention = exponent / denominator[target].clamp_min(1e-12)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        messages = projected[source] * attention.unsqueeze(-1)
        output = projected.new_zeros((node_count, self.heads, self.out_features))
        output.index_add_(0, target, messages)
        if self.concat:
            output = output.reshape(node_count, self.heads * self.out_features)
        else:
            output = output.mean(dim=1)
        return output + self.bias, attention


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, dropout: float) -> None:
        super().__init__()
        self.lstm_1 = nn.LSTM(input_dim, hidden_1, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm_1(sequences)
        hidden = self.dropout(hidden)
        hidden, _ = self.lstm_2(hidden)
        return hidden[:, -1, :]


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_1: int,
        heads_1: int,
        hidden_2: int,
        heads_2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gat_1 = GraphAttentionLayer(
            input_dim, hidden_1, heads=heads_1, concat=True, dropout=dropout
        )
        self.gat_2 = GraphAttentionLayer(
            hidden_1 * heads_1,
            hidden_2,
            heads=heads_2,
            concat=False,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden, attention_1 = self.gat_1(x, edge_index, edge_attr)
        hidden = self.dropout(F.elu(hidden))
        hidden, attention_2 = self.gat_2(hidden, edge_index, edge_attr)
        return F.elu(hidden), (attention_1, attention_2)


class AttentionFusion(nn.Module):
    def __init__(
        self,
        temporal_dim: int,
        spatial_dim: int,
        fusion_dim: int,
        attention_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_projection = nn.Linear(temporal_dim, fusion_dim)
        self.spatial_projection = nn.Linear(spatial_dim, fusion_dim)
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, attention_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, 2),
        )

    def forward(
        self, temporal: torch.Tensor, spatial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = torch.tanh(self.temporal_projection(temporal))
        spatial = torch.tanh(self.spatial_projection(spatial))
        weights = torch.softmax(self.attention(torch.cat([temporal, spatial], dim=-1)), dim=-1)
        fused = weights[:, :1] * temporal + weights[:, 1:] * spatial
        return fused, weights


@dataclass
class ModelOutput:
    fused_logits: torch.Tensor
    temporal_logits: torch.Tensor
    spatial_logits: torch.Tensor
    fusion_weights: torch.Tensor
    graph_attention: tuple[torch.Tensor, torch.Tensor]


class V2XRiskModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.temporal = TemporalEncoder(
            config["input_dim"],
            config["lstm_hidden_1"],
            config["lstm_hidden_2"],
            config["dropout"],
        )
        self.spatial = SpatialEncoder(
            config["input_dim"],
            config["gat_hidden_1"],
            config["gat_heads_1"],
            config["gat_hidden_2"],
            config["gat_heads_2"],
            config["dropout"],
        )
        self.temporal_classifier = nn.Linear(config["lstm_hidden_2"], config["num_classes"])
        self.spatial_classifier = nn.Linear(config["gat_hidden_2"], config["num_classes"])
        self.fusion = AttentionFusion(
            config["lstm_hidden_2"],
            config["gat_hidden_2"],
            config["fusion_dim"],
            config["fusion_attention_hidden"],
            config["dropout"],
        )
        self.fused_classifier = nn.Linear(config["fusion_dim"], config["num_classes"])

    def forward(self, graph: GraphBatch) -> ModelOutput:
        temporal_nodes = self.temporal(graph.sequences)
        temporal_graphs = global_mean_pool(temporal_nodes, graph.batch)
        spatial_nodes, graph_attention = self.spatial(graph.x, graph.edge_index, graph.edge_attr)
        spatial_graphs = global_mean_pool(spatial_nodes, graph.batch)
        fused, fusion_weights = self.fusion(temporal_graphs, spatial_graphs)
        return ModelOutput(
            fused_logits=self.fused_classifier(fused),
            temporal_logits=self.temporal_classifier(temporal_graphs),
            spatial_logits=self.spatial_classifier(spatial_graphs),
            fusion_weights=fusion_weights,
            graph_attention=graph_attention,
        )
