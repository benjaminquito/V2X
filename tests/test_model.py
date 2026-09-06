import torch

from v2x_risk.data import GraphSample, collate_graphs
from v2x_risk.model import V2XRiskModel


def make_sample(label: int, frame_id: int) -> GraphSample:
    return GraphSample(
        x=torch.rand(3, 4),
        sequences=torch.rand(3, 15, 4),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.ones(4, 1),
        y=torch.tensor(label),
        frame_id=torch.tensor(frame_id),
        vehicle_ids=torch.tensor([1, 2, 3]),
    )


def test_model_outputs_three_class_logits_and_normalized_fusion_weights() -> None:
    config = {
        "input_dim": 4,
        "lstm_hidden_1": 12,
        "lstm_hidden_2": 8,
        "gat_hidden_1": 8,
        "gat_heads_1": 2,
        "gat_hidden_2": 8,
        "gat_heads_2": 1,
        "fusion_dim": 8,
        "fusion_attention_hidden": 6,
        "num_classes": 3,
        "dropout": 0.0,
    }
    output = V2XRiskModel(config)(collate_graphs([make_sample(0, 20), make_sample(2, 21)]))
    assert output.fused_logits.shape == (2, 3)
    assert output.temporal_logits.shape == (2, 3)
    assert output.spatial_logits.shape == (2, 3)
    torch.testing.assert_close(output.fusion_weights.sum(dim=1), torch.ones(2))
