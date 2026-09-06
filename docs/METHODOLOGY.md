# Methodology and manuscript reconciliation

## Implemented experiment unit

One sample is a traffic-frame graph. A node is included only when that vehicle has a complete
15-frame history ending at the graph frame. Each node therefore contains:

- a current four-feature vector `[Local_X, Local_Y, v_Vel, v_Acc]`; and
- an aligned sequence with shape `[15, 4]` for the same vehicle.

Directed edges join different vehicles whose Euclidean separation is strictly less than 20 m.
Inverse metric distance is supplied to the GAT as an edge feature. Self-loops are added inside each
attention layer. The graph target defaults to the maximum node risk class in the frame, making the
frame label reflect the most safety-critical participating vehicle.

This definition fixes the earlier development notebook's arbitrary index-based pairing between
independent LSTM sequences and graph snapshots.

## Preprocessing

1. Parse the required numeric NGSIM fields. When `v_Vel` or `v_Acc` is absent, derive it by finite
   differences within contiguous vehicle runs and record that fallback in the manifest.
2. Sort and deduplicate records by location, vehicle, frame, and timestamp when available.
3. Interpolate selected numeric features within each contiguous vehicle run, then drop unresolved
   records.
4. Use a supplied `Risk_Class` when present, or apply the configured heuristic.
5. If a row limit is requested, make a memory-bounded first CSV pass to count trajectories, select
   complete vehicles with a seeded shuffle, and load only those trajectories on a second pass.
   Sampling complete trajectories avoids destroying temporal continuity.
6. Construct consecutive 15-step histories and aligned proximity graphs. For combined datasets,
   separate trajectories by location and contiguous run and key snapshots by location and time.
7. Select at most 10,988 graph frames, evenly across chronological coverage when more are available.
8. Create chronological 70/15/15 splits with a configurable boundary gap.
9. Fit feature-wise min-max scaling on training histories only and transform all splits.
10. Save the arrays, scaler, source hash, processed-data hash, counts, and class distribution.

## Risk labels

The final class encoding follows the user's later correction:

| Class | Meaning | Example downstream action |
|---:|---|---|
| 0 | Low risk | No alert |
| 1 | Medium risk | Advisory warning |
| 2 | High risk | Immediate intervention |

The manuscript says labels were derived from time headway, space headway, and acceleration but does
not publish numerical cutoffs. Consequently, the YAML thresholds are implementation assumptions,
not paper-derived facts. Existing labels are preferred. If heuristic labels are used, the precise
configuration must accompany any reported result.

## Model

### Temporal branch

Each node's `[15, 4]` history passes through stacked LSTMs with 128 and 64 hidden units. The final
hidden state is the node representation. Mean pooling yields one temporal representation per graph,
and an auxiliary classifier produces temporal-only logits.

### Spatial branch

Current node features pass through two sparse graph-attention layers:

- layer 1: 32 features per head, two heads, concatenated;
- layer 2: 16 features, one head; and
- mean pooling plus an auxiliary three-class classifier.

Attention uses source and destination node terms plus a learned transform of inverse-distance edge
features. This is a native PyTorch implementation of attention-weighted message passing.

### Fusion

Temporal and spatial graph representations are projected to a common dimension. An MLP produces two
softmax weights for every sample. Their weighted representation is passed to the final three-class
classifier. The weights are learned end to end; no fixed alpha or beta is used.

Training minimizes fused cross-entropy plus configurable auxiliary cross-entropy losses for the two
branches. Class-balanced loss is enabled for full-data training. The checkpoint with the highest
validation macro F1 is evaluated once on the held-out test split.

## Reconciliation table

| Topic | Uploaded manuscript | Later revision | Repository |
|---|---|---|---|
| Temporal input | 50 steps, 3 features in experimental text | 15 steps, 4 features | 15 steps, 4 features |
| Spatial model | GCN/GNN, 32 then 16 | GAT, 32 x 2 heads then 16 | GAT, 32 x 2 heads then 16 |
| Target | Continuous score and threshold alerts | Three-class softmax | Three-class softmax |
| Fusion | Fixed 0.6 LSTM / 0.4 GNN | Learned attention | Learned representation attention |
| Alignment | Not defined | Earlier code truncated unrelated lists | Exact vehicle/frame alignment |
| Evaluation | Regression MAE/RMSE | Accuracy and weighted F1 claims | Accuracy, macro/weighted F1, per-class metrics, confusion matrix, ordinal errors |

The repository therefore reproduces the specified revised design, not every contradictory statement
in the earlier manuscript.
