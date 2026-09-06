# NGSIM data preparation

NGSIM trajectory files are not included in this repository. Download them from the
[FHWA NGSIM page](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) or the
[U.S. DOT NGSIM Open Data portal](https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/)
under the applicable terms and keep them in `data/raw/`, which Git ignores.

The loader accepts genuine CSV files and Excel OOXML workbooks. It detects the content signature,
so a workbook accidentally stored with a `.csv` suffix remains readable; a correct `.xlsx` suffix
is still recommended for clarity.

## Expected schema

Required fields:

| Column | Use |
|---|---|
| `Vehicle_ID` | Vehicle trajectory identifier |
| `Frame_ID` | Ordered frame identifier |
| `Local_X`, `Local_Y` | Position and proximity graph construction |
| `v_Vel` | Temporal and spatial feature; derived from `Local_Y` and time when absent |
| `v_Acc` | Temporal/spatial feature and risk heuristic; derived from speed when absent |

Label options:

- Preferred: an existing `Risk_Class` containing `0`, `1`, `2` or recognized low/medium/high names.
- Fallback: `Time_Headway` and `Space_Headway`, plus `v_Acc`, using configured thresholds.

When `v_Vel` or `v_Acc` is absent and `data.kinematics.derive_missing` is enabled, the loader uses
finite differences within each contiguous vehicle run. With the default NGSIM units,
`Global_Time` is converted from milliseconds to seconds, speed is the absolute derivative of
`Local_Y` in ft/s, and acceleration is the derivative of speed in ft/s². Existing fields are
preserved. The manifest records whether source or derived values were used.

## Units

Common NGSIM trajectory exports express local coordinates in feet. The default
`coordinate_scale_to_m: 0.3048` converts these coordinates before applying the 20 m graph radius.
If your file already uses metres, set the scale to `1.0`. Feature values are min-max scaled after
the split, but the graph must always be constructed in known physical units first.

The default acceleration heuristic is stated in feet per second squared because it is evaluated
before normalization. Change it if the source file uses SI units.

## Multiple sites or recording periods

For combined site or recording files, retain `Location` and `Global_Time`. The loader separates
locations, splits reused vehicle IDs at discontinuous frame runs, and constructs snapshots from
location plus timestamp. Record the exact source files and hashes for each experiment.

## Sampling

The paper reports a one-million-row subset. This implementation samples complete vehicle
trajectories until the target is met, rather than sampling isolated rows, because isolated row
sampling breaks consecutive histories. The seeded selection and resulting row count are captured in
the preprocessing manifest.
