# NGSIM data preparation

NGSIM trajectory files are not included in this repository. Download them from the
[FHWA NGSIM page](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) or the
[U.S. DOT NGSIM Open Data portal](https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/)
under the applicable terms and keep them in `data/raw/`, which Git ignores.

## Expected schema

Required fields:

| Column | Use |
|---|---|
| `Vehicle_ID` | Vehicle trajectory identifier |
| `Frame_ID` | Ordered frame identifier |
| `Local_X`, `Local_Y` | Position and proximity graph construction |
| `v_Vel` | Temporal and spatial feature |
| `v_Acc` | Temporal/spatial feature and optional risk heuristic |

Label options:

- Preferred: an existing `Risk_Class` containing `0`, `1`, `2` or recognized low/medium/high names.
- Fallback: `Time_Headway` and `Space_Headway`, plus `v_Acc`, using configured thresholds.

`Global_Time` is retained when available for traceability but is not currently a model feature.

## Units

Common NGSIM trajectory exports express local coordinates in feet. The default
`coordinate_scale_to_m: 0.3048` converts these coordinates before applying the 20 m graph radius.
If your file already uses metres, set the scale to `1.0`. Feature values are min-max scaled after
the split, but the graph must always be constructed in known physical units first.

The default acceleration heuristic is stated in feet per second squared because it is evaluated
before normalization. Change it if the source file uses SI units.

## Multiple sites or recording periods

Process one coherent NGSIM site/recording CSV at a time unless vehicle and frame identifiers have
been made globally unique. Concatenating files with reused `Vehicle_ID` or `Frame_ID` values can
merge unrelated trajectories or graph snapshots. Record the exact source files and hashes for each
experiment.

## Sampling

The paper reports a one-million-row subset. This implementation samples complete vehicle
trajectories until the target is met, rather than sampling isolated rows, because isolated row
sampling breaks consecutive histories. The seeded selection and resulting row count are captured in
the preprocessing manifest.
