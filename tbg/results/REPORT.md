# TBG charge-density timing comparison

GPU environment extraction and Bohrium queue time are excluded. Aiden reports structure/graph, model loading, cached full-grid forward, export and end-to-end time separately. Charge3Net reports official per-sample full-grid time and the outer official-run wall time; its required CHGCAR-format loader target is a placeholder and is not used for final accuracy.

Primary density metric: raw-grid NMAE = sum(|prediction − DFT|) / sum(|DFT|). A separately reported charge-normalized NMAE is diagnostic only.

See `timing_per_sample.csv`, `density_nmae_per_sample.csv`, and `summary.json`.
