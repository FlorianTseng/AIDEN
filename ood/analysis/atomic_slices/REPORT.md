# Atom-rich 2D charge-density slices

Each system contributes one atom-rich grid plane normal to x. Planes are selected from atomic positions only; model errors do not affect selection.

NMAE is evaluated only on the displayed 2D plane. Mean and population variance refer to the plotted absolute-error distribution. Cyan circles mark projected atom centers within the reported plane tolerance.

| sample | plane | near atoms | method | slice NMAE (%) | mean |error| (e/Å³) | variance |error| ((e/Å³)²) |
|---|---|---:|---|---:|---:|---:|
| water | x=191 | 9 | aiden | 1.886887 | 5.42187455e-03 | 1.57298954e-04 |
| water | x=191 | 9 | charge3net | 1.641169 | 4.71581555e-03 | 1.22803774e-04 |
| water | x=191 | 9 | sad | 12.103453 | 3.47786670e-02 | 3.56522953e-03 |
| almg | x=36 | 13 | aiden | 6.672537 | 4.57519968e-02 | 5.71491995e-02 |
| almg | x=36 | 13 | charge3net | 6.682523 | 4.58204640e-02 | 5.65528590e-02 |
| almg | x=36 | 13 | sad | 2.459126 | 1.68616399e-02 | 2.02902368e-04 |
| asi_011 | x=97 | 5 | aiden | 1.302058 | 3.00561050e-03 | 7.89894603e-06 |
| asi_011 | x=97 | 5 | charge3net | 1.618370 | 3.73577199e-03 | 1.13744876e-05 |
| asi_011 | x=97 | 5 | sad | 9.772403 | 2.25581697e-02 | 4.22025970e-04 |

Density panel (a) and common error panels (b–d) use separate stacked logarithmic color bars. Values at or below the positive plotting floor are clipped for display; metrics use the original unclipped arrays.
