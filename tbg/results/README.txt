TBG DFT / Aiden / Charge3Net plotting export

Contents
--------
energy_summary.csv
    One row per method and TBG cell. Energies are in eV; errors are relative
    to the original converged DFT SCF energy. DFT is the zero-error reference.

forces_per_atom.csv
    One row per atom, method, and cell. Forces and force differences are in
    eV/Angstrom. DFT forces come from the original Gamma-centered PBE SCF.

bands_long.csv
    Long-format eigenvalues for DFT, Aiden, and Charge3Net for all three
    cells. Energies are in eV. kx/ky/kz are fractional reciprocal coordinates.
    path_distance_inv_angstrom is a continuous x-axis computed from the POSCAR
    reciprocal lattice. The path is Gamma-M-K-Gamma.

bands_wide.csv
    One row per cell, k point, and band, with the three methods in columns.
    This is convenient for direct plotting in pandas/matplotlib.

band_kpoints.csv
    Unique path/k-point metadata for plotting.

timings_detailed.csv / timings_summary.csv
    Wall times in seconds and hours. DFT includes the original full SCF plus
    the line-mode NSCF band job. Model calculations include fixed-density
    energy/force plus line-mode NSCF bands. The Aiden m03 energy stage was
    preserved from valid job 23292647, so its actual 12750 s is used rather
    than the 0 s bookkeeping entry in the band-only retry 23299653.

property_metrics.csv / band_metrics.csv
    The final benchmark metrics from analysis/. Band aligned errors remove one
    rigid energy shift per method/cell. Current calculations use raw model
    smooth densities plus the same DFT PAW augmentation tail; they are not
    end-to-end self-consistent PAW model energies.

overview_summary.csv
    One row per method and cell combining energy, force, band, and timing
    summary fields.
