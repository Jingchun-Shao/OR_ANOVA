# OR-ANOVA MATLAB experiments

MATLAB code for ANOVA decomposition, low-order interaction models, and global and local optimal recovery. The entry point for the paper's numerical experiments is [`reproduce_OR_ANOVA.mlx`](reproduce_OR_ANOVA.mlx).

## Requirements

- MATLAB with Symbolic Math Toolbox for the sextic local-parameter solver.
- CVX with SeDuMi only for the optional Beck–Eldar SDP comparison tests; the main live script does not use CVX.

## Reproduce the experiments

Open `reproduce_OR_ANOVA.mlx` in the MATLAB Live Editor and run it from the repository folder. The script sets the random seed and configures the MATLAB path. Its four parts cover FFT-based ANOVA projection, Sobol-function recovery, Energy Efficiency recovery, and global recovery on multinary perovskite oxides.

The Energy Efficiency and oxide experiments read [`ENB2012_data.xlsx`](datasets/energy_efficiency/data_set/ENB2012_data.xlsx) and [`multinary_oxides.csv`](datasets/multinary%20perovskite%20oxides/data_set/multinary_oxides.csv), respectively. These are the two datasets included in this release. The full run can take substantial time and memory: the Sobol experiment includes a dense SVD on a 6,561-cell tensor and a long power-method calculation.

## Small recovery examples

From the repository folder, run:

```matlab
addpath('core');
set_path();
recovery_map_global
recovery_map_local
```

These examples use synthetic data. The global example fixes the regularization parameter; the local example estimates it from model-mismatch and noise bounds.

## Repository layout

- `core/`: ANOVA routines, path setup, and examples.
- `optimal_param/`: parameter solvers and worst-case-error calculations.
- `optimal_recovery_map/`: recovery solvers.
- `datasets/energy_efficiency/`: Energy Efficiency data and experiment scripts.
- `datasets/multinary perovskite oxides/`: multinary oxide data and experiment scripts.
- `tests/`: MATLAB test and comparison scripts.
