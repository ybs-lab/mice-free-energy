# mice-fee

## Machine Learning the Entropy to Estimate Free Energy Differences without Sampling Transitions

### Yamin Ben-Shimon, Barak Hirshberg, Yohai Bar-Sinai
[![arXiv](http://img.shields.io/badge/arXiv-1612.03235-B31B1B.svg)](https://arxiv.org/abs/2510.24930)

The tooling in this repository reproduces the data preparation and neural-network pipelines used for the entropy-learning workflow described in [arXiv:2510.24930](https://arxiv.org/abs/2510.24930). It includes scripts for packing LAMMPS trajectories into HDF5, voxelizing atomic configurations, and training mutual-information estimators on the generated tensors.

## Simulation Pipeline
### Requirements

- **LAMMPS:** 15 Jun 2023 development build (`patch_15Jun2023-30-gc5d9f901d9-modified`).
- **PLUMED:** 2.8.3 (`git f1e636b5b`) with the PairEntropy plugin enabled.
- **Python packages:** the tooling and training scripts rely on `numpy`, `torch`, `pandas`, `h5py`, `matplotlib`, `seaborn`, and `scipy`. Install them via:
  ```bash
  pip install numpy torch pandas h5py matplotlib seaborn scipy
  ```

**Note:** All scripts should be run from the main repository directory (the directory containing `data/` and `train/`).

### One-liners

Build all *solid + liquid* train/val:
```bash
./data/bin/run_all.sh -D m_Na365 -t 365 -n 6 -e Na
./data/bin/run_all.sh -D m_Al933 -t 900 -n 6 -e Al
```

Build a single phase + seeds, and write `coordinates.h5` to `data/coordinates_h5/<dataset>/`:
```bash
./data/bin/run_phase.sh -d m_Na365_L_train -p liquid -t 365 -n 6 -s data/seeds/seeds_train -o coordinates.h5 -k 200 -e Na
./data/bin/run_phase.sh -d m_Al933_S_train -p solid -t 900 -n 6 -s data/seeds/seeds_train -o coordinates.h5 -k 200 -e Al
```

- `-k` controls how many initial frames to skip (equilibration).
- The HDF5 file is written to `data/coordinates_h5/<dataset>/coordinates.h5` and has a group per seed:
  `/<seed>/positions` shaped `(frames, 1024, 3)` in `float32`, default compression `lzf`.

### Gather and Redirect LAMMPS Trajectories to HDF5: pack_coordinates.py

`pack_coordinates.py` collects multiple LAMMPS trajectory files and automatically redirects the packed HDF5 output to a standardized location under `data/coordinates_h5`.

Example usage:

```bash
python data/tools/python/pack_coordinates.py \
  --dataset data/simulation/m_Na365_S_train \
  --seeds-file data/seeds/seeds_train \
  --output coordinates.h5 \
  --phase solid \
  --temp 365 \
  --skip-frames 200
```

### Voxelization with make_dataset.py

The script `make_dataset.py` takes atomic coordinate data stored in HDF5 files and converts (voxelizes) the positions of atoms into 3D grids—called "tensors"—that can be efficiently used as input for neural network training. This step is essential for transforming continuous atomic coordinates into a fixed-size, regularly spaced representation suitable for machine learning models.

#### Input Parameters:
- `--h5`: Path to the input HDF5 coordinate file. This should point to the `.h5` file produced in the previous step (e.g., `data/coordinates_h5/.../coordinates.h5`).
- `--element`: The chemical element (such as `Na` or `Al`) to extract and voxelize.
- `--bins`: The number of discretization bins per axis for the voxel grid (e.g., `32` for a 32x32x32 grid).
- `--bf` (`--box-fraction`): Fraction of the simulation box size to use for the voxel grid. This controls how much of the original atomic environment is included.
- `--outdir`: Directory to save the resulting `.npy` tensor files.


These parameters let you control the shape and coverage of the voxelization, and organize your dataset splits and outputs for downstream neural net training workflows.

**Example usage:**
```bash
# Voxelize Sodium (Na) training and validation coordinate datasets, using 32 bins and a box fraction of 0.4
python data/tools/python/make_dataset.py --h5 data/coordinates_h5/m_Na365_S_train/coordinates.h5 --element Na --bins 32 --bf 0.4 --outdir data/coordinates
python data/tools/python/make_dataset.py --h5 data/coordinates_h5/m_Na365_S_val/coordinates.h5 --element Na --bins 32 --bf 0.4 --outdir data/coordinates
```
This step produces `.npy` tensors that are used directly in the neural network training stage.

## Training The Neural Network

The training scripts now load `.npy` tensors from `data/coordinates` directly (no Weights & Biases required). By default the trainer looks for `coordinates_train_<ELEMENT>_bf<BIN>_bin<BINS>.npy` and the matching validation file, but you can override the filenames explicitly.

Example runs that train on the provided datasets:

```bash
# Sodium dataset
python train/mw_train.py \
  --train-file coordinates_train_Na_bf0.4_bin32 \
  --val-file coordinates_val_Na_bf0.4_bin32 \
  --run-name Na_bf0.4_bin32 \
  --element Na \
  --bins 32 \
  --seed 42 \
  --log-freq 10

# Aluminum dataset
python train/mw_train.py \
  -train-file coordinates_train_Na_bf0.4_bin32 \
  --val-file coordinates_val_Na_bf0.4_bin32 \
  --run-name Al_bf0.3_bin32 \
  --element Al \
  --bins 32 \
  --seed 42
  --log-freq 100
```

## Parsing Results

Parse training results from metrics files and create a DataFrame for analysis:

```bash
python train/parse_results.py --output analysis/results_summary.csv
```
