"""Voxelization utility for atomic coordinates stored in HDF5.

``DataGen`` loads atomic positions from an HDF5 file, applies geometric transforms
(rotation, centering and random shifts), crops a cubic region, and
voxelizes atoms into boolean occupancy grids of shape
``(bin_num, bin_num, bin_num)``.

The resulting tensors are written as ``.npy`` files and are the direct input to
the mutual-information trainers in ``train/``.
"""

import numpy as np
from pathlib import Path
import h5py
import sys


class DataGen:
    """One-hot voxel dataset generator for (N, M, 3) atomic coordinate arrays."""
    def __init__(
        self,
        bin_num: int,
        box_fraction: float = 1.0,
        lattice: float = 4.228,
        side: int = 8,
        rotate: bool = True,
        shift: bool = True,
        scaled: bool = False,
        testMode: int | None = None,
        center: bool = True,
        verbose: bool = True,
    ) -> None:
        self.data = []
        self.coords = None
        self.numParticles = []
        self.bin_num = int(bin_num)
        self.shift = bool(shift)
        self.lattice = float(lattice)
        self.side = int(side)
        self.box_fraction = float(box_fraction)
        self.scaled = bool(scaled)
        self.testMode = testMode
        self.rotate = bool(rotate)
        self.center = bool(center)
        self.verbose = bool(verbose)

        # The cubic crop side length used for voxelization.
        if self.box_fraction is not None:
            self.length = self.lattice * self.side * self.box_fraction
        else:
            self.length = 1.0 if self.scaled else self.lattice * self.side

        self.middle = None  # centers per-sample

    # ------------------------- Loading -------------------------

    def _find_positions_dataset(self, h5: h5py.File, dataset_key: str | None):
        """Locate a positions dataset with trailing xyz dimension in an HDF5 file."""
        if dataset_key is not None:
            if dataset_key in h5:
                ds = h5[dataset_key]
                if isinstance(ds, h5py.Dataset) and ds.ndim >= 2 and ds.shape[-1] == 3:
                    return ds
            # allow nested path (e.g., "group/subgroup/positions")
            try:
                ds = h5[dataset_key]
                if isinstance(ds, h5py.Dataset) and ds.ndim >= 2 and ds.shape[-1] == 3:
                    return ds
            except KeyError:
                pass

        # BFS across groups to find a dataset with last dim = 3
        queue = [h5]
        while queue:
            node = queue.pop(0)
            for k, v in node.items():
                if isinstance(v, h5py.Dataset):
                    if v.ndim >= 2 and v.shape[-1] == 3:
                        return v
                elif isinstance(v, h5py.Group):
                    queue.append(v)
        raise ValueError("Could not locate a dataset with trailing dimension 3 in the HDF5 file. "
                         "Please pass --dataset-key to specify the dataset explicitly.")

    def _collect_all_seed_positions(self, h5: h5py.File):
        """Collect ``<seed>/positions`` datasets into a list, one array per seed."""
        all_positions = []
        for key in h5.keys():
            if isinstance(h5[key], h5py.Group):
                if "positions" in h5[key]:
                    pos_ds = h5[key]["positions"]
                    if pos_ds.ndim >= 2 and pos_ds.shape[-1] == 3:
                        all_positions.append(pos_ds[...])
        return all_positions

    def load_from_h5(self, path: str | Path, dataset_key: str | None = None):
        """Load coordinates from an HDF5 file into ``self.coords`` as (N, M, 3)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        with h5py.File(path, 'r') as h5:
            # If dataset_key is provided, use it directly
            if dataset_key is not None:
                ds = self._find_positions_dataset(h5, dataset_key)
                arr = ds[...]
            else:
                # Auto-detect: check if root has groups with "positions" (multi-seed format)
                # Otherwise fall back to finding first dataset
                seed_positions = self._collect_all_seed_positions(h5)
                if seed_positions:
                    # Concatenate all seed positions along the first dimension (frames)
                    if self.verbose:
                        total_frames = sum(p.shape[0] for p in seed_positions)
                        print(f"[INFO] Found {len(seed_positions)} seed group(s) with {total_frames} total frames", file=sys.stderr)
                    arr = np.concatenate(seed_positions, axis=0)
                else:
                    # Fall back to original behavior
                    ds = self._find_positions_dataset(h5, None)
                    arr = ds[...]
        
        if arr.ndim < 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected a positions array shaped (N, M, 3) or similar, got {arr.shape}")
        # Normalize to (N, M, 3)
        self.coords = np.asarray(arr, dtype=np.float32)
        if self.testMode is not None:
            self.coords = self.coords[: self.testMode]

    # ------------------------- Geometry helpers -------------------------

    def calculateCenter(self, coords: np.ndarray) -> np.ndarray:
        """Center of the bounding box (average of min/max corners)."""
        return (coords.max(axis=0) + coords.min(axis=0)) / 2.0

    def rotateCoordinates(self):
        """Apply random 3D rotations to each sample (optionally about its center)."""
        n = len(self.coords)
        phis = np.random.uniform(-np.pi/2, np.pi/2, size=(n, 3)).astype(np.float32)
        # Build rotation matrices Rx, Ry, Rz in vectorized form
        Rx = np.array([
            [np.ones(n), np.zeros(n), np.zeros(n)],
            [np.zeros(n), np.cos(phis[:, 0]), -np.sin(phis[:, 0])],
            [np.zeros(n), np.sin(phis[:, 0]),  np.cos(phis[:, 0])],
        ]).transpose(2, 0, 1)
        Ry = np.array([
            [ np.cos(phis[:, 1]), np.zeros(n), np.sin(phis[:, 1])],
            [ np.zeros(n),        np.ones(n), np.zeros(n)       ],
            [-np.sin(phis[:, 1]), np.zeros(n), np.cos(phis[:, 1])],
        ]).transpose(2, 0, 1)
        Rz = np.array([
            [np.cos(phis[:, 2]), -np.sin(phis[:, 2]), np.zeros(n)],
            [np.sin(phis[:, 2]),  np.cos(phis[:, 2]), np.zeros(n)],
            [np.zeros(n),         np.zeros(n),        np.ones(n) ],
        ]).transpose(2, 0, 1)

        # If we're not centered, temporarily center, rotate, then re-shift.
        if not self.center:
            self.coords = self.coords - self.middle[:, np.newaxis, :]

        for i in range(self.coords.shape[1]):  # per-particle
            self.coords[:, i, :] = np.einsum("ijk,ik->ij", Rz, np.einsum("ijk,ik->ij", Ry, np.einsum("ijk,ik->ij", Rx, self.coords[:, i, :])))

        if not self.center:
            self.coords = self.coords + self.middle[:, np.newaxis, :]

    # ------------------------- Core pipeline -------------------------

    def _filter_cube(self, coords: np.ndarray, middle: np.ndarray) -> np.ndarray:
        """Crop atoms to a cube of side ``self.length`` centered at ``middle``."""
        l = self.length
        x_ok = (coords[:, 0] > middle[0] - l/2) & (coords[:, 0] < middle[0] + l/2)
        y_ok = (coords[:, 1] > middle[1] - l/2) & (coords[:, 1] < middle[1] + l/2)
        z_ok = (coords[:, 2] > middle[2] - l/2) & (coords[:, 2] < middle[2] + l/2)
        mask = x_ok & y_ok & z_ok
        if self.verbose:
            self.numParticles.append(coords[mask].shape[0])
        return coords[mask]

    def _one_hot_voxels(self, coords: np.ndarray, middle: np.ndarray) -> np.ndarray:
        """Voxelize atomic positions into a boolean occupancy grid."""
        n = self.bin_num
        l = self.length
        max_val = middle + l/2
        min_val = middle - l/2
        grid = (max_val - min_val) / n
        idx = np.floor((coords - min_val) / grid).astype(int)
        idx = np.clip(idx, 0, n - 1)
        vox = np.zeros((n, n, n), dtype=np.bool_)
        np.add.at(vox, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
        vox[vox > 1] = 1
        return vox

    def process(self):
        """Center/augment/crop coordinates and voxelize them into one-hot grids."""
        if self.coords is None:
            raise RuntimeError("No coordinates loaded. Call load_from_h5(...) first.")

        # Per-sample centers (before any recentering)
        self.middle = np.array([self.calculateCenter(c) for c in self.coords], dtype=np.float32)

        # Optional centering around origin
        if self.center:
            self.coords = self.coords - self.middle[:, np.newaxis, :]
            self.middle = np.zeros_like(self.middle)

        if self.rotate:
            self.rotateCoordinates()

        if self.shift:
            if self.scaled:
                shift = np.random.uniform(-self.length/(2*self.side), self.length/(2*self.side), size=(len(self.coords), 3)).astype(np.float32)
            else:
                shift = np.random.uniform(-self.lattice/2, self.lattice/2, size=(len(self.coords), 3)).astype(np.float32)
            self.middle = self.middle + shift

        # Crop & voxelize
        self.data = []
        filtered_coords = []
        for i, coord in enumerate(self.coords):
            m = self.middle[i]
            cropped = self._filter_cube(coord, m)
            vox = self._one_hot_voxels(cropped, m)
            self.data.append(vox)
            filtered_coords.append(cropped)
        self.coords = filtered_coords  # keep cropped coords

    # ------------------------- Saving -------------------------

    def save_tensor(self, out_path: str | Path) -> None:
        """Persist the generated voxel grids to a ``.npy`` file on disk."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            np.save(f, np.asarray(self.data, dtype=np.bool_), allow_pickle=False)
