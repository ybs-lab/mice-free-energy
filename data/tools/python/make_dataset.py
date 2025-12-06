"""CLI wrapper around ``DataGen`` to build voxel datasets from coordinates.h5.
"""

import argparse
from pathlib import Path
import numpy as np
from DataGen import DataGen

CHEM = {
    "Na": {"lattice": 4.228, "side": 8},
    "Al": {"lattice": 4.049, "side": 8},
}

def detect_split_tag(h5_path: Path) -> str | None:
    """Heuristically return 'train' or 'val' if the file path hints at a split."""
    markers = ("train", "val")
    name_chain = [h5_path.stem] + [parent.name for parent in h5_path.parents]
    for name in name_chain:
        lower = name.lower()
        for marker in markers:
            if lower == marker or lower.endswith(f"_{marker}"):
                return marker
    return None


def main():
    """Parse CLI arguments and drive HDF5→voxel→.npy dataset generation."""
    p = argparse.ArgumentParser(description="Create OneHot voxel datasets from positions.h5")
    p.add_argument("--h5", required=True, help="Path to positions.h5 file")
    p.add_argument("--dataset-key", default=None, help="Optional dataset path inside the HDF5 (auto-detect if omitted)")
    p.add_argument("--element", required=True, choices=list(CHEM.keys()), help="Chemical element (sets lattice and side)")
    p.add_argument("--bins", required=True, type=int, nargs="+", help="One or more voxel grid sizes, e.g. 32 48 64")
    p.add_argument("--bf", type=float, default=1.0, help="Box fraction (default: 1.0)")
    p.add_argument("--test", type=int, default=None, help="If set, limit the number of samples processed")
    p.add_argument("--center", action=argparse.BooleanOptionalAction, default=True, help="Center each sample before processing")
    p.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True, help="Randomly rotate samples")
    p.add_argument("--shift", action=argparse.BooleanOptionalAction, default=True, help="Randomly shift centers")
    p.add_argument("--scaled", action=argparse.BooleanOptionalAction, default=False, help="Treat input as scaled coords in [0,1]")
    p.add_argument("--outdir", default="data/coordinates", help="Directory to place the generated .npy files")
    p.add_argument("--prefix", default=None, help="Optional prefix for output filenames")
    args = p.parse_args()

    chem = CHEM[args.element]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h5_path = Path(args.h5).expanduser().resolve()
    split_tag = detect_split_tag(h5_path)
    # Process once per bin size to avoid holding multiple copies in memory.
    for nbin in args.bins:
        dg = DataGen(
            bin_num=nbin,
            box_fraction=args.bf,
            lattice=chem["lattice"],
            side=chem["side"],
            rotate=args.rotate,
            shift=args.shift,
            scaled=args.scaled,
            testMode=args.test,
            center=args.center,
            verbose=True,
        )
        dg.load_from_h5(str(h5_path), dataset_key=args.dataset_key)
        dg.process()

        stem = args.prefix or h5_path.stem
        suffix = f"_{split_tag}" if split_tag else ""
        out_path = outdir / f"{stem}{suffix}_{args.element}_bf{args.bf}_bin{nbin}.npy"
        dg.save_tensor(out_path)

        print(f"created {out_path}")
        print(f"number of samples: {len(dg.data)}")
        print(f"average number of particles: {np.mean(dg.numParticles) if dg.numParticles else 'n/a'}")
        if len(dg.data) > 0:
            # centers after (optional) centering/shift steps
            print(f"average middle: {np.mean(dg.middle, axis=0)}")

if __name__ == "__main__":
    main()
