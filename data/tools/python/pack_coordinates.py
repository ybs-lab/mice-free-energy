#!/usr/bin/env python3
import argparse, os, re, sys, glob, h5py, numpy as np

HDR_TS = "ITEM: TIMESTEP"
HDR_N  = "ITEM: NUMBER OF ATOMS"
HDR_BB = "ITEM: BOX BOUNDS"
HDR_AT = "ITEM: ATOMS"

def parse_dump_frames(path, skip_frames=0):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
        frame = -1
        n_atoms = None
        while line:
            if not line.startswith("ITEM:"):
                line = f.readline(); continue

            if line.startswith(HDR_TS):
                _ = f.readline()  # timestep value
                frame += 1
                line = f.readline(); continue

            if line.startswith(HDR_N):
                n_atoms = int(f.readline().strip())
                line = f.readline(); continue

            if line.startswith(HDR_BB):
                _ = f.readline(); _ = f.readline(); _ = f.readline()
                line = f.readline(); continue

            if line.startswith(HDR_AT):
                cols = line.strip().split()
                try:
                    id_idx = cols.index("id") - 2
                    x_idx = cols.index("x") - 2
                    y_idx = cols.index("y") - 2
                    z_idx = cols.index("z") - 2
                except ValueError:
                    raise RuntimeError(f"Expected id/x/y/z columns in: {line.strip()}")
                if n_atoms is None:
                    raise RuntimeError("NUMBER OF ATOMS header missing before ATOMS block")

                import numpy as np
                ids = np.empty((n_atoms,), dtype=np.int64)
                pos = np.empty((n_atoms, 3), dtype=np.float32)
                for i in range(n_atoms):
                    parts = f.readline().split()
                    if not parts:
                        raise RuntimeError(f"Unexpected EOF while reading atoms for frame {frame}")
                    ids[i] = int(parts[id_idx])
                    pos[i, 0] = float(parts[x_idx])
                    pos[i, 1] = float(parts[y_idx])
                    pos[i, 2] = float(parts[z_idx])

                if frame >= skip_frames:
                    order = np.argsort(ids, kind="stable")
                    yield (frame - skip_frames), pos[order]
                line = f.readline(); continue

            line = f.readline()

def pack_dataset_to_h5(dataset_dir, seeds, output, phase=None, temperature=None, compression="lzf", chunk=128, skip_frames=200):
    import h5py
    total_seeds = len(seeds)
    print(f"[INFO] Processing {total_seeds} seed(s)...", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(output, "w") as h5:
        h5.attrs["format"] = "positions"
        if phase is not None:
            h5.attrs["phase"] = str(phase)
        if temperature is not None:
            h5.attrs["temperature_K"] = float(temperature)

        for seed_idx, seed in enumerate(seeds, 1):
            seed = seed.strip()
            if not seed:
                continue
            print(f"[{seed_idx}/{total_seeds}] Processing seed: {seed}", file=sys.stderr)
            seed_dir = os.path.join(dataset_dir, seed)
            import glob
            dump_glob = glob.glob(os.path.join(seed_dir, "dump", "dump*.lammpstrj"))
            if not dump_glob:
                print(f"[WARN] No dump found for seed {seed}", file=sys.stderr)
                continue
            dump_path = dump_glob[0]

            grp = h5.create_group(seed)
            n_atoms = None
            dset = None
            frames_written = 0
            for fidx, pos in parse_dump_frames(dump_path, skip_frames=skip_frames):
                if n_atoms is None:
                    n_atoms = pos.shape[0]
                    dset = grp.create_dataset(
                        "positions",
                        shape=(0, n_atoms, 3),
                        maxshape=(None, n_atoms, 3),
                        chunks=(chunk, n_atoms, 3),
                        dtype="float32",
                        compression=compression
                    )
                    grp.attrs["n_atoms"] = n_atoms
                    if temperature is not None:
                        grp.attrs["temperature_K"] = float(temperature)
                    if phase is not None:
                        grp.attrs["phase"] = str(phase)

                dset.resize((frames_written + 1, n_atoms, 3))
                dset[frames_written, :, :] = pos
                frames_written += 1
                
                # Print progress every 1000 frames
                # if frames_written - last_progress_frame >= 1000:
                #     print(f"  [{seed_idx}/{total_seeds}] {seed}: {frames_written} frames written...", file=sys.stderr)
                #     last_progress_frame = frames_written

            grp.attrs["frames"] = frames_written
            if frames_written == 0:
                print(f"[WARN] No frames written for seed {seed} (after skipping)", file=sys.stderr)
            else:
                print(f"  [{seed_idx}/{total_seeds}] {seed}: {frames_written} frames written", file=sys.stderr)

def read_seed_list(path):
    import os
    print(f"Reading seeds from {path}", file=sys.stderr)
    with open(path, "r", encoding="utf-8") as f:
        seeds = [ln.strip() for ln in f if ln.strip()]
    print(f"Found {len(seeds)} seed(s) in file", file=sys.stderr)
    return seeds

def main():
    ap = argparse.ArgumentParser(description="Pack LAMMPS lammpstrj coordinates into a single HDF5 (group per seed).")
    ap.add_argument("--dataset", "-d", required=True, help="Dataset directory containing <seed>/dump/dump*.lammpstrj")
    ap.add_argument("--seeds-file", "-s", required=True, help="Path to seed list file used for the dataset")
    ap.add_argument("--output", "-o", default="coordinates.h5", help="Output HDF5 filename (will be written to data/coordinates_h5/<dataset>/<output>)")
    ap.add_argument("--phase", "-p", choices=["solid","liquid"], help="Phase metadata to attach")
    ap.add_argument("--temp", "-t", type=float, help="Temperature in K to attach as metadata")
    ap.add_argument("--skip-frames", "-k", type=int, default=200, help="Frames to skip at start (equilibration)")
    ap.add_argument("--chunk", type=int, default=128, help="Chunk length in frames")
    ap.add_argument("--no-compress", action="store_true", help="Disable compression")
    args = ap.parse_args()

    # Resolve dataset directory to absolute path
    dataset_dir = os.path.abspath(args.dataset)
    
    # Extract dataset name (last component of dataset directory path)
    dataset_name = os.path.basename(dataset_dir.rstrip(os.sep))
    
    # Find repo root (directory containing data/)
    # Try to find data/ by going up from dataset directory
    current = dataset_dir
    repo_root = None
    while current != os.path.dirname(current):  # Stop at root
        data_dir = os.path.join(current, "data")
        if os.path.isdir(data_dir):
            repo_root = current
            break
        current = os.path.dirname(current)
    
    if repo_root is None:
        # Fallback: if we can't find repo root, use original output path
        print(f"[WARN] Could not find repo root (directory containing data/), using original output path", file=sys.stderr)
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(dataset_dir, output_path)
    else:
        # Construct new output path: data/coordinates_h5/<dataset>/<output>
        coordinates_h5_dir = os.path.join(repo_root, "data", "coordinates_h5", dataset_name)
        os.makedirs(coordinates_h5_dir, exist_ok=True)
        output_filename = os.path.basename(args.output)
        output_path = os.path.join(coordinates_h5_dir, output_filename)
    
    seeds = read_seed_list(args.seeds_file)
    compression = None if args.no_compress else "lzf"
    print(f"[INFO] Output file: {output_path}", file=sys.stderr)
    pack_dataset_to_h5(args.dataset, seeds, output_path, phase=args.phase, temperature=args.temp, compression=compression, chunk=args.chunk, skip_frames=args.skip_frames)
    print(f"Wrote {output_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
