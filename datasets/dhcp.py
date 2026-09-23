"""dHCP cortical surface classification dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from torch.utils.data import Dataset

from .build import DATASETS


@DATASETS.register_module()
class DHCPSurfaceClassification(Dataset):
    """One hemisphere per sample; label 1 denotes preterm birth."""

    def __init__(self, config):
        self.root = Path(config.DATA_PATH)
        self.split_root = Path(config.SPLIT_PATH)
        self.subset = config.subset
        self.npoints = int(config.N_POINTS)
        if self.subset not in {"train", "val", "test"}:
            raise ValueError(f"Unknown subset: {self.subset}")

        records = pd.read_csv(config.csv_path, sep="\t", engine="python", on_bad_lines="skip")
        all_samples = []
        for _, row in records.iterrows():
            birth_age, scan_age = row["birth_age"], row["scan_age"]
            preterm = birth_age < 37
            if preterm and scan_age <= 37:
                continue
            participant, session = row["participant_id"], row["session_id"]
            anat = self.root / f"sub-{participant}" / f"ses-{session}" / "anat"
            for hemi, side in (("L", "left"), ("R", "right")):
                path = anat / f"sub-{participant}_hemi-{hemi}_surface_ico5.vtk"
                if path.is_file():
                    name = f"sub-{participant}_ses-{session}_{side}"
                    all_samples.append((name, path, int(preterm)))

        def read_ids(subset):
            path = self.split_root / f"dhcp_{subset}_ids_seed42.txt"
            return set(path.read_text(encoding="utf-8").splitlines())

        splits = {part: read_ids(part) for part in ("train", "val", "test")}
        if any(splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("dHCP split files contain overlapping sample IDs")
        self.samples = [record for record in all_samples if record[0] in splits[self.subset]]
        if not self.samples:
            raise ValueError(f"No {self.subset} samples found; check data and split paths")

        # Preserve the original preprocessing scale across the available cohort.
        self.global_scale = max(
            np.linalg.norm(points - points.mean(axis=0), axis=1).max()
            for _, path, _ in all_samples
            for points in (pv.read(path).points,)
        )
        if self.global_scale <= 0:
            raise ValueError("Invalid global cortical surface scale")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, path, label = self.samples[index]
        mesh = pv.read(path)
        points = np.asarray(mesh.points, dtype=np.float32)
        if points.shape != (self.npoints, 3):
            raise ValueError(f"Expected {self.npoints} vertices in {path}, got {points.shape}")
        points = (points - points.mean(axis=0)) / self.global_scale
        attributes = np.stack(
            [np.asarray(mesh.point_data[key], dtype=np.float32).reshape(self.npoints, -1)[:, 0]
             for key in ("sulc", "curv", "thickness")], axis=1
        )
        return {
            "name": name,
            "points": torch.from_numpy(points),
            "attributes": torch.from_numpy(attributes),
            "label": torch.tensor(label, dtype=torch.long),
        }
