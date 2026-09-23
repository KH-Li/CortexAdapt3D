# CortexAdapt3D

CortexAdapt3D classifies preterm birth from dHCP cortical surface geometry and vertex attributes. Each left or right hemisphere forms one sample. The model combines surface patches with `sulc`, `curv`, and `thickness` features through a spectral adapter and a transformer encoder.

## Task and model

The classification target uses `1` for preterm birth and `0` for term birth. The cohort includes preterm samples scanned after 37 weeks.

The model registry name is `CortexAdapt3D`. Its `SpectralAdapter` integrates cortical vertex attributes with patch features in the spectral domain. The configuration in [`cfgs/mae/finetune_dHCP_classification.yaml`](cfgs/mae/finetune_dHCP_classification.yaml) defines the architecture and training settings.

## Environment

Use Python 3.9 or later with a PyTorch installation suited to the available hardware. Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Data layout

Provide a root directory containing cortical surface files with this structure:

```text
<data-root>/
└── sub-<participant>/
    └── ses-<session>/
        └── anat/
            ├── sub-<participant>_hemi-L_surface_ico5.vtk
            └── sub-<participant>_hemi-R_surface_ico5.vtk
```

Each surface has 10,242 vertices and the `sulc`, `curv`, and `thickness` vertex attributes. The participant metadata file is tab separated and contains `participant_id`, `session_id`, `birth_age`, and `scan_age` columns.

The split directory contains these files:

```text
<split-root>/
├── dhcp_train_ids_seed42.txt
├── dhcp_val_ids_seed42.txt
└── dhcp_test_ids_seed42.txt
```

Each line holds a hemisphere sample ID, such as `sub-001_ses-01_left` or `sub-001_ses-01_right`. Use distinct IDs across the training, validation, and test splits.

## Training

Run the following command from the repository root with a compatible pretrained encoder checkpoint:

```bash
python main.py \
  --data-root /path/to/dhcp \
  --csv-path /path/to/combined_03.csv \
  --split-root /path/to/splits \
  --ckpts /path/to/pretrained.pth \
  --exp-name dhcp_preterm
```

Training computes the mean and standard deviation of the three vertex attributes from the training split. The checkpoint with the highest validation accuracy is saved at:

```text
experiments/finetune_dHCP_classification/dhcp_preterm/ckpt-best.pth
```

The checkpoint includes the model weights and attribute statistics. After training, the selected checkpoint is evaluated on the test split.

## Evaluation

Evaluate a checkpoint produced by this training workflow:

```bash
python main.py \
  --data-root /path/to/dhcp \
  --csv-path /path/to/combined_03.csv \
  --split-root /path/to/splits \
  --ckpts experiments/finetune_dHCP_classification/dhcp_preterm/ckpt-best.pth \
  --test \
  --exp-name dhcp_test
```

Evaluation writes `metrics.csv` and `predictions.csv` under `experiments/finetune_dHCP_classification/dhcp_test/`. Metrics include accuracy, precision, recall, F1, ROC AUC, sensitivity, and specificity. The prediction file records each sample ID, label, predicted class, and class probabilities.
