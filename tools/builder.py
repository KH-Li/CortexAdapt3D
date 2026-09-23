"""Build the configured cortical surface dataset and model."""

from torch.utils.data import DataLoader

from datasets import build_dataset_from_cfg
from models import build_model_from_cfg


def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    training = config.others.subset == "train"
    batch_size = config.others.bs if training else 2
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=args.num_workers, drop_last=training)


def model_builder(config):
    return build_model_from_cfg(config)
