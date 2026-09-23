"""CortexAdapt3D training and evaluation entry point."""

import argparse
from pathlib import Path

import torch

from tools.runner_finetune import run_net, test_net
from utils.config import cfg_from_yaml_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfgs/mae/finetune_dHCP_classification.yaml")
    parser.add_argument("--data-root", required=True, help="Directory containing sub-*/ses-*/anat/*.vtk")
    parser.add_argument("--csv-path", required=True, help="Participant metadata TSV/CSV")
    parser.add_argument("--split-root", required=True, help="Directory containing dhcp_*_ids_seed42.txt")
    parser.add_argument("--ckpts", help="Pretrained checkpoint for training, or trained checkpoint for --test")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.ckpts:
        parser.error("--ckpts is required (pretrained weights for training, trained weights for --test)")
    args.experiment_path = str(Path("experiments") / Path(args.config).stem / args.exp_name)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = cfg_from_yaml_file(args.config)
    for part in ("train", "val", "test"):
        dataset = config.dataset[part]._base_
        dataset.DATA_PATH = args.data_root
        dataset.csv_path = args.csv_path
        dataset.SPLIT_PATH = args.split_root
        config.dataset[part].others.bs = config.total_bs
    if args.test:
        test_net(args, config)
    else:
        run_net(args, config)


if __name__ == "__main__":
    main()
