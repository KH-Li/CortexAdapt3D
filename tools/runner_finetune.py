"""Training and evaluation for dHCP cortical surface classification."""

import csv
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from tools.builder import dataset_builder, model_builder
from tools.loss import FocalLoss


def attribute_stats(loader, device):
    total = torch.zeros(3, device=device)
    total_sq = torch.zeros(3, device=device)
    count = 0
    for batch in loader:
        attributes = batch["attributes"].to(device).reshape(-1, 3)
        total += attributes.sum(0)
        total_sq += attributes.square().sum(0)
        count += attributes.shape[0]
    if count == 0:
        raise ValueError("Training set is empty")
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min(1e-6).sqrt()
    return mean, std


def metrics_from_logits(logits, labels):
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    truth = labels.cpu().numpy()
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    try:
        auc = roc_auc_score(truth, probabilities[:, 1])
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(truth, predictions),
        "precision": precision_score(truth, predictions, zero_division=0),
        "recall": recall_score(truth, predictions, zero_division=0),
        "f1": f1_score(truth, predictions, zero_division=0),
        "auc": auc,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }, probabilities, predictions


def evaluate(model, loader, mean, std, device, output=None):
    model.eval()
    all_logits, all_labels, all_names = [], [], []
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            attributes = (batch["attributes"].to(device) - mean) / (std + 1e-6)
            all_logits.append(model(points, attributes).cpu())
            all_labels.append(batch["label"].cpu())
            all_names.extend(batch["name"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics, probabilities, predictions = metrics_from_logits(logits, labels)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["sample", "true_label", "pred_label", "prob_class_0", "prob_class_1"])
            for name, truth, pred, prob in zip(all_names, labels.tolist(), predictions, probabilities):
                writer.writerow([name, truth, int(pred), float(prob[0]), float(prob[1])])
    return metrics


def _load_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("base_model", checkpoint.get("model", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_net(args, config, train_writer=None, val_writer=None):
    device = _device()
    train_loader = dataset_builder(args, config.dataset.train)
    val_loader = dataset_builder(args, config.dataset.val)
    test_loader = dataset_builder(args, config.dataset.test)
    mean, std = attribute_stats(train_loader, device)
    model = model_builder(config.model).to(device)
    if args.ckpts:
        model.load_model_from_ckpt(args.ckpts)

    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if any(term in name for term in ("adapt", "cls", "attr_encoder", "rel_pos_bias"))]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=config.optimizer.kwargs.lr,
        weight_decay=config.optimizer.kwargs.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch)
    criterion = FocalLoss()
    best_accuracy = -1.0
    output = Path(args.experiment_path)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.max_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader, 1):
            points = batch["points"].to(device)
            labels = batch["label"].to(device)
            attributes = (batch["attributes"].to(device) - mean) / (std + 1e-6)
            loss = criterion(model(points, attributes), labels)
            (loss / config.step_per_update).backward()
            losses.append(loss.item())
            if step % config.step_per_update == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        print(f"epoch={epoch + 1} loss={sum(losses) / len(losses):.4f}")
        if (epoch + 1) % args.val_freq == 0:
            validation = evaluate(model, val_loader, mean, std, device)
            print(f"validation: {validation}")
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                torch.save({"base_model": model.state_dict(), "attr_mean": mean.cpu(),
                            "attr_std": std.cpu(), "epoch": epoch + 1,
                            "validation": validation}, output / "ckpt-best.pth")

    checkpoint = output / "ckpt-best.pth"
    if checkpoint.exists():
        _load_weights(model, checkpoint)
        print(f"test: {evaluate(model, test_loader, mean, std, device, output)}")
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def test_net(args, config):
    device = _device()
    loader = dataset_builder(args, config.dataset.test)
    model = model_builder(config.model).to(device)
    checkpoint = _load_weights(model, args.ckpts)
    if "attr_mean" not in checkpoint or "attr_std" not in checkpoint:
        raise ValueError("Checkpoint lacks attribute normalization statistics")
    mean = checkpoint["attr_mean"].to(device)
    std = checkpoint["attr_std"].to(device)
    print(f"test: {evaluate(model, loader, mean, std, device, args.experiment_path)}")
