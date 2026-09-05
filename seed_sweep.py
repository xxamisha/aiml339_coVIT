"""
Multi-seed training wrapper.

Runs the same model/training setup across N seeds, with:
  - full reproducibility (torch/numpy/random/cuda all seeded per run)
  - PER-SEED checkpoint isolation, so a bad save/overwrite in one seed's
    run can't clobber another seed's progress (this is what went wrong
    with the single shared latest.pt earlier)
  - a persistent results CSV that survives disconnects — so if you're
    partway through 30 seeds and get disconnected, you know exactly
    which seeds are already done and don't have to guess or redo them

Usage in Colab (after Cells 1-5 have defined build_model(), train_loader,
val_loader, test_loader, device):

    from seed_sweep import set_seed, run_seed_sweep
    seeds = list(range(30))  # or any 30 fixed seeds you want to report
    run_seed_sweep(seeds, build_model_fn=build_model, num_epochs=1)
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW


def set_seed(seed):
    """Seed every source of randomness that affects a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: this makes cuDNN deterministic at some speed cost. If you need
    # every run to be bit-for-bit reproducible (common ask alongside
    # "track all seeds"), keep this. If you only need seed-to-seed
    # variance (not exact reproducibility), you can drop these two lines
    # for a modest speed gain.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_results_log_path(base_dir):
    return os.path.join(base_dir, "seed_results.csv")


def load_completed_seeds(results_path):
    """Read which seeds already have a logged result, so a rerun after a
    disconnect doesn't redo work that's already done and safely saved."""
    completed = set()
    if os.path.exists(results_path):
        with open(results_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["seed"]))
    return completed


def append_result(results_path, row, fieldnames):
    """Append one seed's result as a new row — never overwrites prior
    rows, unlike the single-checkpoint mistake from earlier."""
    file_exists = os.path.exists(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y, metadata in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return correct / total


def run_seed_sweep(
    seeds,
    build_model_fn,
    train_loader,
    val_loader,
    test_loader,
    device,
    ckpt_base_dir="/content/drive/MyDrive/seed_sweep_checkpoints",
    num_epochs=1,
    lr=1e-5,
    log_every=100,
):
    """
    Args:
        seeds: list of ints, e.g. list(range(30)) — the exact seeds you
            will report, so keep this list itself somewhere durable too
            (e.g. paste it in your writeup) in case this file changes later.
        build_model_fn: a zero-arg function returning a fresh, untrained
            model instance (e.g. lambda: PhikonGPSAClassifier(...)).
            Must be called AFTER set_seed() for that seed's init to matter.
        ckpt_base_dir: each seed gets its own subfolder here — e.g.
            seed_sweep_checkpoints/seed_7/latest.pt — so seeds never
            share a file.
    """
    os.makedirs(ckpt_base_dir, exist_ok=True)
    results_path = get_results_log_path(ckpt_base_dir)
    fieldnames = ["seed", "epoch", "final_train_loss", "val_ood_acc", "test_ood_acc"]

    completed = load_completed_seeds(results_path)
    print(f"seeds already completed: {sorted(completed)}")

    for seed in seeds:
        if seed in completed:
            print(f"seed {seed}: already done, skipping")
            continue

        print(f"\n=== seed {seed} ===")
        set_seed(seed)  # must happen BEFORE model construction for init to be seeded

        seed_ckpt_dir = os.path.join(ckpt_base_dir, f"seed_{seed}")
        os.makedirs(seed_ckpt_dir, exist_ok=True)
        seed_ckpt_path = os.path.join(seed_ckpt_dir, "latest.pt")

        model = build_model_fn().to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        start_epoch, start_step = 0, 0
        if os.path.exists(seed_ckpt_path):
            ckpt = torch.load(seed_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch, start_step = ckpt["epoch"], ckpt["step"] + 1
            print(f"  resumed seed {seed} from epoch {start_epoch} step {start_step}")

        model.train()
        final_loss = None
        for epoch in range(start_epoch, num_epochs):
            for step, (x, y, metadata) in enumerate(train_loader):
                if epoch == start_epoch and step < start_step:
                    continue

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

                if step % log_every == 0:
                    print(f"  seed {seed} epoch {epoch} step {step} loss {loss.item():.4f}")

                if step % 1500 == 0 and step > 0:
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step,
                    }, seed_ckpt_path)

            start_step = 0

        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"  seed {seed}: val_ood_acc={val_acc:.4f} test_ood_acc={test_acc:.4f}")

        append_result(results_path, {
            "seed": seed,
            "epoch": num_epochs - 1,
            "final_train_loss": final_loss,
            "val_ood_acc": val_acc,
            "test_ood_acc": test_acc,
        }, fieldnames)

    print(f"\nall done. results log: {results_path}")


def summarize_results(ckpt_base_dir="/content/drive/MyDrive/seed_sweep_checkpoints"):
    """Mean/std across all completed seeds — this is what actually goes
    in your report (not any single seed's number)."""
    results_path = get_results_log_path(ckpt_base_dir)
    val_accs, test_accs = [], []
    with open(results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_accs.append(float(row["val_ood_acc"]))
            test_accs.append(float(row["test_ood_acc"]))

    val_accs, test_accs = np.array(val_accs), np.array(test_accs)
    print(f"n = {len(val_accs)} seeds")
    print(f"val_ood_acc:  mean={val_accs.mean():.4f}  std={val_accs.std():.4f}")
    print(f"test_ood_acc: mean={test_accs.mean():.4f}  std={test_accs.std():.4f}")
    return val_accs, test_accs
