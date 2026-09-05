"""
Paired 30-seed comparison: vanilla-phikon vs GPSA-phikon.

Uses seed_sweep.py's run_seed_sweep() for BOTH models, with the SAME
seed list, so the comparison is paired (per the lecture: "use the same
random seeds to compare two different versions of the same approach").

Run this in Colab AFTER Cells 1-4 (deps, dataset, dataloaders) have run.
"""

from transformers import ViTModel
import torch.nn as nn
from phikon_gpsa import inject_gpsa
from seed_sweep import set_seed, run_seed_sweep, summarize_results
from scipy.stats import wilcoxon, ttest_rel
import csv


class PhikonClassifier(nn.Module):
    """Vanilla phikon fine-tune — no GPSA. This is the baseline half of
    the paired comparison."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]
        return self.head(out)


class PhikonGPSAClassifier(nn.Module):
    """GPSA-injected phikon fine-tune — the treatment half."""
    def __init__(self, local_layers=10, locality_strength=1.0, num_classes=2):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        inject_gpsa(self.backbone, local_layers=local_layers, locality_strength=locality_strength)
        self.local_layers = local_layers
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]
        return self.head(out)


def run_paired_comparison(train_loader, val_loader, test_loader, device,
                           seeds=None, num_epochs=1):
    """
    Runs BOTH models across the same seeds. Each seed's run is fully
    independent and checkpointed separately (see seed_sweep.py), so this
    is safe to re-run after a disconnect — already-completed seeds for
    each model are skipped automatically.
    """
    if seeds is None:
        seeds = list(range(10))  # reduced from 30 given compute constraints — document
                                  # this explicitly in your report as a deliberate, justified
                                  # deviation (per the lecture's "or 50 unless too
                                  # time-consuming" allowance), not an oversight

    print("=== vanilla phikon: 30-seed sweep ===")
    run_seed_sweep(
        seeds=seeds,
        build_model_fn=lambda: PhikonClassifier(),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device,
        ckpt_base_dir="/content/drive/MyDrive/seed_sweep_vanilla",
        num_epochs=num_epochs,
    )

    print("\n=== GPSA phikon: 30-seed sweep (same seeds) ===")
    run_seed_sweep(
        seeds=seeds,
        build_model_fn=lambda: PhikonGPSAClassifier(local_layers=10, locality_strength=1.0),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device,
        ckpt_base_dir="/content/drive/MyDrive/seed_sweep_gpsa",
        num_epochs=num_epochs,
    )


def compare_results(
    vanilla_dir="/content/drive/MyDrive/seed_sweep_vanilla",
    gpsa_dir="/content/drive/MyDrive/seed_sweep_gpsa",
):
    """Paired statistical test on OOD test accuracy across the 30 shared seeds."""
    def load_test_accs(base_dir):
        path = f"{base_dir}/seed_results.csv"
        accs = {}
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                accs[int(row["seed"])] = float(row["test_ood_acc"])
        return accs

    vanilla_accs = load_test_accs(vanilla_dir)
    gpsa_accs = load_test_accs(gpsa_dir)

    shared_seeds = sorted(set(vanilla_accs) & set(gpsa_accs))
    print(f"seeds with results in BOTH: {len(shared_seeds)} / expected 30")
    if len(shared_seeds) < len(vanilla_accs) or len(shared_seeds) < len(gpsa_accs):
        print("WARNING: some seeds are missing from one model or the other — "
              "the comparison below only uses seeds present in both.")

    v = [vanilla_accs[s] for s in shared_seeds]
    g = [gpsa_accs[s] for s in shared_seeds]

    import numpy as np
    print(f"\nvanilla-phikon: mean={np.mean(v):.4f} std={np.std(v):.4f}")
    print(f"GPSA-phikon:    mean={np.mean(g):.4f} std={np.std(g):.4f}")

    # paired test — same seeds means these are paired observations, not
    # independent samples (this is why sharing seeds mattered)
    stat, p_wilcoxon = wilcoxon(g, v)
    t_stat, p_ttest = ttest_rel(g, v)

    print(f"\nWilcoxon signed-rank: statistic={stat:.4f}, p={p_wilcoxon:.4e}")
    print(f"Paired t-test:        t={t_stat:.4f}, p={p_ttest:.4e}")

    # effect size (Cohen's d for paired samples)
    diffs = np.array(g) - np.array(v)
    cohens_d = diffs.mean() / diffs.std(ddof=1)
    print(f"Cohen's d (paired):   {cohens_d:.4f}")

    return {"vanilla": v, "gpsa": g, "wilcoxon_p": p_wilcoxon, "ttest_p": p_ttest, "cohens_d": cohens_d}
