"""
Local runner for the vanilla-phikon vs GPSA-phikon seed sweep.
Run directly from VS Code's terminal: python run_sweep_local.py

Prerequisites (run once in your terminal, ideally inside a venv):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install transformers datasets scipy numpy

Files needed in the same folder as this script:
    gpsa.py, phikon_gpsa.py, seed_sweep.py, paired_seed_comparison.py

WINDOWS NOTE: everything below is wrapped in `if __name__ == "__main__":`.
This is REQUIRED on Windows whenever DataLoader uses num_workers > 0 —
Windows' multiprocessing "spawn" mode re-imports this whole file in each
worker process, so any code sitting at module level (not inside this
guard) gets re-executed once per worker. Without the guard, you'd see
the dataset get re-downloaded/re-filtered multiple times, with worker
processes racing to write the same cache file — exactly what the
FileExistsError/WinError 1224 errors were.
"""

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from paired_seed_comparison import run_paired_comparison, compare_results

# ── config ──────────────────────────────────────────────────────────────
BATCH_SIZE = 16          # halved from Colab's 32 — 3070 has 8GB VRAM vs T4's 16GB
GRAD_ACCUM_STEPS = 2     # 16 * 2 = effective batch 32, exact match (see seed_sweep.py docstring)
NUM_SEEDS = 10
NUM_EPOCHS = 1
NUM_WORKERS = 4          # if you still see DataLoader worker errors after this fix,
                         # try dropping this to 0 or 2 as a fallback


class Camelyon17HFDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        image = ex["image"].convert("RGB")
        label = ex["label"]
        metadata = {"center": ex["center"], "patient": ex["patient"], "node": ex["node"]}
        return self.transform(image), label, metadata


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, labels, metadata


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected — this will be extremely slow. Check your torch install has CUDA support:")
        print("  py -c \"import torch; print(torch.cuda.is_available())\"")

    # ── dataset ─────────────────────────────────────────────────────────
    print("loading dataset (cached locally after first run)...")
    hf_dataset = load_dataset("wltjr1007/Camelyon17-WILDS")
    all_data = concatenate_datasets([hf_dataset["train"], hf_dataset["validation"], hf_dataset["test"]])

    TRAIN_CENTERS = {0, 3, 4}
    VAL_OOD_CENTER = 1
    TEST_OOD_CENTER = 2

    train_hf = all_data.filter(lambda ex: ex["center"] in TRAIN_CENTERS)
    val_ood_hf = all_data.filter(lambda ex: ex["center"] == VAL_OOD_CENTER)
    test_ood_hf = all_data.filter(lambda ex: ex["center"] == TEST_OOD_CENTER)

    print(f"train: {len(train_hf)}, val (OOD): {len(val_ood_hf)}, test (OOD): {len(test_ood_hf)}")
    # expect ~302,436 / ~34,904 / ~85,054

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data = Camelyon17HFDataset(train_hf, transform)
    val_ood_data = Camelyon17HFDataset(val_ood_hf, transform)
    test_ood_data = Camelyon17HFDataset(test_ood_hf, transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ood_data, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ood_data, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # ── run the sweep ────────────────────────────────────────────────────
    run_paired_comparison(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    seeds=list(range(NUM_SEEDS)),
    num_epochs=NUM_EPOCHS,
    log_every=50,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    vanilla_ckpt_dir="./checkpoints/seed_sweep_vanilla",
    gpsa_ckpt_dir="./checkpoints/seed_sweep_gpsa_v2",  # NEW folder — don't overwrite your valid v1 results
    gpsa_new_lr=5e-4,       # 50x higher than backbone's 1e-5 — new params adapt faster
    gating_init=0.0,        # sigmoid(0)=0.5, gentler start than the paper's default 0.73
    )

    print("\n=== final comparison ===")
    compare_results(
        vanilla_dir="./checkpoints/seed_sweep_vanilla",
        gpsa_dir="./checkpoints/seed_sweep_gpsa_v2",  # NEW folder — don't overwrite your valid v1 results  
    )
