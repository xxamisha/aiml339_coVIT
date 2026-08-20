# ── Cell 1: install deps ─────────────────────────────────────────────
!pip install -q transformers datasets torch torchvision

# ── Cell 2: upload gpsa.py and phikon_gpsa.py to the Colab session ────
# Either drag-and-drop them into the Files pane on the left, or:
from google.colab import files
uploaded = files.upload()  # select gpsa.py and phikon_gpsa.py

# ── Cell 3: download Camelyon17-WILDS ──────────────────────────────────
# NOTE: wilds's own get_dataset(download=True) pulls from CodaLab, whose
# servers frequently truncate/timeout mid-download without raising an
# error — this is a known, long-standing issue (not something specific
# to your setup). If you hit FileNotFoundError on metadata.csv after a
# "successful" download, that's what happened.
#
# Workaround: use the maintained HuggingFace mirror instead, which is
# the same data (455,954 images, same patient/node/x/y metadata) served
# as parquet — far more reliable to pull.
from datasets import load_dataset

hf_dataset = load_dataset("wltjr1007/Camelyon17-WILDS")
# hf_dataset has 'train' / 'validation' / 'test' splits, but those don't
# line up 1:1 with WILDS's OFFICIAL hospital split — reconstruct it
# using the 'center' column instead (this mirrors camelyon17_dataset.py's
# own split logic): centers 0,3,4 are the 3 source/training hospitals;
# center 1 is the OOD validation hospital; center 2 is the OOD test hospital.
from datasets import concatenate_datasets

all_data = concatenate_datasets([hf_dataset["train"], hf_dataset["validation"], hf_dataset["test"]])

TRAIN_CENTERS = {0, 3, 4}
VAL_OOD_CENTER = 1
TEST_OOD_CENTER = 2

train_hf     = all_data.filter(lambda ex: ex["center"] in TRAIN_CENTERS)
val_ood_hf   = all_data.filter(lambda ex: ex["center"] == VAL_OOD_CENTER)
test_ood_hf  = all_data.filter(lambda ex: ex["center"] == TEST_OOD_CENTER)

print(f"train: {len(train_hf)}, val (OOD): {len(val_ood_hf)}, test (OOD): {len(test_ood_hf)}")
# expect ~302,436 / ~34,904 / ~85,054 respectively (WILDS's reported counts)

# ── Cell 4: wrap as a PyTorch Dataset + build dataloaders ──────────────
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # phikon's expected normalization
])

class Camelyon17HFDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        image = ex["image"].convert("RGB")  # PIL image, 96x96
        label = ex["label"]                  # 0 = non-tumor, 1 = tumor
        metadata = {"center": ex["center"], "patient": ex["patient"], "node": ex["node"]}
        return self.transform(image), label, metadata

train_data    = Camelyon17HFDataset(train_hf, transform)
val_ood_data  = Camelyon17HFDataset(val_ood_hf, transform)
test_ood_data = Camelyon17HFDataset(test_ood_hf, transform)

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, labels, metadata

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(val_ood_data, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader  = DataLoader(test_ood_data, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

# ── Cell 5: build the GPSA-injected phikon model ───────────────────────
import torch
import torch.nn as nn
from transformers import ViTModel
from phikon_gpsa import inject_gpsa, get_gating_params

class PhikonGPSAClassifier(nn.Module):
    def __init__(self, local_layers=10, locality_strength=1.0, num_classes=2):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        inject_gpsa(self.backbone, local_layers=local_layers, locality_strength=locality_strength)
        self.local_layers = local_layers
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]  # cls token
        return self.head(out)

    def get_gating_params(self):
        return get_gating_params(self.backbone, local_layers=self.local_layers)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhikonGPSAClassifier(local_layers=10, locality_strength=1.0).to(device)
print(f"Using device: {device}")
print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ── Cell 6: training loop ───────────────────────────────────────────────
import torch.nn.functional as F
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # low LR: fine-tuning pretrained backbone
num_epochs = 3
log_every = 50

model.train()
for epoch in range(num_epochs):
    for step, (x, y, metadata) in enumerate(train_loader):
        # metadata contains hospital id — useful later for per-hospital analysis
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"epoch {epoch} step {step} loss {loss.item():.4f} acc {acc:.3f}")

    # log gating params per epoch — this is your key diagnostic:
    # watch sigmoid(lambda) drift from ~0.73 (positional-heavy at init)
    # toward content-based attention as training progresses
    gates = model.get_gating_params()
    print(f"epoch {epoch} gates (layer 0, mean over heads):", gates[0].mean().item())

# ── Cell 7: OOD evaluation (held-out hospitals) ────────────────────────
@torch.no_grad()
def evaluate(loader, name):
    model.eval()
    correct, total = 0, 0
    for x, y, metadata in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f"{name} accuracy: {acc:.4f}")
    return acc

evaluate(val_loader, "OOD val (held-out hospital)")
evaluate(test_loader, "OOD test (held-out hospital)")
