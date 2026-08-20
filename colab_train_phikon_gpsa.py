# ── Cell 1: install deps ─────────────────────────────────────────────
!pip install -q transformers wilds torch torchvision

# ── Cell 2: upload gpsa.py and phikon_gpsa.py to the Colab session ────
# Either drag-and-drop them into the Files pane on the left, or:
from google.colab import files
uploaded = files.upload()  # select gpsa.py and phikon_gpsa.py

# ── Cell 3: download Camelyon17-WILDS ──────────────────────────────────
from wilds import get_dataset

dataset = get_dataset(dataset="camelyon17", download=True, root_dir="./data")
# this is a few GB — first run will take a while

# ── Cell 4: build train / OOD-val / OOD-test splits ────────────────────
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # phikon's expected normalization
])

train_data = dataset.get_subset("train", transform=transform)
# Camelyon17-WILDS splits by hospital: id_val/test are same-hospital,
# val/test are held-out hospitals — this is your OOD generalization eval
val_ood_data   = dataset.get_subset("val",   transform=transform)
test_ood_data  = dataset.get_subset("test",  transform=transform)

from wilds.common.data_loaders import get_train_loader, get_eval_loader

train_loader = get_train_loader("standard", train_data, batch_size=32)
val_loader   = get_eval_loader("standard", val_ood_data, batch_size=32)
test_loader  = get_eval_loader("standard", test_ood_data, batch_size=32)

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

# WILDS also provides its own eval() method that computes the official
# metrics/leaderboard format if you want directly comparable numbers:
# dataset.eval(all_preds, all_y_true, all_metadata)
