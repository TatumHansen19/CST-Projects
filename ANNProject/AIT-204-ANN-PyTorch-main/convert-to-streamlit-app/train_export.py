# train_export.py
import json, joblib, numpy as np, torch, torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Progress bars ---
try:
    from tqdm.auto import tqdm
except ImportError:
    # Minimal fallback if tqdm isn't installed
    def tqdm(x=None, **k): return x if x is not None else range(0)
    print("tqdm not found; install with: pip install tqdm for progress bars.")

SEED = 42
rng = np.random.default_rng(SEED)

tqdm.write("1) Loading dataset…")
data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)
tqdm.write(f"   Loaded {len(X_raw)} documents across {num_classes} classes.")

tqdm.write("2) Vectorizing text with TF-IDF (fit)…")
vectorizer = TfidfVectorizer(
    max_features=5000, lowercase=True, stop_words='english',
    strip_accents='unicode', token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
# fit_transform is a single call; we still show stage updates
X = vectorizer.fit_transform(X_raw).toarray()
tqdm.write(f"   Vectorized to shape {X.shape} (features={X.shape[1]}).")

tqdm.write("3) Train/test split…")
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
tqdm.write(f"   Train: {Xtr.shape[0]}  Test: {Xte.shape[0]}.")

# 4) Model
class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewsMLP(Xtr.shape[1], num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# 5) Torch training with progress bars (mini-batch)
from torch.utils.data import TensorDataset, DataLoader

batch_size = 512
epochs = 8

# Build datasets/loaders
Xtr_t = torch.from_numpy(Xtr).float()
ytr_t = torch.from_numpy(ytr).long()
train_ds = TensorDataset(Xtr_t, ytr_t)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

tqdm.write(f"4) Training on {len(train_ds)} samples "
           f"(batch_size={batch_size}, epochs={epochs}) on {device}…")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=True)
    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = crit(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / (pbar.n * train_dl.batch_size if pbar.n > 0 else 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

# (Optional) quick progress-noted eval on test set
tqdm.write("5) Quick eval on test set…")
with torch.no_grad():
    model.eval()
    Xte_t = torch.from_numpy(Xte).float().to(device)
    yte_t = torch.from_numpy(yte).long().to(device)
    logits = model(Xte_t)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == yte_t).float().mean().item()
tqdm.write(f"   Test accuracy: {acc:.4f}")

# 6) Export artifacts
tqdm.write("6) Saving artifacts…")
torch.save(model.state_dict(), "model_state_dict.pt")
joblib.dump(vectorizer, "vectorizer.pkl")
with open("label_names.json","w") as f:
    json.dump(data.target_names, f)
tqdm.write("   Exported: model_state_dict.pt, vectorizer.pkl, label_names.json")
print("Done.")
