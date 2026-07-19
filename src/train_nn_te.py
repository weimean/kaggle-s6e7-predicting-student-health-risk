"""Torch MLP member + exact-value target encoding (E7).

Variant of train_nn.py adding the one verified public edge: exact-value
categorization of numerics -> cross-fitted multiclass target encoding,
appended (standardized) to the numeric block. The public ablation showed
this lifts NNs specifically (RealMLP OOF 0.95062); our E6 showed it hurts
trees. Architecture unchanged from train_nn.py so the delta isolates TE.

Runs in .venv-torch (python 3.12 + torch 2.2.2 — the last Intel-mac build).
Notebook 06 documents the design and consumes the artifacts this writes.

Design notes (RealMLP-lite):
- categorical embeddings (dim 8, "missing" is its own token — the E2 lesson)
- numerics: clip to train-fold [0.1%, 99.9%] quantiles, median-impute, standardize,
  plus missingness flags; all statistics fit per training fold
- 512-256 SiLU MLP, dropout 0.15
- class-weighted cross-entropy (the loss-metric alignment lever, now available
  since we control the training loop), AdamW + cosine schedule
- early stopping on validation balanced accuracy, best-state restore
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
from time import time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = 'cpu'

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
NUM = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
       'step_count', 'exercise_duration', 'water_intake']
CAT = ['diet_type', 'stress_level', 'sleep_quality',
       'physical_activity_level', 'smoking_alcohol', 'gender']
CLASSES = ['at-risk', 'fit', 'unhealthy']
y = train['health_condition'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

# Exact fold contract: reuse the persisted assignment, don't recompute.
fold_id = np.load('artifacts/fold_id.npy')

# Global categorical vocabularies (value lists are unsupervised — no leakage).
vocabs = {c: ['missing'] + sorted(train[c].dropna().unique()) for c in CAT}
def cat_codes(df):
    return np.stack([pd.Categorical(df[c].fillna('missing'), categories=vocabs[c]).codes
                     for c in CAT], axis=1).astype('int64')

cat_tr_all, cat_te = cat_codes(train), cat_codes(test)

# Exact-value categoricals (coarse set, per the verified public recipe)
def exact_value_cats(df, sleep_bins):
    out = pd.DataFrame(index=df.index)
    out['ev_steps'] = (df['step_count'] / 10).round()
    out['ev_cal'] = df['calorie_expenditure'] // 5
    out['ev_water'] = (df['water_intake'] * 50).round()
    out['ev_sleep'] = pd.cut(df['sleep_duration'], bins=sleep_bins, labels=False)
    out['ev_hr_bmi'] = (df['heart_rate'].round(0).astype(str) + '_'
                        + df['bmi'].round(0).astype(str))
    return out.astype(str)

_sleep_bins = np.unique(train['sleep_duration'].quantile(np.linspace(0, 1, 71)))
ev_tr_all = exact_value_cats(train, _sleep_bins)
ev_te_all = exact_value_cats(test, _sleep_bins)


def prep_numeric(tr_idx, df):
    """Fit clip/impute/scale stats on the training fold; apply to df."""
    ref = train[NUM].iloc[tr_idx]
    lo, hi, med = ref.quantile(0.001), ref.quantile(0.999), ref.median()
    x = df[NUM].clip(lo, hi, axis=1).fillna(med)
    mu, sd = x.iloc[:0].pipe(lambda _: (ref.clip(lo, hi, axis=1).fillna(med).mean(),
                                        ref.clip(lo, hi, axis=1).fillna(med).std()))
    x = (x - mu) / sd
    flags = df[NUM + CAT].isna().astype('float32')
    nmiss = flags.sum(axis=1) / len(NUM + CAT)
    return np.hstack([x.to_numpy('float32'), flags.to_numpy('float32'),
                      nmiss.to_numpy('float32')[:, None]])


class TabMLP(nn.Module):
    def __init__(self, n_num, emb_dim=8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(len(vocabs[c]), emb_dim) for c in CAT])
        d_in = n_num + emb_dim * len(CAT)
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(256, 3))

    def forward(self, xn, xc):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat([xn] + e, dim=1))


def predict(model, xn, xc, bs=16384):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xn), bs):
            out.append(torch.softmax(model(xn[i:i+bs], xc[i:i+bs]), 1))
    return torch.cat(out).numpy()


counts = np.bincount(y)
class_w = torch.tensor(len(y) / (3.0 * counts), dtype=torch.float32)

oof = np.zeros((len(train), 3), dtype='float64')
testp = np.zeros((len(test), 3), dtype='float64')
EPOCHS, BS, PATIENCE = 30, 2048, 5

for k in range(5):
    t0 = time()
    va_idx = np.where(fold_id == k)[0]
    tr_idx = np.where(fold_id != k)[0]
    enc = TargetEncoder(target_type='multiclass', smooth='auto', cv=5,
                        random_state=SEED)
    Z_tr = enc.fit_transform(ev_tr_all.iloc[tr_idx], y[tr_idx]).astype('float32')
    Z_va = enc.transform(ev_tr_all.iloc[va_idx]).astype('float32')
    Z_te = enc.transform(ev_te_all).astype('float32')
    z_mu, z_sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8

    def with_te(base, Z):
        return np.hstack([base, (Z - z_mu) / z_sd])

    Xn_tr = torch.from_numpy(with_te(prep_numeric(tr_idx, train.iloc[tr_idx]), Z_tr))
    Xn_va = torch.from_numpy(with_te(prep_numeric(tr_idx, train.iloc[va_idx]), Z_va))
    Xn_te = torch.from_numpy(with_te(prep_numeric(tr_idx, test), Z_te))
    Xc_tr = torch.from_numpy(cat_tr_all[tr_idx])
    Xc_va = torch.from_numpy(cat_tr_all[va_idx])
    Xc_te = torch.from_numpy(cat_te)
    y_tr = torch.from_numpy(y[tr_idx])

    torch.manual_seed(SEED + k)
    model = TabMLP(Xn_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(weight=class_w)

    best_score, best_state, since_best = -1, None, 0
    n = len(tr_idx)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = lossf(model(Xn_tr[idx], Xc_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        sched.step()
        score = balanced_accuracy_score(y[va_idx], predict(model, Xn_va, Xc_va).argmax(1))
        if score > best_score:
            best_score, since_best = score, 0
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break
    model.load_state_dict(best_state)
    oof[va_idx] = predict(model, Xn_va, Xc_va)
    testp += predict(model, Xn_te, Xc_te) / 5
    print(f"fold {k}: {best_score:.5f} (epoch {epoch+1-since_best}, {time()-t0:.0f}s)",
          flush=True)

print(f"nn OOF balanced accuracy: {balanced_accuracy_score(y, oof.argmax(1)):.5f}")
np.save('artifacts/nn_te_oof.npy', oof)
np.save('artifacts/nn_te_testp.npy', testp)
print("saved artifacts/nn_te_oof.npy, artifacts/nn_te_testp.npy")
