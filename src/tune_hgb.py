"""E11: give HistGradientBoosting (blend weight 0.5!) the tuning attention
it never got. Small targeted grid around the two knobs that matter for HGB:
max_leaf_nodes (public kernel uses 23 — smaller than default 31) and
l2_regularization; fold-0 proxy for ranking, full CV for the winner."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from time import time

SEED = 42
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
NUM = ['sleep_duration','heart_rate','bmi','calorie_expenditure','step_count','exercise_duration','water_intake']
CAT = ['diet_type','stress_level','sleep_quality','physical_activity_level','smoking_alcohol','gender']
CLASSES = ['at-risk','fit','unhealthy']
y = train['health_condition'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
fold_id = np.load('artifacts/fold_id.npy')

X = train[NUM+CAT].copy(); X_test = test[NUM+CAT].copy()
for c in CAT:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype(pd.CategoricalDtype(X[c].cat.categories))

def make(params):
    return HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.05, categorical_features='from_dtype',
        class_weight='balanced', early_stopping=True, n_iter_no_change=50,
        validation_fraction=0.1, random_state=SEED, **params)

va0 = np.where(fold_id == 0)[0]; tr0 = np.where(fold_id != 0)[0]
grid = [{'max_leaf_nodes': m, 'l2_regularization': l2}
        for m in (23, 31, 63) for l2 in (0.0, 1.0, 10.0)]
results = []
for p in grid:
    t0 = time()
    mdl = make(p); mdl.fit(X.iloc[tr0], y[tr0])
    s = balanced_accuracy_score(y[va0], mdl.predict(X.iloc[va0]))
    results.append((s, p))
    print(f"{p}: fold0 {s:.5f} ({time()-t0:.0f}s)", flush=True)
best_p = max(results)[1]
print("winner:", best_p, "— re-verifying on full 5-fold")

oof = np.zeros((len(train), 3)); testp = np.zeros((len(test), 3))
for k in range(5):
    va = np.where(fold_id == k)[0]; tr = np.where(fold_id != k)[0]
    mdl = make(best_p); mdl.fit(X.iloc[tr], y[tr])
    oof[va] = mdl.predict_proba(X.iloc[va])
    testp += mdl.predict_proba(X_test) / 5
    print(f"fold {k}: {balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f}", flush=True)
cv = balanced_accuracy_score(y, oof.argmax(1))
print(f"hgb tuned OOF: {cv:.5f} (baseline hgb 0.94956, Δ {cv-0.94956:+.5f})")
np.save('artifacts/hgb_tuned_oof.npy', oof); np.save('artifacts/hgb_tuned_testp.npy', testp)
