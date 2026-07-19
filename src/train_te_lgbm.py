"""E6: exact-value categorization + cross-fitted multiclass target encoding.

The one feature-engineering move with verified lift in this competition
(public ablation: +0.0009 OOF). Mechanism: the synthetic generator leaves
signal at exact repeated numeric values; raw-numeric splits never isolate
"value == 7.2" but a categorical level can, and target encoding compresses
each level's class distribution into 3 dense columns.

Leakage control: sklearn TargetEncoder is fit inside each training fold and
its fit_transform is itself internally cross-fitted (cv=5), so no row's
encoding ever saw its own target.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from time import time

SEED = 42
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
NUM = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
       'step_count', 'exercise_duration', 'water_intake']
CAT = ['diet_type', 'stress_level', 'sleep_quality',
       'physical_activity_level', 'smoking_alcohol', 'gender']
CLASSES = ['at-risk', 'fit', 'unhealthy']
y = train['health_condition'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
fold_id = np.load('artifacts/fold_id.npy')


def exact_value_cats(df):
    """Numerics as exact-value categoricals (strings; NaN its own level)."""
    out = pd.DataFrame(index=df.index)
    out['ev_sleep'] = df['sleep_duration'].round(2)
    out['ev_hr'] = df['heart_rate'].round(1)
    out['ev_bmi'] = df['bmi'].round(2)
    out['ev_cal'] = (df['calorie_expenditure'] // 5)
    out['ev_steps'] = (df['step_count'] / 10).round()
    out['ev_ex'] = df['exercise_duration'].round(1)
    out['ev_water'] = (df['water_intake'] * 50).round()
    out['ev_hr_bmi'] = (df['heart_rate'].round(0).astype(str) + '_'
                        + df['bmi'].round(0).astype(str))
    return out.astype(str)  # NaN -> 'nan' level

ev_tr, ev_te = exact_value_cats(train), exact_value_cats(test)
EV = list(ev_tr.columns)

X_raw = train[NUM + CAT].copy()
X_test_raw = test[NUM + CAT].copy()
for c in CAT:
    X_raw[c] = X_raw[c].astype('category')
    X_test_raw[c] = X_test_raw[c].astype(pd.CategoricalDtype(X_raw[c].cat.categories))

oof = np.zeros((len(train), 3))
testp = np.zeros((len(test), 3))
for k in range(5):
    t0 = time()
    va = np.where(fold_id == k)[0]
    tr = np.where(fold_id != k)[0]
    te = TargetEncoder(target_type='multiclass', smooth='auto', cv=5, random_state=SEED)
    Z_tr = te.fit_transform(ev_tr.iloc[tr], y[tr])
    Z_va, Z_te = te.transform(ev_tr.iloc[va]), te.transform(ev_te)
    te_cols = [f'te_{c}_{j}' for c in EV for j in range(3)]

    def assemble(raw, Z, idx=None):
        base = raw if idx is None else raw.iloc[idx]
        return pd.concat([base.reset_index(drop=True),
                          pd.DataFrame(Z, columns=te_cols)], axis=1)

    Xtr = assemble(X_raw, Z_tr, tr)
    Xva = assemble(X_raw, Z_va, va)
    Xte = assemble(X_test_raw, Z_te)

    model = lgb.LGBMClassifier(objective='multiclass', num_class=3,
                               n_estimators=2000, learning_rate=0.05,
                               class_weight='balanced', random_state=SEED,
                               n_jobs=-1, verbose=-1)
    model.fit(Xtr, y[tr], eval_set=[(Xva, y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof[va] = model.predict_proba(Xva)
    testp += model.predict_proba(Xte) / 5
    print(f"fold {k}: {balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f} "
          f"(iters {model.best_iteration_}, {time()-t0:.0f}s)", flush=True)

cv = balanced_accuracy_score(y, oof.argmax(1))
print(f"lgbm+TE OOF balanced accuracy: {cv:.5f}  (baseline 0.94894, Δ {cv-0.94894:+.5f})")
np.save('artifacts/lgbm_te_oof.npy', oof)
np.save('artifacts/lgbm_te_testp.npy', testp)
print("saved artifacts/lgbm_te_oof.npy, artifacts/lgbm_te_testp.npy")
