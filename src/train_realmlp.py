"""E10: RealMLP-TD (pytabkit) member.

The published honest single-model frontier for this competition is a RealMLP
at OOF 0.95062. This runs the real thing (periodic numeric embeddings,
tuned-defaults config) on our folds, in the torch side-venv, on CPU.

- exact-value categoricals included as native categorical features — RealMLP
  embeds them; this is the verified TE-for-NNs edge in its natural form
- numerics median-imputed per fold + missing flags (E2 discipline)
- no class-weight support in pytabkit -> handled downstream by the blend's
  per-class threshold search (equivalent lever, measured in milestone 4)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
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
    out = pd.DataFrame(index=df.index)
    out['ev_steps'] = (df['step_count'] / 10).round()
    out['ev_cal'] = df['calorie_expenditure'] // 5
    out['ev_water'] = (df['water_intake'] * 50).round()
    out['ev_sleep'] = df['sleep_duration'].round(2)
    out['ev_hr_bmi'] = (df['heart_rate'].round(0).astype(str) + '_'
                        + df['bmi'].round(0).astype(str))
    out = out.astype('object').where(out.notna(), 'missing').astype(str)
    return out

EV = ['ev_steps', 'ev_cal', 'ev_water', 'ev_sleep', 'ev_hr_bmi']


def build(df):
    X = pd.concat([df[NUM].reset_index(drop=True),
                   df[NUM + CAT].isna().astype('float32')
                     .add_suffix('_miss').reset_index(drop=True),
                   df[CAT].reset_index(drop=True),
                   exact_value_cats(df).reset_index(drop=True)], axis=1)
    for c in CAT:
        X[c] = X[c].fillna('missing')
    return X

X_all, X_test = build(train), build(test)
# categorical dtypes aligned across train/test so codes match
for c in CAT + EV:
    joint = pd.CategoricalDtype(pd.concat([X_all[c], X_test[c]]).unique())
    X_all[c] = X_all[c].astype(joint)
    X_test[c] = X_test[c].astype(joint)

oof = np.zeros((len(train), 3))
testp = np.zeros((len(test), 3))
for k in range(5):
    t0 = time()
    va = np.where(fold_id == k)[0]
    tr = np.where(fold_id != k)[0]
    Xtr, Xva = X_all.iloc[tr].copy(), X_all.iloc[va].copy()
    Xte = X_test.copy()
    med = Xtr[NUM].median()
    for f in (Xtr, Xva, Xte):
        f[NUM] = f[NUM].fillna(med)
    model = RealMLP_TD_Classifier(
        device='cpu', n_threads=7, random_state=SEED + k,
        n_epochs=6, use_early_stopping=True,
        val_metric_name='1-balanced_accuracy', verbosity=0)
    model.fit(Xtr, y[tr])
    oof[va] = model.predict_proba(Xva)
    testp += model.predict_proba(Xte) / 5
    print(f"fold {k}: argmax {balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f} "
          f"({time()-t0:.0f}s)", flush=True)

cv = balanced_accuracy_score(y, oof.argmax(1))
print(f"realmlp OOF (argmax, unweighted loss): {cv:.5f}")
np.save('artifacts/realmlp_oof.npy', oof)
np.save('artifacts/realmlp_testp.npy', testp)
print("saved artifacts/realmlp_oof.npy, artifacts/realmlp_testp.npy")
