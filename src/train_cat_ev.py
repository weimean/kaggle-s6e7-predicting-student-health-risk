"""E9: exact-value categoricals fed to CatBoost natively.

The tree-safe version of the E6 idea. E6 failed because we pre-computed
target encodings and handed a booster dense columns to overfit. CatBoost's
ordered target statistics ARE target encoding — computed internally, per
permutation, with leakage protection built in. So we feed the raw
exact-value categoricals and let CatBoost do the encoding its own way.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
    # pandas 3: astype(str) preserves NaN, and CatBoost rejects NaN in cats —
    # mask explicitly.
    out = out.astype('object').where(out.notna(), 'missing').astype(str)
    assert out.notna().all().all()
    return out

EV = ['ev_steps', 'ev_cal', 'ev_water', 'ev_sleep', 'ev_hr_bmi']

def build(df):
    X = pd.concat([df[NUM].reset_index(drop=True),
                   df[CAT].reset_index(drop=True),
                   exact_value_cats(df).reset_index(drop=True)], axis=1)
    for c in CAT:
        X[c] = X[c].fillna('missing').astype(str)
    return X

X, X_test = build(train), build(test)
cat_features = CAT + EV

oof = np.zeros((len(train), 3))
testp = np.zeros((len(test), 3))
for k in range(5):
    t0 = time()
    va = np.where(fold_id == k)[0]
    tr = np.where(fold_id != k)[0]
    model = CatBoostClassifier(
        loss_function='MultiClass', iterations=2000, learning_rate=0.05,
        auto_class_weights='Balanced', cat_features=cat_features,
        early_stopping_rounds=50, random_seed=SEED, verbose=0)
    model.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
    oof[va] = model.predict_proba(X.iloc[va])
    testp += model.predict_proba(X_test) / 5
    print(f"fold {k}: {balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f} "
          f"(iters {model.get_best_iteration()}, {time()-t0:.0f}s)", flush=True)

cv = balanced_accuracy_score(y, oof.argmax(1))
print(f"catboost+EV OOF: {cv:.5f}  (cat baseline 0.94923, Δ {cv-0.94923:+.5f})")
np.save('artifacts/cat_ev_oof.npy', oof)
np.save('artifacts/cat_ev_testp.npy', testp)
print("saved artifacts/cat_ev_oof.npy, artifacts/cat_ev_testp.npy")
