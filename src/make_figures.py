"""Generate the README figures into assets/.

Data is inlined from experiments.md / the submission log; the correlation
heatmap is computed live from artifacts/ (same members and prior-correction
as notebooks/07-final-push.ipynb).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / 'assets'
ASSETS.mkdir(exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'savefig.dpi': 180,
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': '#d8d7d2', 'axes.linewidth': 0.8,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def score_arc():
    stages = [('majority\n(metric probe)', 1 / 3),
              ('LightGBM,\ndefault loss', 0.87542),
              ('+ class\nweights', 0.94894),
              ('+ decision\nthresholds', 0.94961),
              ('+ 3-GBM\nblend', 0.94968),
              ('+ NN\nmember', 0.95009),
              ('+ cat_ev &\ngreedy blend', 0.95036)]
    labels, cv = zip(*stages)
    x = np.arange(len(cv))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                 gridspec_kw={'width_ratios': [1, 1.3]})
    # Left: full arc — the two alignment cliffs
    a1.plot(x, cv, drawstyle='steps-post', color=BLUE, lw=2, zorder=3)
    a1.scatter(x, cv, color=BLUE, s=28, zorder=4)
    a1.set_xticks(x, labels, fontsize=6.8, rotation=30)
    a1.set_ylim(0.28, 1.0)
    a1.set_ylabel('OOF balanced accuracy')
    a1.set_title('The whole arc')
    a1.grid(axis='y', color='#ebeae6', lw=0.7, zorder=0)
    a1.annotate('metric discovered:\nbalanced accuracy', (0, 1 / 3),
                xytext=(0.25, 0.46), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7))
    a1.annotate('align loss\nwith metric (+0.074)', (2, 0.94894),
                xytext=(1.1, 0.80), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7))

    # Right: zoom on the grind above 0.948
    zx, zcv = x[2:], cv[2:]
    a2.plot(zx, zcv, drawstyle='steps-post', color=BLUE, lw=2, zorder=3)
    a2.scatter(zx, zcv, color=BLUE, s=28, zorder=4)
    a2.scatter([zx[-1] + 0.25], [0.95040], color=ORANGE, s=46, zorder=5,
               marker='D')
    a2.annotate('final private LB 0.95040\nrank 209 of 3356 (top 6.2%)',
                (zx[-1] + 0.25, 0.95040), xytext=(3.6, 0.95055), fontsize=8.5,
                color=INK, ha='right', fontweight='bold')
    for xi, yi in zip(zx, zcv):
        a2.annotate(f'{yi:.5f}', (xi, yi), xytext=(0, -13),
                    textcoords='offset points', ha='center', fontsize=7.5,
                    color=INK2)
    a2.set_xticks(zx, labels[2:], fontsize=7.2)
    a2.set_ylim(0.9484, 0.9509)
    a2.set_title('The grind above 0.949, zoomed')
    a2.grid(axis='y', color='#ebeae6', lw=0.7, zorder=0)

    fig.suptitle('Score arc — every step is a ledger entry', fontsize=12,
                 fontweight='bold', y=1.0)
    fig.tight_layout()
    fig.savefig(ASSETS / 'score-arc.png', bbox_inches='tight')
    plt.close(fig)


def cv_lb_calibration():
    subs = [('nb02 LightGBM', 0.94894, 0.94667, 0.94704),
            ('nb03 features', 0.94916, 0.94942, 0.94949),
            ('nb04 blend', 0.94968, 0.94946, 0.94965),
            ('nb04 robust', 0.94961, 0.94918, 0.94964),
            ('nb05 blend', 0.94990, 0.94978, 0.95000),
            ('nb06 +NN', 0.95009, 0.95006, 0.95002),
            ('nb07 grand', 0.95036, 0.95048, 0.95040)]
    names, cvs, pubs, privs = zip(*subs)
    lo, hi = 0.9460, 0.9512

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([lo, hi], [lo, hi], color='#c9c8c3', lw=1, ls='--', zorder=1)
    ax.fill_between([lo, hi], [lo - 0.003, hi - 0.003], [lo + 0.003, hi + 0.003],
                    color=BLUE, alpha=0.06, zorder=0)
    for c, pu, pr in zip(cvs, pubs, privs):
        ax.plot([c, c], [pu, pr], color='#c9c8c3', lw=0.9, zorder=2)
    ax.scatter(cvs, pubs, color=BLUE, s=42, zorder=3, label='public LB')
    ax.scatter(cvs, privs, color=ORANGE, s=38, zorder=3, marker='D',
               label='private LB')
    offsets = {'nb02 LightGBM': (8, -3), 'nb03 features': (-8, 6),
               'nb04 blend': (8, -10), 'nb04 robust': (-8, 3),
               'nb05 blend': (8, -6), 'nb06 +NN': (8, -3),
               'nb07 grand': (-8, 6)}
    for n, c, pu, pr in subs:
        dx, dy = offsets[n]
        ax.annotate(n, (c, pu), xytext=(dx, dy), textcoords='offset points',
                    fontsize=8, color=INK2,
                    ha='left' if dx > 0 else 'right')
    ax.annotate('y = x\n(CV predicts LB)', (0.9505, 0.9508), fontsize=8,
                color=INK2, ha='right')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('cross-validation (OOF balanced accuracy)')
    ax.set_ylabel('leaderboard score')
    ax.set_title('CV as a calibrated instrument\nall 7 submissions, public and private, within ±0.003 (shaded)')
    ax.grid(color='#ebeae6', lw=0.7, zorder=0)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(ASSETS / 'cv-lb-calibration.png', bbox_inches='tight')
    plt.close(fig)


def member_correlations():
    train = pd.read_csv(ROOT / 'data/train.csv')
    y = train['health_condition'].map({'at-risk': 0, 'fit': 1, 'unhealthy': 2}).to_numpy()
    A = ROOT / 'artifacts'
    oof = {'lgbm': np.mean([np.load(A / f'{n}_oof.npy')
                            for n in ('lgbm_tuned', 'lgbm_seed1', 'lgbm_seed2')], axis=0),
           'cat_ev': np.load(A / 'cat_ev_oof.npy'),
           'hgb': np.load(A / 'hgb_oof.npy'),
           'hgb_tuned': np.load(A / 'hgb_tuned_oof.npy'),
           'nn_te': np.mean([np.load(A / 'nn_te_oof.npy'),
                             np.load(A / 'nn_te_s1_oof.npy'),
                             np.load(A / 'nn_te_s2_oof.npy')], axis=0),
           'lgbm_te': np.load(A / 'lgbm_te_oof.npy')}
    weights = {'lgbm': 0.013, 'cat_ev': 0.388, 'hgb': 0.113,
               'hgb_tuned': 0.325, 'nn_te': 0.150, 'lgbm_te': 0.013}
    names = list(oof)
    err = pd.DataFrame({n: (o.argmax(1) != y).astype(int) for n, o in oof.items()})
    corr = err.corr().to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr, cmap=plt.matplotlib.colors.LinearSegmentedColormap
                   .from_list('seq', ['#f3f7fc', '#1d4e8f']),
                   vmin=0.75, vmax=1.0)
    ticklabels = [f"{n}\n(w={weights[n]:.2f})" for n in names]
    ax.set_xticks(range(len(names)), ticklabels, fontsize=7.5)
    ax.set_yticks(range(len(names)), ticklabels, fontsize=7.5)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if corr[i, j] > 0.93 else INK)
    ax.set_title('Error correlation between final blend members\n'
                 '(greedy-selection weights in axis labels)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='error correlation')
    fig.tight_layout()
    fig.savefig(ASSETS / 'member-correlations.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    score_arc()
    cv_lb_calibration()
    member_correlations()
    print('wrote:', sorted(p.name for p in ASSETS.glob('*.png')))
