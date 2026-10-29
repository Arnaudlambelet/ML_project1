"""Reproduces every number of the report (5-fold stratified cross-validation).

    python experiments.py            # everything (~85 min)
    python experiments.py baselines  # Table I   : the 6 required methods (~30 min)
    python experiments.py ablation   # Table II  : step-by-step improvements (~50 min)
    python experiments.py other      # Table III : ideas that did not help (~10 min)

Each line printed (and appended to results/results.txt) is
    name | F1 (mean ± std over folds) | accuracy | OOF-F1
F1/accuracy: threshold tuned on the 4 training folds, applied to the 5th.
OOF-F1: one threshold tuned on all out-of-fold predictions (used by run.py).
Preprocessing (no labels involved) is fit once on the whole training set;
everything that uses the labels is fit inside each fold.
"""

import os
import sys

import numpy as np

from models import gbdt, logistic, mlp, select_top, table1_model
from utils import Preprocessor, best_threshold, cross_validate, load_data, tree_features

os.makedirs("results", exist_ok=True)
x, x_test, y, _, names = load_data()


def features(**kw):
    return Preprocessor(**kw).fit(x, names).transform(x)


def report(name, res):
    line = (
        f"{name:<42s} F1 {res['f1']:.4f} ± {res['f1_std']:.4f}   "
        f"acc {res['acc']:.4f}   OOF-F1 {res['oof_f1']:.4f}"
    )
    print(line, flush=True)
    with open("results/results.txt", "a") as f:
        f.write(line + "\n")


def run(name, tx, fit, **kw):
    res = cross_validate(tx, y, fit, **kw)
    report(name, res)
    np.save(f"results/oof_{name.split()[0]}.npy", res["oof"])
    return res


def baselines():
    """Table I: the 6 methods of implementations.py on the base features.
    Step sizes: L = largest eigenvalue of X^T X / N = 33.8 on these features,
    so gamma = 1/L = 0.03 for the MSE and 4/L = 0.1 for the logistic loss
    (its curvature is at most L/4). Larger steps diverge."""
    tx = features()
    run(
        "I.1 mean_squared_error_gd",
        tx,
        table1_model("mean_squared_error_gd", gamma=0.03, max_iters=1000),
    )
    run(
        "I.2 mean_squared_error_sgd",
        tx,
        table1_model("mean_squared_error_sgd", gamma=1e-3, max_iters=20000),
    )
    run("I.3 least_squares", tx, table1_model("least_squares"))
    run("I.4 ridge_regression", tx, table1_model("ridge_regression", lambda_=1e-4))
    run(
        "I.5 logistic_regression",
        tx,
        table1_model("logistic_regression", gamma=0.1, max_iters=1000),
    )
    run(
        "I.6 reg_logistic_regression",
        tx,
        table1_model(
            "reg_logistic_regression", lambda_=1e-4, gamma=0.1, max_iters=1000
        ),
    )


def ablation():
    """Table II: improvements added one at a time, then the 3 models and the blend."""
    run("II.1 Newton logistic (base features)", features(), logistic())
    run("II.2 + keep columns <=90% missing", features(nan_threshold=0.9), logistic())
    tx = features(nan_threshold=0.9, codebook=True, log_skewed=True)
    run("II.3 + codebook decoding + log", tx, logistic())
    run("II.4 + interactions top-20 (model A)", tx, logistic(interactions=20))
    run("II.5 MLP 128-64 x3 (model B)", tx, mlp())
    del tx
    xb, _ = tree_features(x, x_test, names)
    run("II.6 GBDT 300 trees depth 5 (model C)", xb, gbdt())
    blend()


def blend():
    """Average of the probabilities of models A, B, C (out-of-fold)."""
    oof = {k: np.load(f"results/oof_{k}.npy") for k in ("II.4", "II.5", "II.6")}
    for combo in (
        ("II.4", "II.5"),
        ("II.4", "II.6"),
        ("II.5", "II.6"),
        ("II.4", "II.5", "II.6"),
    ):
        t, f1 = best_threshold(np.mean([oof[c] for c in combo], axis=0), y)
        line = f"blend {' + '.join(combo):<36s} OOF-F1 {f1:.4f}   threshold {t:.5f}"
        print(line)
        with open("results/results.txt", "a") as f:
            f.write(line + "\n")
    threshold_figure(np.mean([oof[c] for c in ("II.4", "II.5", "II.6")], axis=0))


def other():
    """Table III: ideas tested on the features of II.3 that did not help."""
    tx = features(nan_threshold=0.9, codebook=True, log_skewed=True)
    run("III.1 class weight 10 for positives", tx, logistic(pos_weight=10))
    run("III.2 feature selection top-200", tx, logistic(), select=select_top(200))
    del tx  # free memory before building larger feature sets
    run(
        "III.3 + squared numeric features",
        features(nan_threshold=0.9, codebook=True, log_skewed=True, poly_degree=2),
        logistic(),
    )
    run(
        "III.4 no one-hot encoding",
        features(nan_threshold=0.9, codebook=True, log_skewed=True, one_hot=False),
        logistic(),
    )


def threshold_figure(scores):
    """Figure for the report: precision / recall / F1 versus the threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.linspace(0.02, 0.6, 120)
    prec = [np.mean(y[scores >= t]) for t in ts]
    rec = [np.mean(scores[y == 1] >= t) for t in ts]
    f1 = [2 * p * r / (p + r) for p, r in zip(prec, rec)]
    plt.figure(figsize=(3.4, 2.2))
    plt.plot(ts, prec, label="precision")
    plt.plot(ts, rec, label="recall")
    plt.plot(ts, f1, label="F1", lw=2)
    plt.axvline(ts[int(np.argmax(f1))], color="gray", ls="--", lw=0.8)
    plt.xlabel("decision threshold on predicted probability")
    plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig("report/threshold.pdf")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["baselines", "ablation", "other"]
    for p in parts:
        {"baselines": baselines, "ablation": ablation, "other": other, "blend": blend}[
            p
        ]()
