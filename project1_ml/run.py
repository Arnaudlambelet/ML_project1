"""Reproduces our final AIcrowd submission:  python run.py  (~15 min)

Final prediction = average of the probabilities of three models
    A. logistic regression (Newton's method) + interactions of the top-20 features
    B. 3 neural networks (128-64 hidden units), averaged
    C. gradient-boosted trees (300 trees, depth 5)
A person is predicted to have MICHD (+1) if the average probability is
>= THRESHOLD, the value maximising F1 in cross-validation
(see `python experiments.py ablation`, last line).
"""

import os

import numpy as np

from helpers import create_csv_submission
from models import gbdt, logistic, mlp
from utils import Preprocessor, load_data, tree_features

THRESHOLD = (
    0.21470  # from `python experiments.py ablation` (blend A+B+C, OOF-F1 0.4345)
)

x, x_test, y, test_ids, names = load_data()

prep = Preprocessor(nan_threshold=0.9, codebook=True, log_skewed=True).fit(x, names)
tx, tx_test = prep.transform(x), prep.transform(x_test)
xb, xb_test = tree_features(x, x_test, names)

probs = []
for name, model, train, test in [
    ("A logistic + interactions", logistic(interactions=20), tx, tx_test),
    ("B neural networks", mlp(), tx, tx_test),
    ("C boosted trees", gbdt(), xb, xb_test),
]:
    print("training", name, flush=True)
    probs.append(model(y, train)(test))

pred = np.where(np.mean(probs, axis=0) >= THRESHOLD, 1, -1)
os.makedirs("submissions", exist_ok=True)
create_csv_submission(test_ids, pred, "submissions/submission.csv")
print(f"saved submissions/submission.csv ({np.mean(pred == 1):.1%} predicted positive)")
