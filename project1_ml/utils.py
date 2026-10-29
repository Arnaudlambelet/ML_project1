"""Data loading, cleaning/feature engineering, metrics and cross-validation.

Only the Python standard library and NumPy are used.
"""

import csv
import os

import numpy as np

from helpers import load_csv_data

# 1. Loading


def load_data(data_dir="data"):
    """Load the CSVs (slow, ~1.5 min) once and cache them in data/cache.npz.

    Returns x_train (N, D), x_test (M, D), y (N,) in {0, 1}, test_ids, names.
    """
    cache = os.path.join(data_dir, "cache.npz")
    if not os.path.exists(cache):
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_dir)
        with open(os.path.join(data_dir, "x_train.csv")) as f:
            names = next(csv.reader(f))[1:]
        np.savez(
            cache,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            train_ids=train_ids,
            test_ids=test_ids,
            feature_names=names,
        )
    d = np.load(cache)
    y = (d["y_train"] == 1).astype(float)  # {-1, 1} -> {0, 1}
    names = [str(s) for s in d["feature_names"]]
    return d["x_train"], d["x_test"], y, d["test_ids"], names


# 2. Cleaning and feature engineering

#  describe the survey, not the person:
DROP_COLUMNS = {
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "CTELNUM1",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_LLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_CLLCPWT",
}


def special_codes(col):
    """BRFSS answer codes for "don't know"/"refused" (-> NaN) and "none" (-> 0).

    Guessed from the column's maximum: 9 -> {7, 9} missing (7 kept if 8 also
    occurs, then it is a real category), 99 -> {77, 99} missing and 88 = none,
    999 -> {777, 999} missing and 888 = none, 9999 -> {7777, 9999} missing.
    """
    v = col[~np.isnan(col)]
    m = v.max() if v.size else 0
    if m == 9:
        return ([9] if np.any(v == 8) else [7, 9]), []
    return {
        99: ([77, 99], [88]),
        999: ([777, 999], [888]),
        9999: ([7777, 9999], []),
    }.get(m, ([], []))


def _frequency(v, f1=None, f2=None, f3=None, none=None):
    """Decode 3-digit BRFSS frequency codes 'kxx' (k = period, xx = count).
    f1, f2, f3 convert "xx per period k" to a common unit."""
    out = np.full_like(v, np.nan)
    for k, factor in ((1, f1), (2, f2), (3, f3)):
        if factor is not None:
            m = (v > 100 * k) & (v < 100 * k + 100)
            out[m] = (v[m] - 100 * k) * factor
    if none is not None:
        out[v == none] = 0.0
    return out


def _fruit_veg(v):  # 1xx per day, 2xx per week, 3xx per month -> per day
    return np.where(v == 300, 0.02, _frequency(v, 1.0, 1 / 7, 1 / 30, none=555))


# columns whose coding mixes units -> decoded explicitly (BRFSS 2015 codebook)
CODEBOOK = {
    "WEIGHT2": lambda v: np.where(
        (v >= 50) & (v <= 776),
        v * 0.4536,  # lb -> kg
        np.where((v >= 9000) & (v <= 9998), v - 9000, np.nan),
    ),
    "HEIGHT3": lambda v: np.where(
        (v >= 200) & (v <= 711),  # ft/in -> cm
        (v // 100 * 12 + v % 100) * 2.54,
        np.where((v >= 9000) & (v <= 9998), v - 9000, np.nan),
    ),
    "ALCDAY5": lambda v: _frequency(v, 30 / 7, 1.0, none=888),  # days / month
    "EXEROFT1": lambda v: _frequency(v, 1.0, 7 / 30),  # times / week
    "STRENGTH": lambda v: _frequency(v, 1.0, 7 / 30, none=888),
    "EXERHMM1": lambda v: np.where(
        np.isin(v, (777, 999)), np.nan, v // 100 * 60 + v % 100  # h:mm -> min
    ),
    "_DRNKWEK": lambda v: np.where(v == 99900, np.nan, v),
    "DROCDY3_": lambda v: np.where(v == 900, np.nan, v),
    **{
        c: _fruit_veg
        for c in ("FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1")
    },
}


class Preprocessor:
    """All statistics are learned on the training data in fit() and applied
    unchanged to any data in transform().

    Steps: special codes / codebook -> drop admin and mostly-missing columns
    -> columns with <= 10 values: one-hot (+ NaN level), others: optional
    log(1+x), median imputation, standardisation, optional powers -> bias.
    """

    def __init__(
        self,
        nan_threshold=0.5,
        codebook=False,
        log_skewed=False,
        one_hot=True,
        poly_degree=1,
    ):
        self.nan_threshold = nan_threshold
        self.codebook = codebook
        self.log_skewed = log_skewed
        self.one_hot = one_hot
        self.poly_degree = poly_degree

    def clean(self, x):
        """Decode special codes (learned in fit). Keeps all columns, NaN = missing."""
        x = x.copy()
        for j, (to_nan, to_zero) in self.codes.items():
            x[:, j][np.isin(x[:, j], to_nan)] = np.nan
            x[:, j][np.isin(x[:, j], to_zero)] = 0.0
        for j, f in self.decode.items():
            with np.errstate(invalid="ignore"):
                x[:, j] = f(x[:, j])
        return x

    def fit(self, x, names):
        self.decode = {
            j: CODEBOOK[n]
            for j, n in enumerate(names)
            if self.codebook and n in CODEBOOK
        }
        self.codes = {
            j: special_codes(x[:, j]) for j in range(x.shape[1]) if j not in self.decode
        }
        xc = self.clean(x)
        nan_frac = np.isnan(xc).mean(0)
        self.cat, self.cont, self.levels, self.has_nan, self.log = [], [], {}, {}, {}
        for j, name in enumerate(names):
            vals = np.unique(xc[:, j][~np.isnan(xc[:, j])])
            if (
                name in DROP_COLUMNS
                or nan_frac[j] > self.nan_threshold
                or vals.size < 2
            ):
                continue
            if self.one_hot and vals.size <= 10:
                self.cat.append(j)
                self.levels[j] = vals[
                    1:
                ]  # first level = reference (avoids collinearity)
                self.has_nan[j] = bool(nan_frac[j] > 0)
            else:
                self.cont.append(j)
                v = xc[:, j][~np.isnan(xc[:, j])]
                skew = np.mean((v - v.mean()) ** 3) / (v.std() ** 3 + 1e-12)
                self.log[j] = self.log_skewed and v.min() >= 0 and skew > 2
        self.kept = sorted(self.cat + self.cont)

        z = self._log(xc[:, self.cont])
        self.median = np.nanmedian(z, 0)
        z = np.where(np.isnan(z), self.median, z)
        self.mean, self.std = z.mean(0), z.std(0) + 1e-12
        z = (z - self.mean) / self.std
        self.poly = [
            ((z**d).mean(0), (z**d).std(0) + 1e-12)
            for d in range(2, self.poly_degree + 1)
        ]
        return self

    def _log(self, z):
        z = z.copy()
        for k, j in enumerate(self.cont):
            if self.log[j]:
                z[:, k] = np.log1p(np.maximum(z[:, k], 0))
        return z

    def transform(self, x):
        xc = self.clean(x)
        z = self._log(xc[:, self.cont])
        z = (np.where(np.isnan(z), self.median, z) - self.mean) / self.std
        blocks = [np.ones((len(x), 1)), z]
        blocks += [(z**d - m) / s for d, (m, s) in enumerate(self.poly, start=2)]
        for j in self.cat:
            col = xc[:, j]
            blocks.append((col[:, None] == self.levels[j][None, :]).astype(float))
            if self.has_nan[j]:
                blocks.append(np.isnan(col)[:, None].astype(float))
        return np.hstack(blocks)


def make_bins(x, n_bins=32):
    """Bin edges per column (quantiles). Bin 0 is reserved for NaN."""
    edges = []
    for col in x.T:
        v = col[~np.isnan(col)]
        u = np.unique(v)
        if len(u) < n_bins:
            edges.append((u[:-1] + u[1:]) / 2)
        else:
            edges.append(np.unique(np.quantile(v, np.linspace(0, 1, n_bins)[1:-1])))
    return edges


def apply_bins(x, edges):
    xb = np.zeros(x.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        xb[:, j] = np.where(
            np.isnan(x[:, j]), 0, 1 + np.searchsorted(e, x[:, j], "right")
        )
    return xb


def tree_features(x_train, x_test, names):
    """Inputs for the trees: cleaned columns (no one-hot / imputation /
    scaling, NaN kept as its own bin), binned into 32 bins learned on train."""
    prep = Preprocessor(nan_threshold=0.9, codebook=True, one_hot=False).fit(
        x_train, names
    )
    xr, xr_test = prep.clean(x_train)[:, prep.kept], prep.clean(x_test)[:, prep.kept]
    edges = make_bins(xr)
    return apply_bins(xr, edges), apply_bins(xr_test, edges)


# 3. Metrics


def f1_score(y, pred):
    """F1 of the positive class (y, pred in {0, 1} / bool)."""
    y, pred = y.astype(bool), pred.astype(bool)
    tp = np.sum(y & pred)
    return 2 * tp / max(np.sum(y) + np.sum(pred), 1)


def accuracy(y, pred):
    return np.mean(y.astype(bool) == pred.astype(bool))


def best_threshold(scores, y, n=300):
    """Threshold on the scores that maximises F1 (candidates = upper quantiles)."""
    cands = np.unique(np.quantile(scores, np.linspace(0.5, 0.995, n)))
    f1s = [f1_score(y, scores >= t) for t in cands]
    i = int(np.argmax(f1s))
    return cands[i], f1s[i]


# 4. Cross-validation


def stratified_folds(y, k=5, seed=1):
    """k folds with the same positive rate in each."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        for f, part in enumerate(np.array_split(idx, k)):
            folds[f].append(part)
    return [np.sort(np.concatenate(f)) for f in folds]


def cross_validate(tx, y, fit, k=5, seed=1, select=None):
    """k-fold CV of a model. `fit(y, tx)` must return a score function.

    The F1-optimal threshold is chosen on the training folds and applied to the
    validation fold. Also returns the out-of-fold scores (oof) and the F1 with
    one threshold tuned on all of them (oof_f1).
    """
    folds = stratified_folds(y, k, seed)
    f1s, accs, oof = [], [], np.zeros(len(y))
    for f in range(k):
        va = folds[f]
        tr = np.concatenate([folds[g] for g in range(k) if g != f])
        cols = select(y[tr], tx[tr]) if select else slice(None)
        score = fit(y[tr], tx[tr][:, cols])
        t, _ = best_threshold(score(tx[tr][:, cols]), y[tr])
        oof[va] = score(tx[va][:, cols])
        f1s.append(f1_score(y[va], oof[va] >= t))
        accs.append(accuracy(y[va], oof[va] >= t))
    return dict(
        f1=np.mean(f1s),
        f1_std=np.std(f1s),
        acc=np.mean(accs),
        oof_f1=best_threshold(oof, y)[1],
        oof=oof,
    )
