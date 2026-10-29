import numpy as np

import implementations as impl
from implementations import sigmoid

# Baselines: the 6 methods of Table 1


def table1_model(method, lambda_=1e-4, gamma=0.1, max_iters=500):
    def fit(y, tx):
        w0 = np.zeros(tx.shape[1])
        if method == "least_squares":
            # derived BRFSS variables are exact functions of others, so X^T X
            # is singular: np.linalg.solve then returns noise. Use the
            # min-norm solution (pinv is allowed, lstsq is not).
            if np.linalg.cond(tx.T @ tx) < 1e10:
                w = impl.least_squares(y, tx)[0]
            else:
                w = np.linalg.pinv(tx.T @ tx, hermitian=True) @ (tx.T @ y)
        elif method == "ridge_regression":
            w = impl.ridge_regression(y, tx, lambda_)[0]
        elif method == "mean_squared_error_gd":
            w = impl.mean_squared_error_gd(y, tx, w0, max_iters, gamma)[0]
        elif method == "mean_squared_error_sgd":
            np.random.seed(0)
            w = impl.mean_squared_error_sgd(y, tx, w0, max_iters, gamma)[0]
        elif method == "logistic_regression":
            w = impl.logistic_regression(y, tx, w0, max_iters, gamma)[0]
        elif method == "reg_logistic_regression":
            w = impl.reg_logistic_regression(y, tx, lambda_, w0, max_iters, gamma)[0]
        return lambda m: m @ w

    return fit


# Logistic regression with Newton's method


def newton_logistic(y, tx, lambda_=1e-4, pos_weight=1.0, max_iters=30):
    """Minimise mean c_n * logistic loss + lambda ||w||^2 (bias not penalised).

    Newton step: w <- w - t H^-1 g, with H = X^T S X / N + 2 lambda I.
    Converges in ~7 iterations (vs. thousands of GD steps); t is halved
    until the loss decreases (line search).
    """
    n, d = tx.shape
    c = np.where(y == 1, pos_weight, 1.0)
    reg = np.full(d, 2 * lambda_)
    reg[0] = 0.0

    def loss(w):
        z = tx @ w
        return np.mean(c * (np.logaddexp(0, z) - y * z)) + lambda_ * np.sum(w[1:] ** 2)

    w, cur = np.zeros(d), np.inf
    for _ in range(max_iters):
        p = sigmoid(tx @ w)
        g = tx.T @ (c * (p - y)) / n + reg * w
        s = c * p * (1 - p) / n
        h = np.diag(reg + 1e-10)
        for a in range(0, n, 32768):  # in chunks to avoid copying tx
            h += tx[a : a + 32768].T @ (tx[a : a + 32768] * s[a : a + 32768, None])
        step, t = np.linalg.solve(h, g), 1.0
        while loss(w - t * step) > cur and t > 1e-4:
            t /= 2
        w = w - t * step
        new = loss(w)
        if cur - new < 1e-8:
            break
        cur = new
    return w


def _corr(tx, y):
    """|correlation| of each column with y (bias column -> 0)."""
    xc, yc = tx - tx.mean(0), y - y.mean()
    r = np.abs(xc.T @ yc) / (np.sqrt((xc**2).sum(0) * (yc**2).sum()) + 1e-12)
    r[0] = 0
    return r


def logistic(lambda_=1e-4, pos_weight=1.0, interactions=0):
    """Model A. If interactions=k, adds all products of the k features most
    correlated with y (chosen on the training data only)."""

    def fit(y, tx):
        top = np.argsort(-_corr(tx, y))[:interactions]
        i, j = np.triu_indices(len(top), k=1)
        a, b = top[i], top[j]
        p = tx[:, a] * tx[:, b]
        mu, sd = p.mean(0), p.std(0) + 1e-12

        def expand(m):
            return np.hstack([m, (m[:, a] * m[:, b] - mu) / sd])

        w = newton_logistic(y, expand(tx), lambda_, pos_weight)
        return lambda m: sigmoid(expand(m) @ w)

    return fit


def select_top(m):
    """Feature selection: keep the bias + the m columns most correlated with y."""
    return lambda y, tx: np.r_[0, np.sort(np.argsort(-_corr(tx, y))[:m])]


# B. Neural network (multi-layer perceptron)


def mlp(hidden=(128, 64), epochs=6, lr=5e-4, weight_decay=1e-4, batch=512, n_models=3):
    """Model B. ReLU MLP with sigmoid output, logistic loss, Adam optimiser.
    n_models networks with different seeds are trained and averaged."""

    def train_one(y, tx, seed):
        rng = np.random.default_rng(seed)
        sizes = [tx.shape[1], *hidden, 1]
        W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        B = [np.zeros(b) for b in sizes[1:]]
        params = W + B
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            perm = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                idx = perm[s : s + batch]
                acts = [tx[idx]]  # forward pass
                for l in range(len(W)):
                    z = acts[-1] @ W[l] + B[l]
                    acts.append(np.maximum(z, 0) if l < len(W) - 1 else z)
                g = ((sigmoid(acts[-1][:, 0]) - y[idx]) / len(idx))[:, None]
                gW, gB = [None] * len(W), [None] * len(W)
                for l in reversed(range(len(W))):  # backward pass
                    gW[l] = acts[l].T @ g + weight_decay * W[l]
                    gB[l] = g.sum(0)
                    if l > 0:
                        g = (g @ W[l].T) * (acts[l] > 0)
                t += 1  # Adam update
                for k, (p, gr) in enumerate(zip(params, gW + gB)):
                    m1[k] = 0.9 * m1[k] + 0.1 * gr
                    m2[k] = 0.999 * m2[k] + 0.001 * gr**2
                    p -= (
                        lr
                        * (m1[k] / (1 - 0.9**t))
                        / (np.sqrt(m2[k] / (1 - 0.999**t)) + 1e-8)
                    )

        def score(m):
            out = []
            for s in range(0, len(m), 65536):
                h = m[s : s + 65536]
                for l in range(len(W)):
                    h = h @ W[l] + B[l]
                    h = np.maximum(h, 0) if l < len(W) - 1 else h
                out.append(sigmoid(h[:, 0]))
            return np.concatenate(out)

        return score

    def fit(y, tx):
        nets = [train_one(y, tx, seed) for seed in range(n_models)]
        return lambda m: np.mean([net(m) for net in nets], axis=0)

    return fit


# C. Gradient-boosted decision trees


def gbdt(
    n_trees=300,
    depth=5,
    lr=0.1,
    lambda_=1.0,
    min_hess=1.0,
    subsample=0.5,
    n_bins=32,
    seed=0,
):
    """Model C. Each tree fits the 2nd-order approximation of the logistic loss
    (as in XGBoost): for gradient g and hessian h summed in each child,
        gain = GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda),
        leaf value = -lr * G / (H + lambda).
    Trees are grown level by level using histograms of g and h per bin.
    Each tree uses a random half of the rows and of the features.
    Input: binned uint8 matrix (see utils.tree_features)."""

    def predict_tree(tree, xb):
        feats, thrs, leaf = tree
        node, rows = np.zeros(len(xb), dtype=np.int64), np.arange(len(xb))
        for f, t in zip(feats, thrs):
            node = 2 * node + (xb[rows, f[node]] > t[node])
        return leaf[node]

    def fit(y, xb):
        rng = np.random.default_rng(seed)
        n, F = xb.shape
        base = np.log(y.mean() / (1 - y.mean()))
        f, trees = np.full(n, base), []
        for _ in range(n_trees):
            p = sigmoid(f)
            rows = rng.choice(n, int(subsample * n), replace=False)
            cols = np.sort(rng.choice(F, int(subsample * F), replace=False))
            xs = xb[rows][:, cols].astype(np.int32)
            g, h = (p - y)[rows], (p * (1 - p))[rows]
            node, feats, thrs = np.zeros(len(rows), dtype=np.int32), [], []
            for level in range(depth):
                nn = 2**level  # histograms of g and h per (node, feature, bin)
                key = (
                    (node[:, None] * len(cols) + np.arange(len(cols))) * n_bins + xs
                ).ravel()
                shape, size = (nn, len(cols), n_bins), nn * len(cols) * n_bins
                G = np.bincount(key, np.repeat(g, len(cols)), size).reshape(shape)
                H = np.bincount(key, np.repeat(h, len(cols)), size).reshape(shape)
                GL, HL = G.cumsum(2)[:, :, :-1], H.cumsum(2)[:, :, :-1]
                Gt, Ht = G.sum(2, keepdims=True), H.sum(2, keepdims=True)
                gain = (
                    GL**2 / (HL + lambda_)
                    + (Gt - GL) ** 2 / (Ht - HL + lambda_)
                    - Gt**2 / (Ht + lambda_)
                )
                gain[(HL < min_hess) | (Ht - HL < min_hess)] = -np.inf
                best = gain.reshape(nn, -1).argmax(1)
                ok = gain.reshape(nn, -1)[np.arange(nn), best] > 0
                bf, bt = best // (n_bins - 1), best % (n_bins - 1)
                feats.append(np.where(ok, cols[bf], 0))
                thrs.append(np.where(ok, bt, n_bins))  # no split: all go left
                node = 2 * node + (
                    xs[np.arange(len(rows)), np.where(ok, bf, 0)[node]] > thrs[-1][node]
                )
            leaf = (
                -lr
                * np.bincount(node, g, 2**depth)
                / (np.bincount(node, h, 2**depth) + lambda_)
            )
            trees.append((feats, thrs, leaf))
            f += predict_tree(trees[-1], xb)
        return lambda m: sigmoid(base + sum(predict_tree(t, m) for t in trees))

    return fit
