"""The learners, written out rather than imported.

Three estimators, in increasing order of what they can represent:

- ``RidgeRegression`` — the linear baseline, solved in closed form by SVD.
- ``DecisionTreeRegressor`` — a CART regression tree with an exact best-split
  search over pre-binned features.
- ``GradientBoostingRegressor`` — Friedman's stagewise additive model on top of
  those trees, with shrinkage and row subsampling.

There is no scikit-learn dependency, and the reason is the same one that kept
`statsmodels` out of project 03: the interesting parts of these methods are
exactly the parts an import hides. Ridge's behaviour is a shrinkage factor
``s^2/(s^2 + alpha)`` applied to each singular direction, and seeing that in
the code is what makes it obvious why a ridge on 14 collinear features barely
differs from OLS. A boosted tree's capacity is set by the interaction between
depth, learning rate and tree count, and a model that can represent
interactions is precisely what a linear factor model cannot — which is the
question this project is asking.

**Why the features are binned.** The split search is histogram-based, as in
LightGBM: each feature is quantised to 64 bins once, and each candidate split
is then evaluated by accumulating gradient sums into those bins with
``np.bincount``. The alternative — sorting every feature at every node — is
``O(n log n)`` per feature per node and is what makes a pure-Python CART
unusable on 470,000 rows. Binning makes it ``O(n)`` per feature per *depth
level*, and the whole 200-tree fit runs in seconds. The cost is that a split
can only fall on a bin edge; with 64 bins on features that are already
cross-sectional ranks, that is not a meaningful restriction, and
``validation.gate_tree_split`` checks the binned search against a brute-force
optimum to prove it.

References
----------
- Hoerl, A. and Kennard, R. (1970). *Ridge regression: biased estimation for
  nonorthogonal problems.* Technometrics.
- Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984). *Classification
  and Regression Trees.*
- Friedman, J. (2001). *Greedy function approximation: a gradient boosting
  machine.* Annals of Statistics.
- Friedman, J. (2002). *Stochastic gradient boosting.* Computational Statistics
  and Data Analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Linear
# --------------------------------------------------------------------------


class RidgeRegression:
    """Ridge regression solved through the SVD.

    ``alpha`` is the penalty on the coefficients *in standardized units*: the
    design is centred and scaled to unit variance before the penalty is
    applied, so a single ``alpha`` means the same amount of shrinkage for a
    feature measured in basis points and one measured in log dollars. The
    intercept is never penalised — it is recovered from the means afterwards,
    which is the same thing as leaving its column out of the penalty matrix
    and is less error-prone.

    Solving by SVD rather than by inverting ``X'X + alpha*I`` costs a little
    more arithmetic and buys two things. It is stable when the features are
    nearly collinear, which 14 overlapping technical signals certainly are;
    and it exposes the per-direction shrinkage factor ``s^2/(s^2+alpha)``
    directly, which is the quantity ``effective_dof`` reports and the
    orthonormal-design identity in the validation gates checks.
    """

    def __init__(self, alpha: float = 1.0, standardize: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.standardize = standardize
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._singular: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self._mean = X.mean(axis=0)
        if self.standardize:
            scale = X.std(axis=0, ddof=0)
            scale = np.where(scale > 0, scale, 1.0)
        else:
            scale = np.ones(X.shape[1])
        self._scale = scale

        z = (X - self._mean) / scale
        y_mean = y.mean()
        yc = y - y_mean

        u, s, vt = np.linalg.svd(z, full_matrices=False)
        self._singular = s
        # d_j = s_j / (s_j^2 + alpha) — the ridge filter, per singular direction.
        d = s / (s ** 2 + self.alpha)
        coef_z = vt.T @ (d * (u.T @ yc))

        self.coef_ = coef_z / scale
        self.intercept_ = float(y_mean - self._mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    @property
    def effective_dof(self) -> float:
        """``sum_j s_j^2 / (s_j^2 + alpha)`` — the model's effective parameter count.

        Equals the number of features at ``alpha=0`` and falls toward zero as
        the penalty grows. The honest way to say how much a ridge has actually
        constrained the fit; the nominal count of 14 does not change with
        ``alpha`` and so says nothing.
        """
        if self._singular is None:
            raise RuntimeError("model is not fitted")
        s2 = self._singular ** 2
        return float(np.sum(s2 / (s2 + self.alpha)))


def ols(X: np.ndarray, y: np.ndarray) -> RidgeRegression:
    """Ordinary least squares — ridge with the penalty switched off."""
    return RidgeRegression(alpha=0.0).fit(X, y)


# --------------------------------------------------------------------------
# Feature binning
# --------------------------------------------------------------------------


class QuantileBinner:
    """Quantise each feature to at most ``n_bins`` levels at its own quantiles.

    Fitted on the training rows only. Applying a binner fitted on the test set
    — or on both — would let the test period's feature distribution set the
    split candidates, which is a mild but genuine leak of the kind that is
    almost impossible to see in results.
    """

    def __init__(self, n_bins: int = 64) -> None:
        if not 2 <= n_bins <= 256:
            raise ValueError("n_bins must be in [2, 256]")
        self.n_bins = n_bins
        self.edges_: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "QuantileBinner":
        X = np.asarray(X, dtype=float)
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges_ = []
        for j in range(X.shape[1]):
            # Unique-ing collapses the edges of a feature with ties (a boolean,
            # or a rank feature on a small cross-section) so that empty bins
            # are not created for splits that can never separate anything.
            e = np.unique(np.quantile(X[:, j], qs))
            self.edges_.append(e)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges_):
            out[:, j] = np.searchsorted(e, X[:, j], side="left")
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def bin_counts(self) -> list[int]:
        return [len(e) + 1 for e in self.edges_]

    def threshold(self, feature: int, bin_index: int) -> float:
        """The feature value a split at ``bin_index`` corresponds to."""
        e = self.edges_[feature]
        if bin_index < 0 or bin_index >= len(e):
            return np.nan
        return float(e[bin_index])


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------


@dataclass
class Node:
    """One node of a regression tree. Leaves have ``feature == -1``."""

    value: float = 0.0
    feature: int = -1
    bin_split: int = -1          # go left when binned value <= bin_split
    left: "Node | None" = None
    right: "Node | None" = None
    n_samples: int = 0
    gain: float = 0.0


def _best_split(
    binned: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    n_bins: list[int],
    features: np.ndarray,
    min_samples_leaf: int,
) -> tuple[int, int, float]:
    """Exact best split of ``rows`` over the candidate ``features``.

    Returns ``(feature, bin_split, gain)`` with ``gain`` the reduction in sum
    of squared error, or ``(-1, -1, 0.0)`` if no admissible split exists.

    The criterion is the standard variance reduction, written in the form that
    needs only sums and counts:

        gain = S_L^2/n_L + S_R^2/n_R - S^2/n

    where ``S`` is the sum of the target over a side. Both are accumulated per
    bin with ``bincount`` and turned into all candidate splits at once by a
    cumulative sum, so a feature costs one pass over the node's rows regardless
    of how many split points it has.
    """
    t = target[rows]
    total_sum = float(t.sum())
    n = len(rows)
    parent = total_sum * total_sum / n

    # Gather the node's rows once, transposed so each feature is a contiguous
    # vector. Indexing ``binned[rows, j]`` inside the feature loop instead
    # repeats the same expensive gather 14 times per node, and on 300 trees
    # that difference is most of the runtime.
    sub = np.ascontiguousarray(binned[rows].T)

    best_feature, best_bin, best_gain = -1, -1, 0.0
    for j in features:
        col = sub[j]
        nb = n_bins[j]
        if nb < 2:
            continue
        sums = np.bincount(col, weights=t, minlength=nb)
        counts = np.bincount(col, minlength=nb).astype(np.int64)

        left_sum = np.cumsum(sums)[:-1]
        left_n = np.cumsum(counts)[:-1]
        right_sum = total_sum - left_sum
        right_n = n - left_n

        ok = (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
        if not ok.any():
            continue

        gains = np.full(len(left_n), -np.inf)
        gains[ok] = (
            left_sum[ok] ** 2 / left_n[ok]
            + right_sum[ok] ** 2 / right_n[ok]
            - parent
        )
        k = int(np.argmax(gains))
        if gains[k] > best_gain:
            best_feature, best_bin, best_gain = int(j), k, float(gains[k])

    return best_feature, best_bin, best_gain


class DecisionTreeRegressor:
    """A CART regression tree over pre-binned features, squared-error loss.

    Grown greedily to ``max_depth``, splitting on the exact best variance
    reduction at each node. ``min_gain`` stops a node splitting on a
    difference that is not there; without it a deep tree will happily
    manufacture leaves separating a handful of rows on noise, which on
    financial data is most of what there is to find.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        min_gain: float = 0.0,
        max_features: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.root_: Node | None = None
        self.n_features_: int = 0

    def fit(self, binned: np.ndarray, y: np.ndarray, n_bins: list[int]) -> "DecisionTreeRegressor":
        binned = np.asarray(binned)
        y = np.asarray(y, dtype=float).ravel()
        self.n_features_ = binned.shape[1]
        rows = np.arange(len(y))
        self.root_ = self._grow(binned, y, rows, n_bins, depth=0)
        return self

    def _grow(
        self,
        binned: np.ndarray,
        y: np.ndarray,
        rows: np.ndarray,
        n_bins: list[int],
        depth: int,
    ) -> Node:
        node = Node(value=float(y[rows].mean()), n_samples=len(rows))
        if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf:
            return node

        if self.max_features is None or self.max_features >= self.n_features_:
            candidates = np.arange(self.n_features_)
        else:
            candidates = self.rng.choice(
                self.n_features_, size=self.max_features, replace=False
            )

        feature, bin_split, gain = _best_split(
            binned, y, rows, n_bins, candidates, self.min_samples_leaf
        )
        if feature < 0 or gain <= self.min_gain:
            return node

        mask = binned[rows, feature] <= bin_split
        node.feature, node.bin_split, node.gain = feature, bin_split, gain
        node.left = self._grow(binned, y, rows[mask], n_bins, depth + 1)
        node.right = self._grow(binned, y, rows[~mask], n_bins, depth + 1)
        return node

    def predict(self, binned: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("model is not fitted")
        out = np.empty(len(binned), dtype=float)
        self._fill(self.root_, binned, np.arange(len(binned)), out)
        return out

    def _fill(self, node: Node, binned: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        """Push whole blocks of rows down the tree rather than one row at a time."""
        if node.feature < 0 or len(rows) == 0:
            out[rows] = node.value
            return
        mask = binned[rows, node.feature] <= node.bin_split
        self._fill(node.left, binned, rows[mask], out)
        self._fill(node.right, binned, rows[~mask], out)

    def importances(self) -> np.ndarray:
        """Total squared-error reduction attributable to each feature."""
        out = np.zeros(self.n_features_)
        stack = [self.root_] if self.root_ is not None else []
        while stack:
            node = stack.pop()
            if node is None or node.feature < 0:
                continue
            out[node.feature] += node.gain
            stack.extend([node.left, node.right])
        return out


# --------------------------------------------------------------------------
# Gradient boosting
# --------------------------------------------------------------------------


@dataclass
class GradientBoostingRegressor:
    """Friedman's gradient boosting machine with squared-error loss.

    Under squared error the negative gradient of the loss *is* the residual,
    so each stage is simply a tree fitted to what the model so far has got
    wrong, added with a shrinkage factor. That makes the implementation short
    and the behaviour easy to reason about, which is the point of writing it
    out; the same skeleton with a different gradient gives logistic or Huber
    boosting.

    ``subsample < 1`` fits each tree on a random subset of rows (Friedman
    2002). On this data it is not a speed optimisation — it is regularisation,
    and it matters more than the tree count.

    The defaults are deliberately timid: depth 3, 300 trees, learning rate
    0.03. A cross-sectional equity signal has a signal-to-noise ratio around
    1%, and a model with the capacity to fit it exactly is a model that has
    fitted the noise.
    """

    n_estimators: int = 300
    learning_rate: float = 0.03
    max_depth: int = 3
    min_samples_leaf: int = 200
    subsample: float = 0.7
    max_features: int | None = None
    n_bins: int = 64
    seed: int = 0

    trees_: list[DecisionTreeRegressor] = field(default_factory=list, repr=False)
    binner_: QuantileBinner | None = field(default=None, repr=False)
    base_: float = 0.0
    train_loss_: list[float] = field(default_factory=list, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.seed)

        self.binner_ = QuantileBinner(self.n_bins)
        binned = self.binner_.fit_transform(X)
        n_bins = self.binner_.bin_counts

        self.base_ = float(y.mean())
        pred = np.full(len(y), self.base_)
        self.trees_ = []
        self.train_loss_ = []

        n_sub = max(1, int(round(self.subsample * len(y))))
        for _ in range(self.n_estimators):
            residual = y - pred
            if self.subsample < 1.0:
                rows = rng.choice(len(y), size=n_sub, replace=False)
                sub_binned, sub_resid = binned[rows], residual[rows]
            else:
                sub_binned, sub_resid = binned, residual

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=rng,
            ).fit(sub_binned, sub_resid, n_bins)

            pred += self.learning_rate * tree.predict(binned)
            self.trees_.append(tree)
            self.train_loss_.append(float(np.mean((y - pred) ** 2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.binner_ is None:
            raise RuntimeError("model is not fitted")
        binned = self.binner_.transform(np.asarray(X, dtype=float))
        out = np.full(len(binned), self.base_)
        for tree in self.trees_:
            out += self.learning_rate * tree.predict(binned)
        return out

    def staged_predict(self, X: np.ndarray, every: int = 1):
        """Yield ``(n_trees, prediction)`` as the ensemble is built up.

        Used by the figure that shows out-of-sample skill against tree count —
        the plot that says whether more capacity was helping or overfitting,
        which a single final number cannot.
        """
        if self.binner_ is None:
            raise RuntimeError("model is not fitted")
        binned = self.binner_.transform(np.asarray(X, dtype=float))
        out = np.full(len(binned), self.base_)
        for i, tree in enumerate(self.trees_, start=1):
            out += self.learning_rate * tree.predict(binned)
            if i % every == 0 or i == len(self.trees_):
                yield i, out.copy()

    def importances(self, normalize: bool = True) -> np.ndarray:
        """Summed squared-error reduction per feature across every tree."""
        if not self.trees_:
            raise RuntimeError("model is not fitted")
        total = np.zeros(self.trees_[0].n_features_)
        for tree in self.trees_:
            total += tree.importances()
        if normalize and total.sum() > 0:
            total = total / total.sum()
        return total
