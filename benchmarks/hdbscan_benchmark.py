#!/usr/bin/env python3
# ==============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Benchmark: scikit-learn-intelex HDBSCAN vs stock sklearn HDBSCAN

Apples-to-apples comparison: same algorithm, same metric, same parameters.

Usage:
    python benchmarks/hdbscan_benchmark.py
"""

import time

import numpy as np
from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from sklearnex.cluster import HDBSCAN as sklearnex_HDBSCAN


def time_fit(cls, X, n_runs=3, warmup=1, **kwargs):
    """Time fit() over multiple runs, return (median_ms, labels)."""
    times = []
    labels = None
    for i in range(warmup + n_runs):
        h = cls(**kwargs)
        t0 = time.perf_counter()
        h.fit(X)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
        labels = h.labels_
    return np.median(times) * 1000, labels


def run_comparison(X, y_true, description, sklearn_kw, sklearnex_kw, n_runs=3):
    """Run one comparison: sklearn vs sklearnex with given params."""
    sk_ms, sk_labels = time_fit(
        sklearn_HDBSCAN, X, n_runs=n_runs, **sklearn_kw
    )
    ex_ms, ex_labels = time_fit(
        sklearnex_HDBSCAN, X, n_runs=n_runs, **sklearnex_kw
    )

    ari_cross = adjusted_rand_score(sk_labels, np.asarray(ex_labels))
    ari_sk = adjusted_rand_score(y_true, sk_labels) if y_true is not None else None
    ari_ex = adjusted_rand_score(y_true, np.asarray(ex_labels)) if y_true is not None else None
    nc_sk = len(set(sk_labels)) - (1 if -1 in sk_labels else 0)
    nc_ex = len(set(np.asarray(ex_labels))) - (1 if -1 in np.asarray(ex_labels) else 0)
    speedup = sk_ms / ex_ms if ex_ms > 0 else float("inf")

    return {
        "description": description,
        "sklearn_ms": sk_ms,
        "sklearnex_ms": ex_ms,
        "speedup": speedup,
        "ari_cross": ari_cross,
        "ari_sk": ari_sk,
        "ari_ex": ari_ex,
        "nc_sk": nc_sk,
        "nc_ex": nc_ex,
    }


def print_result(r):
    ari_str = f"ARI_sk={r['ari_sk']:.4f}" if r["ari_sk"] is not None else ""
    print(
        f"  sklearn:   {r['sklearn_ms']:9.1f}ms  ({r['nc_sk']} clusters)  {ari_str}"
    )
    ari_str = f"ARI_ex={r['ari_ex']:.4f}" if r["ari_ex"] is not None else ""
    print(
        f"  sklearnex: {r['sklearnex_ms']:9.1f}ms  ({r['nc_ex']} clusters)  {ari_str}"
    )
    print(f"  speedup:   {r['speedup']:9.2f}x    cross-ARI={r['ari_cross']:.4f}")
    if r["ari_cross"] < 0.9:
        print("  ** WARNING: cross-ARI < 0.9 — possible correctness mismatch **")


def main():
    print("=" * 80)
    print("HDBSCAN Benchmark: scikit-learn-intelex vs stock scikit-learn")
    print("  Method-vs-method, metric-vs-metric comparison")
    print("=" * 80)

    # =========================================================================
    # 1. METHOD COMPARISON (euclidean metric, varying data sizes)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: METHOD COMPARISON (euclidean metric)")
    print("=" * 80)

    # sklearn: brute = brute-force pairwise distances
    #          kd_tree = KDTree-accelerated mutual reachability
    # oneDAL:  brute_force = BLAS GEMM distance + Prim's MST
    #          kd_tree = kd-tree core distances + Boruvka MST

    datasets = [
        ("1K x 2D", 1000, 2, 3, 0.5),
        ("5K x 2D", 5000, 2, 5, 0.5),
        ("10K x 2D", 10000, 2, 7, 0.5),
        ("50K x 2D", 50000, 2, 7, 0.8),
        ("5K x 10D", 5000, 10, 5, 1.0),
        ("10K x 10D", 10000, 10, 5, 1.0),
        ("50K x 10D", 50000, 10, 7, 1.5),
    ]

    methods = [
        # (label, sklearn_algorithm, sklearnex_algorithm)
        ("brute", "brute", "brute"),
        ("kd_tree", "kd_tree", "kd_tree"),
        ("auto", "auto", "auto"),
    ]

    for name, n, d, k, std in datasets:
        X, y = make_blobs(n_samples=n, n_features=d, centers=k,
                          cluster_std=std, random_state=42)
        mcs = max(15, n // 100)

        print(f"\n--- {name}, {k} centers, mcs={mcs} ---")
        for mlabel, sk_algo, ex_algo in methods:
            desc = f"{name} {mlabel}"
            sk_kw = dict(
                min_cluster_size=mcs, min_samples=5, algorithm=sk_algo,
                metric="euclidean", copy=True,
            )
            ex_kw = dict(
                min_cluster_size=mcs, min_samples=5, algorithm=ex_algo,
                metric="euclidean",
            )
            r = run_comparison(X, y, desc, sk_kw, ex_kw, n_runs=3)
            print(f"  [{mlabel:8s}]  sklearn: {r['sklearn_ms']:9.1f}ms  "
                  f"sklearnex: {r['sklearnex_ms']:9.1f}ms  "
                  f"speedup: {r['speedup']:7.1f}x  "
                  f"cross-ARI: {r['ari_cross']:.4f}")

    # =========================================================================
    # 2. METRIC COMPARISON (brute-force, 10K points)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: METRIC COMPARISON (brute-force method, 10K x 2D)")
    print("=" * 80)

    X, y = make_blobs(n_samples=10000, n_features=2, centers=5,
                      cluster_std=0.5, random_state=42)
    mcs = 100

    # Metrics supported by both sklearn and oneDAL
    metrics_brute = ["euclidean", "manhattan", "chebyshev", "cosine"]

    for metric in metrics_brute:
        sk_kw = dict(
            min_cluster_size=mcs, min_samples=5, algorithm="brute",
            metric=metric, copy=True,
        )
        ex_kw = dict(
            min_cluster_size=mcs, min_samples=5, algorithm="brute",
            metric=metric,
        )
        r = run_comparison(X, y, f"brute+{metric}", sk_kw, ex_kw, n_runs=3)
        print(f"\n  [{metric:12s}]")
        print_result(r)

    # =========================================================================
    # 3. METRIC COMPARISON (kd_tree, 10K points)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: METRIC COMPARISON (kd_tree method, 10K x 2D)")
    print("=" * 80)

    # kd_tree in sklearn supports: euclidean, manhattan, chebyshev (NOT cosine)
    # kd_tree in oneDAL supports: euclidean, manhattan, chebyshev, minkowski (NOT cosine)
    metrics_kdtree = ["euclidean", "manhattan", "chebyshev"]

    for metric in metrics_kdtree:
        sk_kw = dict(
            min_cluster_size=mcs, min_samples=5, algorithm="kd_tree",
            metric=metric, copy=True,
        )
        ex_kw = dict(
            min_cluster_size=mcs, min_samples=5, algorithm="kd_tree",
            metric=metric,
        )
        r = run_comparison(X, y, f"kd_tree+{metric}", sk_kw, ex_kw, n_runs=3)
        print(f"\n  [{metric:12s}]")
        print_result(r)

    # =========================================================================
    # 4. SCALING TEST (auto algorithm, euclidean)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: SCALING TEST (auto algorithm, euclidean, 2D)")
    print("=" * 80)

    scaling_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
    print(f"\n  {'N':>8s}  {'sklearn':>10s}  {'sklearnex':>10s}  {'speedup':>8s}  {'cross-ARI':>10s}")
    print(f"  {'':->8s}  {'':->10s}  {'':->10s}  {'':->8s}  {'':->10s}")

    for n in scaling_sizes:
        X, y = make_blobs(n_samples=n, n_features=2, centers=5,
                          cluster_std=0.5, random_state=42)
        mcs = max(15, n // 100)
        sk_kw = dict(
            min_cluster_size=mcs, min_samples=5, algorithm="auto",
            metric="euclidean", copy=True,
        )
        ex_kw = dict(
            min_cluster_size=mcs, min_samples=5, algorithm="auto",
            metric="euclidean",
        )
        r = run_comparison(X, y, f"scale-{n}", sk_kw, ex_kw,
                           n_runs=3 if n <= 50000 else 1)
        print(f"  {n:>8d}  {r['sklearn_ms']:>9.1f}ms  {r['sklearnex_ms']:>9.1f}ms  "
              f"{r['speedup']:>7.1f}x  {r['ari_cross']:>10.4f}")

    # =========================================================================
    # 5. HIGH-DIMENSIONAL SCALING (auto, euclidean)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 5: DIMENSIONALITY TEST (auto algorithm, euclidean, 10K points)")
    print("=" * 80)

    dims = [2, 5, 10, 20, 50]
    print(f"\n  {'dim':>5s}  {'sklearn':>10s}  {'sklearnex':>10s}  {'speedup':>8s}  {'cross-ARI':>10s}")
    print(f"  {'':->5s}  {'':->10s}  {'':->10s}  {'':->8s}  {'':->10s}")

    for d in dims:
        X, y = make_blobs(n_samples=10000, n_features=d, centers=5,
                          cluster_std=1.0, random_state=42)
        sk_kw = dict(
            min_cluster_size=100, min_samples=5, algorithm="auto",
            metric="euclidean", copy=True,
        )
        ex_kw = dict(
            min_cluster_size=100, min_samples=5, algorithm="auto",
            metric="euclidean",
        )
        r = run_comparison(X, y, f"dim-{d}", sk_kw, ex_kw, n_runs=3)
        print(f"  {d:>5d}  {r['sklearn_ms']:>9.1f}ms  {r['sklearnex_ms']:>9.1f}ms  "
              f"{r['speedup']:>7.1f}x  {r['ari_cross']:>10.4f}")

    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
