import numpy as np
import pytest

from src.datasets.adapter import DatasetAdapter


def test_partition_iid_cover_and_balance():
    N = 103
    num_clients = 5
    parts = DatasetAdapter.partition_iid(N, num_clients, seed=42)

    # cover all indices exactly once
    all_idx = sorted(sum([v for v in parts.values()], []))
    assert all_idx == list(range(N))

    # sizes differ by at most 1
    sizes = [len(v) for v in parts.values()]
    assert max(sizes) - min(sizes) <= 1


def test_partition_dirichlet_cover_and_no_duplicates():
    N = 200
    num_classes = 10
    labels = np.array([i % num_classes for i in range(N)])
    num_clients = 7

    parts = DatasetAdapter.partition_dirichlet(labels, num_clients, alpha=0.5, seed=123)

    all_idx = sorted(sum([v for v in parts.values()], []))
    assert len(all_idx) == N
    assert len(set(all_idx)) == N  # no duplicates

    # per-class counts preserved
    counts_orig = np.bincount(labels, minlength=num_classes)
    counts_reconstructed = np.zeros(num_classes, dtype=int)
    for idxs in parts.values():
        if len(idxs) == 0:
            continue
        lbls = labels[idxs]
        counts_reconstructed += np.bincount(lbls, minlength=num_classes)
    assert np.array_equal(counts_reconstructed, counts_orig)


def test_dirichlet_small_alpha_leads_to_skewness():
    # With a small alpha we expect concentrated (skewed) partitions for some clients
    N = 300
    num_classes = 5
    labels = np.repeat(np.arange(num_classes), N // num_classes)
    num_clients = 10

    parts = DatasetAdapter.partition_dirichlet(labels, num_clients, alpha=0.01, seed=1)

    # ensure at least one client has a dominant class (>50% of its samples)
    dominated = False
    for idxs in parts.values():
        if len(idxs) == 0:
            continue
        lbls = labels[idxs]
        vals = np.bincount(lbls, minlength=num_classes)
        if vals.max() > 0.5 * len(idxs):
            dominated = True
            break
    assert dominated
