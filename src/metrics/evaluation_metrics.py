
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.stats import binomtest


def compute_brier_and_confidence_ece(
    probs: np.ndarray,
    true_bits: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, object]:
    p = np.asarray(probs, dtype=np.float64).ravel()
    b = np.asarray(true_bits, dtype=np.uint8).ravel()
    m_total = len(p)

    pred = (p >= 0.5).astype(np.uint8)
    correct = (pred == b).astype(np.float64)
    conf = np.maximum(p, 1.0 - p)
    true_prob = np.where(b == 1, p, 1.0 - p)
    brier = float(np.mean((1.0 - true_prob) ** 2))

    edges = np.linspace(0.5, 1.0, num_bins + 1)
    bins_out = []
    ece = 0.0
    for m in range(num_bins):
        lo, hi = edges[m], edges[m + 1]
        mask = ((conf >= lo) & (conf < hi)) if m < num_bins - 1 else ((conf >= lo) & (conf <= hi + 1e-12))
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_c = float(np.mean(conf[mask]))
            mean_a = float(np.mean(correct[mask]))
            gap = abs(mean_a - mean_c)
            ece += (cnt / m_total) * gap
        else:
            mean_c, mean_a, gap = 0.5 * (lo + hi), 0.0, 0.0
        bins_out.append({
            "bin_index": m + 1,
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "count": cnt,
            "mean_confidence": mean_c,
            "empirical_accuracy": mean_a,
            "abs_calibration_gap": gap,
        })

    return {
        "total_samples": m_total,
        "accuracy": float(np.mean(correct)),
        "num_errors": int(m_total - np.sum(correct)),
        "mean_true_bit_prob": float(np.mean(true_prob)),
        "brier_score": brier,
        "ece_10bin": float(ece),
        "bins": bins_out,
    }


def compute_2x2_error_contingency(
    e_dct: np.ndarray,
    e_neu: np.ndarray,
) -> Dict[str, float]:
    ed = np.asarray(e_dct, dtype=np.int32).ravel()
    en = np.asarray(e_neu, dtype=np.int32).ravel()
    n_cc = int(np.sum((ed == 0) & (en == 0)))
    n_cw = int(np.sum((ed == 1) & (en == 0)))
    n_wc = int(np.sum((ed == 0) & (en == 1)))
    n_ww = int(np.sum((ed == 1) & (en == 1)))

    n_dct_err = n_cw + n_ww
    n_neu_err = n_wc + n_ww
    n_dct_ok = n_cc + n_wc
    n_neu_ok = n_cc + n_cw

    denom = np.sqrt(float(n_dct_err) * float(n_dct_ok) * float(n_neu_err) * float(n_neu_ok))
    phi = float((n_cc * n_ww - n_cw * n_wc) / denom) if denom > 0 else 0.0

    return {
        "n_CC": n_cc,
        "n_CW": n_cw,
        "n_WC": n_wc,
        "n_WW": n_ww,
        "dct_errors": n_dct_err,
        "neural_errors": n_neu_err,
        "pearson_phi": phi,
        "prob_neu_correct_given_dct_err": float(n_cw / max(n_dct_err, 1)),
        "prob_dct_correct_given_neu_err": float(n_wc / max(n_neu_err, 1)),
    }


def paired_sign_test_and_clustered_bootstrap(
    success_p9: np.ndarray,
    success_base: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    s9 = np.asarray(success_p9, dtype=np.int32).ravel()
    sb = np.asarray(success_base, dtype=np.int32).ravel()
    diffs = (s9 - sb).astype(np.float64) * 100.0
    gain_pp = float(np.mean(diffs))

    n_10 = int(np.sum((s9 == 1) & (sb == 0)))
    n_01 = int(np.sum((s9 == 0) & (sb == 1)))
    n_disc = n_10 + n_01
    p_val = 1.0 if n_disc == 0 else float(binomtest(n_10, n_disc, 0.5, alternative="two-sided").pvalue)

    rng = np.random.RandomState(seed)
    if cluster_ids is not None:
        cids = np.asarray(cluster_ids)
        uniq, inv = np.unique(cids, return_inverse=True)
        n_clusters = len(uniq)
        cluster_sums = np.zeros(n_clusters, dtype=np.float64)
        counts = np.bincount(inv).astype(np.float64)
        np.add.at(cluster_sums, inv, diffs)
        idx = rng.randint(0, n_clusters, size=(n_boot, n_clusters))
        boot_gains = np.sum(cluster_sums[idx], axis=1) / np.maximum(np.sum(counts[idx], axis=1), 1.0)
    else:
        n_clusters = len(diffs)
        idx = rng.randint(0, n_clusters, size=(n_boot, n_clusters))
        boot_gains = np.mean(diffs[idx], axis=1)

    ci_lo, ci_hi = np.percentile(boot_gains, [2.5, 97.5])
    return {
        "gain_pp": gain_pp,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "ci95_low_pp": float(ci_lo),
        "ci95_high_pp": float(ci_hi),
        "n_10": n_10,
        "n_01": n_01,
        "p_value": p_val,
        "exact_sign_test_p": p_val,
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
