"""
Qiskit Fall Fest 2026 - Quantum Machine Learning for Healthcare
Small-cohort breast-cancer diagnosis: classical kernel SVM vs. quantum-kernel SVM vs. a
quantum-enhanced (hybrid) kernel SVM, all on identical preprocessing and identical tuning budgets.

Usage:
    python experiment.py --stage dev     # design-stage splits (used while building the method)
    python experiment.py --stage final   # fresh, untouched splits for the reported results
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qkernel import N_QUBITS, bloch_features, fidelity_kernel, projected_kernel, zz_statevectors

TRAIN_SIZES = [16, 32, 64, 128, 256]
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
GAMMA_GRID = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]   # classical RBF kernel width
BANDWIDTH_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]      # quantum bandwidth (angle = bw * z)
PQK_GAMMA_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]                 # projected-kernel width
WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]                                # hybrid mixing weight (0 = pure classical)
SEEDS = {"dev": 11, "final": 2026}

MODELS = ["Logistic regression", "Random forest", "RBF SVM", "Quantum kernel SVM", "Projected quantum kernel SVM",
          "Hybrid RBF+linear SVM", "Hybrid RBF+quantum SVM"]


# ------------------------------------------------------------------ helpers
def cv_splits(y, seed):
    k = int(min(3, np.bincount(y).min()))
    return list(RepeatedStratifiedKFold(n_splits=k, n_repeats=2, random_state=seed).split(np.zeros(len(y)), y))


def cv_kernel(K, y, C, splits):
    """Mean inner-CV balanced accuracy of a precomputed-kernel SVM."""
    s = []
    for tr, va in splits:
        clf = SVC(C=C, kernel="precomputed").fit(K[np.ix_(tr, tr)], y[tr])
        s.append(balanced_accuracy_score(y[va], clf.predict(K[np.ix_(va, tr)])))
    return float(np.mean(s))


def tune_kernel(kernels, y, splits):
    """kernels: list of (setting, K_train). Returns best (setting, C) by inner CV; ties keep the earlier one."""
    best, best_s = None, -1.0
    for setting, K in kernels:
        for C in C_GRID:
            s = cv_kernel(K, y, C, splits)
            if s > best_s + 1e-12:
                best, best_s = (setting, C), s
    return best


def rbf(A, B, gamma):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d, 0))


def lin_norm(A, B, scale):
    return (A @ B.T) / scale


def evaluate(y_true, score, threshold=0.0):
    pred = (score > threshold).astype(int)
    return {"bacc": balanced_accuracy_score(y_true, pred), "auc": roc_auc_score(y_true, score),
            "sens": recall_score(y_true, pred, pos_label=1), "spec": recall_score(y_true, pred, pos_label=0)}


# ------------------------------------------------------------------ one split
def run_split(X, y, n, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    s1 = StandardScaler().fit(Xtr)
    pca = PCA(n_components=N_QUBITS, random_state=seed).fit(s1.transform(Xtr))
    s2 = StandardScaler().fit(pca.transform(s1.transform(Xtr)))
    Ztr = s2.transform(pca.transform(s1.transform(Xtr)))
    Zte = s2.transform(pca.transform(s1.transform(Xte)))
    splits = cv_splits(ytr, seed)
    out, info = {}, {}

    # classical references
    best_c = max(C_GRID, key=lambda C: np.mean([balanced_accuracy_score(
        ytr[va], LogisticRegression(C=C, max_iter=5000).fit(Ztr[tr], ytr[tr]).predict(Ztr[va])) for tr, va in splits]))
    lr = LogisticRegression(C=best_c, max_iter=5000).fit(Ztr, ytr)
    out["Logistic regression"] = evaluate(yte, lr.decision_function(Zte))
    rf = RandomForestClassifier(n_estimators=300, random_state=seed).fit(Ztr, ytr)
    out["Random forest"] = evaluate(yte, rf.predict_proba(Zte)[:, 1], threshold=0.5)

    # kernels on the training set
    rbf_tr = {g: rbf(Ztr, Ztr, g) for g in GAMMA_GRID}
    t = time.perf_counter()
    S_tr = {b: zz_statevectors(b * Ztr) for b in BANDWIDTH_GRID}
    q_tr = {b: fidelity_kernel(S_tr[b], S_tr[b]) for b in BANDWIDTH_GRID}
    info["quantum_sim_seconds"] = time.perf_counter() - t
    lin_scale = float(np.mean(np.sum(Ztr ** 2, axis=1)))

    # RBF SVM (comparable classical baseline)
    g_star, C_rbf = tune_kernel([(g, rbf_tr[g]) for g in GAMMA_GRID], ytr, splits)
    # Quantum kernel SVM
    b_star, C_q = tune_kernel([(b, q_tr[b]) for b in BANDWIDTH_GRID], ytr, splits)
    # Projected quantum kernel SVM: tune (bandwidth, pqk gamma) jointly
    F_tr = {b: bloch_features(S_tr[b]) for b in BANDWIDTH_GRID}
    (pb, pg), C_p = tune_kernel([((b, g), projected_kernel(F_tr[b], F_tr[b], g))
                                 for b in BANDWIDTH_GRID for g in PQK_GAMMA_GRID], ytr, splits)
    # kernel-concentration diagnostic: spread of off-diagonal fidelity-kernel values
    off = q_tr[b_star][~np.eye(n, dtype=bool)]
    info["qk_offdiag_mean"], info["qk_offdiag_var"] = float(off.mean()), float(off.var())
    # Hybrids: keep each component's own tuned width, then tune the mixing weight and C
    w_q, C_hq = tune_kernel([(w, (1 - w) * rbf_tr[g_star] + w * q_tr[b_star]) for w in WEIGHT_GRID], ytr, splits)
    lin_tr = lin_norm(Ztr, Ztr, lin_scale)
    w_l, C_hl = tune_kernel([(w, (1 - w) * rbf_tr[g_star] + w * lin_tr) for w in WEIGHT_GRID], ytr, splits)

    S_te = zz_statevectors(b_star * Zte)
    K = {
        "RBF SVM": (rbf_tr[g_star], rbf(Zte, Ztr, g_star), C_rbf),
        "Quantum kernel SVM": (q_tr[b_star], fidelity_kernel(S_te, S_tr[b_star]), C_q),
        "Projected quantum kernel SVM": (projected_kernel(F_tr[pb], F_tr[pb], pg),
                                         projected_kernel(bloch_features(zz_statevectors(pb * Zte)), F_tr[pb], pg), C_p),
        "Hybrid RBF+quantum SVM": ((1 - w_q) * rbf_tr[g_star] + w_q * q_tr[b_star],
                                   (1 - w_q) * rbf(Zte, Ztr, g_star) + w_q * fidelity_kernel(S_te, S_tr[b_star]), C_hq),
        "Hybrid RBF+linear SVM": ((1 - w_l) * rbf_tr[g_star] + w_l * lin_tr,
                                  (1 - w_l) * rbf(Zte, Ztr, g_star) + w_l * lin_norm(Zte, Ztr, lin_scale), C_hl),
    }
    for name, (Ktr, Kte, C) in K.items():
        clf = SVC(C=C, kernel="precomputed").fit(Ktr, ytr)
        out[name] = evaluate(yte, clf.decision_function(Kte))
    info.update({"gamma": g_star, "C_rbf": C_rbf, "bandwidth": b_star, "C_q": C_q,
                 "pqk_bandwidth": pb, "pqk_gamma": pg, "C_pqk": C_p,
                 "w_quantum": w_q, "C_hq": C_hq, "w_linear": w_l, "C_hl": C_hl,
                 "n_test": int(len(yte)),
                 "kernel_entries_train": int(n * (n - 1) // 2), "kernel_entries_test": int(len(yte) * n)})
    return out, info


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "final"], default="dev")
    ap.add_argument("--repeats", type=int, default=None)
    args = ap.parse_args()
    repeats = args.repeats or (10 if args.stage == "dev" else 30)

    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y  # 1 = malignant
    rng = np.random.RandomState(SEEDS[args.stage])
    res = {n: {m: [] for m in MODELS} for n in TRAIN_SIZES}
    infos = {n: [] for n in TRAIN_SIZES}
    t0 = time.time()
    for n in TRAIN_SIZES:
        for r in range(repeats):
            out, info = run_split(X, y, n, int(rng.randint(1e9)))
            for m in MODELS:
                res[n][m].append(out[m])
            infos[n].append(info)
        print(f"n={n:>3} done  ({time.time() - t0:.0f}s)", flush=True)

    # summary tables
    print("\nmean over splits:  bacc | AUC | sensitivity | specificity")
    summary = []
    for n in TRAIN_SIZES:
        for m in MODELS:
            arr = {k: np.array([d[k] for d in res[n][m]]) for k in ("bacc", "auc", "sens", "spec")}
            row = {"train_size": n, "model": m}
            for k, v in arr.items():
                row[f"{k}_mean"], row[f"{k}_sd"] = float(v.mean()), float(v.std(ddof=1))
            summary.append(row)
            print(f"n={n:>3} {m:<24} {row['bacc_mean']:.3f} | {row['auc_mean']:.3f} | {row['sens_mean']:.3f} | {row['spec_mean']:.3f}")

    comparisons = [("Quantum kernel SVM", "RBF SVM"), ("Projected quantum kernel SVM", "RBF SVM"),
                   ("Projected quantum kernel SVM", "Logistic regression"), ("Hybrid RBF+quantum SVM", "RBF SVM"),
                   ("Hybrid RBF+quantum SVM", "Hybrid RBF+linear SVM"), ("Hybrid RBF+quantum SVM", "Logistic regression")]
    tests = []
    print("\npaired comparisons (balanced accuracy and AUC; Wilcoxon signed-rank)")
    for a, b in comparisons:
        for n in TRAIN_SIZES:
            row = {"model": a, "baseline": b, "train_size": n}
            for k in ("bacc", "auc"):
                d = np.array([x[k] for x in res[n][a]]) - np.array([x[k] for x in res[n][b]])
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                row[f"{k}_diff"], row[f"{k}_ci95"], row[f"{k}_p"] = float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), p
                row[f"{k}_wins"], row[f"{k}_losses"] = int((d > 0).sum()), int((d < 0).sum())
            tests.append(row)
            print(f"{a:<24} vs {b:<22} n={n:>3}  bacc {row['bacc_diff']:+.3f}±{row['bacc_ci95']:.3f} (p={row['bacc_p']:.3g}, "
                  f"{row['bacc_wins']}W/{row['bacc_losses']}L)   AUC {row['auc_diff']:+.4f}±{row['auc_ci95']:.4f} (p={row['auc_p']:.3g})")

    print("\nchosen hyper-parameters (most common)")
    for n in TRAIN_SIZES:
        from collections import Counter
        c = lambda k: Counter(i[k] for i in infos[n]).most_common(3)
        print(f"n={n:>3} gamma {c('gamma')}  bw {c('bandwidth')}  pqk {c('pqk_bandwidth')} {c('pqk_gamma')}  "
              f"C_q {c('C_q')}  w_q {c('w_quantum')}  QK offdiag mean {np.mean([i['qk_offdiag_mean'] for i in infos[n]]):.3f}")

    Path("results").mkdir(exist_ok=True)
    with open(f"results/{args.stage}_results.json", "w") as f:
        json.dump({"stage": args.stage, "repeats": repeats, "train_sizes": TRAIN_SIZES, "summary": summary,
                   "tests": tests, "per_split": {str(n): res[n] for n in TRAIN_SIZES},
                   "info": {str(n): infos[n] for n in TRAIN_SIZES}}, f, indent=1)
    print(f"\nsaved results/{args.stage}_results.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
