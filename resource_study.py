"""
Quantum resource / robustness study.

On real hardware the kernels are not exact:
  * Fidelity kernel: each entry K(x, x') is the probability of measuring |0...0> after U(x')^dagger U(x),
    estimated from S shots -> Binomial(S, K) / S. Needs one circuit per PAIR of patients.
  * Projected kernel: the Bloch vector of each qubit is estimated by measuring all qubits in the
    X, Y and Z bases (3 circuits per PATIENT, S shots each).
Noise: approximate global depolarising model with per-CNOT error eps. The surviving-signal fraction
f = (1 - eps)^(number of CNOTs) damps overlaps towards 1/2^n and Pauli expectations towards 0.
This is a simple approximation, not a full device simulation.

Hyper-parameters are tuned on exact kernels (as in experiment.py); the selected models are then
re-trained and tested on kernels estimated under each measurement condition.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment import BANDWIDTH_GRID, GAMMA_GRID, PQK_GAMMA_GRID, cv_splits, evaluate, rbf, tune_kernel
from qkernel import N_QUBITS, bloch_features, circuit_resources, fidelity_kernel, projected_kernel, zz_statevectors

SIZES = [32, 128]
REPEATS = 15
SEED = 7
CONDITIONS = [("exact", None, 0.0), ("10000 shots", 10000, 0.0), ("1000 shots", 1000, 0.0),
              ("100 shots", 100, 0.0), ("1000 shots + noise (eps=0.005)", 1000, 0.005)]


def noisy_fidelity(K, shots, f, rng, symmetric):
    p = f * K + (1 - f) / 2 ** N_QUBITS
    if shots is None:
        return p
    est = rng.binomial(shots, np.clip(p, 0, 1)) / shots
    if symmetric:
        est = np.triu(est, 1)
        est = est + est.T + np.eye(len(est))   # diagonal is 1 by definition, not measured
    return est


def noisy_bloch(F, shots, f, rng):
    m = f * F
    if shots is None:
        return m
    return 2 * rng.binomial(shots, np.clip((1 + m) / 2, 0, 1)) / shots - 1


def main():
    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y
    res = circuit_resources()
    rng_split = np.random.RandomState(SEED)
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for n in SIZES:
        acc = {c[0]: {"Quantum kernel SVM": [], "Projected quantum kernel SVM": []} for c in CONDITIONS}
        rbf_ref = []
        for _ in range(REPEATS):
            seed = int(rng_split.randint(1e9))
            Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
            s1 = StandardScaler().fit(Xtr)
            pca = PCA(n_components=N_QUBITS, random_state=seed).fit(s1.transform(Xtr))
            s2 = StandardScaler().fit(pca.transform(s1.transform(Xtr)))
            Ztr, Zte = (s2.transform(pca.transform(s1.transform(A))) for A in (Xtr, Xte))
            splits = cv_splits(ytr, seed)

            g, Cr = tune_kernel([(g, rbf(Ztr, Ztr, g)) for g in GAMMA_GRID], ytr, splits)
            rbf_ref.append(evaluate(yte, SVC(C=Cr, kernel="precomputed").fit(rbf(Ztr, Ztr, g), ytr)
                                    .decision_function(rbf(Zte, Ztr, g)))["bacc"])
            S_tr = {b: zz_statevectors(b * Ztr) for b in BANDWIDTH_GRID}
            b, Cq = tune_kernel([(b, fidelity_kernel(S_tr[b], S_tr[b])) for b in BANDWIDTH_GRID], ytr, splits)
            F_tr = {bb: bloch_features(S_tr[bb]) for bb in BANDWIDTH_GRID}
            (pb, pg), Cp = tune_kernel([((bb, gg), projected_kernel(F_tr[bb], F_tr[bb], gg))
                                        for bb in BANDWIDTH_GRID for gg in PQK_GAMMA_GRID], ytr, splits)

            Kq_tr, Kq_te = fidelity_kernel(S_tr[b], S_tr[b]), fidelity_kernel(zz_statevectors(b * Zte), S_tr[b])
            Fp_tr, Fp_te = F_tr[pb], bloch_features(zz_statevectors(pb * Zte))
            for name, shots, eps in CONDITIONS:
                f = (1 - eps) ** res["kernel_circuit_cx"]
                f_local = (1 - eps) ** res["feature_map_cx"]
                Ktr = noisy_fidelity(Kq_tr, shots, f, rng, True)
                Kte = noisy_fidelity(Kq_te, shots, f, rng, False)
                acc[name]["Quantum kernel SVM"].append(
                    evaluate(yte, SVC(C=Cq, kernel="precomputed").fit(Ktr, ytr).decision_function(Kte))["bacc"])
                Ftr, Fte = noisy_bloch(Fp_tr, shots, f_local, rng), noisy_bloch(Fp_te, shots, f_local, rng)
                acc[name]["Projected quantum kernel SVM"].append(
                    evaluate(yte, SVC(C=Cp, kernel="precomputed").fit(projected_kernel(Ftr, Ftr, pg), ytr)
                             .decision_function(projected_kernel(Fte, Ftr, pg)))["bacc"])
        n_test = len(y) - n
        circuits = {"Quantum kernel SVM": n * (n - 1) // 2 + n_test * n,
                    "Projected quantum kernel SVM": 3 * (n + n_test)}
        for name, shots, eps in CONDITIONS:
            for model, v in acc[name].items():
                rows.append({"train_size": n, "condition": name, "model": model,
                             "bacc_mean": float(np.mean(v)), "bacc_sd": float(np.std(v, ddof=1)),
                             "rbf_svm_bacc_mean": float(np.mean(rbf_ref)),
                             "circuits": circuits[model],
                             "total_shots": None if shots is None else circuits[model] * shots})
        print(f"n={n} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\ncircuit resources per kernel circuit: {res}")
    print(f"\n{'n':>4} {'condition':<32} {'model':<30} {'bacc':>13} {'RBF ref':>8} {'circuits':>9} {'total shots':>12}")
    for r in rows:
        ts = "-" if r["total_shots"] is None else f"{r['total_shots']:,}"
        print(f"{r['train_size']:>4} {r['condition']:<32} {r['model']:<30} {r['bacc_mean']:.3f} ± {r['bacc_sd']:.3f} "
              f"{r['rbf_svm_bacc_mean']:>8.3f} {r['circuits']:>9,} {ts:>12}")
    Path("results").mkdir(exist_ok=True)
    with open("results/resource_study.json", "w") as fh:
        json.dump({"circuit_resources": res, "repeats": REPEATS, "rows": rows}, fh, indent=1)
    print("\nsaved results/resource_study.json")


if __name__ == "__main__":
    main()
