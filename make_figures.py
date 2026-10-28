"""Build the README figures from results/final_results.json and results/resource_study.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Path("figures").mkdir(exist_ok=True)
STYLE = {  # classical = muted/dashed, quantum = solid
    "Logistic regression": ("#7a7a7a", "o", "--"),
    "Random forest": ("#a89f91", "^", "--"),
    "RBF SVM": ("#2f5d8a", "s", "--"),
    "Hybrid RBF+linear SVM": ("#6f93b8", "v", ":"),
    "Quantum kernel SVM": ("#b5541c", "D", "-"),
    "Projected quantum kernel SVM": ("#7d3c98", "P", "-"),
    "Hybrid RBF+quantum SVM": ("#2e8b57", "X", "-"),
}

final = json.load(open("results/final_results.json"))
sizes = final["train_sizes"]
reps = final["repeats"]

# 1. learning curves
fig, ax = plt.subplots(figsize=(8, 4.8))
for m, (c, mk, ls) in STYLE.items():
    rows = [r for r in final["summary"] if r["model"] == m]
    mu = np.array([r["bacc_mean"] for r in rows])
    se = np.array([r["bacc_sd"] for r in rows]) / np.sqrt(reps)
    ax.errorbar(sizes, mu, yerr=1.96 * se, color=c, marker=mk, ls=ls, lw=2 if ls == "-" else 1.3, capsize=3, label=m)
ax.set_xscale("log", base=2)
ax.set_xticks(sizes, [str(s) for s in sizes])
ax.set_xlabel("Labelled training patients")
ax.set_ylabel("Balanced accuracy (held-out patients)")
ax.set_title(f"Breast-cancer diagnosis, 6 PCA features / 6 qubits, {reps} splits per size (95% CI)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig("figures/learning_curves.png", dpi=150)

# 2. paired differences vs the comparable classical baseline
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, label in [(axes[0], "bacc", "Balanced accuracy"), (axes[1], "auc", "ROC AUC")]:
    for m in ["Quantum kernel SVM", "Projected quantum kernel SVM", "Hybrid RBF+quantum SVM"]:
        rows = [t for t in final["tests"] if t["model"] == m and t["baseline"] == "RBF SVM"]
        c, mk, _ = STYLE[m]
        ax.errorbar(sizes, [t[f"{metric}_diff"] for t in rows], yerr=[t[f"{metric}_ci95"] for t in rows],
                    color=c, marker=mk, capsize=3, lw=1.8, label=m)
    ax.axhline(0, color="black", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("Labelled training patients")
    ax.set_ylabel(f"{label}: model − RBF SVM")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Paired difference to the comparable classical baseline (RBF-kernel SVM), mean ± 95% CI")
fig.tight_layout()
fig.savefig("figures/paired_differences.png", dpi=150)

# 3. measurement / noise robustness
rs = json.load(open("results/resource_study.json"))
conds = list(dict.fromkeys(r["condition"] for r in rs["rows"]))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, n in zip(axes, sorted({r["train_size"] for r in rs["rows"]})):
    x = np.arange(len(conds))
    for k, m in enumerate(["Quantum kernel SVM", "Projected quantum kernel SVM"]):
        rows = {r["condition"]: r for r in rs["rows"] if r["train_size"] == n and r["model"] == m}
        c, mk, _ = STYLE[m]
        ax.bar(x + (k - 0.5) * 0.38, [rows[cd]["bacc_mean"] for cd in conds], 0.36, color=c, label=m,
               yerr=[1.96 * rows[cd]["bacc_sd"] / np.sqrt(rs["repeats"]) for cd in conds], capsize=2)
    ref = [r for r in rs["rows"] if r["train_size"] == n][0]["rbf_svm_bacc_mean"]
    ax.axhline(ref, color=STYLE["RBF SVM"][0], ls="--", lw=1.5, label="RBF SVM (exact, classical)")
    ax.set_xticks(x, [c.replace(" + noise", "\n+ noise") for c in conds], fontsize=7.5, rotation=0)
    ax.set_title(f"{n} training patients")
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Balanced accuracy")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Effect of finite measurement shots and approximate hardware noise")
fig.tight_layout()
fig.savefig("figures/shots_and_noise.png", dpi=150)
print("saved figures/learning_curves.png, figures/paired_differences.png, figures/shots_and_noise.png")
