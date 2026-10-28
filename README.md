# Quantum Kernels for Small-Cohort Breast-Cancer Diagnosis

**Qiskit Fall Fest 2026 · Quantum Machine Learning for Healthcare track**

A fair, controlled comparison of quantum-kernel classifiers against a comparable classical kernel
classifier for diagnosing breast cancer from biopsy measurements when only **16–256 labelled
patients** are available.

> **Summary of findings** (details and numbers in [Results](#4-results))
>
> - **No quantum advantage was found.** On 30 fresh splits per size, the tuned quantum-kernel SVM
>   *matches* the comparable classical RBF-kernel SVM at 128 patients (balanced accuracy 0.947 vs
>   0.945, p = 0.95) but is **significantly worse** at 16, 32, 64 and 256 patients. Simple logistic
>   regression is the best model at every size from 32 patients upward.
> - **Why:** to be competitive, cross-validation pushes the quantum feature map to a tiny bandwidth,
>   where all kernel values sit at ≈ 0.99 (*kernel concentration*). The circuit then behaves like a
>   smooth, nearly classical kernel, and the entangling structure is not used.
> - **Resource cost:** that same tuning makes the kernel fragile on hardware. With 10,000 shots per
>   circuit, accuracy drops from 0.95 to 0.76 at 128 patients (about 646 million shots in total),
>   while the classical baseline runs in milliseconds.
> - **The one positive quantum-vs-quantum result:** the projected quantum kernel matches the fidelity
>   kernel's exact accuracy at 128 patients (0.947 vs 0.949) with **38× fewer circuits** (1,707 vs
>   64,576), and its cost grows linearly rather than quadratically with the number of patients.
> - We report these as negative results on purpose, with a protocol designed so that an advantage
>   *could* have shown up (Section 2.3–2.4).

---

## 1. The healthcare problem

**Task.** Classify a breast mass as **malignant or benign** from measurements of cell nuclei in a
fine-needle aspirate (FNA) biopsy image.

**Why it matters.** FNA is a cheap, minimally invasive first test, and an automated second reader
helps where specialist cytopathologists are scarce. The costly error is a **missed malignancy**, so
we report **sensitivity** (malignant cases caught) and **specificity** alongside overall metrics.

**Why small cohorts.** A new clinic, a new imaging device or an under-represented population often
starts with only a few dozen labelled cases. Classical models are weakest there, and small data is
also where the QML literature most often suggests quantum models *might* help. So we study the
whole learning curve (16, 32, 64, 128 and 256 training patients), not one large split.

**Data.** Breast Cancer Wisconsin (Diagnostic) dataset: 569 patients (212 malignant, 357 benign) and
30 real-valued features per patient (radius, texture, concavity and so on, each as mean, standard
error and worst value). Source: W. N. Street, W. H. Wolberg and O. L. Mangasarian (1993), UCI Machine
Learning Repository, loaded with `sklearn.datasets.load_breast_cancer`. Malignant is the positive class.

**Intended outcome.** Find out whether a quantum kernel gives a measurable benefit over a comparable
classical kernel in this small-data regime, and at what quantum resource cost.

---

## 2. Methods

### 2.1 Identical pipeline for every model
For every random split, fitted on the **training patients only** (no test leakage):

```
30 features -> standardise -> PCA to 6 components -> standardise -> model
```

Six components are kept so that the quantum models use one qubit per feature (**6 qubits**).
Every classical model gets exactly the same 6 features.

### 2.2 Models

| Model | Type | Role |
|---|---|---|
| **RBF-kernel SVM** | classical kernel | **The comparable baseline.** Same SVM, same features, same tuning budget; only the kernel differs |
| Logistic regression | classical | Strong simple reference |
| Random forest (300 trees) | classical | Non-kernel reference |
| Hybrid RBF + linear SVM | classical kernel mix | Control for the hybrid below: same "mix in a second kernel" flexibility, with no quantum part |
| **Quantum kernel SVM** | quantum | SVM on the fidelity kernel \|⟨ψ(x)\|ψ(x′)⟩\|² of Qiskit's ZZ feature map |
| **Projected quantum kernel SVM** | quantum | SVM on an RBF kernel over each qubit's measured Bloch vector (Huang et al., 2021), designed to avoid kernel concentration |
| **Hybrid RBF + quantum SVM** | quantum-enhanced classical | (1−w)·K_RBF + w·K_quantum, with w chosen by cross-validation (w = 0 recovers the classical model) |

**Quantum feature map.** `qiskit.circuit.library.zz_feature_map(feature_dimension=6, reps=2,
entanglement="linear")`. Each feature is scaled by a tuned **bandwidth** before encoding
(angle = bandwidth × feature). Bandwidth tuning is known to be essential for quantum kernels
(Shaydulin & Wild, 2022); our first, untuned experiment confirms this (Section 4.5).

**Why a quantum kernel here?** Its stated purpose is to address the classical model's main
small-data limitation, the **inductive bias** of the kernel. The ZZ feature map embeds 6 features
into a 2⁶ = 64-dimensional Hilbert space with entangling, product-of-features terms that an RBF
kernel does not express directly. If that bias suits the data, it should show up as better accuracy
from fewer patients.

### 2.3 Fair tuning
Every kernel model chooses its hyper-parameters by **repeated stratified 3-fold cross-validation
inside the training set** (2 repeats), maximising balanced accuracy. Grids:

- C ∈ {0.1, 1, 10, 10², 10³, 10⁴, 10⁵} for all SVMs (and logistic regression)
- RBF width γ: 10 values from 3·10⁻⁵ to 1
- quantum bandwidth: 10 values from 0.001 to 0.8
- projected-kernel width: 7 values from 0.03 to 30
- hybrid weight w ∈ {0, 0.25, 0.5, 0.75, 1}

Ties go to the earlier setting, so the hybrids fall back to the purely classical kernel unless the
quantum part strictly helps.

### 2.4 Evaluation protocol
- **Two separate stages.** All design decisions (models, grids) were made on *development* splits
  (`--stage dev`, seed 11). The reported numbers come from a single run on **fresh splits**
  (`--stage final`, seed 2026) that were not looked at during development.
- **30 random stratified splits** per training size. Every patient not used for training is a test
  patient (313–553 per split).
- Metrics: **balanced accuracy** (primary), ROC AUC, sensitivity and specificity.
- **Paired** comparison on identical splits, using the Wilcoxon signed-rank test, mean difference
  with 95% CI, and win/loss counts.

### 2.5 Simulation, and how it was checked against Qiskit
Kernels are computed from exact statevectors. Because every gate after each Hadamard layer of the
ZZ map is diagonal, we simulate the circuit in closed form with NumPy (`qkernel.py`), which is about
50× faster than building a Qiskit circuit per patient. `fastsim_check.py` verifies the statevectors
against `qiskit.quantum_info.Statevector` (fidelity 1.000000000000), and the Bloch features against
Qiskit Pauli expectation values (error < 10⁻¹⁰).

---

## 3. Quantum resources

Gate counts come from Qiskit after transpiling to the basis {rz, sx, x, cx} at optimisation level 1:

| Resource | Value |
|---|---|
| Qubits | 6 |
| Feature-map repetitions | 2 |
| Feature map U(x): depth / CNOTs | 27 / 20 |
| Kernel circuit U(x′)†U(x) + measurement: depth / CNOTs / 1-qubit gates | 52 / 38 / 91 |
| Trainable quantum parameters | 0 (kernel method; only the SVM is trained classically) |

How many circuits each method needs, for n training and m test patients:

| | Circuits to train | Circuits to predict | Example: n = 128, m = 441 |
|---|---|---|---|
| Fidelity quantum kernel | n(n−1)/2 | m·n | 64,576 circuits |
| Projected quantum kernel | 3n (X, Y, Z bases) | 3m | 1,707 circuits |
| Classical RBF SVM | 0 | 0 | none (runs in milliseconds on a laptop) |

Each circuit also needs repeated **shots**. Section 4.4 shows how accuracy depends on the number of
shots and on approximate hardware noise.

---

## 4. Results

All numbers below come from the **final stage** (`results/final_results.json`): 30 fresh stratified
splits per training size, never used during development. The development run (10 splits, seed 11)
showed the same pattern.

### 4.1 Learning curves (classical ML metrics)

![Learning curves](figures/learning_curves.png)

Balanced accuracy on held-out patients, mean ± standard deviation over 30 splits (best in **bold**):

| Model | n = 16 | n = 32 | n = 64 | n = 128 | n = 256 |
|---|---|---|---|---|---|
| Logistic regression | 0.875 ± 0.055 | **0.922 ± 0.030** | **0.943 ± 0.023** | **0.954 ± 0.011** | **0.963 ± 0.013** |
| Random forest | **0.887 ± 0.032** | 0.905 ± 0.027 | 0.921 ± 0.026 | 0.929 ± 0.016 | 0.941 ± 0.015 |
| RBF SVM *(comparable baseline)* | 0.855 ± 0.055 | 0.910 ± 0.035 | 0.931 ± 0.024 | 0.945 ± 0.019 | 0.958 ± 0.012 |
| Quantum kernel SVM | 0.813 ± 0.080 | 0.895 ± 0.030 | 0.924 ± 0.029 | 0.947 ± 0.016 | 0.953 ± 0.011 |
| Projected quantum kernel SVM | 0.758 ± 0.112 | 0.881 ± 0.045 | 0.922 ± 0.030 | 0.942 ± 0.016 | 0.954 ± 0.013 |
| Hybrid RBF+linear SVM | 0.856 ± 0.052 | 0.911 ± 0.033 | 0.932 ± 0.025 | 0.947 ± 0.016 | 0.958 ± 0.013 |
| Hybrid RBF+quantum SVM | 0.847 ± 0.058 | 0.909 ± 0.034 | 0.930 ± 0.024 | 0.948 ± 0.016 | 0.957 ± 0.012 |

AUC, sensitivity (malignant cases caught) and specificity, mean over 30 splits:

| Model | n = 32: AUC | Sens. | Spec. | n = 128: AUC | Sens. | Spec. |
|---|---|---|---|---|---|---|
| Logistic regression | 0.981 | 0.880 | 0.964 | 0.992 | 0.931 | 0.978 |
| Random forest | 0.971 | 0.865 | 0.945 | 0.982 | 0.908 | 0.950 |
| RBF SVM | 0.974 | 0.862 | 0.959 | 0.988 | 0.918 | 0.973 |
| Quantum kernel SVM | 0.959 | 0.846 | 0.944 | 0.988 | 0.920 | 0.975 |
| Projected quantum kernel SVM | 0.952 | 0.827 | 0.935 | 0.985 | 0.912 | 0.973 |
| Hybrid RBF+linear SVM | 0.976 | 0.858 | 0.964 | 0.990 | 0.918 | 0.975 |
| Hybrid RBF+quantum SVM | 0.973 | 0.860 | 0.958 | 0.989 | 0.923 | 0.972 |

At 32 patients the quantum kernel misses about 1.6 % more malignant cases than the RBF SVM. At 128
patients its sensitivity is slightly higher (0.920 vs 0.918), but the difference is far inside the
noise. Results for every size are in `results/final_results.json`.

### 4.2 Paired comparison with the comparable classical baseline

![Paired differences](figures/paired_differences.png)

Same splits, model minus RBF SVM. Δ is the mean difference with its 95% CI; p is the Wilcoxon
signed-rank test; W/L counts the splits each model won or lost.

| Comparison | n | Δ balanced acc. | p | W/L | Δ AUC | p |
|---|---|---|---|---|---|---|
| Quantum kernel vs RBF | 16 | −0.042 ± 0.024 | 0.001 | 7/22 | −0.052 | <0.001 |
| | 32 | −0.015 ± 0.013 | 0.007 | 8/22 | −0.015 | <0.001 |
| | 64 | −0.007 ± 0.006 | 0.043 | 10/20 | −0.002 | 0.15 |
| | 128 | **+0.002 ± 0.007** | 0.95 | 13/15 | −0.001 | 0.10 |
| | 256 | −0.005 ± 0.003 | 0.001 | 6/20 | −0.002 | 0.005 |
| Projected quantum kernel vs RBF | 16 | −0.097 ± 0.042 | <0.001 | 8/22 | −0.120 | <0.001 |
| | 32 | −0.029 ± 0.015 | <0.001 | 7/23 | −0.022 | <0.001 |
| | 64 | −0.008 ± 0.008 | 0.063 | 10/20 | −0.005 | 0.025 |
| | 128 | −0.003 ± 0.006 | 0.13 | 12/18 | −0.003 | 0.007 |
| | 256 | −0.004 ± 0.003 | 0.032 | 11/17 | −0.002 | 0.013 |
| Hybrid RBF+quantum vs RBF | 16 | −0.008 ± 0.011 | 0.16 | 3/5 | −0.008 | 0.09 |
| | 32 | −0.001 ± 0.006 | 0.66 | 6/5 | −0.002 | 0.29 |
| | 64 | −0.001 ± 0.003 | 0.97 | 7/6 | +0.001 | 0.31 |
| | 128 | +0.002 ± 0.004 | 0.60 | 7/6 | +0.000 | 0.50 |
| | 256 | −0.001 ± 0.001 | 0.11 | 2/4 | −0.000 | 0.13 |
| Hybrid RBF+quantum vs Hybrid RBF+linear | 16–256 | between −0.009 and +0.001 | all ≥ 0.17 | — | — | — |

**Reading the table.**
- The quantum kernel **rivals** the comparable classical kernel only at 128 patients. Everywhere else
  it is worse, and most clearly in the small-data regime it was meant to help (−4.2 points at 16
  patients).
- The hybrid never differs significantly from the RBF SVM. Cross-validation gave the quantum part a
  non-zero weight in only 7–15 of the 30 splits per size. Mixing in a quantum kernel helped no more
  than mixing in a plain linear kernel (the control).
- Several splits have ties (W + L < 30), because both models made exactly the same predictions.

### 4.3 Why the quantum kernel behaves classically: bandwidth and concentration

For every split we recorded the tuned bandwidth and the off-diagonal entries of the quantum
training kernel (median over 30 splits):

| n | Tuned bandwidth (median) | Kernel value between two *different* patients | Spread of those values (SD) |
|---|---|---|---|
| 16 | 0.002 | 0.999 | 0.0010 |
| 32 | 0.002 | 0.999 | 0.0012 |
| 64 | 0.004 | 0.995 | 0.0040 |
| 128 | 0.002 | 0.999 | 0.0015 |
| 256 | 0.005 | 0.992 | 0.0081 |

Larger bandwidths (0.05–0.8) make the kernel values spread out, and the circuit then expresses its
entangling ZZ structure. Those settings were available, and cross-validation rejected them in almost
every split, because they **overfit** small cohorts. At bandwidth ≈ 0.002, every rotation angle is
tiny. The kernel becomes approximately 1 − bandwidth² × (a quadratic form in the features), which is
a smooth, low-order, essentially classical similarity. So the best quantum model is the one that is
least quantum. This matches the bandwidth and concentration results of Shaydulin & Wild (2022) and
Thanasilp et al. (2024).

### 4.4 Quantum resources: shots and noise

The exact results above assume perfect kernel values. On hardware every kernel entry (fidelity
kernel) or Bloch-vector component (projected kernel) is estimated from a finite number of shots
(`resource_study.py`, 15 splits per size, hyper-parameters tuned on exact kernels):

![Shots and noise](figures/shots_and_noise.png)

| n | Condition | Quantum kernel | Projected QK | RBF SVM (classical) | Circuits QK / PQK | Total shots QK / PQK |
|---|---|---|---|---|---|---|
| 32 | exact | 0.877 | 0.876 | 0.901 | 17,680 / 1,707 | — |
| 32 | 10,000 shots | 0.707 | 0.660 | 0.901 | 17,680 / 1,707 | 177 M / 17 M |
| 32 | 1,000 shots | 0.620 | 0.589 | 0.901 | | 17.7 M / 1.7 M |
| 32 | 100 shots | 0.582 | 0.542 | 0.901 | | 1.8 M / 0.17 M |
| 32 | 1,000 shots + noise | 0.604 | 0.560 | 0.901 | | 17.7 M / 1.7 M |
| 128 | exact | **0.949** | **0.947** | 0.947 | 64,576 / 1,707 | — |
| 128 | 10,000 shots | 0.759 | 0.677 | 0.947 | 64,576 / 1,707 | 646 M / 17 M |
| 128 | 1,000 shots | 0.682 | 0.574 | 0.947 | | 64.6 M / 1.7 M |
| 128 | 100 shots | 0.599 | 0.509 | 0.947 | | 6.5 M / 0.17 M |
| 128 | 1,000 shots + noise | 0.548 | 0.557 | 0.947 | | 64.6 M / 1.7 M |

"Noise" is a global depolarising approximation with an error of 0.005 per CNOT (38 CNOTs per
fidelity circuit, 20 per projected-kernel circuit). It is not a full device model.

**Why shots hurt so much.** Two different patients differ in kernel value by only ~0.001–0.008
(Section 4.3). With S shots, an estimate near 0.99 has a statistical error of about 0.1/√S, which is
0.001 even at 10,000 shots, the same size as the signal. The projected kernel is hit even harder:
its Bloch components move by roughly the bandwidth (~0.002), while each one has an error of about
1/√S. The large C values chosen in tuning (10⁴–10⁵) then make the SVM fit this noise.

**Main resource finding.** The bandwidth that gives the best *exact-simulation* accuracy makes the
kernel nearly impossible to estimate on hardware. Benchmarks that use only exact statevectors, like
our Sections 4.1–4.2, therefore **overstate** how well quantum kernels would do on a real device.

### 4.5 The first, untuned experiment (archive)

Our first version (`archive/v1_qml_healthcare.py`) used a narrow grid. There the quantum bandwidth
always hit the grid edge (0.05) and C stayed at 100. In that version the quantum kernel lost to the
RBF SVM by **−0.124, −0.084, −0.049 and −0.029** balanced accuracy at 16, 32, 64 and 128 patients.
Widening the grids for *all* models (Section 2.3) closed most of that gap. This shows how easily a
poorly tuned quantum model, or a poorly tuned classical baseline, can produce a misleading
comparison.

---

## 5. Discussion

### 5.1 Did the quantum model offer a unique advantage?
**Not over the classical baseline.** The track asks for an advantage "without sacrificing other
aspects of performance". Our quantum kernels:
- **accuracy and AUC:** equal to or worse than the comparable classical kernel, and worse than
  logistic regression;
- **sensitivity:** no consistent gain in catching malignant cases;
- **cost:** need tens of thousands of circuits and up to hundreds of millions of shots, while the
  classical SVM trains in milliseconds;
- **robustness:** lose most of their accuracy under realistic shot counts.

**Quantum vs quantum:** the projected quantum kernel is clearly the better *quantum* design. It
reaches the same exact accuracy as the fidelity kernel at 128 patients with 38× fewer circuits, and
its circuit count grows as 3(n + m) instead of n²/2 + n·m. Any future hardware study of this problem
should start from it.

### 5.2 Trade-offs at a glance

| | RBF SVM / logistic regression | Fidelity quantum kernel | Projected quantum kernel |
|---|---|---|---|
| Best balanced accuracy (n = 128) | 0.945 / **0.954** | 0.947 | 0.942 |
| Small-data accuracy (n = 16) | 0.855 / 0.875 | 0.813 | 0.758 |
| Circuits (n = 128, all 569 patients) | 0 | 64,576 | 1,707 |
| Scaling of circuits with patients | — | quadratic | linear |
| Accuracy at 10,000 shots (n = 128) | unaffected | 0.759 | 0.677 |
| Interpretability | coefficients / support vectors | kernel only | kernel only |

### 5.3 Limitations
- **Simulation only.** All quantum results are exact or simulated-shot statevector computations. No
  real IBM device was used, and the noise model is a simple approximation.
- **One dataset, one feature map.** A low-dimensional, almost linearly separable dataset is a hard
  place for any non-linear model to beat logistic regression. Other datasets, feature maps or
  encodings might behave differently.
- **6 qubits, chosen via PCA.** That keeps the simulation exact, but it also limits how much
  structure the circuit can express.
- **Hyper-parameters tuned on exact kernels.** On hardware, tuning should use shot-estimated kernels.
  That would probably choose larger bandwidths, trading some exact accuracy for robustness. We did
  not test this.

### 5.4 What would be worth trying next
1. **Shot-aware tuning:** choose the bandwidth on kernels estimated with the planned shot budget.
2. **Trainable or data-adapted feature maps** (quantum kernel alignment), so that a larger bandwidth
   does not overfit.
3. **A dataset with known non-classical structure,** where a quantum inductive bias has a reason to
   help.
4. **A small run on real IBM hardware** with the projected kernel, which needs only 3 circuits per
   patient.

**Bottom line.** For small-cohort breast-cancer diagnosis from biopsy features, a carefully tuned
quantum kernel can *match* a comparable classical kernel in exact simulation at one cohort size. It
does not beat a good classical model, and it is much more expensive and fragile on realistic quantum
hardware. We think a reproducible, honestly reported null result, with an explanation of *why*,
is more useful to the community than a fragile advantage claim.

---

## 6. Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python fastsim_check.py           # verify the fast simulator against Qiskit (seconds)
python experiment.py --stage dev  # development splits  (~12 min)
python experiment.py --stage final  # reported results   (~45 min)
python resource_study.py          # shots / noise study  (~10 min)
python make_figures.py            # figures/ from results/
```

Tested with Python 3.14, Qiskit 2.6.1 and scikit-learn 1.9.1 (exact versions in `requirements.txt`).
No IBM Quantum account is needed; everything runs as an exact local simulation.

| File | Purpose |
|---|---|
| `qkernel.py` | ZZ feature map simulation, fidelity and projected kernels, Qiskit resource counts |
| `experiment.py` | Main comparison: all models, tuning, metrics and paired tests |
| `resource_study.py` | Finite-shot and noise robustness, circuit and shot counts |
| `fastsim_check.py` | Correctness check of the simulator against Qiskit |
| `make_figures.py` | Figures from the saved results |
| `results/` | Raw results (JSON), including every split |
| `archive/` | First (untuned) experiment, kept for transparency |

## 7. References
- W. N. Street, W. H. Wolberg, O. L. Mangasarian. *Nuclear feature extraction for breast tumor diagnosis.* 1993. UCI ML Repository: Breast Cancer Wisconsin (Diagnostic).
- V. Havlíček et al. *Supervised learning with quantum-enhanced feature spaces.* Nature 567, 209 (2019). (ZZ feature map / quantum kernel SVM)
- H.-Y. Huang et al. *Power of data in quantum machine learning.* Nature Communications 12, 2631 (2021). (projected quantum kernel)
- R. Shaydulin, S. M. Wild. *Importance of kernel bandwidth in quantum machine learning.* Physical Review A 106, 042407 (2022).
- S. Thanasilp et al. *Exponential concentration in quantum kernel methods.* Nature Communications 15, 5200 (2024).
- J. Bowles, S. Ahmed, M. Schuld. *Better than classical? The subtle art of benchmarking quantum machine learning models.* arXiv:2403.07059 (2024).
- Qiskit: https://github.com/Qiskit/qiskit · scikit-learn: https://scikit-learn.org

## Acknowledgement
Developed with the help of an AI assistant (Claude, by Anthropic) for code, analysis and write-up.
All results were produced by the code in this repository.
