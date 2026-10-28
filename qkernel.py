"""
Quantum kernel utilities.

The feature map is Qiskit's ZZ feature map (linear entanglement). Because every gate after the
Hadamard layer is diagonal, each repetition is
    |psi> <- D(x) H^{(x)n} |psi>,
    D(x) = exp(i * [ sum_i 2 x_i b_i + sum_{(i,j)} 2 (pi - x_i)(pi - x_j) (b_i XOR b_j) ])
for computational-basis bitstring b. We simulate it exactly with NumPy (verified against
qiskit.quantum_info.Statevector in fastsim_check.py), which is ~100x faster than building a
circuit per sample. Circuits for resource counting are still built with Qiskit.
"""
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map

N_QUBITS = 6
REPS = 2
PAIRS = [(i, i + 1) for i in range(N_QUBITS - 1)]  # "linear" entanglement

_DIM = 2 ** N_QUBITS
# bits[k, i] = value of qubit i in basis state k (Qiskit little-endian ordering)
_BITS = ((np.arange(_DIM)[:, None] >> np.arange(N_QUBITS)[None, :]) & 1).astype(float)
_XOR = np.stack([np.logical_xor(_BITS[:, i], _BITS[:, j]) for i, j in PAIRS], axis=1).astype(float)
_H = np.array([[1.0]])
for _ in range(N_QUBITS):
    _H = np.kron(_H, np.array([[1, 1], [1, -1]]) / np.sqrt(2))


def zz_statevectors(X):
    """X: (n_samples, N_QUBITS) rotation angles -> (n_samples, 2**N_QUBITS) statevectors."""
    X = np.asarray(X, dtype=float)
    single = 2 * X @ _BITS.T                                    # (n, dim)
    pair_coef = np.stack([2 * (np.pi - X[:, i]) * (np.pi - X[:, j]) for i, j in PAIRS], axis=1)
    phase = np.exp(1j * (single + pair_coef @ _XOR.T))          # (n, dim)
    psi = np.zeros((len(X), _DIM), dtype=complex)
    psi[:, 0] = 1.0
    for _ in range(REPS):
        psi = phase * (psi @ _H.T)
    return psi


def fidelity_kernel(A, B):
    """K[i, j] = |<psi(a_i)|psi(b_j)>|^2 for statevector arrays A, B."""
    return np.abs(A.conj() @ B.T) ** 2


def bloch_features(S):
    """Projected quantum features: the Bloch vector (<X>, <Y>, <Z>) of every single-qubit reduced
    density matrix, concatenated -> (n_samples, 3 * N_QUBITS). On hardware these are local
    Pauli expectation values, measured directly instead of estimating state overlaps."""
    n = len(S)
    T = S.reshape((n,) + (2,) * N_QUBITS)
    feats = []
    for q in range(N_QUBITS):
        axis = N_QUBITS - q  # reshape axis 1 is the most significant bit (qubit N_QUBITS-1)
        t = np.moveaxis(T, axis, 1).reshape(n, 2, -1)
        rho = t @ t.conj().transpose(0, 2, 1)                   # (n, 2, 2) reduced density matrix
        feats += [2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag, (rho[:, 0, 0] - rho[:, 1, 1]).real]
    return np.stack(feats, axis=1)


def projected_kernel(FA, FB, gamma):
    """Projected quantum kernel (Huang et al., Nat. Commun. 2021): exp(-gamma * ||F(x) - F(x')||^2)."""
    d = (FA ** 2).sum(1)[:, None] + (FB ** 2).sum(1)[None, :] - 2 * FA @ FB.T
    return np.exp(-gamma * np.maximum(d, 0))


def circuit_resources():
    """Resource counts for one kernel-entry circuit U(x)^dagger U(x') on a generic hardware basis."""
    fm = zz_feature_map(feature_dimension=N_QUBITS, reps=REPS, entanglement="linear")
    p = fm.parameters
    x1 = dict(zip(p, np.linspace(0.1, 0.6, N_QUBITS)))
    x2 = dict(zip(p, np.linspace(0.2, 0.7, N_QUBITS)))
    kernel_circ = fm.assign_parameters(x1).compose(fm.assign_parameters(x2).inverse())
    kernel_circ.measure_all()
    t = transpile(kernel_circ, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=0)
    ops = t.count_ops()
    fm_t = transpile(fm.assign_parameters(x1), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=0)
    return {
        "qubits": N_QUBITS,
        "feature_map_reps": REPS,
        "feature_map_depth": fm_t.depth(),
        "feature_map_cx": int(fm_t.count_ops().get("cx", 0)),
        "kernel_circuit_depth": t.depth(),
        "kernel_circuit_cx": int(ops.get("cx", 0)),
        "kernel_circuit_1q": int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))),
        "trainable_quantum_parameters": 0,
    }
