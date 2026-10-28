"""Check that the fast NumPy simulation of the ZZ feature map matches Qiskit's Statevector exactly."""
import time

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

from qkernel import zz_statevectors, N_QUBITS, REPS

rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, size=(20, N_QUBITS))
fm = zz_feature_map(feature_dimension=N_QUBITS, reps=REPS, entanglement="linear")

t = time.time()
ref = np.array([Statevector(fm.assign_parameters(dict(zip(fm.parameters, x)))).data for x in X])
t_q = time.time() - t
t = time.time()
fast = zz_statevectors(X)
t_f = time.time() - t

# global phase can differ, so compare fidelities |<a|b>|^2
fid = np.abs(np.sum(ref.conj() * fast, axis=1)) ** 2
print(f"min fidelity vs Qiskit: {fid.min():.12f}  (1.0 = identical)")
print(f"Qiskit: {t_q * 1000 / len(X):.2f} ms/sample   NumPy: {t_f * 1000 / len(X):.4f} ms/sample")
assert fid.min() > 1 - 1e-10

# Bloch-vector features must equal Qiskit's single-qubit Pauli expectation values
from qiskit.quantum_info import SparsePauliOp
from qkernel import bloch_features
F = bloch_features(fast[:5])
for s in range(5):
    sv = Statevector(ref[s])
    for q in range(N_QUBITS):
        for k, P in enumerate("XYZ"):
            label = "".join(P if i == q else "I" for i in reversed(range(N_QUBITS)))
            ev = sv.expectation_value(SparsePauliOp(label)).real
            assert abs(ev - F[s, 3 * q + k]) < 1e-10, (s, q, P, ev, F[s, 3 * q + k])
print("Bloch features match Qiskit Pauli expectation values")
print("OK")
