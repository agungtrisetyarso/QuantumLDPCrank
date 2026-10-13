"""Cross-check of Lemma 2, Icoh(S) = k - H(ell|sigma), against a direct
density-matrix evaluation of S(B) - S(RB) for the maximally mixed code input.

Run: python dm_check.py
"""
import itertools
import sys

import numpy as np

from codes import FIVE, iceberg, rep_Z, STEANE
from nogain import Code, Exact, bitflip, dephasing, depolarizing

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}


def op(s):
    out = np.array([[1.0]])
    for ch in s:
        out = np.kron(out, PAULI[ch])
    return out


def vn(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-14]
    return float(-(w * np.log2(w)).sum())


def icoh_dm(code: Code, q):
    n, k = code.n, code.k
    proj = np.eye(2 ** n, dtype=complex)
    for g in code.gens:
        proj = proj @ (np.eye(2 ** n) + op(g)) / 2
    w, v = np.linalg.eigh(proj)
    V = v[:, w > 0.5]                       # orthonormal code-space basis, 2^n x 2^k
    assert V.shape[1] == 2 ** k
    psi = V.reshape(2 ** n, 2 ** k).T.reshape(-1) / np.sqrt(2 ** k)  # |psi>_{RB}
    rho = np.zeros((2 ** (n + k), 2 ** (n + k)), dtype=complex)
    probs = dict(zip("IXZY", q))
    for s in itertools.product("IXZY", repeat=n):
        pe = np.prod([probs[c] for c in s])
        if pe == 0:
            continue
        U = np.kron(np.eye(2 ** k), op("".join(s)))
        phi = U @ psi
        rho += pe * np.outer(phi, phi.conj())
    rhoB = np.einsum("rbrc->bc", rho.reshape(2 ** k, 2 ** n, 2 ** k, 2 ** n))
    return vn(rhoB) - vn(rho)


worst = 0.0
cases = [(rep_Z(3), dephasing(0.1)), (rep_Z(3), bitflip(0.2)), (iceberg(4), depolarizing(0.15)),
         (FIVE, depolarizing(0.19)), (FIVE, (0.7, 0.1, 0.15, 0.05)),
         (Code("mixed", ["XZIY", "ZXZI"]), (0.6, 0.2, 0.1, 0.1))]
for code, q in cases:
    a = icoh_dm(code, q)
    b = Exact(code, q).evaluate().Icoh
    worst = max(worst, abs(a - b))
    print(f"{code.name:22s} q={tuple(round(x, 3) for x in q)}  density matrix {a:+.12f}  Lemma 2 {b:+.12f}")
print(f"largest difference {worst:.2e}")
sys.exit(0 if worst < 1e-9 else 1)
