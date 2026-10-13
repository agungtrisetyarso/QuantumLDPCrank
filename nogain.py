"""Exact evaluation of the gain over hashing of stabilizer inner codes.

Reference implementation for "Screening inner codes for concatenated quantum
communication: a proved no-gain region for stabilizer codes over Pauli
channels". Written directly from the definitions in the paper; no other module
is required beyond numpy and scipy.

Conventions
-----------
* An n-qubit Pauli operator modulo phases is an integer bitmask of 2n bits:
  bit i is the x-part on qubit i, bit n+i is the z-part on qubit i
  (coordinates x_1..x_n, z_1..z_n in the paper, 0-indexed here).
* Symplectic form <u,v> = u^x.v^z + u^z.v^x (mod 2).
* Noise is i.i.d. with single-qubit distribution q = (qI, qX, qZ, qY).
* All entropies are in bits.

The coset c = e + S is represented by a canonical representative obtained by
clearing the pivot coordinates of the reduced row-echelon form of S, so no
symplectic (pure-error / logical) basis is needed: H(c) = H(sigma, ell) by
Lemma 1, hence H(ell | sigma) = H(c) - H(sigma).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.optimize import brentq

LOG2 = np.log2

# --------------------------------------------------------------------------
# elementary entropy helpers
# --------------------------------------------------------------------------

def h2(x: float) -> float:
    """Binary entropy in bits, h(0) = h(1) = 0."""
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return float(-x * LOG2(x) - (1 - x) * LOG2(1 - x))


def shannon(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * LOG2(p)).sum())


def depolarizing(p: float):
    return (1 - p, p / 3, p / 3, p / 3)


def dephasing(p: float):
    return (1 - p, 0.0, p, 0.0)


def bitflip(p: float):
    return (1 - p, p, 0.0, 0.0)


def H1(q) -> float:
    return shannon(q)


# --------------------------------------------------------------------------
# Pauli strings and linear algebra over F_2
# --------------------------------------------------------------------------

def pauli_to_int(s: str) -> int:
    n = len(s)
    v = 0
    for i, ch in enumerate(s):
        if ch in "XY":
            v |= 1 << i
        if ch in "ZY":
            v |= 1 << (n + i)
        if ch not in "IXYZ":
            raise ValueError(f"bad Pauli character {ch!r}")
    return v


def int_to_pauli(v: int, n: int) -> str:
    out = []
    for i in range(n):
        x = (v >> i) & 1
        z = (v >> (n + i)) & 1
        out.append("IXZY"[x + 2 * z])
    return "".join(out)


def symp(u: int, v: int, n: int) -> int:
    mask = (1 << n) - 1
    ux, uz = u & mask, u >> n
    vx, vz = v & mask, v >> n
    return (bin(ux & vz).count("1") + bin(uz & vx).count("1")) & 1


def swap_xz(v: int, n: int) -> int:
    mask = (1 << n) - 1
    return ((v & mask) << n) | (v >> n)


def weight(v: int, n: int) -> int:
    mask = (1 << n) - 1
    return bin((v & mask) | (v >> n)).count("1")


def rref(rows, ncols: int, col_order=None):
    """Reduced row echelon form over F_2.

    rows: iterable of ints (bit j = column j). col_order: order in which
    columns are tried as pivots (default 0..ncols-1). Returns (rows, pivots)
    with rows[i] having pivot column pivots[i] and zeros in every other pivot
    column.
    """
    rows = [int(r) for r in rows]
    order = list(range(ncols)) if col_order is None else list(col_order)
    piv_rows, pivots = [], []
    for c in order:
        bit = 1 << c
        sel = None
        for i, r in enumerate(rows):
            if r & bit:
                sel = i
                break
        if sel is None:
            continue
        pr = rows.pop(sel)
        rows = [r ^ pr if r & bit else r for r in rows]
        piv_rows = [r ^ pr if r & bit else r for r in piv_rows]
        piv_rows.append(pr)
        pivots.append(c)
        rows = [r for r in rows if r]
    return piv_rows, pivots


def gf2_rank(rows, ncols: int) -> int:
    return len(rref(rows, ncols)[0])


# --------------------------------------------------------------------------
# stabilizer codes
# --------------------------------------------------------------------------

@dataclass
class Code:
    name: str
    gens: list  # listed generators as Pauli strings (may be dependent)
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.gens[0])
        assert all(len(g) == self.n for g in self.gens)
        v = [pauli_to_int(g) for g in self.gens]
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                if symp(v[a], v[b], self.n):
                    raise ValueError(f"{self.name}: generators {a},{b} anticommute")

    @cached_property
    def vecs(self):
        return [pauli_to_int(g) for g in self.gens]

    @cached_property
    def basis(self):
        rows, piv = rref(self.vecs, 2 * self.n)
        return rows

    @property
    def m(self) -> int:
        return len(self.basis)

    @property
    def k(self) -> int:
        return self.n - self.m

    @property
    def rate(self) -> float:
        return self.k / self.n

    @cached_property
    def rZ(self) -> int:
        """dim of the Z-type subgroup = m - rank(G^x)."""
        mask = (1 << self.n) - 1
        return self.m - gf2_rank([v & mask for v in self.basis], self.n)

    @cached_property
    def rX(self) -> int:
        mask = (1 << self.n) - 1
        return self.m - gf2_rank([v >> self.n for v in self.basis], self.n)

    @property
    def max_weight(self) -> int:
        return max(weight(v, self.n) for v in self.vecs)

    def info_set(self, col_order):
        _, piv = rref(self.vecs, 2 * self.n, col_order)
        assert len(piv) == self.m
        return piv

    def coord_name(self, c: int) -> str:
        return (f"x{c + 1}" if c < self.n else f"z{c - self.n + 1}")


# --------------------------------------------------------------------------
# exact enumeration
# --------------------------------------------------------------------------

def enumerate_errors(n: int, q):
    """All errors with nonzero probability, as (masks uint64, probs float64)."""
    paulis = [(0, 0, q[0]), (1, 0, q[1]), (0, 1, q[2]), (1, 1, q[3])]
    paulis = [t for t in paulis if t[2] > 0]
    masks = np.zeros(1, dtype=np.uint64)
    probs = np.ones(1)
    for i in range(n):
        ms, ps = [], []
        for x, z, w in paulis:
            bits = (x << i) | (z << (n + i))
            ms.append(masks | np.uint64(bits))
            ps.append(probs * w)
        masks = np.concatenate(ms)
        probs = np.concatenate(ps)
    return masks, probs


def parity(a: np.ndarray) -> np.ndarray:
    return (np.bitwise_count(a) & np.uint8(1)).astype(np.uint64)


def entropy_of_keys(keys: np.ndarray, probs: np.ndarray) -> float:
    _, inv = np.unique(keys, return_inverse=True)
    return shannon(np.bincount(inv, weights=probs))


@dataclass
class Evaluation:
    code: str
    n: int
    k: int
    m: int
    H1: float
    H_sigma: float
    H_c: float
    H_ell_given_sigma: float
    H_e_given_c: float
    R: float
    Icoh: float
    hashing: float
    Delta: float
    Pf_DML: float
    Pf_ND: float


class Exact:
    """Exact enumeration of one code under one i.i.d. Pauli distribution."""

    def __init__(self, code: Code, q):
        self.code, self.q = code, tuple(float(x) for x in q)
        n = code.n
        self.e, self.p = enumerate_errors(n, self.q)
        # syndrome w.r.t. an independent basis (any basis gives the same entropies)
        sig = np.zeros_like(self.e)
        for a, g in enumerate(code.basis):
            gs = np.uint64(swap_xz(g, n))
            sig |= parity(self.e & gs) << np.uint64(a)
        self.sigma = sig
        # canonical coset representative
        rows, piv = rref(code.vecs, 2 * n)
        rep = self.e.copy()
        for r, c in zip(rows, piv):
            hit = (rep >> np.uint64(c)) & np.uint64(1)
            rep ^= hit * np.uint64(r)
        self.coset = rep

    # ---- basic entropies ----
    @cached_property
    def H_sigma(self):
        return entropy_of_keys(self.sigma, self.p)

    @cached_property
    def H_c(self):
        return entropy_of_keys(self.coset, self.p)

    def H_eP(self, P):
        """H(e_P) by the product formula (12) of the paper."""
        n = self.code.n
        qI, qX, qZ, qY = self.q
        tot = 0.0
        for i in range(n):
            hasx, hasz = i in P, (n + i) in P
            if hasx and hasz:
                tot += H1(self.q)
            elif hasx:
                tot += h2(qX + qY)
            elif hasz:
                tot += h2(qZ + qY)
        return tot

    def eP_key(self, P):
        key = np.zeros_like(self.e)
        for j, c in enumerate(P):
            key |= ((self.e >> np.uint64(c)) & np.uint64(1)) << np.uint64(j)
        return key

    def H_eP_sigma(self, P):
        m = self.code.m
        key = self.eP_key(P) | (self.sigma << np.uint64(m))
        return entropy_of_keys(key, self.p)

    def H_eP_enum(self, P):
        return entropy_of_keys(self.eP_key(P), self.p)

    # ---- decoding ----
    @cached_property
    def Pf_DML(self):
        # max over cosets with a given syndrome of p(c), summed over syndromes
        cos, inv = np.unique(self.coset, return_inverse=True)
        pc = np.bincount(inv, weights=self.p)
        # syndrome of each coset (constant on cosets)
        first = np.zeros(len(cos), dtype=np.int64)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        sc = self.sigma[first]
        order = np.lexsort((-pc, sc))
        sc_sorted = sc[order]
        keep = np.ones(len(order), dtype=bool)
        keep[1:] = sc_sorted[1:] != sc_sorted[:-1]
        return 1.0 - float(pc[order][keep].sum())

    @cached_property
    def Pf_ND(self):
        # most likely single error per syndrome; success = its coset's mass
        cos, inv = np.unique(self.coset, return_inverse=True)
        pc = np.bincount(inv, weights=self.p)
        order = np.lexsort((-self.p, self.sigma))
        s_sorted = self.sigma[order]
        keep = np.ones(len(order), dtype=bool)
        keep[1:] = s_sorted[1:] != s_sorted[:-1]
        best = order[keep]
        return 1.0 - float(pc[inv[best]].sum())

    def evaluate(self) -> Evaluation:
        c = self.code
        Hq = H1(self.q)
        Hs, Hc = self.H_sigma, self.H_c
        Hl = Hc - Hs
        Hec = c.n * Hq - Hc
        R = c.m - Hs
        Ic = c.k - Hl
        hashing = c.n * (1 - Hq)
        return Evaluation(c.name, c.n, c.k, c.m, Hq, Hs, Hc, Hl, Hec, R, Ic,
                          hashing, Ic - hashing, self.Pf_DML, self.Pf_ND)


# --------------------------------------------------------------------------
# certificates
# --------------------------------------------------------------------------

def alpha(g: int, n: int, q, i: int) -> float:
    qI, qX, qZ, qY = q
    x = (g >> i) & 1
    z = (g >> (n + i)) & 1
    if x and not z:
        return qZ + qY
    if z and not x:
        return qX + qY
    if x and z:
        return qX + qZ
    return 0.0


def pi_check(g: int, n: int, q) -> float:
    prod = 1.0
    for i in range(n):
        prod *= 1 - 2 * alpha(g, n, q, i)
    return 0.5 * (1 - prod)


def greedy_basis(code: Code, q):
    """Matroid-greedy basis of minimal sum h(pi(g)) from the listed generators."""
    n = code.n
    scored = sorted(range(len(code.vecs)),
                    key=lambda a: (h2(pi_check(code.vecs[a], n, q)), a))
    chosen, rows = [], []
    for a in scored:
        if gf2_rank(rows + [code.vecs[a]], 2 * n) > len(rows):
            rows.append(code.vecs[a])
            chosen.append(a)
        if len(chosen) == code.m:
            break
    return chosen


def marginal_order(code: Code, q):
    """Columns in order of increasing marginal entropy, ties by index."""
    n = code.n
    qI, qX, qZ, qY = q
    ent = [h2(qX + qY)] * n + [h2(qZ + qY)] * n
    return sorted(range(2 * n), key=lambda c: (round(ent[c], 15), c))


def cap_value(code: Code, q, P, J) -> float:
    ex_HeP = Exact.H_eP(_Dummy(code, q), P)
    return ex_HeP - code.m + sum(h2(pi_check(code.vecs[a], code.n, q)) for a in J)


class _Dummy:
    """Light object so H_eP can be called without enumeration."""
    def __init__(self, code, q):
        self.code, self.q = code, tuple(q)


def H_eP_formula(code: Code, q, P) -> float:
    return Exact.H_eP(_Dummy(code, q), P)


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------

def first_root(f, lo=1e-6, hi=0.3, grid=600):
    """Smallest p in (lo, hi] at which f changes sign from <= 0 to > 0.

    Returns (root, ok) where ok says f <= 0 on every grid point below the root
    (so the level is the end of the interval [0, root] on which f <= 0).
    """
    xs = np.linspace(lo, hi, grid)
    vals = [f(x) for x in xs]
    for a in range(len(xs) - 1):
        if vals[a] <= 0 < vals[a + 1]:
            r = brentq(f, xs[a], xs[a + 1], xtol=1e-12)
            return r, all(v <= 0 for v in vals[: a + 1])
        if vals[a] > 0:
            return xs[a], False
    return None, True


# --------------------------------------------------------------------------
# Algorithm 1 (certified screening)
# --------------------------------------------------------------------------

def screen(code: Code, q, exact_T5=True):
    """Return (verdict, test) with verdict in {"no-gain", "inconclusive"}."""
    n, m, k = code.n, code.m, code.k
    qI, qX, qZ, qY = q
    Hq = H1(q)
    if k - n * (1 - Hq) <= 0:                                   # T1
        return "no-gain", "T1"
    w = code.max_weight
    al = max(alpha(g, n, q, i) for g in code.vecs for i in range(n))
    theta = max(h2(qX + qY), h2(qZ + qY))
    if al <= 0.5 and theta + h2(0.5 * (1 - (1 - 2 * al) ** w)) <= 1:  # T2
        return "no-gain", "T2"
    if (qX == 0 and qY == 0) or (qZ == 0 and qY == 0):            # T3
        return "no-gain", "T3"
    if -k - n * (1 - Hq) > 0:                                    # stopping rule
        return "inconclusive", "stop"
    P = code.info_set(marginal_order(code, q))
    J = greedy_basis(code, q)
    if cap_value(code, q, P, J) <= 0:                            # T4
        return "no-gain", "T4"
    if exact_T5 and Exact(code, q).H_eP_sigma(P) - m <= 0:       # T5
        return "no-gain", "T5"
    return "inconclusive", "-"


def H1_depol(p):
    return H1(depolarizing(p))


def p_rate(code: Code):
    rho = code.rate
    return brentq(lambda p: H1_depol(p) - (1 - rho), 1e-9, 0.75)


def p_star(w: int):
    def f(p):
        a = 2 * p / 3
        return h2(a) + h2(0.5 * (1 - (1 - 2 * a) ** w)) - 1
    return brentq(f, 1e-6, 0.5)


def p_for_H1(target):
    return brentq(lambda p: H1_depol(p) - target, 1e-12, 0.75)
