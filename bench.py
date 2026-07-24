
import numpy as np
import time
import itertools
from collections import defaultdict
from deltac_exact import zcheck_matrix, h2, rank_gf2


def echelon(Hz, n):
    R = Hz.copy() % 2
    piv, r = [], 0
    for c in range(n):
        pv = None
        for i in range(r, R.shape[0]):
            if R[i, c]:
                pv = i
                break
        if pv is None:
            continue
        R[[r, pv]] = R[[pv, r]]
        for i in range(R.shape[0]):
            if i != r and R[i, c]:
                R[i] = (R[i] + R[r]) % 2
        piv.append(c)
        r += 1
        if r == R.shape[0]:
            break
    return R[:r], piv


class ScalableCosetDecoder:
    """Coset decoder whose table is built from low-weight errors only.

    Cost: O(sum_{w<=W} C(n,w)) instead of O(2^n). Exact whenever the
    truncation tail is negligible; we report the tail so the approximation is
    auditable.
    """

    def __init__(self, code, p, max_weight=None):
        self.code = code
        self.n = n = code.n
        self.p = p
        self.Hz = zcheck_matrix(code)
        self.R, self.piv = echelon(self.Hz, n)
        self.rank = self.R.shape[0]
        if max_weight is None:
            # enough weight to capture the bulk of the mass
            max_weight = min(n, max(2, int(np.ceil(3 * p * n)) + 2))
        self.W = max_weight

        t0 = time.perf_counter()
        table = defaultdict(lambda: defaultdict(float))
        mass = 0.0
        for w in range(self.W + 1):
            pw = (p ** w) * ((1 - p) ** (n - w))
            for pos in itertools.combinations(range(n), w):
                e = np.zeros(n, dtype=np.uint8)
                for q in pos:
                    e[q] = 1
                s = tuple((self.Hz @ e) % 2)
                v = e.copy()
                for i, c in enumerate(self.piv):
                    if v[c]:
                        v = (v + self.R[i]) % 2
                table[s][bytes(v)] += pw
                mass += pw
        self.table = {s: max(d.items(), key=lambda kv: kv[1])[0]
                      for s, d in table.items()}
        self.build_s = time.perf_counter() - t0
        self.tail = 1.0 - mass
        self.state_size = max(len(d) for d in table.values())
        self.table_entries = len(self.table)

    def decode(self, s):
        return self.table.get(tuple(s))


class MinWeightDecoder:
    """Minimum-weight (MWPM-style) decoder: lowest-weight consistent error.

    This is the decoding rule that PyMatching implements exactly for
    matchable codes and that BP-OSD approximates. We build it by low-weight
    enumeration so the comparison is like-for-like on cost accounting.
    """

    def __init__(self, code, p, max_weight=None):
        self.code = code
        self.n = n = code.n
        self.p = p
        self.Hz = zcheck_matrix(code)
        self.R, self.piv = echelon(self.Hz, n)
        if max_weight is None:
            max_weight = min(n, max(2, int(np.ceil(3 * p * n)) + 2))
        self.W = max_weight
        t0 = time.perf_counter()
        best = {}
        for w in range(self.W + 1):
            for pos in itertools.combinations(range(n), w):
                e = np.zeros(n, dtype=np.uint8)
                for q in pos:
                    e[q] = 1
                s = tuple((self.Hz @ e) % 2)
                if s not in best:
                    v = e.copy()
                    for i, c in enumerate(self.piv):
                        if v[c]:
                            v = (v + self.R[i]) % 2
                    best[s] = bytes(v)
        self.table = best
        self.build_s = time.perf_counter() - t0
        self.state_size = 1          # stores one correction per syndrome
        self.table_entries = len(best)

    def decode(self, s):
        return self.table.get(tuple(s))


def canon_of(code, e, R=None, piv=None):
    n = code.n
    Hz = zcheck_matrix(code)
    if R is None:
        R, piv = echelon(Hz, n)
    v = e.copy() % 2
    for i, c in enumerate(piv):
        if v[c]:
            v = (v + R[i]) % 2
    return bytes(v)


def benchmark(code, p, n_trials=2000, seed=0):
    """Compare ScalableCosetDecoder against MinWeightDecoder."""
    rng = np.random.default_rng(seed)
    n = code.n
    Hz = zcheck_matrix(code)
    R, piv = echelon(Hz, n)

    cd = ScalableCosetDecoder(code, p)
    mw = MinWeightDecoder(code, p)

    errs = (rng.random((n_trials, n)) < p).astype(np.uint8)
    syns = (errs @ Hz.T) % 2
    truth = [canon_of(code, e, R, piv) for e in errs]

    t0 = time.perf_counter()
    out_c = [cd.decode(s) for s in syns]
    t_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_m = [mw.decode(s) for s in syns]
    t_m = time.perf_counter() - t0

    acc_c = np.mean([a == b for a, b in zip(out_c, truth)])
    acc_m = np.mean([a == b for a, b in zip(out_m, truth)])
    agree = np.mean([a == b for a, b in zip(out_c, out_m)])

    return dict(n=n, rank=cd.rank, k=n - cd.rank, W=cd.W,
                tail=cd.tail,
                entries_c=cd.table_entries, entries_m=mw.table_entries,
                state_c=cd.state_size,
                build_c=cd.build_s, build_m=mw.build_s,
                t_c=t_c, t_m=t_m, n_trials=n_trials,
                acc_c=acc_c, acc_m=acc_m, agree=agree,
                err_space=2.0 ** n, coset_space=2.0 ** (n - cd.rank))
