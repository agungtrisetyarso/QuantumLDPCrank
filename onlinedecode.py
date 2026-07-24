import numpy as np
import time
import itertools
from deltac_exact import zcheck_matrix


def rref_with_transform(M):
    """Return (R, pivots, T) with R = T M in reduced row echelon form."""
    m, n = M.shape
    A = np.concatenate([M.copy() % 2, np.eye(m, dtype=np.uint8)], axis=1)
    piv, r = [], 0
    for c in range(n):
        pv = None
        for i in range(r, m):
            if A[i, c]:
                pv = i
                break
        if pv is None:
            continue
        A[[r, pv]] = A[[pv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] = (A[i] + A[r]) % 2
        piv.append(c)
        r += 1
    return A[:, :n], piv, A[:, n:], r


def nullspace_basis(M, n):
    R, piv, _, r = rref_with_transform(M)
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, c in enumerate(piv):
            if R[i, f]:
                v[c] = 1
        basis.append(v)
    return basis


class OnlineCosetDecoder:
    """Coset decoder with O(n^2) storage and O(2^(n-r) * n) decode cost."""

    def __init__(self, code, p, verify=True):
        self.code = code
        self.n = n = code.n
        self.p = p
        self.Hz = zcheck_matrix(code)
        t0 = time.perf_counter()

        self.R, self.piv, self.T, self.rank = rref_with_transform(self.Hz)
        # basis of rowsp(Hz) is R (rank rows); basis of ker(Hz)
        self.ker = nullspace_basis(self.Hz, n)

        # quotient ker/(ker cap rowsp): build by Gaussian elimination of ker
        # vectors modulo rowsp
        self.quot = []
        span = [r.copy() for r in self.R]
        for v in self.ker:
            w = self._reduce(v, span)
            if w.any():
                span.append(w)
                self.quot.append(v.copy())
        self.state_size = 2 ** len(self.quot)
        self.expected = 2 ** (n - self.rank)
        # VALIDITY GUARD: the quotient ker/(ker cap rowsp) enumerates the
        # consistent cosets only when rowsp(Hz) is contained in ker(Hz), i.e.
        # Hz Hz^T = 0. Otherwise this construction is NOT correct and we say so
        # rather than returning wrong answers silently.
        self.self_orthogonal = not ((self.Hz @ self.Hz.T) % 2).any()
        self.ok = self.self_orthogonal and (self.state_size <= self.expected)
        self.setup_s = time.perf_counter() - t0
        # persistent storage in bits: echelon form + transform + quotient basis
        self.storage_bits = (self.R.size + self.T.size +
                             sum(v.size for v in self.quot))
        if verify and not self.ok:
            raise ValueError(
                "online construction requires Hz Hz^T = 0 (self-orthogonal "
                "Z-checks); this code does not satisfy it")
        # stabilizer elements for exact coset marginalisation
        self._stab = [np.zeros(n, dtype=np.uint8)]
        for row in self.R:
            self._stab = self._stab + [(v + row) % 2 for v in self._stab]

    @staticmethod
    def _reduce(v, span):
        w = v.copy() % 2
        for b in span:
            nz = np.nonzero(b)[0]
            if len(nz) and w[nz[0]]:
                w = (w + b) % 2
        return w

    def _canon(self, v):
        w = v.copy() % 2
        for i, c in enumerate(self.piv):
            if w[c]:
                w = (w + self.R[i]) % 2
        return bytes(w)

    def _particular(self, s):
        """Any e with Hz e = s. O(n^2), no table."""
        sv = (self.T @ (np.asarray(s, dtype=np.uint8) % 2)) % 2
        e = np.zeros(self.n, dtype=np.uint8)
        for i, c in enumerate(self.piv):
            if sv[i]:
                e[c] = 1
        if self.rank < len(sv) and sv[self.rank:].any():
            return None
        return e

    def decode(self, s):
        e0 = self._particular(s)
        if e0 is None:
            return None
        L = len(self.quot)
        best, bestp = None, -1.0
        # belief state: one score per element of the 2^L quotient
        for mask in range(1 << L):
            cand = e0.copy()
            for i in range(L):
                if (mask >> i) & 1:
                    cand = (cand + self.quot[i]) % 2
            # score = SUMMED mass over the stabilizer coset. Exact
            # marginalisation over rowsp(Hz); this is what makes the decoder
            # degenerate rather than minimum-weight.
            pr = 0.0
            for g in self._stab:
                v = (cand + g) % 2
                w = int(v.sum())
                pr += (self.p ** w) * ((1 - self.p) ** (self.n - w))
            if pr > bestp:
                bestp, best = pr, self._canon(cand)
        return best
