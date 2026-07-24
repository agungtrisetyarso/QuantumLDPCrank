
import numpy as np
import itertools
from collections import defaultdict
from deltac_core import shannon
from deltac_exact import h2, rank_gf2


def coset_entropy_linear(H, p, n=None):
    """Generic: entropy of the coset of a Bernoulli(p) source modulo the
    row space of a binary matrix H. This single function underlies all three
    settings below -- that is the point.
    """
    m, n_ = H.shape
    n = n_ if n is None else n
    # canonical reduction modulo rowspace
    R = H.copy() % 2
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
    R = R[:r]

    def canon(v):
        w = v.copy() % 2
        for i, c in enumerate(piv):
            if w[c]:
                w = (w + R[i]) % 2
        return tuple(w)

    cos = defaultdict(float)
    syn = defaultdict(float)
    for mask in range(1 << n):
        e = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)
        w = int(e.sum())
        pr = (p ** w) * ((1 - p) ** (n - w))
        cos[canon(e)] += pr
        syn[tuple((H @ e) % 2)] += pr
    Hc = shannon(np.array(list(cos.values())))
    Hs = shannon(np.array(list(syn.values())))
    return dict(H_source=n * h2(p), H_coset=Hc, H_syndrome=Hs,
                H_given_coset=n * h2(p) - Hc,
                H_given_syndrome=n * h2(p) - Hs,
                rank=r, n=n)


# ------------------------------------------------------- (1) Slepian-Wolf

def slepian_wolf_check(H, p):
    """Syndrome-based Slepian-Wolf: encoder sends s = H x, decoder knows the
    correlation. Achievable rate = rank(H)/n bits per symbol; the SW bound
    says H(X|Y) is the minimum. With X = Y + e, e ~ Bern(p), H(X|Y) = n h(p).

    The syndrome conveys H(s) bits; the residual H(x | s, y) = n h(p) - H(s)
    is exactly what our invariant computes. Ground truth: the decoder succeeds
    iff the residual vanishes on the typical set.
    """
    r = coset_entropy_linear(H, p)
    n = r['n']
    return dict(sw_bound=n * h2(p),            # H(X|Y), the SW rate
                syndrome_rate=r['rank'],        # bits actually sent
                residual=r['H_given_syndrome'],  # our invariant
                slack=r['rank'] - r['H_syndrome'])


# ------------------------------------------------------- (2) quantization

def quantization_check(H, p):
    """Binary linear quantizer: the coset label IS the quantized value.
    Distortion-side quantity = entropy discarded = H(e|c), our invariant.
    """
    r = coset_entropy_linear(H, p)
    return dict(rate=np.log2(2 ** (r['n'] - r['rank'])),   # coset space
                discarded=r['H_given_coset'],
                retained=r['H_coset'])


# ------------------------------------- (3) privacy amplification / hashing

def hashing_check(H, p):
    """Linear universal hash x -> Hx. The leftover hash lemma bounds the
    extractable randomness by H_min(X) - (n - rank). The entropy NOT captured
    by the hash is H(x | Hx), our invariant with the syndrome as the label.
    """
    r = coset_entropy_linear(H, p)
    n = r['n']
    Hmin = -n * np.log2(max(p, 1 - p))       # min-entropy of the source
    return dict(H_min=Hmin, digest_bits=r['rank'],
                leftover_bound=max(Hmin - (n - r['rank']), 0.0),
                unhashed=r['H_given_syndrome'])


def hamming74():
    return np.array([[1, 0, 1, 0, 1, 0, 1],
                     [0, 1, 1, 0, 0, 1, 1],
                     [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)


def random_H(m, n, seed):
    rng = np.random.default_rng(seed)
    return (rng.random((m, n)) < 0.5).astype(np.uint8)
