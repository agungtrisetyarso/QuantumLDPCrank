"""Second, independent implementation of the code-induced coherent information.

nogain.py identifies a coset by its canonical representative modulo a
row-echelon basis of S. This script instead builds the symplectic basis of
Section 2.2 explicitly -- generators g_a, pure errors t_a with <g_a,t_b> =
delta_ab, and logical pairs (xbar_j, zbar_j) -- reads off (sigma, ell) by the
symplectic products (6), and evaluates Lemma 2, Icoh = k - H(ell|sigma),
directly. It shares no code with nogain.py: enumeration, linear algebra and
entropies are re-implemented here. Only the generator lists are imported.

Run: python crosscheck_symplectic.py   (exits non-zero on any disagreement)
"""
import itertools
import math
import random
import sys
from collections import defaultdict

import codes as C


# ---------------------------------------------------------------- F_2 helpers
def to_vec(s):
    n = len(s)
    x = sum(1 << i for i, c in enumerate(s) if c in "XY")
    z = sum(1 << i for i, c in enumerate(s) if c in "ZY")
    return x | (z << n)


def sp(u, v, n):
    m = (1 << n) - 1
    return (bin((u & m) & (v >> n)).count("1") + bin((u >> n) & (v & m)).count("1")) & 1


def independent(vecs):
    basis = []                      # xor basis keyed by leading bit
    out = []
    for v in vecs:
        w = v
        for b in basis:
            w = min(w, w ^ b)
        if w:
            basis.append(w)
            basis.sort(reverse=True)
            out.append(v)
    return out


def solve(rows, rhs, nbits):
    """Find v (int over nbits) with parity(rows[i] & v) = rhs[i] for all i."""
    aug = [(r, b) for r, b in zip(rows, rhs)]
    piv = []
    for col in range(nbits):
        sel = next((i for i in range(len(piv), len(aug)) if (aug[i][0] >> col) & 1), None)
        if sel is None:
            continue
        r = len(piv)
        aug[r], aug[sel] = aug[sel], aug[r]
        for i in range(len(aug)):
            if i != r and (aug[i][0] >> col) & 1:
                aug[i] = (aug[i][0] ^ aug[r][0], aug[i][1] ^ aug[r][1])
        piv.append(col)
    for i in range(len(piv), len(aug)):
        assert aug[i][1] == 0, "inconsistent system"
    v = 0
    for i, col in enumerate(piv):
        if aug[i][1]:
            v |= 1 << col
    return v


def swap(v, n):
    m = (1 << n) - 1
    return ((v & m) << n) | (v >> n)


def symplectic_basis(gens_str):
    n = len(gens_str[0])
    g = independent([to_vec(s) for s in gens_str])
    m, k = len(g), n - len(g)
    rows = [swap(x, n) for x in g]                 # <g_a, v> = parity(swap(g_a) & v)
    t = [solve(rows, [int(a == b) for a in range(m)], 2 * n) for b in range(m)]
    for b in range(m):                             # make the t_a mutually commuting
        for a in range(b):
            if sp(t[a], t[b], n):
                t[b] ^= g[a]
    # W = vectors commuting with every g_a and t_a, dimension 2k
    cons = rows + [swap(x, n) for x in t]
    W = nullspace(cons, 2 * n)
    xs, zs = [], []
    while W:
        x = W.pop(0)
        j = next(i for i, w in enumerate(W) if sp(x, w, n))
        z = W.pop(j)
        W = [w ^ (sp(w, z, n) * x) ^ (sp(w, x, n) * z) for w in W]
        xs.append(x)
        zs.append(z)
    assert len(xs) == k
    return n, m, k, g, t, xs, zs


def nullspace(rows, nbits):
    rows = list(rows)
    piv_rows, piv_cols = [], []
    for col in range(nbits):
        sel = next((r for r in rows if (r >> col) & 1), None)
        if sel is None:
            continue
        rows.remove(sel)
        rows = [r ^ sel if (r >> col) & 1 else r for r in rows]
        piv_rows = [r ^ sel if (r >> col) & 1 else r for r in piv_rows]
        piv_rows.append(sel)
        piv_cols.append(col)
    free = [c for c in range(nbits) if c not in piv_cols]
    basis = []
    for f in free:
        v = 1 << f
        for r, c in zip(piv_rows, piv_cols):
            if (r >> f) & 1:
                v |= 1 << c
        basis.append(v)
    return basis


def check_basis(n, m, k, g, t, xs, zs):
    for a in range(m):
        for b in range(m):
            assert sp(g[a], t[b], n) == (a == b) and sp(t[a], t[b], n) == 0 and sp(g[a], g[b], n) == 0
        for j in range(k):
            assert sp(g[a], xs[j], n) == sp(g[a], zs[j], n) == sp(t[a], xs[j], n) == sp(t[a], zs[j], n) == 0
    for i in range(k):
        for j in range(k):
            assert sp(xs[i], zs[j], n) == (i == j) and sp(xs[i], xs[j], n) == 0 and sp(zs[i], zs[j], n) == 0


def random_codes(count, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        n = rng.randint(3, 7)
        m = rng.randint(1, n - 1)
        vecs = []
        for _ in range(5000):
            v = rng.getrandbits(2 * n)
            if v and all(sp(v, u, n) == 0 for u in vecs) and len(independent(vecs + [v])) == len(vecs) + 1:
                vecs.append(v)
            if len(vecs) == m:
                break
        if len(vecs) == m:
            out.append(["".join("IXZY"[((v >> i) & 1) + 2 * ((v >> (n + i)) & 1)] for i in range(n))
                        for v in vecs])
    return out


def H(dist):
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def evaluate(gens_str, q):
    n, m, k, g, t, xs, zs = symplectic_basis(gens_str)
    check_basis(n, m, k, g, t, xs, zs)
    paulis = [(0, 0, q[0]), (1, 0, q[1]), (0, 1, q[2]), (1, 1, q[3])]
    p_sl = defaultdict(float)
    p_s = defaultdict(float)
    for combo in itertools.product([P for P in paulis if P[2] > 0], repeat=n):
        e, pr = 0, 1.0
        for i, (x, z, w) in enumerate(combo):
            e |= (x << i) | (z << (n + i))
            pr *= w
        sig = tuple(sp(ga, e, n) for ga in g)
        ell = tuple(sp(z, e, n) for z in zs) + tuple(sp(x, e, n) for x in xs)
        p_sl[sig + ell] += pr
        p_s[sig] += pr
    H1 = -sum(w * math.log2(w) for w in q if w > 0)
    Hs, Hsl = H(p_s), H(p_sl)
    Ic = k - (Hsl - Hs)                    # Lemma 2
    return {"Icoh": Ic, "H_sigma": Hs, "H_e_given_c": n * H1 - Hsl,
            "Delta": Ic - n * (1 - H1), "k": k, "m": m}


def lemma1_check(gens_str, trials=300):
    """(sigma, ell) = 0 exactly on S: compare with membership in span(S)."""
    n, m, k, g, t, xs, zs = symplectic_basis(gens_str)
    span = {0}
    for v in g:
        span |= {s ^ v for s in span}
    rng = random.Random(1)
    for _ in range(trials):
        e = rng.choice(list(span)) if rng.random() < 0.5 else rng.getrandbits(2 * n)
        zero = all(sp(ga, e, n) == 0 for ga in g) and all(sp(z, e, n) == 0 for z in zs) \
            and all(sp(x, e, n) == 0 for x in xs)
        assert zero == (e in span)


if __name__ == "__main__":
    from nogain import Exact, depolarizing, dephasing   # only for the comparison
    worst = 0.0
    named = [C.rep_Z(3), C.rep_Z(5), C.rep_Z(9), C.iceberg(4), C.iceberg(6), C.FIVE, C.STEANE, C.SHOR,
             C.ROT_SURFACE, C.padded(9, 4)]
    noises = [depolarizing(0.19), dephasing(0.15), depolarizing(0.05), (0.7, 0.1, 0.15, 0.05)]
    rng = random.Random(7)
    count = 0
    for code in named:
        lemma1_check(code.gens)
        for q in noises:
            a = evaluate(code.gens, q)
            b = Exact(code, q).evaluate()
            for key, val in (("Icoh", b.Icoh), ("H_sigma", b.H_sigma), ("H_e_given_c", b.H_e_given_c),
                             ("Delta", b.Delta)):
                worst = max(worst, abs(a[key] - val))
            count += 1
    # random stabilizer codes (mostly non-CSS) under asymmetric noise
    from nogain import Code
    for gens in random_codes(40, seed=11):
        lemma1_check(gens)
        q = (0.7, 0.1, 0.12, 0.08)
        a = evaluate(gens, q)
        b = Exact(Code("r", gens), q).evaluate()
        worst = max(worst, abs(a["Icoh"] - b.Icoh), abs(a["H_e_given_c"] - b.H_e_given_c))
        count += 1
    print(f"{count} code-noise pairs, symplectic-basis route vs canonical-coset route: "
          f"largest difference {worst:.2e}")
    sys.exit(0 if worst < 1e-10 else 1)
