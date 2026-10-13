"""High-precision certification of every printed level and interval claim.

The paper's certified levels have the form "Delta <= 0 for all p <= p_hat",
where p_hat is the printed (rounded-down) value. By Lemma 3 (monotonicity),
each bound used -- the rate bound, the obstruction (Cor. 2), the cap (Thm 4,
P and J fixed) and the joint bound (Thm 5, P fixed) -- is non-decreasing in the
depolarizing rate p on [0, 3/4]. Hence it suffices to show that the bound is
<= 0 AT p_hat. This script evaluates every bound at the printed value with the
exact rational probabilities q = (1 - p_hat, p_hat/3, p_hat/3, p_hat/3) and
60-digit arithmetic (mpmath), propagating the distribution of the relevant
linear functionals of the error by exact XOR convolution over qubits.

It also brackets every printed onset of a gain: Delta < 0 at the lower end and
Delta > 0 at the upper end of the printed rounding interval, so a sign change
of the continuous function Delta lies inside it. Uniqueness of that sign change
is NOT claimed.

Run: python certify.py      Writes results/certification.json; exits non-zero
if any printed claim fails.
"""
import json
import os
import sys

import mpmath as mp

import codes as C
from crosscheck_symplectic import symplectic_basis
from nogain import Code, greedy_basis, marginal_order, depolarizing

mp.mp.dps = 60
os.makedirs("results", exist_ok=True)
CERTS = {c["code"]: c for c in json.load(open("supplement_certificates.json"))["codes"]}


def F(x):
    return mp.mpf(x) if not isinstance(x, str) else mp.mpf(x)


def h(x):
    x = mp.mpf(x)
    if x <= 0 or x >= 1:
        return mp.mpf(0)
    return -x * mp.log(x, 2) - (1 - x) * mp.log(1 - x, 2)


def qdep(p_str):
    p = mp.mpf(p_str)
    return (1 - p, p / 3, p / 3, p / 3)


def H1(q):
    return -sum(w * mp.log(w, 2) for w in q if w > 0)


def parity(x):
    return bin(x).count("1") & 1


def swap(v, n):
    m = (1 << n) - 1
    return ((v & m) << n) | (v >> n)


def functional_entropy(n, masks, q, keep_marginal=None):
    """Exact entropy of the vector (parity(mask_j & e))_j for i.i.d. Pauli e.

    masks act on the bitmask encoding of e (bit i = x_i, bit n+i = z_i).
    Returns H(all) and, if keep_marginal (a list of indices) is given, also the
    entropy of that sub-vector.
    """
    t = len(masks)
    dist = [mp.mpf(0)] * (1 << t)
    dist[0] = mp.mpf(1)
    paulis = [(0, 0, q[0]), (1, 0, q[1]), (0, 1, q[2]), (1, 1, q[3])]
    for i in range(n):
        vals = []
        for x, z, w in paulis:
            if w == 0:
                continue
            e = (x << i) | (z << (n + i))
            v = sum(parity(mk & e) << j for j, mk in enumerate(masks))
            vals.append((v, w))
        new = [mp.mpf(0)] * (1 << t)
        for s, ps in enumerate(dist):
            if ps == 0:
                continue
            for v, w in vals:
                new[s ^ v] += ps * w
        dist = new
    Hall = -sum(p * mp.log(p, 2) for p in dist if p > 0)
    if keep_marginal is None:
        return Hall
    marg = {}
    for s, ps in enumerate(dist):
        key = sum(((s >> j) & 1) << a for a, j in enumerate(keep_marginal))
        marg[key] = marg.get(key, mp.mpf(0)) + ps
    return Hall, -sum(p * mp.log(p, 2) for p in marg.values() if p > 0)


def coord_index(code, name):
    i = int(name[1:]) - 1
    return i if name[0] == "x" else code.n + i


def rate_bound(code, p_str):
    return code.k - code.n * (1 - H1(qdep(p_str)))


def obstruction(w, p_str):
    a = 2 * mp.mpf(p_str) / 3
    return h(a) + h((1 - (1 - 2 * a) ** w) / 2) - 1


def cap_bound(code, p_str, P, J):
    q = qdep(p_str)
    n = code.n
    HeP = mp.mpf(0)
    for i in range(n):
        hx, hz = i in P, (n + i) in P
        if hx and hz:
            HeP += H1(q)
        elif hx or hz:
            HeP += h(2 * mp.mpf(p_str) / 3)
    s = mp.mpf(0)
    for a in J:
        g = code.vecs[a]
        prod = mp.mpf(1)
        for i in range(n):
            if (g >> i) & 1 or (g >> (n + i)) & 1:
                prod *= 1 - 2 * (2 * mp.mpf(p_str) / 3)
        s += h((1 - prod) / 2)
    return HeP - code.m + s


def joint_bound(code, p_str, P):
    n = code.n
    masks = [1 << c for c in P] + [swap(g, n) for g in code.basis]
    return functional_entropy(n, masks, qdep(p_str)) - code.m


def delta(code, p_str):
    n, m, k, g, t, xs, zs = symplectic_basis(code.gens)
    masks = [swap(x, n) for x in g] + [swap(z, n) for z in zs] + [swap(x, n) for x in xs]
    q = qdep(p_str)
    Hsl, Hs = functional_entropy(n, masks, q, keep_marginal=list(range(m)))
    Ic = k - (Hsl - Hs)
    return Ic - n * (1 - H1(q))


results, failures = [], []


def record(claim, value, ok, kind):
    results.append({"claim": claim, "value": mp.nstr(value, 12), "holds": bool(ok), "type": kind})
    if not ok:
        failures.append(claim)
    print(f"{'OK  ' if ok else 'FAIL'} {kind:22s} {claim:62s} {mp.nstr(value, 6)}")


def main():
    # ---- Table 4: obstruction levels (rounded down)
    for w, s in [(2, "0.119"), (3, "0.096"), (4, "0.082"), (5, "0.072"), (6, "0.065"),
                 (8, "0.054"), (10, "0.047"), (12, "0.042")]:
        record(f"Table 4: p*({w}) >= {s}", obstruction(w, s), obstruction(w, s) <= 0, "certified level")
    # ---- Tables 5 and A1: rate, cap and joint levels as printed
    q10 = depolarizing(0.10)
    printed = {
        "iceberg [[4,2,2]]": ("0.074", "0.088", "0.111"), "iceberg [[6,4,2]]": ("0.044", "0.069", "0.083"),
        "iceberg [[8,6,2]]": ("0.031", "0.058", "0.068"), "iceberg [[10,8,2]]": ("0.023", "0.050", "0.058"),
        "repetition n=3": ("0.108", "0.119", "0.142"), "repetition n=5": ("0.138", "0.119", "0.159"),
        "[[5,1,3]]": ("0.138", "0.082", "0.145"), "Steane [[7,1,3]]": ("0.152", "0.088", "0.160"),
        "Shor [[9,1,3]]": ("0.160", "0.108", "0.165"), "rotated surface [[9,1,3]]": ("0.160", "0.103", "0.165"),
    }
    onsets = {"iceberg [[4,2,2]]": "0.1926", "iceberg [[6,4,2]]": "0.1929", "iceberg [[8,6,2]]": "0.1927",
              "iceberg [[10,8,2]]": "0.1922", "repetition n=3": "0.1889", "repetition n=5": "0.1891",
              "[[5,1,3]]": "0.1895", "Steane [[7,1,3]]": "0.1898", "Shor [[9,1,3]]": "0.1891",
              "rotated surface [[9,1,3]]": "0.1896"}
    for name, (pr, pc, pj) in printed.items():
        ce = CERTS[name]
        code = Code(name, ce["generators"])
        P = [coord_index(code, s) for s in ce["information_set_P"]]
        J = [code.gens.index(s) for s in ce["check_basis_J"]]
        assert J == greedy_basis(code, q10) or sorted(J) == sorted(greedy_basis(code, q10))
        v = rate_bound(code, pr)
        record(f"{name}: p_rate >= {pr}", v, v <= 0, "certified level")
        v = cap_bound(code, pc, P, J)
        record(f"{name}: p_cap >= {pc}", v, v <= 0, "certified level")
        v = joint_bound(code, pj, P)
        record(f"{name}: p_joint >= {pj}", v, v <= 0, "certified level")
        lo = str(mp.mpf(onsets[name]) - mp.mpf("0.00005"))
        hi = str(mp.mpf(onsets[name]) + mp.mpf("0.00005"))
        dlo, dhi = delta(code, lo), delta(code, hi)
        record(f"{name}: Delta changes sign in [{mp.nstr(mp.mpf(lo), 5)}, {mp.nstr(mp.mpf(hi), 5)}]",
               dhi, dlo < 0 < dhi, "onset bracket")
    # ---- Figure 5 caption: every curve changes sign in [0.1888, 0.1898]
    for code in (C.rep_Z(3), C.rep_Z(5), C.rep_Z(9), C.FIVE, C.STEANE, C.SHOR):
        dlo, dhi = delta(code, "0.1888"), delta(code, "0.1898")
        record(f"Fig. 5: {code.name} changes sign in [0.1888, 0.1898]", dhi, dlo < 0 < dhi, "onset bracket")
    # ---- Proposition 7: Delta > 0 for p >= p_+(w); printed upper bounds on p_+
    for w, s in [(4, "0.355"), (6, "0.430"), (8, "0.475"), (10, "0.505")]:
        v = H1(qdep(s)) - (2 - mp.mpf(2) / w)
        record(f"Prop. 7: H1({s}) > 2 - 2/{w}, so Delta > 0 for p >= {s}", v, v > 0, "certified level")
    # ---- claims in the text
    bb = 12 - 144 * (1 - H1(qdep("0.167")))
    record("Sec. 5: [[144,12,12]] rate bound excludes gain for p <= 0.167", bb, bb <= 0, "certified level")
    tor = 2 - 200 * (1 - H1(qdep("0.186")))
    record("Sec. 5: L=10 toric [[200,2]] rate bound excludes gain for p <= 0.186", tor, tor <= 0, "certified level")
    sh = h(mp.mpf("0.174")) - mp.mpf(2) / 3
    record("Sec. 5: dephasing penalty beats rate bound for Shor code at p > 0.174", sh, sh > 0, "certified level")
    for w, s in [(4, "0.459"), (6, "0.549"), (10, "0.648")]:
        exact = 1 - H1(qdep(str(mp.findroot(lambda p: obstruction(w, p), 0.07))))
        record(f"Thm. 6: rho*({w}) ~ {s} (to 3 decimals)", exact, abs(exact - mp.mpf(s)) < mp.mpf("0.0005"),
               "rounded value")
    with open(os.path.join("results", "certification.json"), "w") as f:
        json.dump({"precision_digits": mp.mp.dps, "claims": results, "failures": failures}, f, indent=1)
    print(f"\n{len(results)} claims checked, {len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
