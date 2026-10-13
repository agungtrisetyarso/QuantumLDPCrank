"""Randomized and structural validation of every identity and bound in the paper.

Run:  python validate.py            (prints a summary; exits non-zero on any violation)
Writes validation_report.json.
"""
import itertools
import json
import random
import sys

import numpy as np

from nogain import (Code, Exact, H1, alpha, entropy_of_keys, rref, bitflip, cap_value, dephasing,
                    depolarizing, gf2_rank, greedy_basis, h2, int_to_pauli,
                    marginal_order, pauli_to_int, pi_check, screen, swap_xz,
                    symp)

TOL = 1e-9
rng = random.Random(20261013)
np_rng = np.random.default_rng(20261013)
viol = {}
count = {}


def check(name, ok, info=""):
    count[name] = count.get(name, 0) + 1
    if not ok:
        viol.setdefault(name, []).append(info)


# ---------------------------------------------------------------- random codes
def random_stabilizer(n, m):
    vecs = []
    tries = 0
    while len(vecs) < m and tries < 20000:
        tries += 1
        v = rng.getrandbits(2 * n)
        if v == 0 or any(symp(v, u, n) for u in vecs):
            continue
        if gf2_rank(vecs + [v], 2 * n) == len(vecs) + 1:
            vecs.append(v)
    return [int_to_pauli(v, n) for v in vecs]


def random_css(n):
    while True:
        rz = rng.randint(1, n - 2)
        HZ = []
        for _ in range(rz):
            w = rng.randint(2, min(5, n))
            HZ.append(sorted(rng.sample(range(n), w)))
        HZm = np.zeros((rz, n), dtype=int)
        for i, s in enumerate(HZ):
            HZm[i, s] = 1
        # X checks: random vectors orthogonal to all Z checks
        ker = [v for v in range(1, 1 << n)
               if all(bin(v & sum(1 << j for j in s)).count("1") % 2 == 0 for s in HZ)]
        rx = rng.randint(0, max(0, n - 1 - rz))
        X = rng.sample(ker, min(rx, len(ker))) if ker else []
        gens = ["".join("Z" if j in s else "I" for j in range(n)) for s in HZ]
        gens += ["".join("X" if (v >> j) & 1 else "I" for j in range(n)) for v in X]
        try:
            c = Code("css", gens)
        except ValueError:
            continue
        if c.k >= 1:
            return c


def random_q():
    x = np_rng.dirichlet([1, 1, 1, 1])
    x[0] += 2.0          # bias towards low noise
    x /= x.sum()
    return tuple(float(v) for v in x)


codes = []
for _ in range(60):
    n = rng.randint(3, 9)
    m = rng.randint(1, n - 1)
    g = random_stabilizer(n, m)
    codes.append(Code(f"rand n={n}", g))
for _ in range(60):
    codes.append(random_css(rng.randint(3, 9)))

# a code is CSS iff its stabilizer splits into X-type and Z-type parts
noncss = sum(1 for c in codes if c.rX + c.rZ < c.m)

noises = []
for p in (0.05, 0.15, 0.30):
    noises.append(("depolarizing", p, depolarizing(p)))
    noises.append(("dephasing", p, dephasing(p)))
noises.append(("bitflip", 0.15, bitflip(0.15)))
noises.append(("boundary dephasing", 0.5, dephasing(0.5)))
noises.append(("boundary depolarizing", 0.75, depolarizing(0.75)))
noises.append(("noiseless", 0.0, (1.0, 0.0, 0.0, 0.0)))


def random_info_sets(c, q, k=4):
    orders = [marginal_order(c, q)]
    for _ in range(k - 1):
        o = list(range(2 * c.n))
        rng.shuffle(o)
        orders.append(o)
    return [c.info_set(o) for o in orders]


n_instances = 0
algo_counts = {}
for c in codes:
    for label, p, q in noises + [("random", None, random_q()) for _ in range(3)]:
        n_instances += 1
        n, m, k = c.n, c.m, c.k
        ex = Exact(c, q)
        ev = ex.evaluate()
        tag = f"{c.gens} {label} {q}"
        D = ev.Delta
        # rate bound and its converse
        check("rate bound Icoh<=k", ev.Icoh <= k + TOL, tag)
        check("Icoh>=-k", ev.Icoh >= -k - TOL, tag)
        check("R>=0", ev.R >= -TOL, tag)
        for P in random_info_sets(c, q):
            HeP = ex.H_eP(P)
            check("H(e_P) product formula", abs(HeP - ex.H_eP_enum(P)) < 1e-8, tag)
            # Theorem 2 identity H(e|c) = H(e_P|c)
            # joint entropy H(c, e_P); 2n + m <= 27 bits here, so the key fits in 64 bits
            H_c_eP = entropy_of_keys((ex.coset << np.uint64(m)) | ex.eP_key(P), ex.p)
            check("Thm2 identity H(e|c)=H(e_P|c)", abs(ev.H_e_given_c - (H_c_eP - ev.H_c)) < 1e-8, tag)
            check("Thm2 upper H(e|c)<=H(e_P)", ev.H_e_given_c <= HeP + TOL, tag)
            check("Thm2 lower", ev.H_e_given_c >= n * ev.H1 - (2 * n - m) - TOL, tag)
            # Theorem 5 joint identity, with I computed independently
            HePs = ex.H_eP_sigma(P)
            I = HePs + ev.H_c - H_c_eP - ev.H_sigma   # I(e_P; ell | sigma)
            check("Thm5 identity", abs(D - (HePs - m - I)) < 1e-8, tag)
            check("Thm5 0<=I<=2k", -TOL <= I <= 2 * k + TOL, tag)
            # Proposition 5
            check("Prop5 nH1-2k<=H(eP,s)<=nH1", n * ev.H1 - 2 * k - TOL <= HePs <= n * ev.H1 + TOL, tag)
            # Theorem 4 with the greedy basis and with all listed generators
            for J in (greedy_basis(c, q), list(range(len(c.vecs)))):
                if gf2_rank([c.vecs[a] for a in J], 2 * n) < m:
                    continue
                s = sum(h2(pi_check(c.vecs[a], n, q)) for a in J)
                check("Thm4 R>=m-sum h(pi)", ev.R >= m - s - TOL, tag)
                check("Thm4 cap", D <= HeP - m + s + TOL, tag)
        # piling-up formula vs direct syndrome-bit probability (asymmetric noise included)
        for g in c.vecs:
            gs = np.uint64(swap_xz(g, n))
            bit = (np.bitwise_count(ex.e & gs) & np.uint8(1)).astype(float)
            check("piling-up pi(g)", abs(float((bit * ex.p).sum()) - pi_check(g, n, q)) < 1e-12, tag)
        # Corollary 2
        al = max(alpha(g, n, q, i) for g in c.vecs for i in range(n))
        th = max(h2(q[1] + q[3]), h2(q[2] + q[3]))
        w = c.max_weight
        if al <= 0.5:
            check("Cor2 obstruction", D <= m * (th + h2(0.5 * (1 - (1 - 2 * al) ** w)) - 1) + TOL, tag)
        # Theorem 3 and Corollary 1 under dephasing / bit flip
        if q[1] == 0 and q[3] == 0:
            pz = q[2]
            check("Thm3 dephasing penalty", D <= -c.rZ * (1 - h2(pz)) + TOL, tag)
            check("Cor1 lower", ev.H_e_given_c >= n * h2(pz) - (n - c.rZ) - TOL, tag)
            check("Cor1 upper", ev.H_e_given_c <= c.rZ * h2(pz) + TOL, tag)
        if q[2] == 0 and q[3] == 0:
            check("Thm3 bit-flip penalty", D <= -c.rX * (1 - h2(q[1])) + TOL, tag)
        # Proposition 8 (decoding)
        Pf = ev.Pf_DML
        Hl = ev.H_ell_given_sigma
        fano = h2(Pf) + Pf * np.log2(4 ** k - 1) if k > 0 else 0.0
        check("Prop8 Fano", fano >= Hl - 1e-8, tag)
        check("Prop8 min-entropy", 1 - Pf >= 2 ** (-Hl) - 1e-10, tag)
        check("DML <= ND failure", ev.Pf_DML <= ev.Pf_ND + 1e-12, tag)
        # Algorithm 1 soundness
        verdict, test = screen(c, q)
        algo_counts[test] = algo_counts.get(test, 0) + 1
        if verdict == "no-gain":
            check("Algorithm 1 soundness", D <= TOL, f"{test} {tag} Delta={D}")


# ------------------------------------------------- structural checks (Prop. 3, 4)
def rep(n):
    return Code("rep", ["I" * i + "ZZ" + "I" * (n - i - 2) for i in range(n - 1)])


for n in range(3, 17):
    for p in (0.05, 0.1, 0.2, 0.3):
        ev = Exact(rep(n), dephasing(p)).evaluate()
        Podd = 0.5 * (1 - (1 - 2 * p) ** n)
        check("Prop3 closed form", abs(ev.H_e_given_c - (n * h2(p) - h2(Podd))) < 1e-10
              and abs(ev.Icoh - (1 - h2(Podd))) < 1e-10 and abs(ev.H_sigma) < 1e-10, f"n={n} p={p}")
        if n <= 12:
            evb = Exact(rep(n), bitflip(p)).evaluate()
            check("Prop3 bit flip H(e|c)=0", abs(evb.H_e_given_c) < 1e-10, f"n={n} p={p}")

for n in range(2, 8):
    for r in range(0, n + 1):
        for p in (0.0, 0.1, 0.3, 0.5):
            gens = ["I" * i + "Z" + "I" * (n - i - 1) for i in range(r)] or ["I" * n]
            if r == 0:
                continue
            c = Code("Snr", gens)
            q = dephasing(p)
            ev = Exact(c, q).evaluate()
            check("Prop4(1) equality", abs(ev.Icoh - (n - r) * (1 - h2(p))) < 1e-10, f"{n},{r},{p}")
            P = list(range(n, n + r))
            check("Prop4(2) equality", abs(ev.H_e_given_c - Exact(c, q).H_eP(P)) < 1e-10, f"{n},{r},{p}")
            J = list(range(r))
            check("Prop4(4) cap equality Z", abs(ev.Delta - cap_value(c, q, P, J)) < 1e-10, f"{n},{r},{p}")
            cx = Code("SnrX", ["I" * i + "X" + "I" * (n - i - 1) for i in range(r)])
            evx = Exact(cx, q).evaluate()
            check("Prop4(4) cap equality X", abs(evx.Delta - cap_value(cx, q, list(range(r)), J)) < 1e-10
                  and abs(evx.Delta + r * (1 - h2(p))) < 1e-10, f"{n},{r},{p}")
# Prop 4(3): disjoint supports give H(sigma) = sum h(pi)
for _ in range(40):
    n = rng.randint(4, 9)
    perm = list(range(n))
    rng.shuffle(perm)
    blocks, i = [], 0
    while i < n - 1:
        b = rng.randint(1, min(3, n - i))
        blocks.append(perm[i:i + b])
        i += b
    gens = []
    for b in blocks:
        s = ["I"] * n
        for j in b:
            s[j] = rng.choice("XYZ")
        gens.append("".join(s))
    c = Code("disjoint", gens)
    q = random_q()
    ex = Exact(c, q)
    s = sum(h2(pi_check(v, n, q)) for v in c.vecs)
    check("Prop4(3) disjoint equality", abs(ex.H_sigma - s) < 1e-10, str(gens))

# ---------------------------------- Theorem 3: equality holds IFF its two conditions
def z_subgroup_basis(c):
    n = c.n
    span = [0]
    for v in c.basis:
        span = span + [s ^ v for s in span]
    zs = [v for v in span if v and (v & ((1 << n) - 1)) == 0]
    return rref(zs, 2 * n)[0] if zs else []


eq_cases = {"equality": 0, "strict": 0}
thm3_codes = list(codes[:60]) + [Code("Zs", ["I" * i + "Z" + "I" * (6 - i - 1) for i in range(3)]),
                                 Code("Zs+X", ["ZIIIII", "IZIIII", "IIXXII", "IIZZII"]),
                                 Code("Zs+Y", ["ZIIII", "IIYYI", "IIZZI"])]
for c in thm3_codes:
    n = c.n
    Zb = z_subgroup_basis(c)
    PZ = rref(Zb, 2 * n, list(range(n, 2 * n)))[1] if Zb else []
    for p in (0.15, 0.5):
        q = dephasing(p)
        ex = Exact(c, q)
        ev = ex.evaluate()
        lhs_eq = abs(ev.Delta + c.rZ * (1 - h2(p))) < 1e-9
        cond1 = abs(ev.H_sigma - (c.m - c.rZ)) < 1e-9
        if PZ:
            HePc = entropy_of_keys((ex.coset << np.uint64(len(PZ))) | ex.eP_key(PZ), ex.p)
            I = ex.H_eP(PZ) + ev.H_c - HePc
        else:
            I = 0.0
        cond2 = abs(I) < 1e-9
        check("Thm3 equality iff conditions", lhs_eq == (cond1 and cond2), f"{c.gens} p={p}")
        eq_cases["equality" if lhs_eq else "strict"] += 1

# ------------------------- Lemma 3: entropies of linear functionals are monotone in p
mono_grid = np.linspace(0.0, 0.75, 31)
for c in codes[:40]:
    P = c.info_set(list(range(2 * c.n)))
    J = greedy_basis(c, depolarizing(0.1))
    vals_joint, vals_cap, vals_sig = [], [], []
    for p in mono_grid:
        q = depolarizing(p)
        ex = Exact(c, q)
        vals_joint.append(ex.H_eP_sigma(P))
        vals_sig.append(ex.H_sigma)
        vals_cap.append(cap_value(c, q, P, J))
    for name, vals in (("Lemma3 joint bound monotone", vals_joint), ("Lemma3 H(sigma) monotone", vals_sig),
                       ("Lemma3 cap monotone", vals_cap)):
        check(name, all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), str(c.gens))

report = {
    "codes": len(codes), "non_css": noncss, "instances": n_instances,
    "checks": count, "violations": {k: v[:3] for k, v in viol.items()},
    "algorithm1_exits": algo_counts, "thm3_equality_cases": eq_cases,
}
with open("validation_report.json", "w") as f:
    json.dump(report, f, indent=1)
print(f"{len(codes)} codes ({noncss} non-CSS), {n_instances} code-noise instances")
for name in sorted(count):
    print(f"  {name:38s} {count[name]:6d} checks  {len(viol.get(name, []))} violations")
print("Algorithm 1 exits:", algo_counts)
print("Theorem 3 equality / strict cases:", eq_cases)
sys.exit(1 if viol else 0)
