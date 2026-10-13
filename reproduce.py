"""Reproduce every number in Tables 4-7 and A1 and the certificate supplement.

Run:  python reproduce.py
Writes results/*.csv, results/side_claims.json and supplement_certificates.json.

Rounding convention for the manuscript: a certified no-gain level ("no gain for
all p <= level") is reported rounded DOWN, so the printed value is always
covered by the proof; exact onsets of a gain (computed, not certified) are
rounded to nearest.
"""
import csv
import json
import math
import os
import random

import numpy as np

from codes import (FIVE, ICEBERGS, ROT_SURFACE, SHOR, STEANE, SURFACE13,
                   TORIC18, iceberg, padded, rep_Z)
from nogain import (Exact, H1, H1_depol, brentq, cap_value, dephasing,
                    depolarizing, greedy_basis, h2, marginal_order, p_for_H1,
                    p_rate, p_star, pi_check)

os.makedirs("results", exist_ok=True)


def floor3(x):
    return math.floor(x * 1000 + 1e-9) / 1000


def write_csv(name, header, rows):
    with open(os.path.join("results", name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def first_level(f, lo, hi, n_grid):
    """End of the interval [lo, level] on which f <= 0, located on a grid and
    refined by Brent's method. Also returns the grid used for the check."""
    xs = np.linspace(lo, hi, n_grid)
    prev = xs[0]
    if f(prev) > 0:
        raise RuntimeError("bound already positive at the left end")
    for x in xs[1:]:
        if f(x) > 0:
            return brentq(f, prev, x, xtol=1e-12), int(np.searchsorted(xs, x))
        prev = x
    return None, n_grid


Q10 = depolarizing(0.10)
SUPP_P = (0.05, 0.08, 0.10, 0.12, 0.15, 0.19)


def iceberg_info_set(c):
    """P minimizing H(e_P, sigma) at p = 0.10 over 40 information sets:
    the marginal-entropy order plus 39 random column orders (seed 0)."""
    rng = random.Random(0)
    orders = [marginal_order(c, Q10)]
    for _ in range(39):
        o = list(range(2 * c.n))
        rng.shuffle(o)
        orders.append(o)
    ex = Exact(c, Q10)
    # ties (equal entropy up to rounding) go to the earliest order, i.e. the
    # marginal-entropy order, so the choice is reproducible across platforms
    scored = [(round(ex.H_eP_sigma(c.info_set(o)), 10), i) for i, o in enumerate(orders)]
    return list(c.info_set(orders[min(scored)[1]]))


def certify(c, is_iceberg):
    P = iceberg_info_set(c) if is_iceberg else c.info_set(marginal_order(c, Q10))
    J = greedy_basis(c, Q10)
    cap = lambda p: cap_value(c, depolarizing(p), P, J)
    joint = lambda p: Exact(c, depolarizing(p)).H_eP_sigma(P) - c.m
    delta = lambda p: Exact(c, depolarizing(p)).evaluate().Delta
    pr = p_rate(c)
    pc, _ = first_level(cap, 1e-6, 0.3, 301)
    pj, gj = first_level(joint, 1e-6, 0.3, 301)
    po = brentq(delta, 0.18, 0.2, xtol=1e-12)

    def I(p):
        ex = Exact(c, depolarizing(p))
        return ex.H_eP_sigma(P) - c.m - ex.evaluate().Delta

    rows = []
    for p in SUPP_P:
        q = depolarizing(p)
        ex = Exact(c, q)
        rows.append({
            "p": p,
            "H_eP": ex.H_eP(P),
            "sum_h_pi": sum(h2(pi_check(c.vecs[a], c.n, q)) for a in J),
            "cap": cap(p),
            "joint_bound": ex.H_eP_sigma(P) - c.m,
            "rate_bound": c.k - c.n * (1 - H1(q)),
            "Delta_exact": ex.evaluate().Delta,
        })
    return {
        "code": c.name, "n": c.n, "k": c.k, "m": c.m, "rate": c.rate,
        "max_check_weight": c.max_weight,
        "generators": c.gens,
        "information_set_P": [c.coord_name(x) for x in P],
        "check_basis_J": [c.gens[a] for a in J],
        "choices_made_at_p": 0.10,
        "levels": {"p_rate": pr, "p_star": p_star(c.max_weight), "p_cap": pc,
                   "p_joint": pj, "p_on": po,
                   "I_at_p_joint": I(pj), "I_at_p_on": I(po), "max_I_2k": 2 * c.k},
        "levels_reported_floor3": {"p_rate": floor3(pr), "p_star": floor3(p_star(c.max_weight)),
                                   "p_cap": floor3(pc), "p_joint": floor3(pj)},
        "joint_bound_nonpositive_on_grid_below_p_joint": gj,
        "values_at_p": rows,
    }


def main():
    # ------------------------------------------------------------ Table 4 (obstruction)
    t3 = [(w, p_star(w), floor3(p_star(w))) for w in (2, 3, 4, 5, 6, 8, 10, 12)]
    write_csv("table4_obstruction.csv", ["w", "p_star_exact", "p_star_floor3"], t3)
    rho_star = {w: 1 - H1_depol(p_star(w)) for w in (4, 6, 10)}

    # ------------------------------------------- Tables 5 and A1, JSON
    certs = []
    for c in ICEBERGS:
        certs.append(certify(c, True))
    for c in (rep_Z(3), rep_Z(5), FIVE, STEANE, SHOR, ROT_SURFACE):
        certs.append(certify(c, False))
    with open("supplement_certificates.json", "w") as f:
        json.dump({"noise": "depolarizing",
                   "rounding": "certified levels reported rounded down to 3 decimals",
                   "codes": certs}, f, indent=1)
    write_csv("table5_iceberg.csv",
              ["w", "rho", "p_rate", "p_star", "p_cap", "p_joint", "p_on", "I(p_joint)", "I(p_on)"],
              [[c["n"], round(c["rate"], 3), floor3(c["levels"]["p_rate"]), floor3(c["levels"]["p_star"]),
                floor3(c["levels"]["p_cap"]), floor3(c["levels"]["p_joint"]), round(c["levels"]["p_on"], 4),
                round(c["levels"]["I_at_p_joint"], 2), round(c["levels"]["I_at_p_on"], 2)]
               for c in certs[:4]])
    write_csv("tableA1_certificates.csv",
              ["code", "P", "J", "p_rate", "p_cap", "p_joint", "p_on"],
              [[c["code"], " ".join(c["information_set_P"]), " ".join(c["check_basis_J"]),
                floor3(c["levels"]["p_rate"]), floor3(c["levels"]["p_cap"]),
                floor3(c["levels"]["p_joint"]), round(c["levels"]["p_on"], 4)] for c in certs])

    # ------------------------------------------------------------ Table 6 (dephasing)
    q = dephasing(0.15)
    hp = h2(0.15)
    t5 = []
    for c in (rep_Z(9), iceberg(4), FIVE, STEANE, SHOR, ROT_SURFACE, SURFACE13, TORIC18, padded(9, 4)):
        ev = Exact(c, q).evaluate()
        P = c.info_set(marginal_order(c, q))
        t5.append([c.name, c.n, c.k, c.rZ] + [round(x, 3) for x in (
            ev.H_sigma + 0.0, ev.H_ell_given_sigma, ev.H_e_given_c, Exact(c, q).H_eP(P),
            ev.Icoh, (c.n - c.rZ) * (1 - hp), c.n * (1 - hp), ev.Pf_DML)])
    write_csv("table6_dephasing.csv",
              ["code", "n", "k", "rZ", "H(sigma)", "H(ell|sigma)", "H(e|c)", "H(e_P)", "Icoh",
               "penalty (n-rZ)(1-h)", "n(1-h)", "Pf_DML"], t5)

    # ------------------------------------------------------------ Table 7 (depolarizing)
    q = depolarizing(0.19)
    t6 = []
    for c in (rep_Z(3), rep_Z(5), rep_Z(9), iceberg(4), FIVE, STEANE, SHOR, ROT_SURFACE):
        ex = Exact(c, q)
        ev = ex.evaluate()
        # cap: J matroid-greedy from listed generators; P the information set of
        # least H(e_P), found by exhaustive search over all m-subsets of columns
        import itertools
        from nogain import gf2_rank, H_eP_formula
        best = None
        for P in itertools.combinations(range(2 * c.n), c.m):
            sub = [sum(((v >> col) & 1) << j for j, col in enumerate(P)) for v in c.basis]
            if gf2_rank(sub, c.m) == c.m:
                val = H_eP_formula(c, q, P)
                if best is None or val < best[0] - 1e-12:
                    best = (val, P)
        J = greedy_basis(c, q)
        cap = cap_value(c, q, best[1], J)
        t6.append([c.name, c.n, c.k] + [round(x, 4) for x in (
            ev.H_e_given_c, ev.R, ev.Delta, ev.Icoh, ev.hashing)] +
                  [round(cap, 3), round(ev.Pf_DML, 3), round(ev.Pf_ND, 3)])
    write_csv("table7_depolarizing.csv",
              ["code", "n", "k", "H(e|c)", "R", "Delta", "Icoh", "n(1-H1)", "cap", "Pf_DML", "Pf_ND"], t6)

    # ------------------------------------------------------ claims in the text
    onsets = {c.name: brentq(lambda p: Exact(c, depolarizing(p)).evaluate().Delta, 0.18, 0.2, xtol=1e-12)
              for c in (rep_Z(3), rep_Z(5), rep_Z(9), FIVE, STEANE, SHOR, ROT_SURFACE)}
    side = {
        "hashing_threshold_depolarizing": p_for_H1(1.0),
        "rho_star": rho_star,
        "shor_dephasing_penalty_beats_rate_bound_above_p": brentq(lambda p: h2(p) - 2 / 3, 0.01, 0.5),
        "bb_144_12_12_rate_bound_level": p_for_H1(1 - 12 / 144),
        "toric_L10_200_2_rate_bound_level": p_for_H1(1 - 2 / 200),
        "exact_onsets_small_codes": onsets,
        "max_onset_offset_from_hashing": max(abs(v - p_for_H1(1.0)) for v in onsets.values()),
        "iceberg_provable_gain_above_p (H1 > 1 + rho)": {w: p_for_H1(1 + (1 - 2 / w)) for w in (4, 6, 8, 10)},
    }
    with open(os.path.join("results", "side_claims.json"), "w") as f:
        json.dump(side, f, indent=1)
    print("wrote results/*.csv, results/side_claims.json, supplement_certificates.json")


if __name__ == "__main__":
    main()
