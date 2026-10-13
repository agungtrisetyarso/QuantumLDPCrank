"""Codes used in the paper, as listed generators (Pauli strings)."""
import numpy as np

from nogain import Code


def rep_Z(n):
    return Code(f"repetition n={n}",
                ["I" * i + "ZZ" + "I" * (n - i - 2) for i in range(n - 1)])


def iceberg(w):
    return Code(f"iceberg [[{w},{w-2},2]]", ["X" * w, "Z" * w])


def padded(n, r):
    return Code(f"padded Z1..Z{r}", ["I" * i + "Z" + "I" * (n - i - 1) for i in range(r)])


def css_from(HX, HZ, name):
    n = HX.shape[1] if HX.size else HZ.shape[1]
    gens = ["".join("X" if b else "I" for b in row) for row in HX]
    gens += ["".join("Z" if b else "I" for b in row) for row in HZ]
    return Code(name, gens)


def hgp(H1, H2, name):
    r1, n1 = H1.shape
    r2, n2 = H2.shape
    HX = np.concatenate([np.kron(H1, np.eye(n2, dtype=int)),
                         np.kron(np.eye(r1, dtype=int), H2.T)], axis=1) % 2
    HZ = np.concatenate([np.kron(np.eye(n1, dtype=int), H2),
                         np.kron(H1.T, np.eye(r2, dtype=int))], axis=1) % 2
    return css_from(HX, HZ, name)


REP3_OPEN = np.array([[1, 1, 0], [0, 1, 1]])
REP3_CYCLIC = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])

FIVE = Code("[[5,1,3]]", ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
STEANE = Code("Steane [[7,1,3]]", ["XIXIXIX", "IXXIIXX", "IIIXXXX",
                                   "ZIZIZIZ", "IZZIIZZ", "IIIZZZZ"])
SHOR = Code("Shor [[9,1,3]]", ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII",
                               "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ",
                               "XXXXXXIII", "IIIXXXXXX"])
ROT_SURFACE = Code("rotated surface [[9,1,3]]",
                   ["IIXIIXIII", "IIIXIIXII", "ZZIIIIIII", "IIIIIIIZZ",
                    "XXIXXIIII", "IIIIXXIXX", "IZZIZZIII", "IIIZZIZZI"])
SURFACE13 = hgp(REP3_OPEN, REP3_OPEN, "surface [[13,1,3]] (HGP)")
TORIC18 = hgp(REP3_CYCLIC, REP3_CYCLIC, "toric [[18,2,3]] (HGP)")

ICEBERGS = [iceberg(w) for w in (4, 6, 8, 10)]
