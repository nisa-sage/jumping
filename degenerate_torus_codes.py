r"""
Minimum distance of codes on degenerate tori in weighted projective planes
==========================================================================

Companion code for the paper "Minimum distance of codes on degenerate tori in
weighted projective planes". Theorem, proposition and example numbers refer to
the version of the paper dated 2026-11-03.

Setting
-------
For a prime power q and weights 1 <= w1 <= w2 put d_i = (q-1)/gcd(q-1, w_i).
The degenerate torus Y_Q in P(1, w1, w2) has d1*d2 points, and C_{alpha, Y_Q}
is the code obtained by evaluating the homogeneous polynomials of degree alpha
on Y_Q. By Proposition 2.4,

    d_min(C_{alpha, Y_Q}) = min (d1 - a1)(d2 - a2)  over  (a1, a2) in Red(alpha),

    Red(alpha) = {(a1, a2) : 0 <= a_i <= d_i - 1,  w1*a1 + w2*a2 <= alpha}.

All functions accept optional arguments d1, d2 (divisors of q - 1), which
replace the default orders; this covers the products of arbitrary cyclic
subgroups of F_q^* (Remark 2.5).

Main functions
--------------
    dmin(q, w1, w2, alpha)              minimum distance, following Theorem 1.1
    dmin_dual(q, w1, w2, alpha)         minimum distance of the dual code (Thm 5.2)
    dimension(q, w1, w2, alpha)         dimension |Red(alpha)| (Remark 2.1)
    isodual_degree(q, w1, w2)           degree of the isodual code (Cor 5.3(ii))
    jumping_points(q, w1, w2, alpha)    jumping points (Definition 4.2)
    dmin_brute_force(q, w1, w2, alpha)  minimum of L over all rows (for checking)
    generator_matrix(q, w1, w2, alpha)  generator matrix over GF(q) (Sage only)
    check_paper_examples()              checks the numerical examples of the paper
    check_lopez()                       compares Theorem 1.1 with [lopez, Thm 3.8]
    check_gcd_reduction()               checks Lemma 2.6 and Remark 2.7
    random_check(n)                     compares dmin with the brute force

Usage
-----
In SageMath:   load("degenerate_torus_codes.py")
In Python 3:   from degenerate_torus_codes import *
               (everything except generator_matrix works without Sage)

    sage: dmin(157, 3, 4, 50)["dmin"]
    1400
    sage: dmin_dual(11, 3, 4, 12)["dmin"]
    5
"""

# --------------------------------------------------------------------------
# Sage / plain Python compatibility
# --------------------------------------------------------------------------
try:
    from sage.all import ZZ, gcd, inverse_mod
    HAVE_SAGE = True
except ImportError:                      # plain Python 3.8+
    from math import gcd

    def ZZ(x):
        return int(x)

    def inverse_mod(a, m):
        return pow(int(a), -1, int(m))

    HAVE_SAGE = False


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def _parameters(q, w1, w2, d1=None, d2=None):
    """Return (q, w1, w2, d1, d2) with w1 <= w2 (exchanging the roles of x1 and
    x2 if necessary, as in Section 2). By default d_i = (q-1)/gcd(q-1, w_i)."""
    q, w1, w2 = ZZ(q), ZZ(w1), ZZ(w2)
    if q < 2 or w1 < 1 or w2 < 1:
        raise ValueError("need q >= 2 and positive weights")
    if d1 is None:
        d1 = (q - 1) // gcd(q - 1, w1)
    if d2 is None:
        d2 = (q - 1) // gcd(q - 1, w2)
    d1, d2 = ZZ(d1), ZZ(d2)
    if d1 < 1 or d2 < 1 or (q - 1) % d1 != 0 or (q - 1) % d2 != 0:
        raise ValueError("d1 and d2 must be positive divisors of q - 1")
    if w1 > w2:
        w1, w2, d1, d2 = w2, w1, d2, d1
    return q, w1, w2, d1, d2


def _row_max(d1, w1, w2, alpha, a2):
    """Largest a1 with (a1, a2) in Red(alpha), i.e. N_{a2} (Section 3.3)."""
    return min(d1 - 1, (alpha - w2 * a2) // w1)


def _top_row(d2, w2, alpha):
    """B = min(d2 - 1, floor(alpha / w2)) (Section 3.2)."""
    return min(d2 - 1, alpha // w2)


# --------------------------------------------------------------------------
# Dimension, brute force and jumping points
# --------------------------------------------------------------------------
def dimension(q, w1, w2, alpha, d1=None, d2=None):
    """Dimension K = |Red(alpha)| of C_{alpha, Y_Q} (Remark 2.1)."""
    q, w1, w2, d1, d2 = _parameters(q, w1, w2, d1, d2)
    alpha = ZZ(alpha)
    if alpha < 0:
        return ZZ(0)
    B = _top_row(d2, w2, alpha)
    return sum(_row_max(d1, w1, w2, alpha, a2) + 1 for a2 in range(B + 1))


def dmin_brute_force(q, w1, w2, alpha, d1=None, d2=None):
    """Minimum of L(a1, a2) = (d1 - a1)(d2 - a2) over Red(alpha), computed row
    by row with the largest admissible a1 on each row (equation (2))."""
    q, w1, w2, d1, d2 = _parameters(q, w1, w2, d1, d2)
    alpha = ZZ(alpha)
    B = _top_row(d2, w2, alpha)
    return min((d1 - _row_max(d1, w1, w2, alpha, a2)) * (d2 - a2)
               for a2 in range(B + 1))


def jumping_points(q, w1, w2, alpha, d1=None, d2=None):
    """Jumping points j in {1, ..., B}, i.e. the rows with step
    N_{j-1} - N_j = floor(w2/w1) + 1 (Definition 4.2), computed with the
    reduced weights of Lemma 2.6. Assumes alpha <= w1(d1 - 1)."""
    q, w1, w2, d1, d2 = _parameters(q, w1, w2, d1, d2)
    alpha = ZZ(alpha)
    if alpha > w1 * (d1 - 1):
        raise ValueError("jumping points are defined for alpha <= w1(d1-1)")
    g = gcd(w1, w2)
    u1, u2, al = w1 // g, w2 // g, alpha // g
    M = u2 // u1
    B = _top_row(d2, u2, al)
    N = [(al - u2 * a2) // u1 for a2 in range(B + 1)]
    return [j for j in range(1, B + 1) if N[j - 1] - N[j] == M + 1]


# --------------------------------------------------------------------------
# Minimum distance
# --------------------------------------------------------------------------
def dmin(q, w1, w2, alpha, d1=None, d2=None):
    """Minimum distance of C_{alpha, Y_Q}, following Theorem 1.1.

    * alpha >= w1(d1-1) + w2(d2-1): trivial code, d_min = 1 (Proposition 3.1).
    * w1(d1-1) <= alpha:            d_min = d2 - beta (Theorem 3.3).
    * otherwise: reduction to coprime weights (Lemma 2.6); if the reduced
      w1 equals 1 (i.e. w1 | w2), Proposition 3.10; else the candidates of
      Corollary 4.14 (natural slope, first and last period; Theorems 4.9 and
      4.13), with residues of jumping points from Lemma 4.4.

    Returns a dictionary; the key "dmin" contains the minimum distance and
    "minimizer" a point of Red(alpha) where it is attained.
    """
    q, w1, w2, d1, d2 = _parameters(q, w1, w2, d1, d2)
    alpha = ZZ(alpha)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    out = {"d1": d1, "d2": d2, "length": d1 * d2,
           "dimension": dimension(q, w1, w2, alpha, d1, d2)}

    # ---- large degrees (Section 3.1)
    low = w1 * (d1 - 1)
    top = low + w2 * (d2 - 1)
    if alpha >= top:
        out.update(regime="trivial code", dmin=ZZ(1),
                   minimizer=(d1 - 1, d2 - 1))
        return out
    if alpha >= low:
        beta = (alpha - low) // w2
        out.update(regime="large degree", beta=beta, dmin=d2 - beta,
                   minimizer=(d1 - 1, beta))
        return out

    # ---- small degrees: reduction to coprime weights (Lemma 2.6)
    g = gcd(w1, w2)
    u1, u2, al = w1 // g, w2 // g, alpha // g
    M = u2 // u1
    B = _top_row(d2, u2, al)
    out.update(M=M, B=B, reduced_weights=(u1, u2), reduced_degree=al)

    def N(a2):                       # alpha <= w1(d1-1): no cap needed, (3)
        return (al - u2 * a2) // u1

    def Ltilde(a2):
        return (d1 - N(a2)) * (d2 - a2)

    def finish(regime, candidates, **extra):
        values = {a2: Ltilde(a2) for a2 in sorted(set(candidates))}
        a2min = min(values, key=lambda y: values[y])
        out.update(regime=regime, candidates=sorted(values), values=values,
                   dmin=values[a2min], minimizer=(N(a2min), a2min), **extra)
        return out

    if B == 0:
        return finish("single row", [0])
    if u1 == 1:                      # w1 | w2: Proposition 3.10
        return finish("w1 divides w2", [0, B])

    # ---- Section 4: jumping points, natural slope and periodicity
    k2 = u2 - M * u1
    r = al % u1
    inv = inverse_mod(k2, u1)
    # residues c_t (in 1..u1) of the jumping points, Lemma 4.4
    jump_res = set()
    for t in range(1, k2 + 1):
        c = (inv * (r + t)) % u1
        jump_res.add(u1 if c == 0 else c)

    if k2 <= u1 - k2:
        # natural slope M: exceptional steps M+1, candidates = predecessors j-1
        slope = "M"
        cand_res = sorted(jump_res)                       # residues of j
        shift = 1                                         # candidate y = j - 1
    else:
        # natural slope M+1: exceptional steps M, candidates = the j in Flat
        slope = "M+1"
        cand_res = [c for c in range(1, u1 + 1) if c not in jump_res]
        shift = 0                                         # candidate y = j

    candidates = {0, B}
    for c in cand_res:
        if c > B:                          # no exceptional step of this class
            continue
        j_first = c                        # smallest j >= 1 in the class
        j_last = c + u1 * ((B - c) // u1)  # largest  j <= B in the class
        candidates.add(j_first - shift)    # first period
        candidates.add(j_last - shift)     # last period
    return finish("small degree (Section 4)", candidates, k2=k2,
                  natural_slope=slope, jump_residues=sorted(jump_res))


def dmin_dual(q, w1, w2, alpha, d1=None, d2=None):
    """Minimum distance of the dual code. By Theorem 5.2 the dual of
    C_alpha is monomially equivalent to C_{alpha_perp},
    alpha_perp = w1(d1-1) + w2(d2-1) - alpha - 1 (Corollary 5.3)."""
    q, w1, w2, d1, d2 = _parameters(q, w1, w2, d1, d2)
    alpha = ZZ(alpha)
    top = w1 * (d1 - 1) + w2 * (d2 - 1)
    if alpha >= top:
        return {"regime": "zero code", "length": d1 * d2, "dimension": ZZ(0),
                "dmin": None}
    out = dmin(q, w1, w2, top - alpha - 1, d1, d2)
    out["alpha_perp"] = top - alpha - 1
    return out


def isodual_degree(q, w1, w2, d1=None, d2=None):
    """Corollary 5.3(ii): if T = w1(d1-1) + w2(d2-1) is odd, the code of degree
    (T-1)/2 is isodual (monomially equivalent to its dual), of dimension
    d1*d2/2. Returns this degree, or None if T is even."""
    q, w1, w2, d1, d2 = _parameters(q, w1, w2, d1, d2)
    top = w1 * (d1 - 1) + w2 * (d2 - 1)
    return (top - 1) // 2 if top % 2 == 1 else None


# --------------------------------------------------------------------------
# Generator matrix (SageMath only)
# --------------------------------------------------------------------------
def generator_matrix(q, w1, w2, alpha, d1=None, d2=None):
    """Generator matrix of C_{alpha, Y_Q} over GF(q), with rows ev(x^a),
    a in Red(alpha) (Remark 2.1, equation (8)). Requires SageMath.

    Example (Sage):
        sage: G = generator_matrix(11, 3, 4, 12)
        sage: LinearCode(G).minimum_distance() == dmin(11, 3, 4, 12)["dmin"]
        True
    """
    if not HAVE_SAGE:
        raise RuntimeError("generator_matrix requires SageMath")
    from sage.all import GF, matrix
    q, w1, w2, d1, d2 = _parameters(q, w1, w2, d1, d2)
    alpha = ZZ(alpha)
    F = GF(q)
    eta = F.multiplicative_generator()
    eta1 = eta ** ((q - 1) // d1)        # generator of the subgroup of order d1
    eta2 = eta ** ((q - 1) // d2)
    points = [(eta1 ** k, eta2 ** l) for k in range(d1) for l in range(d2)]
    B = _top_row(d2, w2, alpha)
    rows = [[x1 ** a1 * x2 ** a2 for (x1, x2) in points]
            for a2 in range(B + 1)
            for a1 in range(_row_max(d1, w1, w2, alpha, a2) + 1)]
    return matrix(F, rows)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_paper_examples(verbose=True):
    """Check the numerical examples of the paper (version 2026-11-03)."""
    checks = [
        # (description, computed value, value in the paper)
        ("Example 3.5(1), alpha = 137", dmin(31, 4, 9, 137)["dmin"], 1),
        ("Example 3.5(2), alpha = 70", dmin(31, 4, 9, 70)["dmin"], 9),
        ("Example 3.7", dmin(31, 4, 9, 34)["dmin"], 70),
        ("Remark 4.8, jumping points of Example 3.7",
         jumping_points(31, 4, 9, 34), [3]),
        ("Example 3.13", dmin(11, 3, 4, 12)["dmin"], 20),
        ("Remark 3.11(2)", dmin(13, 2, 4, 5)["dmin"], 12),
        ("Remark 3.11(3)", dmin(13, 1, 2, 5, d1=12, d2=2)["dmin"], 9),
        ("Table 1, row 1", dmin(97, 3, 4, 25)["dmin"], 575),
        ("Table 1, row 1, candidates", dmin(97, 3, 4, 25)["candidates"],
         [0, 1, 4, 6]),
        ("Table 1, row 2", dmin(31, 8, 9, 34)["dmin"], 104),
        ("Table 1, row 2, candidates", dmin(31, 8, 9, 34)["candidates"],
         [0, 2, 3]),
        ("Table 1, row 3", dmin(157, 3, 4, 50)["dmin"], 1400),
        ("Table 1, row 3, candidates", dmin(157, 3, 4, 50)["candidates"],
         [0, 2, 11, 12]),
        ("Table 1, row 3, jumping points", jumping_points(157, 3, 4, 50),
         [3, 6, 9, 12]),
        ("Example 4.11",
         dmin(20003000029, 50000, 100001, 10000099999)["dmin"],
         500220022250145),
        ("Example 4.11, candidates",
         dmin(20003000029, 50000, 100001, 10000099999)["candidates"],
         [0, 49999, 99999]),
        ("Example 4.15", dmin(241, 5, 8, 83)["dmin"], 957),
        ("Example 4.15, candidates (Theorem 4.13)",
         dmin(241, 5, 8, 83)["candidates"], [0, 1, 4, 6, 9, 10]),
        ("Example 4.15, minimizer", dmin(241, 5, 8, 83)["minimizer"], (15, 1)),
        ("Example 4.15, jumping points", jumping_points(241, 5, 8, 83),
         [2, 3, 5, 7, 8, 10]),
        ("Remark 4.17(3)", dmin(17, 5, 9, 75)["dmin"], 16),
        ("Example 5.4, dimension", dimension(11, 3, 4, 12), 11),
        ("Example 5.4, dual dimension", dimension(11, 3, 4, 30), 39),
        ("Example 5.4, dual distance", dmin_dual(11, 3, 4, 12)["dmin"], 5),
        ("Example 5.4, isodual degree", isodual_degree(11, 3, 4), 21),
        ("Example 5.4, isodual code [50, 25, 9]",
         (dimension(11, 3, 4, 21), dmin(11, 3, 4, 21)["dmin"]), (25, 9)),
        ("Example 5.4, isodual minimizer", dmin(11, 3, 4, 21)["minimizer"],
         (1, 4)),
        ("Corollary 5.3(ii), no isodual degree for T even",
         isodual_degree(17, 5, 9), None),
    ]
    ok = True
    for name, got, expected in checks:
        good = (got == expected)
        ok = ok and good
        if verbose or not good:
            print("%-48s %s" % (name, "ok" if good else
                                "FAILED: got %s, expected %s" % (got, expected)))
    return ok


def check_lopez(qs=(7, 11, 13, 16, 17, 19, 25, 31, 37, 41), verbose=True):
    """Remark 2.5: for w1 = w2 = 1 the codes on H1 x H2 are the affine
    cartesian codes of Lopez, Renteria-Marquez and Villarreal, and
    Theorem 1.1 agrees with their Theorem 3.8 for n = 2. Compares the two
    formulas for all subgroup orders 2 <= d1, d2 dividing q - 1 and all degrees
    1 <= alpha <= (d1 - 1) + (d2 - 1) + 1."""
    def lopez_n2(d1, d2, d):
        a, b = sorted((d1, d2))
        if d >= (a - 1) + (b - 1):
            return 1
        if d <= a - 1:                   # k = 0, l = d
            return (a - d) * b
        return b - (d - (a - 1))         # k = 1, l = d - (a - 1)
    failures = []
    count = 0
    for q in qs:
        divs = [d for d in range(2, q) if (q - 1) % d == 0]
        for d1 in divs:
            for d2 in divs:
                for alpha in range(1, d1 + d2):
                    count += 1
                    if dmin(q, 1, 1, alpha, d1, d2)["dmin"] != \
                            lopez_n2(d1, d2, alpha):
                        failures.append((q, d1, d2, alpha))
    if verbose:
        print("check_lopez: %d cases, %d failures" % (count, len(failures)))
    return failures


def check_gcd_reduction(qs=(7, 11, 13, 16, 17, 25, 31), verbose=True):
    """Lemma 2.6 and Remark 2.7: dividing the weights by g = gcd(w1, w2) and
    replacing alpha by floor(alpha/g), with d1, d2 kept fixed, does not change
    Red(alpha), the integers floor(alpha/w1), B and N_{a2}, and preserves the
    regime conditions. Checked for all orders d1, d2 dividing q - 1
    (Remark 2.5), weights 1 <= w1 <= w2 <= 24 with gcd > 1, and all degrees."""
    def red(d1, d2, w1, w2, alpha):
        return {(a1, a2) for a2 in range(d2) for a1 in range(d1)
                if w1 * a1 + w2 * a2 <= alpha}
    failures = []
    count = 0
    for q in qs:
        divs = [d for d in range(1, q) if (q - 1) % d == 0]
        for d1 in divs:
            for d2 in divs:
                for w1 in range(1, 13):
                    for w2 in range(w1, 25):
                        g = gcd(w1, w2)
                        if g == 1:
                            continue
                        u1, u2 = w1 // g, w2 // g
                        for alpha in range(0, w1 * (d1 - 1) + w2 * (d2 - 1) + 2 * g + 2, g):
                            for al in (alpha, alpha + g - 1):
                                ap = al // g
                                count += 1
                                ok = red(d1, d2, w1, w2, al) == red(d1, d2, u1, u2, ap) \
                                    and al // w1 == ap // u1 \
                                    and min(d2 - 1, al // w2) == min(d2 - 1, ap // u2) \
                                    and all(min(d1 - 1, (al - w2 * a2) // w1) ==
                                            min(d1 - 1, (ap - u2 * a2) // u1)
                                            for a2 in range(d2)) \
                                    and (al > w1 * (d1 - 1) or ap <= u1 * (d1 - 1)) \
                                    and (al >= w1 * (d1 - 1) or ap < u1 * (d1 - 1))
                                if not ok:
                                    failures.append((q, d1, d2, w1, w2, al))
    # Remark 2.7: for q = 13 and (w1, w2) = (4, 6) the reduced problem uses
    # (d1, d2) = (3, 2), not the orders (6, 4) of the degenerate torus of P(1,2,3)
    if _parameters(13, 4, 6)[3:] != (3, 2) or _parameters(13, 2, 3)[3:] != (6, 4):
        failures.append("Remark 2.7 example")
    if verbose:
        print("check_gcd_reduction: %d cases, %d failures" % (count, len(failures)))
    return failures


def random_check(n=2000, seed=0, verbose=True):
    """Compare dmin with dmin_brute_force (and the duality relation
    |Red(alpha)| + |Red(alpha_perp)| = d1*d2) on n random parameter sets,
    including arbitrary orders d1, d2."""
    import random
    rng = random.Random(seed)
    qs = [7, 11, 13, 16, 17, 19, 25, 27, 29, 31, 37, 41, 49, 61, 64, 97, 101]
    failures = []
    for _ in range(n):
        q = rng.choice(qs)
        w1, w2 = rng.randint(1, 15), rng.randint(1, 40)
        if rng.random() < 0.5:
            divs = [d for d in range(1, q) if (q - 1) % d == 0]
            d1, d2 = rng.choice(divs), rng.choice(divs)
        else:
            d1 = d2 = None
        _, a, b, D1, D2 = _parameters(q, w1, w2, d1, d2)
        top = a * (D1 - 1) + b * (D2 - 1)
        alpha = rng.randint(0, top + 3)
        if dmin(q, w1, w2, alpha, d1, d2)["dmin"] != \
                dmin_brute_force(q, w1, w2, alpha, d1, d2):
            failures.append((q, w1, w2, alpha, d1, d2, "dmin"))
        if alpha < top and dimension(q, w1, w2, alpha, d1, d2) + \
                dimension(q, w1, w2, top - alpha - 1, d1, d2) != D1 * D2:
            failures.append((q, w1, w2, alpha, d1, d2, "duality"))
    if verbose:
        print("random_check: %d cases, %d failures" % (n, len(failures)))
    return failures


if __name__ == "__main__":
    all_ok = check_paper_examples()
    fails = random_check() + check_lopez() + check_gcd_reduction()
    print("all checks passed" if all_ok and not fails else "SOME CHECKS FAILED")
