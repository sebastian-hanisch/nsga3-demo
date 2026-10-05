"""Orakel-Tests (unabhängiger Rechenweg): Das-Dennis-Referenzpunkte gegen Aufzählung aller Kompositionen (Anzahl C(p+M-1, M-1)),
Extrempunkte per Schleife, Achsenabschnitte exakt mit Brüchen (Hyperebene durch die Extrempunkte), Zuordnung über
|x|² - (x·r)²/|r|², Nischenbildung gegen eine Schleifenfassung des Algorithmus aus Deb & Jain (identischer Zufallsstrom),
Abdeckungskennzahlen per Schleife und die Brute-Force-Front gegen eine Aufzählung (Regression: gespiegelte Touren erzeugten
Scheinpunkte)."""

import itertools
import math
from fractions import Fraction

import numpy as np

import nsga3_algorithm as A
import nsga3_evaluation as E


def _dom(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def test_das_dennis_points_match_enumeration():
    for m in (2, 3, 4):
        for p in (1, 2, 3, 5):
            pts = A.generate_reference_points(m, p)
            assert len(pts) == math.comb(p + m - 1, m - 1)
            ref = {tuple(Fraction(v, p) for v in c) for c in itertools.product(range(p + 1), repeat=m) if sum(c) == p}
            got = {tuple(Fraction(int(round(x * p)), p) for x in r) for r in pts}
            assert got == ref and len(got) == len(pts)


def _ref_extreme(t, m):
    out = []
    for j in range(m):
        w = [1e-6] * m
        w[j] = 1.0
        out.append(list(min(t, key=lambda row: max(row[k] / w[k] for k in range(m)))))
    return np.array(out)


def _exact_intercepts(ex):
    n = len(ex)
    mx = [[Fraction(float(v)) for v in row] + [Fraction(1)] for row in ex]
    for c in range(n):
        piv = next((r for r in range(c, n) if mx[r][c] != 0), None)
        if piv is None:
            return None
        mx[c], mx[piv] = mx[piv], mx[c]
        mx[c] = [v / mx[c][c] for v in mx[c]]
        for r in range(n):
            if r != c and mx[r][c] != 0:
                f = mx[r][c]
                mx[r] = [a - f * b for a, b in zip(mx[r], mx[c])]
    a = [mx[r][n] for r in range(n)]
    return None if any(v == 0 for v in a) else [1 / v for v in a]


def test_extreme_points_and_intercepts_match_exact_arithmetic():
    rng = np.random.default_rng(11)
    for _ in range(80):
        m, n = int(rng.integers(2, 6)), int(rng.integers(6, 25))
        f = rng.random((n, m)) * 100 + rng.random(m) * 50
        t = f - A.ideal_point(f)
        ex = A.extreme_points(t, m)
        assert np.allclose(ex, _ref_extreme(t.tolist(), m))
        ri = _exact_intercepts(ex)
        if ri is not None and all(float(v) > 1e-10 for v in ri) and np.linalg.cond(ex) < 1e8:
            inc = A.intercepts(ex, t, m)
            assert np.allclose(inc, [float(v) for v in ri], rtol=1e-6)
            assert all(abs(sum(ex[j, k] / inc[k] for k in range(m)) - 1) < 1e-6 for j in range(m))
    assert np.allclose(A.intercepts(np.array([[4., 0.], [0., 2.]]), np.array([[4., 0.], [0., 2.], [1., 1.]]), 2), [4., 2.])


def test_intercepts_fall_back_to_nadir_for_numerically_singular_extreme_points():
    """Regression: ein Individuum, das für zwei Achsen Extrempunkt ist, macht das System exakt singulär; np.linalg.solve
    wirft hier nicht immer, sondern liefert Rauschen (Achsenabschnitte 253,7 / 88,3 / 32,0) - Rückfall auf den Nadir fehlte."""
    ex = np.array([[41.7052445208726, 0.0, 26.740580926619252], [0.0, 88.25187180347044, 0.0], [41.7052445208726, 0.0, 26.740580926619252]])
    t = np.vstack([ex, [[10.0, 20.0, 5.0]]])
    assert np.allclose(A.intercepts(ex, t, 3), t.max(axis=0))


def test_associate_matches_perpendicular_distance_formula():
    rng = np.random.default_rng(12)
    for _ in range(60):
        m, n, p = int(rng.integers(2, 6)), int(rng.integers(1, 20)), int(rng.integers(1, 5))
        refp = A.generate_reference_points(m, p)
        x = rng.random((n, m)) * 1.5
        assoc, d = A.associate(x, refp)
        for i in range(n):
            ds = [math.sqrt(max(0.0, float(x[i] @ x[i] - (x[i] @ r) ** 2 / (r @ r)))) for r in refp]
            assert abs(min(ds) - d[i]) < 1e-7 and abs(ds[assoc[i]] - min(ds)) < 1e-7


def _deb_fronts(f):
    n = len(f)
    s = [[j for j in range(n) if j != i and _dom(f[i], f[j])] for i in range(n)]
    cnt = [sum(1 for j in range(n) if j != i and _dom(f[j], f[i])) for i in range(n)]
    fronts = [[i for i in range(n) if cnt[i] == 0]]
    while fronts[-1]:
        nxt = []
        for p in fronts[-1]:
            for q in s[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        fronts.append(nxt)
    fronts.pop()
    return fronts


def _ref_niching(fronts, assoc, dist, n_refs, size, rng):
    sel, rho, lv = [], [0] * n_refs, 0
    while lv < len(fronts) and len(sel) + len(fronts[lv]) <= size:
        for i in fronts[lv]:
            rho[assoc[i]] += 1
        sel += list(fronts[lv])
        lv += 1
        if len(sel) == size:
            return sel
    if lv >= len(fronts):
        return sel
    last = list(fronts[lv])
    for _ in range(size - len(sel)):
        zr = sorted({assoc[i] for i in last})
        mn = min(rho[z] for z in zr)
        best = [z for z in zr if rho[z] == mn]
        jbar = best[int(rng.integers(len(best)))] if len(best) > 1 else best[0]
        members = [i for i in last if assoc[i] == jbar]
        pick = min(members, key=lambda i: (dist[i], members.index(i))) if rho[jbar] == 0 else members[int(rng.integers(len(members)))]
        sel.append(pick)
        last.remove(pick)
        rho[jbar] += 1
    return sel


def test_niching_matches_paper_algorithm_with_identical_random_stream():
    rng = np.random.default_rng(13)
    for k in range(80):
        n_refs, n, size = int(rng.integers(2, 12)), int(rng.integers(6, 30)), 0
        size = int(rng.integers(2, n))
        f = rng.random((n, 3)) * 10 if k % 2 else rng.integers(0, 4, size=(n, 3)).astype(float)
        fronts = [np.array(x, dtype=np.int64) for x in _deb_fronts(f.tolist())]
        assoc, dist = rng.integers(0, n_refs, size=n), rng.random(n)
        got = A.niching_select(fronts, assoc, dist, n_refs, size, np.random.default_rng(k))
        exp = _ref_niching([x.tolist() for x in fronts], assoc.tolist(), dist.tolist(), n_refs, size, np.random.default_rng(k))
        assert got.tolist() == exp
        assert len(set(got.tolist())) == len(got) == size


def _cover(f, div, ideal, inc):
    refp = A.generate_reference_points(f.shape[1], div)
    hit = set()
    for x in (f - ideal) / np.where(inc < 1e-10, 1.0, inc):
        ds = [math.sqrt(max(0.0, float(x @ x - (x @ r) ** 2 / (r @ r)))) for r in refp]
        hit.add(int(np.argmin(ds)))
    return len(hit), len(refp)


def test_coverage_metrics_match_loop_versions():
    rng = np.random.default_rng(14)
    for _ in range(25):
        f = rng.random((int(rng.integers(4, 30)), 4)) * 100
        div = int(rng.integers(3, 6))
        ideal = f.min(axis=0)
        t = f - ideal
        inc = A.intercepts(_ref_extreme(t.tolist(), 4), t, 4)
        assert E.reference_coverage(f, div) == _cover(f, div, ideal, inc)
        fb = rng.random((12, 4)) * 100 + 20
        comb = np.vstack([f, fb])
        i2 = comb.min(axis=0)
        t2 = comb - i2
        inc2 = A.intercepts(_ref_extreme(t2.tolist(), 4), t2, 4)
        ca, cb, tot = E.shared_reference_coverage(f, fb, div)
        assert (ca, tot) == _cover(f, div, i2, inc2) and cb == _cover(fb, div, i2, inc2)[0]
        hist = [rng.random((int(rng.integers(1, 10)), 2)) for _ in range(int(rng.integers(1, 10)))]
        pop = int(rng.integers(1, 10))
        assert abs(E.saturation_share(hist, pop) - sum(len(h) >= pop for h in hist) / len(hist)) < 1e-12


def test_brute_force_front_matches_enumeration():
    """(n=4, Seed 11) und (n=6, Seed 6) sind die Regression: ohne Rundungs-Deduplikation hatte die Front 7 statt 5 bzw. 20 statt 15 Punkte."""
    for n, seed in ((4, 11), (6, 6)):
        st = E.Settings(n=n, seed=seed)
        fn, n_nodes = E.objective_fn(st)
        inst, d = E.instance(st.n, st.cluster_share, st.seed)
        facs = [None, inst.co2_factor_matrix, inst.time_factor_matrix, inst.risk_factor_matrix]

        def objs(t):
            e = [(t[i], t[(i + 1) % n_nodes]) for i in range(n_nodes)]
            return tuple(round(float(sum(d[a, b] * (1.0 if f is None else f[a, b]) for a, b in e)), 6) for f in facs)

        pts = {objs((0,) + p) for p in itertools.permutations(range(1, n_nodes))}
        ref = {p for p in pts if not any(_dom(q, p) for q in pts)}
        _, front = E.brute_force_front(n_nodes, fn)
        assert {tuple(np.round(r, 6)) for r in front} == ref and len(front) == len(ref)
