import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import e1_power as P  # noqa: E402


class TestE1Power(unittest.TestCase):
    def test_null_support_rate_below_alpha(self):
        # m_A = m_N = K0 rate: true contrast is 0, so support (needs est > 0.03 too) is at most about alpha
        r = P.simulate(24, 24, 0.02, 0.02, 0.02, 0.3, 400, 500, np.random.default_rng(1))
        self.assertLessEqual(r["support_alpha0.025"], 0.05)
        self.assertLessEqual(r["support_alpha0.0125"], r["support_alpha0.025"])

    def test_big_effect_is_detected(self):
        r = P.simulate(24, 24, 0.69, 0.0, 0.0, 0.3, 200, 500, np.random.default_rng(2))
        self.assertGreater(r["support_alpha0.025"], 0.8)

    def test_deterministic_given_seed(self):
        a = P.simulate(24, 24, 0.44, 0.05, 0.0, 0.7, 100, 200, np.random.default_rng(7))
        b = P.simulate(24, 24, 0.44, 0.05, 0.0, 0.7, 100, 200, np.random.default_rng(7))
        self.assertEqual(a, b)

    def test_rate_draw_mean_and_degenerate(self):
        rng = np.random.default_rng(3)
        self.assertAlmostEqual(P.draw_rates(rng, 0.3, 0.3, 200000).mean(), 0.3, places=2)
        self.assertEqual(P.draw_rates(rng, 0.0, 0.3, 5).sum(), 0)

    def test_run_writes_csv_from_seed_file(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            (out / "SEED").write_text("123")
            old = (P.M_A, P.M_N, P.K0_RATES, P.RHOS)
            P.M_A, P.M_N, P.K0_RATES, P.RHOS = (0.44,), (0.0,), (0.0,), (0.3,)
            try:
                p = P.run(out, 20, 50, (24,))
            finally:
                P.M_A, P.M_N, P.K0_RATES, P.RHOS = old
            self.assertEqual(len(p.read_text().strip().splitlines()), 2)

    def test_balanced_known_value(self):
        # A always acts at t3 (K2), never at t0; non-A never acts: contrast = 1 per A person, 0 for non-A.
        # balanced: est = 0.5 exactly; pi: est = 0.159. Both: p = 0.
        b = P.simulate(16, 16, 1.0, 0.0, 0.0, 0.3, 20, 100, np.random.default_rng(1), estimator="balanced")
        self.assertAlmostEqual(b["mean_est"], 0.5)
        self.assertEqual(b["support_alpha0.025"], 1.0)
        self.assertEqual(b["share_est_gt_0.05"], 1.0)
        p = P.simulate(16, 16, 1.0, 0.0, 0.0, 0.3, 20, 100, np.random.default_rng(1), estimator="pi")
        self.assertAlmostEqual(p["mean_est"], P.PI)

    def test_balanced_false_positive_cancels_action(self):
        # A and non-A both act at t3 always (m_A = m_N = 1) vs K0 0: dV = 1/2*1 - 1/2*1 = 0 -> never supported
        b = P.simulate(16, 16, 1.0, 1.0, 0.0, 0.3, 20, 100, np.random.default_rng(1), estimator="balanced")
        self.assertAlmostEqual(b["mean_est"], 0.0)  # 0.5*1 - 0.5*1
        self.assertEqual(b["support_alpha0.025"], 0.0)

    def test_balanced_output_keys(self):
        b = P.simulate(24, 24, 0.0, 0.0, 0.0, 0.3, 50, 100, np.random.default_rng(1), estimator="balanced")
        self.assertEqual(set(b), {"mean_est", "p50_p", "support_alpha0.025", "support_alpha0.0125",
                                  "share_est_gt_0.03", "share_est_gt_0.05"})

    def test_balanced_support_ignores_delta(self):
        # decision 22: an effect below delta_E1 = 0.03 still counts when p <= alpha. A acts at t3 with rate 0.04
        # (est about 0.02 < 0.03), 48 + 48 persons, no non-A actions: most simulations are supported although few
        # have est > 0.03, so support does not require est > delta
        b = P.simulate(48, 48, 0.04, 0.0, 0.0, 0.05, 200, 400, np.random.default_rng(5), estimator="balanced")
        self.assertLess(b["mean_est"], 0.03)
        self.assertGreater(b["support_alpha0.025"], 0.5)
        self.assertLess(b["share_est_gt_0.03"], b["support_alpha0.025"] - 0.3)

    def test_balanced_null_support_rate_below_alpha(self):
        # true contrast 0 under the balanced estimator: support about alpha at most
        b = P.simulate(24, 24, 0.07, 0.07, 0.02, 0.3, 400, 500, np.random.default_rng(11), estimator="balanced")
        self.assertLessEqual(b["support_alpha0.025"], 0.05)

    def test_balanced_p_never_zero(self):
        # p = (b + 1) / (B + 1): with every draw > 0 the smallest p is 1 / (B + 1), not 0 (decision 32)
        b = P.simulate(16, 16, 1.0, 0.0, 0.0, 0.3, 5, 99, np.random.default_rng(1), estimator="balanced")
        self.assertAlmostEqual(b["p50_p"], 1 / 100)

    def test_size_rows_in_the_balanced_grid(self):
        # C17 (判断 38): effect-zero scenarios m_A = m_N = K0 rate are rows of the balanced grid; their support is the size
        import csv
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            (out / "SEED").write_text("123")
            old = (P.BAL_M_A, P.BAL_M_N, P.K0_RATES, P.RHOS, P.SIZE_M)
            P.BAL_M_A, P.BAL_M_N, P.K0_RATES, P.RHOS, P.SIZE_M = (0.44,), (0.0,), (0.0,), (0.3,), (0.0, 0.07)
            try:
                p = P.run(out, 40, 100, (24,), "balanced")
            finally:
                P.BAL_M_A, P.BAL_M_N, P.K0_RATES, P.RHOS, P.SIZE_M = old
            rows = list(csv.DictReader(p.read_text().splitlines()))
            self.assertEqual([r["scenario"] for r in rows], ["power", "size", "size"])
            power, s0, s7 = rows
            self.assertEqual((power["size_alpha0.025"], power["size_alpha0.0125"]), ("", ""))
            for r in (s0, s7):
                self.assertEqual(r["m_A"], r["m_N"])
                self.assertEqual(r["m_N"], r["k0_rate"])                       # K0 at the same rate: the true contrast is 0
                self.assertEqual(r["size_alpha0.025"], r["support_alpha0.025"])
                self.assertEqual(r["size_alpha0.0125"], r["support_alpha0.0125"])
            self.assertEqual(float(s0["support_alpha0.025"]), 0.0)             # nobody ever acts: dV = 0, p = 1
            self.assertEqual(float(s0["p50_p"]), 1.0)
            self.assertEqual(float(s0["mean_est"]), 0.0)
            # the legacy pi grid has no size rows and no scenario column
            old = (P.M_A, P.M_N, P.K0_RATES, P.RHOS)
            P.M_A, P.M_N, P.K0_RATES, P.RHOS = (0.44,), (0.0,), (0.0,), (0.3,)
            try:
                rows_pi = list(csv.DictReader(P.run(out, 20, 50, (24,), "pi").read_text().splitlines()))
            finally:
                P.M_A, P.M_N, P.K0_RATES, P.RHOS = old
            self.assertEqual(len(rows_pi), 1)
            self.assertNotIn("scenario", rows_pi[0])

    def test_size_scenarios_are_near_alpha(self):
        # effect-zero at the planned rates, 24 + 24 persons: the rate of a false "supported" is of the order of alpha, far below
        # a power-like value; the case b level (alpha / 2) is no larger than the case a level
        for m in (0.07, 0.12):
            r = P.simulate(24, 24, m, m, m, 0.3, 300, 400, np.random.default_rng(21), estimator="balanced")
            self.assertLessEqual(r["support_alpha0.025"], 0.10)
            self.assertLessEqual(r["support_alpha0.0125"], r["support_alpha0.025"])
            self.assertLess(abs(r["mean_est"]), 0.02)                         # the estimate is centred on 0

    def test_permutation_test_size_power_and_form(self):
        # 判断 39: --test permutation; no effect -> rate of p <= alpha near / below alpha; a strong effect is detected; p = (b + 1) / (B + 1)
        r = P.simulate(24, 24, 0.07, 0.07, 0.07, 0.3, 300, 400, np.random.default_rng(21), estimator="balanced", test="permutation")
        self.assertLessEqual(r["support_alpha0.025"], 0.06)
        self.assertLessEqual(r["support_alpha0.0125"], r["support_alpha0.025"])
        s = P.simulate(24, 24, 0.69, 0.0, 0.0, 0.3, 100, 400, np.random.default_rng(2), estimator="balanced", test="permutation")
        self.assertGreater(s["support_alpha0.025"], 0.8)
        z = P.simulate(24, 24, 0.0, 0.0, 0.0, 0.3, 20, 99, np.random.default_rng(1), estimator="balanced", test="permutation")
        self.assertEqual((z["p50_p"], z["support_alpha0.025"]), (1.0, 0.0))     # all c_i = 0 -> p = 1
        self.assertAlmostEqual(s["p50_p"] * 401, round(s["p50_p"] * 401))
        with self.assertRaises(ValueError):
            P.simulate(24, 24, 0.1, 0.1, 0.1, 0.3, 5, 50, np.random.default_rng(1), estimator="pi", test="permutation")

    def test_run_permutation_writes_size_and_power_rows(self):
        import csv
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            (out / "SEED").write_text("5")
            old = (P.BAL_M_A, P.BAL_M_N, P.K0_RATES, P.RHOS, P.SIZE_M)
            P.BAL_M_A, P.BAL_M_N, P.K0_RATES, P.RHOS, P.SIZE_M = (0.44,), (0.0,), (0.0,), (0.3,), (0.07,)
            try:
                path = P.run(out, 20, 50, (24,), "balanced", "permutation")
            finally:
                P.BAL_M_A, P.BAL_M_N, P.K0_RATES, P.RHOS, P.SIZE_M = old
            rows = list(csv.DictReader(path.open()))
            self.assertEqual(sorted(r["scenario"] for r in rows), ["power", "size"])
            self.assertTrue(all(r["test"] == "permutation" for r in rows))

    def test_exch_contrasts_are_symmetric(self):
        # attack-me round 3 A2: under the sharp null c_i is symmetric about 0 (mean and third moment ~ 0), also with m != k0
        c = P.person_contrasts_exch(np.random.default_rng(4), 2000, 100, 0.33, 0.0, 0.3).ravel()
        self.assertLess(abs(c.mean()), 0.005)
        self.assertLess(abs((c ** 3).mean()), 0.005)
        self.assertAlmostEqual((c > 0).mean(), (c < 0).mean(), delta=0.005)

    def test_flip_p_rows_matches_sign_flip_p(self):
        from e1_summarise import sign_flip_p
        rng = np.random.default_rng(9)
        c = P.person_contrasts(rng, 4, 20, 0.33, 0.07, 0.3)
        signs = rng.choice(np.array([-1.0, 1.0]), (4, 99, 20))
        got = P.flip_p_rows(c, 10, signs)
        for j in range(4):
            self.assertAlmostEqual(got[j], float(sign_flip_p(c[j:j + 1], 10, signs[j])[0]))

    def test_independent_draws_size_power_and_guards(self):
        for test in ("permutation", "bootstrap"):
            s = P.simulate(24, 24, 0.69, 0.0, 0.0, 0.3, 60, 200, np.random.default_rng(2), estimator="balanced", test=test,
                           independent=True)
            self.assertGreater(s["support_alpha0.025"], 0.8)
        z = P.simulate(24, 24, 0.12, 0.12, 0.0, 0.3, 300, 200, np.random.default_rng(3), estimator="balanced",
                       test="permutation", independent=True, null="exch")
        self.assertLessEqual(z["support_alpha0.025"], 0.06)
        with self.assertRaises(ValueError):
            P.simulate(24, 24, 0.1, 0.1, 0.1, 0.3, 5, 50, np.random.default_rng(1), estimator="pi", independent=True)
        with self.assertRaises(ValueError):
            P.simulate(24, 24, 0.1, 0.1, 0.1, 0.3, 5, 50, np.random.default_rng(1), estimator="balanced", null="other")

    def test_clopper_pearson_known_values(self):
        lo, hi = P.clopper_pearson(250, 10000)
        self.assertAlmostEqual(lo, 0.02203, places=4)
        self.assertAlmostEqual(hi, 0.02827, places=4)
        self.assertEqual(P.clopper_pearson(0, 100)[0], 0.0)
        self.assertEqual(P.clopper_pearson(100, 100)[1], 1.0)

    def test_run_size_nulls_writes_rows_and_flags(self):
        import csv
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            (out / "SEED").write_text("11")
            old = (P.SIZE_NULLS, P.RHOS)
            P.SIZE_NULLS, P.RHOS = (("exch", 0.12, 0.12, 0.0), ("mean0", 0.07, 0.07, 0.07), ("k0zero", 0.12, 0.12, 0.0)), (0.3,)
            try:
                path = P.run_size_nulls(out, 40, 60, 12)
            finally:
                P.SIZE_NULLS, P.RHOS = old
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)
            self.assertEqual({(r["null"], r["test"]) for r in rows},
                             {(n, t) for n in ("exch", "mean0", "k0zero") for t in ("permutation", "bootstrap")})
            for r in rows:
                for a in P.ALPHAS:
                    lo, hi = float(r[f"cp_lo_alpha{a}"]), float(r[f"cp_hi_alpha{a}"])
                    self.assertLessEqual(lo, float(r[f"size_alpha{a}"]))
                    self.assertGreaterEqual(hi, float(r[f"size_alpha{a}"]))
                    self.assertEqual(r[f"exceeds_alpha{a}"], str(lo > a))

    def test_pi_reproduces_existing_result(self):
        # one row of experiments/20261113-e1-power/results/power.csv, same SeedSequence as run()
        import csv
        d = Path(__file__).parent.parent / "experiments" / "20261113-e1-power"
        with (d / "results" / "power.csv").open() as f:
            rows = list(csv.DictReader(f))
        r0 = next(r for r in rows if (r["n_per_stratum"], r["m_A"], r["m_N"], r["k0_rate"], r["rho"]) ==
                  ("24", "0.44", "0.05", "0.0", "0.7"))
        seed = int((d / "SEED").read_text())
        ss = np.random.SeedSequence([seed, 24, 440, 50, 0, 700])
        r = P.simulate(24, 24, 0.44, 0.05, 0.0, 0.7, 1000, 2000, np.random.default_rng(ss), estimator="pi")
        self.assertAlmostEqual(r["support_alpha0.025"], float(r0["support_alpha0.025"]))
        self.assertAlmostEqual(r["mean_est"], float(r0["mean_est"]))


if __name__ == "__main__":
    unittest.main()
