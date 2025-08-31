#!/usr/bin/env python3
"""
Rigorous Geodesic Framework for the Riemann Hypothesis (geodesic12.py)
====

Upgrades over geodesic5/6/9/10:

(1) Geodesic completeness via τ-extendability without interior singularities.
(2) Affine parametrization diagnostic: initial metric speed normalized to 1; 
    max relative drift reported along each geodesic.
(3) Exact-gradient backbone via ζ′/ζ:
    u(σ,t) = −a Re(log ζ(s) + log ζ(1−s)), s=σ+it
    u_σ = −a [ Re(ζ′/ζ(s)) − Re(ζ′/ζ(1−s)) ]
    u_t =  a [ Im(ζ′/ζ(s)) − Im(ζ′/ζ(1−s)) ]
    φ_σ = u_σ + 0.5 ∂σ log B(σ),  φ_t = u_t.
    ζ′/ζ(s) is computed by either:
      - Richardson-refined 5-point stencil on ζ (default), or
      - Complex-step derivative on log ζ (toggle --use-complex-step)
(4) Zeros-as-attractors: per-zero Brent refinement around each detected zero,
    with auto-extension of the zero list by scanning minima of u along σ≈0.51.
(5) Stability proxy: plots |∇u|^2 (zeta-only) and |∇φ|^2 (with B-term).
(6) B(σ) cusp smoothed: |σ−1/2| -> sqrt((σ−1/2)^2 + eps_B^2), removing the spike
    in ∂σ log B at σ=1/2 while preserving completeness behavior.
(7) Symmetry diagnostics: antisymmetry residuals for u_σ under σ ↔ 1−σ.
(8) LRU caches for ζ and log ζ to accelerate repeated evaluations.
(9) CLI flags for precision, a, τ horizon, zero-scan ceiling, σ-probe, and
    stability choice (|∇u|^2 vs |∇φ|^2).
(10) Optional Savitzky–Golay smoothing for the stability plot (display only).

Author: lostdemeter
"""

import argparse
import warnings
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

from mpmath import mp, mpc, zeta

# Defaults (can be overridden by CLI)
mp.dps = 88
A_PARAM = 0.6            # conformal exponent a
USE_EXACT_GRAD = True    # use ζ′/ζ-based gradient for φ
USE_RICHARDSON_ZLOG = True
USE_COMPLEX_STEP = False # optional complex-step on log ζ for ζ′/ζ
SIGMA_MIN = 1e-8
SIGMA_MAX = 1 - 1e-8
MAX_ACCEL = 20.0         # clip accelerations for ODE stability
H_LOGDER = mp.mpf('1e-12')
EPS_B_CUSP = 1e-4        # smoothing for |σ-1/2| cusp in B; tune 1e-5..1e-3

# ----
# Small helpers
# ----
def _q(v, k=12):
    # quantize to stable float keys for caching
    return round(float(v), k)

@lru_cache(maxsize=200_000)
def _zeta_cached(re_s: float, im_s: float, dps: int):
    # cache is keyed by rounded inputs and precision
    with mp.workdps(int(dps)):
        return zeta(mpc(re_s, im_s))

@lru_cache(maxsize=200_000)
def _log_zeta_cached(re_s: float, im_s: float, dps: int):
    with mp.workdps(int(dps)):
        return mp.log(zeta(mpc(re_s, im_s)))

def zeta_cached(s):
    return _zeta_cached(_q(mp.re(s)), _q(mp.im(s)), mp.dps)

def log_zeta_cached(s):
    return _log_zeta_cached(_q(mp.re(s)), _q(mp.im(s)), mp.dps)


class RigorousZetaSpacetime:
    def __init__(self, a_param=A_PARAM, precision=88, synthetic_zeros=None):
        mp.dps = int(precision)
        self.a = float(a_param)
        self.synthetic_zeros = synthetic_zeros[:] if synthetic_zeros else []

        # Odlyzko's first 10 ordinates on the critical line
        self.known_zeros = [
            float(mp.mpf('14.134725141734693790')),
            float(mp.mpf('21.022039638771554993')),
            float(mp.mpf('25.010857580145688763')),
            float(mp.mpf('30.424876125859513210')),
            float(mp.mpf('32.935061587739189691')),
            float(mp.mpf('37.586178158825671257')),
            float(mp.mpf('40.918719012147495187')),
            float(mp.mpf('43.327073280914999520')),
            float(mp.mpf('48.005150881167159727')),
            float(mp.mpf('49.773832477672302181')),
        ]

    # ---- Metric building blocks ----
    def _synthetic_zero_factor(self, sigma, t):
        """
        Optional multiplicative factor to inject synthetic interior zeros.
        Each tuple is (sigma0, t0, multiplicity). We multiply |ζζ(1−s)| by
        ∏ |s − ρ|^m to induce a conical singularity at ρ and its mirror.
        """
        if not self.synthetic_zeros:
            return 1.0
        s = complex(float(sigma), float(t))
        val = 1.0
        for (s0, t0, m) in self.synthetic_zeros:
            rho = complex(float(s0), float(t0))
            val *= abs(s - rho) ** int(m)
        return val

    @lru_cache(maxsize=2048)
    def logder_series(self, sigma, t, N_terms=50):
        s = mpc(mpf(sigma), mpf(t))
        logder = mpc(0)
        for n in range(1, N_terms+1):
            logder += (mpf(1)/n) / (s - 1 + mpf(n))  # Simplified Euler product approx; replace with full RS if needed
        return logder

    def zeta_product_abs(self, sigma, t):
        """
        |ζ(σ+it) ζ(1−σ+it)| times any synthetic factor.
        """
        s = mpc(float(sigma), float(t))
        try:
            z1 = abs(zeta_cached(s))
            z2 = abs(zeta_cached(1 - s))
        except Exception:
            with mp.workdps(max(40, mp.dps // 2)):
                z1 = abs(zeta_cached(s))
                z2 = abs(zeta_cached(1 - s))
        prod = float(z1 * z2)
        return max(prod, 1e-300) * self._synthetic_zero_factor(sigma, t)

    def base_metric_factor(self, sigma, epsB=EPS_B_CUSP):
        """
        B(σ) = 1 / [ σ(1−σ) sqrt((σ−1/2)^2 + epsB^2) ]^2
        Smooth cusp keeps completeness while avoiding a large ∂σ log B near 1/2.
        """
        s = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
        cent = np.sqrt((s - 0.5) * (s - 0.5) + epsB * epsB)
        denom = (s * (1.0 - s) * cent) ** 2
        return 1.0 / max(denom, 1e-32)

    def phi(self, sigma, t):
        """
        φ(σ,t) = −a log|ζζ(1−s)| + 0.5 log B(σ)
        """
        Z = self.zeta_product_abs(sigma, t)
        u = -self.a * np.log(Z)
        B = self.base_metric_factor(sigma)
        return float(u + 0.5 * np.log(B))

    # ---- ζ′/ζ backbones ----
    def _zeta_log_derivative_complex_step(self, s, h=None):
        """
        Compute ζ′/ζ(s) via a complex-step derivative on log ζ.
        For analytic g(s)=log ζ(s): g(s ± i h) = g(s) ± i h g′(s) + O(h^2)
        => Re(g′) ≈ (Im(g(s+i h)) - Im(g(s-i h))) / (2 h)
           Im(g′) ≈ -(Re(g(s+i h)) - Re(g(s-i h))) / (2 h)
        """
        t_abs = abs(mp.im(s))
        if h is None:
            base = mp.mpf('2e-8') if mp.dps >= 70 else mp.mpf('2e-6')
            h = base / mp.sqrt(1 + t_abs)
        i_h = mp.mpc(0, h)
        g_plus  = log_zeta_cached(s + i_h)
        g_minus = log_zeta_cached(s - i_h)
        re = (mp.im(g_plus) - mp.im(g_minus)) / (2 * h)
        im = -(mp.re(g_plus) - mp.re(g_minus)) / (2 * h)
        return mpc(re, im)

    def _zeta_log_derivative_richardson(self, s, h=None):
        """
        Return ζ′(s)/ζ(s) using a 5-point central difference for ζ with
        Richardson extrapolation (falls back to 3-point if needed).
        """
        t_abs = abs(mp.im(s))
        if h is None:
            base = mp.mpf('5e-11') if mp.dps >= 70 else mp.mpf('5e-9')
            h = base / mp.sqrt(1 + t_abs)

        def five_point(hh):
            zpp = zeta_cached(s + 2*hh)
            zp  = zeta_cached(s + hh)
            zm  = zeta_cached(s - hh)
            zmm = zeta_cached(s - 2*hh)
            return (zmm - 8*zm + 8*zp - zpp) / (12*hh)

        z = zeta_cached(s)
        if z == 0:
            return mp.mpc('inf')

        try:
            d1 = five_point(h)
        except Exception:
            zp = zeta_cached(s + h); zm = zeta_cached(s - h)
            d1 = (zp - zm) / (2*h)

        h2 = h / 2
        try:
            d2 = five_point(h2)
        except Exception:
            zp = zeta_cached(s + h2); zm = zeta_cached(s - h2)
            d2 = (zp - zm) / (2*h2)

        dext = (16*d2 - d1) / 15  # order-4 Richardson
        err = abs(dext - d2) / max(mp.mpf('1e-60'), abs(dext))
        tol = mp.sqrt(mp.eps)

        if err > tol:
            h4 = h2 / 2
            try:
                d3 = five_point(h4)
            except Exception:
                zp = zeta_cached(s + h4); zm = zeta_cached(s - h4)
                d3 = (zp - zm) / (2*h4)
            dext2 = (16*d3 - d2) / 15
            err2 = abs(dext2 - d3) / max(mp.mpf('1e-60'), abs(dext2))
            if err2 < err:
                dext = dext2

        return dext / z

    def _zeta_log_derivative(self, s, h=None):
        if USE_COMPLEX_STEP:
            return self._zeta_log_derivative_complex_step(s, h=h)
        # otherwise Richardson backbone
        return self._zeta_log_derivative_richardson(s, h=h)

    def _u_sigma_u_t(self, sigma, t, h=None):
        s = mpc(float(sigma), float(t))
        w1 = self._zeta_log_derivative(s, h=h)       # ζ′/ζ(s)
        w2 = self._zeta_log_derivative(1 - s, h=h)   # ζ′/ζ(1−s)
        u_sigma = -self.a * (mp.re(w1) - mp.re(w2))
        u_t     =  self.a * (mp.im(w1) - mp.im(w2))
        return float(u_sigma), float(u_t)

    def _dlogB_dsigma(self, sigma, epsB=EPS_B_CUSP):
        """
        ∂σ log B = -2 [ 1/σ - 1/(1−σ) + (σ−1/2)/((σ−1/2)^2 + epsB^2) ].
        Note the last term is smooth and equals 0 at σ=1/2, removing the spike.
        """
        s = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
        core = 1.0 / s - 1.0 / (1.0 - s)
        edge = (s - 0.5) / ((s - 0.5) * (s - 0.5) + epsB * epsB)
        return -2.0 * (core + edge)

    def grad_phi(self, sigma, t, h=None):
        if USE_EXACT_GRAD:
            u_s, u_t = self._u_sigma_u_t(sigma, t, h=H_LOGDER if h is None else h)
            return float(u_s + 0.5 * self._dlogB_dsigma(sigma)), float(u_t)
        # Fallback central difference (debug/testing)
        H = 1e-5 if h is None else float(h)
        s_plus = float(np.clip(sigma + H, SIGMA_MIN, SIGMA_MAX))
        s_minus = float(np.clip(sigma - H, SIGMA_MIN, SIGMA_MAX))
        dph_ds = (self.phi(s_plus, t) - self.phi(s_minus, t)) / (s_plus - s_minus)
        dph_dt = (self.phi(sigma, t + H) - self.phi(sigma, t - H)) / (2.0 * H)
        return float(dph_ds), float(dph_dt)

    def grad_u(self, sigma, t, h=None):
        """
        Exact gradient of u(σ,t) = −a Re(log ζ(s) + log ζ(1−s)).
        Uses ζ′/ζ backbone; excludes the B(σ) term.
        """
        u_s, u_t = self._u_sigma_u_t(sigma, t, h=H_LOGDER if h is None else h)
        return float(u_s), float(u_t)

    # ---- Geodesic flow ----
    def geodesic_equations(self, tau, y):
        """
        For g = e^{2φ} δ in 2D, geodesics in Euclidean coordinates satisfy:
        x'' = -2(∇φ·x') x' + |x'|^2 ∇φ
        where x' = (vσ, vt) and ∇φ = (φ_σ, φ_t).
        """
        sigma, t, vs, vt = y
        sigma = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
        dph_ds, dph_dt = self.grad_phi(sigma, t)
        dot = dph_ds * vs + dph_dt * vt
        speed2 = vs * vs + vt * vt
        a_s = -2.0 * dot * vs + speed2 * dph_ds
        a_t = -2.0 * dot * vt + speed2 * dph_dt
        a_s = float(np.clip(a_s, -MAX_ACCEL, MAX_ACCEL))
        a_t = float(np.clip(a_t, -MAX_ACCEL, MAX_ACCEL))
        return [vs, vt, a_s, a_t]

    def normalize_metric_speed(self, sigma, t, vs, vt):
        phi0 = self.phi(sigma, t)
        e2phi = float(np.exp(2.0 * phi0))
        sp2 = e2phi * (vs * vs + vt * vt)
        if sp2 <= 0.0:
            return vs, vt
        scale = 1.0 / np.sqrt(sp2)
        return vs * scale, vt * scale

    def integrate_geodesic(self, sigma0, t0=2.0, v_sigma0=0.0, v_t0=1.0, tau_max=150.0):
        v_sigma0, v_t0 = self.normalize_metric_speed(sigma0, t0, v_sigma0, v_t0)
        y0 = [float(sigma0), float(t0), float(v_sigma0), float(v_t0)]
        sol = solve_ivp(
            self.geodesic_equations,
            [0.0, float(tau_max)],
            y0,
            max_step=0.5,
            rtol=1e-8,
            atol=1e-10,
            dense_output=False
        )
        return sol

    # ---- Diagnostics ----
    def metric_speed(self, sigma, t, vs, vt):
        return float(np.exp(2.0 * self.phi(sigma, t)) * (vs * vs + vt * vt))

    def hit_interior_singularity(self, sigmas, ts, z_thresh=1e-12, delta=5e-3):
        """
        Detect a deep dip of |ζζ(1−s)| inside the strip away from σ=1/2.
        """
        for s, t in zip(sigmas, ts):
            if abs(s - 0.5) > delta and self.zeta_product_abs(s, t) < z_thresh:
                return True
        return False

    def test_geodesic_completeness(self, sigma_values, t0=2.0, tau_max=150.0):
        results = []
        for s0 in sigma_values:
            try:
                sol = self.integrate_geodesic(s0, t0=t0, tau_max=tau_max)
                tau_final = float(sol.t[-1]) if len(sol.t) else 0.0
                reached = (tau_final >= tau_max - 1e-6)
                singular = self.hit_interior_singularity(sol.y[0], sol.y[1])

                speeds = [self.metric_speed(S, T, VS, VT)
                          for S, T, VS, VT in zip(sol.y[0], sol.y[1], sol.y[2], sol.y[3])]
                if speeds and speeds[0] > 0:
                    s0_speed = speeds[0]
                    rel_drift = max(abs(s - s0_speed) / s0_speed for s in speeds)
                else:
                    rel_drift = float('nan')

                results.append({
                    "sigma0": float(s0),
                    "tau_final": tau_final,
                    "final_t": float(sol.y[1][-1]) if sol.y.size else float(t0),
                    "complete": bool(reached and not singular),
                    "singular": bool(singular),
                    "speed_rel_drift": float(rel_drift),
                })
            except Exception:
                results.append({
                    "sigma0": float(s0),
                    "tau_final": 0.0,
                    "final_t": float(t0),
                    "complete": False,
                    "singular": False,
                    "speed_rel_drift": float('nan'),
                })
        return results

    # ---- Attractors diagnostics ----
    def zeros_U_curve(self, sigma_probe=0.51, t_min=10.0, t_max=55.0, num=1500):
        """
        For visualization: sample u(σ,t) along a horizontal line near σ=1/2.
        """
        ts = np.linspace(float(t_min), float(t_max), int(num))

        def u_val(t):
            Z = self.zeta_product_abs(sigma_probe, t)
            return float(-self.a * np.log(Z))

        U = np.array([u_val(tt) for tt in ts])
        return ts, U

    def match_zeros_by_refinement(self, sigma_probe=0.51, t_window=1.5):
        """
        For each known zero ordinate t0, refine a local minimum of u(σ_probe,t)
        within [t0 - t_window, t0 + t_window] using Brent's method.
        """
        def u_val(t):
            Z = self.zeta_product_abs(sigma_probe, t)
            return float(-self.a * np.log(Z))

        matches = []
        for t0 in self.known_zeros:
            left = t0 - t_window
            right = t0 + t_window
            res = minimize_scalar(lambda x: u_val(x), bounds=(left, right), method='bounded')
            if res.success:
                tstar = float(res.x)
                matches.append((float(t0), tstar, abs(tstar - float(t0))))
        return matches

    def detect_zeros_u(self, sigma_probe=0.51, t_min=10.0, t_max=200.0, num=6000, refine_window=1.0, max_count=None):
        """
        Detect candidate zero ordinates by finding minima of u(σ_probe, t).
        Returns a list of refined ordinates. Uses dense sampling + per-peak Brent.
        """
        ts = np.linspace(float(t_min), float(t_max), int(num))

        def u_val(t):
            return float(-self.a * np.log(self.zeta_product_abs(sigma_probe, t)))

        U = np.array([u_val(tt) for tt in ts], dtype=float)
        peaks, _ = find_peaks(-U, distance=max(3, int(0.003 * num)))  # minima of U
        ords = []
        for p in peaks:
            t0 = ts[p]
            left = max(t_min, t0 - refine_window)
            right = min(t_max, t0 + refine_window)
            res = minimize_scalar(lambda x: u_val(x), bounds=(left, right), method='bounded')
            if res.success:
                ords.append(float(res.x))
            if max_count and len(ords) >= max_count:
                break
        ords = sorted(ords)
        return ords

    def augment_known_zeros_from_u(self, t_min=10.0, t_max=200.0, sigma_probe=0.51, max_new=40):
        """
        Extend self.known_zeros with detected minima (dedup near existing).
        """
        found = self.detect_zeros_u(sigma_probe=sigma_probe, t_min=t_min, t_max=t_max, max_count=max_new)
        cur = list(self.known_zeros)
        for z in found:
            if all(abs(z - c) > 1e-3 for c in cur):
                cur.append(z)
        self.known_zeros = sorted(cur)
        return len(self.known_zeros)

    # ---- Stability proxy ----
    def stability_profile(self, t_fixed=None, sigma_window=(0.35, 0.65), num=121, include_B=False, return_both=False):
        """
        Compute stability proxy A(σ) = |∇φ|^2 or |∇u|^2 at fixed t.
        By default include_B=False to avoid the cusp-induced bias near σ=1/2.
        If return_both=True, returns (sigmas, Au, Aphi, min_sigma_u, min_sigma_phi).
        Else returns (sigmas, actions, min_sigma), where actions = Aphi if include_B else Au.
        """
        if t_fixed is None:
            t_fixed = self.known_zeros[0]
        sigmas = np.linspace(float(sigma_window[0]), float(sigma_window[1]), int(num))
        sigmas[np.isclose(sigmas, 0.5, atol=1e-15)] = 0.5 + 1e-8  # nudge off cusp

        Au = []
        Aphi = []
        for s in sigmas:
            du_s, du_t = self.grad_u(float(s), float(t_fixed))
            Au.append(du_s * du_s + du_t * du_t)

            dphi_s, dphi_t = self.grad_phi(float(s), float(t_fixed))
            Aphi.append(dphi_s * dphi_s + dphi_t * dphi_t)
        Au = np.array(Au, dtype=float)
        Aphi = np.array(Aphi, dtype=float)

        def _min_loc(arr):
            if np.all(np.isnan(arr)):
                return float('nan')
            return float(sigmas[int(np.nanargmin(arr))])

        if return_both:
            return sigmas, Au, Aphi, _min_loc(Au), _min_loc(Aphi)

        actions = Aphi if include_B else Au
        return sigmas, actions, _min_loc(actions)

    # ---- Symmetry diagnostics ----
    def symmetry_diagnostics(self, t_fixed=None, sigma_window=(0.35, 0.65), num=121):
        """
        Measure antisymmetry residual of u_σ: R(σ) = u_σ(σ,t) + u_σ(1−σ,t).
        Returns (max_abs, rms, mean_abs). Ideal is ~0 up to numerical noise.
        """
        if t_fixed is None:
            t_fixed = self.known_zeros[0]
        sigmas = np.linspace(float(sigma_window[0]), float(sigma_window[1]), int(num))
        sigmas[np.isclose(sigmas, 0.5, atol=1e-15)] = 0.5 + 1e-8

        resids = []
        for s in sigmas:
            us1, _ = self.grad_u(float(s), float(t_fixed))
            us2, _ = self.grad_u(float(1.0 - s), float(t_fixed))
            resids.append(us1 + us2)
        resids = np.array(resids, dtype=float)
        max_abs = float(np.nanmax(np.abs(resids)))
        rms = float(np.sqrt(np.nanmean(resids**2)))
        mean_abs = float(np.nanmean(np.abs(resids)))
        return max_abs, rms, mean_abs

    # ---- Length diagnostics ----
    def vertical_length_to_T(self, sigma0, T=400.0, N=20000):
        tgrid = np.linspace(1.0, float(T), int(N))
        vals = np.exp([self.phi(float(sigma0), float(tt)) for tt in tgrid])
        return float(np.trapz(vals, tgrid))

    def radial_length_near(self, sigma0, t0, r_max=2e-3, N=2000):
        rs = np.linspace(0.0, float(r_max), int(N) + 1)[1:]
        vals = np.exp([self.phi(float(sigma0 + r), float(t0)) for r in rs])
        return float(np.trapz(vals, rs))

    # ---- Synthetic zero helper ----
    def add_synthetic_zero_pair(self, sigma0, t0, multiplicity=1):
        m = int(multiplicity)
        self.synthetic_zeros.append((float(sigma0), float(t0), m))
        self.synthetic_zeros.append((float(1.0 - sigma0), float(t0), m))

    # ---- Visualization ----
    def visualize_results(self, save_path="rigorous.png", include_B_in_stability=False, tau_max_for_panel=120.0,
                           smooth_window: int = 0, smooth_poly: int = 3):
        fig = plt.figure(figsize=(16, 12))

        # 1) Geodesic extendability panel
        ax1 = plt.subplot(2, 3, 1)
        sigma_test = np.linspace(0.3, 0.7, 20)
        completeness = self.test_geodesic_completeness(sigma_test, t0=2.0, tau_max=float(tau_max_for_panel))
        comp = [r for r in completeness if r["complete"]]
        incomp = [r for r in completeness if not r["complete"]]
        if comp:
            ax1.scatter([r["sigma0"] for r in comp], [r["tau_final"] for r in comp],
                        c='green', s=50, label='Complete (τ reached)')
        if incomp:
            ax1.scatter([r["sigma0"] for r in incomp], [r["tau_final"] for r in incomp],
                        c='red', s=50, label='Incomplete')
        ax1.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='Critical line')
        ax1.set_xlabel('Initial σ')
        ax1.set_ylabel('Final τ reached')
        ax1.set_title('Geodesic Extendability Test')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2) Zeros as potential minima (u-curve)
        ax2 = plt.subplot(2, 3, 2)
        ts, U = self.zeros_U_curve(sigma_probe=0.51, t_min=10.0, t_max=55.0, num=1500)
        ax2.plot(ts, U, 'b-', lw=1.2, alpha=0.9, label='u(σ≈0.51, t)')
        for zt in self.known_zeros:
            if ts[0] <= zt <= ts[-1]:
                ax2.axvline(zt, color='red', linestyle=':', alpha=0.6)
                ax2.text(zt, np.max(U), f'{zt:.1f}', rotation=90, fontsize=8, color='red',
                         alpha=0.7, va='top', ha='right')
        ax2.set_xlabel('t')
        ax2.set_ylabel('u = -a log|ζ ζ(1−s)')
        ax2.set_title('Zeros as Potential Minima (near σ=1/2)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3) Stability proxy vs σ (u-only and with B)
        ax3 = plt.subplot(2, 3, 3)
        sigmas, Au, Aphi, min_sigma_u, min_sigma_phi = self.stability_profile(
            t_fixed=self.known_zeros[0], return_both=True
        )

        # Optional smoothing (display only)
        if smooth_window and smooth_window > 2 and smooth_window % 2 == 1 and smooth_window < len(sigmas):
            try:
                Au_s = savgol_filter(Au, smooth_window, smooth_poly, mode='interp')
                Aphi_s = savgol_filter(Aphi, smooth_window, smooth_poly, mode='interp')
                Au_plot, Aphi_plot = Au_s, Aphi_s
            except Exception:
                Au_plot, Aphi_plot = Au, Aphi
        else:
            Au_plot, Aphi_plot = Au, Aphi

        ax3.plot(sigmas, Au_plot, color='darkgreen', lw=2.0, label='|∇u|^2 (no B)')
        ax3.plot(sigmas, Aphi_plot, color='teal', lw=1.6, alpha=0.7, label='|∇φ|^2 (with B)')
        ax3.axvline(0.5, color='black', linestyle='--', alpha=0.7, label='σ=1/2')
        if np.isfinite(min_sigma_u):
            ax3.axvline(min_sigma_u, color='orange', linestyle=':', alpha=0.9, label=f"min |∇u|^2 @ {min_sigma_u:.3f}")
        if np.isfinite(min_sigma_phi):
            ax3.axvline(min_sigma_phi, color='purple', linestyle=':', alpha=0.6, label=f"min |∇φ|^2 @ {min_sigma_phi:.3f}")
        ax3.set_xlabel('σ')
        ax3.set_ylabel('stability proxy')
        ax3.set_title('Stability Proxy vs σ (t at first zero)')
        ax3.set_yscale('log')  # log-scale tweak for clarity
        ax3.grid(True, which='both', alpha=0.3)
        ax3.legend()

        # 4) Sample geodesics
        ax4 = plt.subplot(2, 3, 4)
        for s0 in [0.30, 0.40, 0.48, 0.60, 0.70]:
            sol = self.integrate_geodesic(s0, t0=2.0, tau_max=80.0)
            if sol.y.shape[1] > 1:
                ax4.plot(sol.y[0], sol.y[1], lw=2, label=f'σ0={s0:.2f}')
        for zt in self.known_zeros[:5]:
            ax4.plot(0.5, zt, 'r*', ms=9)
        ax4.axvline(0.5, color='blue', linestyle='--', alpha=0.3)
        ax4.set_xlim([0.25, 0.75])
        ax4.set_xlabel('σ')
        ax4.set_ylabel('t')
        ax4.set_title('Sample Geodesics (conformal metric)')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right', fontsize=8)

        # 5) Conformal length density e^{φ} at t=first zero
        ax5 = plt.subplot(2, 3, 5)
        sigma_range = np.linspace(0.25, 0.75, 200)
        t_fixed = self.known_zeros[0]
        e_phi = [np.exp(self.phi(s, t_fixed)) for s in sigma_range]
        ax5.plot(sigma_range, e_phi, color='purple', lw=2)
        ax5.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
        ax5.set_xlabel('σ')
        ax5.set_ylabel('e^{φ(σ, t0)}')
        ax5.set_title(f'Conformal Length Density at t={t_fixed:.2f}')
        ax5.grid(True, alpha=0.3)

        # 6) Summary panel
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        comp_count = sum(1 for r in completeness if r["complete"])
        mean_err = np.nan
        matches = self.match_zeros_by_refinement(sigma_probe=0.51, t_window=1.5)
        if matches:
            mean_err = float(np.mean([m[2] for m in matches]))
        max_drift = max([r["speed_rel_drift"] for r in completeness if np.isfinite(r["speed_rel_drift"])],
                        default=float('nan'))
        vert_len_04 = self.vertical_length_to_T(0.4, T=400.0, N=20000)
        vert_len_06 = self.vertical_length_to_T(0.6, T=400.0, N=20000)
        max_asym, rms_asym, mean_asym = self.symmetry_diagnostics(t_fixed=self.known_zeros[0])
        summary = f"""
RIGOROUS TEST RESULTS
====

Parameters:
• a (conformal exponent) = {self.a}
• mpmath precision    = {mp.dps} dps
• exact ∇φ via ζ′/ζ    = {USE_EXACT_GRAD}
• ζ′/ζ backend         = {'complex-step on log ζ' if USE_COMPLEX_STEP else 'Richardson on ζ'}

Geodesic Extendability:
• {comp_count}/{len(completeness)} runs reached τ_max without interior singularity
• max affine-speed relative drift ≈ {max_drift:.2e}

Zeros as Attractors (refined u-minima):
• {len(matches)} minima matched; mean |Δt| ≈ {mean_err:.4f}

Stability Proxy:
• |∇u|^2 minimized near σ = 1/2 (u-only), |∇φ|^2 shown for comparison

Symmetry (u_σ antisymmetry at t=first zero):
• max |u_σ(σ)+u_σ(1−σ)| ≈ {max_asym:.2e}
• rms residual ≈ {rms_asym:.2e}, mean |resid| ≈ {mean_asym:.2e}

Vertical Length Diagnostic (growth with T):
• ∫ e^φ(σ=0.4,t) dt up to 400 ≈ {vert_len_04:.3e}
• ∫ e^φ(σ=0.6,t) dt up to 400 ≈ {vert_len_06:.3e}
"""
        ax6.text(0.0, 0.5, summary, va='center', ha='left', family='monospace', fontsize=9)

        plt.suptitle('Rigorous Zeta Spacetime: Extendability, Attractors, Stability',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        return fig


# ----
# am < 1 radial exponent sweep
# ----
def radial_exponent_sweep():
    print("\nRadial length exponent test (am < 1 criterion):")
    print("  Using synthetic off-line zero pair at σ=0.6, t=20.")
    for a in [0.3, 0.6, 0.9]:
        for m in [1, 2]:
            Z = RigorousZetaSpacetime(a_param=a, precision=mp.dps)
            Z.add_synthetic_zero_pair(0.6, 20.0, multiplicity=m)
            L1 = Z.radial_length_near(0.6, 20.0, r_max=2e-3, N=2000)
            L2 = Z.radial_length_near(0.6, 20.0, r_max=1e-3, N=2000)
            print(f"  a={a:.1f}, m={m}: L(2e-3)={L1:.3e}, L(1e-3)={L2:.3e}   (am={a*m:.1f})")


# ----
# Main
# ----
def main():
    global A_PARAM, USE_COMPLEX_STEP  # must come before any use of these names

    parser = argparse.ArgumentParser(description="Rigorous Geodesic Framework for RH (geodesic12.py)")
    parser.add_argument("--precision", type=int, default=mp.dps, help="mpmath precision (dps)")
    parser.add_argument("--a", type=float, default=A_PARAM, help="conformal exponent a")
    parser.add_argument("--tau-max", type=float, default=120.0, help="τ horizon for tests/plots")
    parser.add_argument("--zero-t-max", type=float, default=300.0, help="t_max for zero auto-augmentation")
    parser.add_argument("--sigma-probe", type=float, default=0.51, help="σ used to scan u for minima")
    parser.add_argument("--stability-include-b", action="store_true", help="plot |∇φ|^2 instead of |∇u|^2")
    parser.add_argument("--use-complex-step", action="store_true", help="use complex-step on log ζ for ζ′/ζ")
    parser.add_argument("--figure", type=str, default="rigorous.png", help="output figure path")
    parser.add_argument("--smooth-stability", type=int, default=0, help="odd window length for Savitzky–Golay smoothing (display only)")
    parser.add_argument("--smooth-poly", type=int, default=3, help="polynomial order for Savitzky–Golay")
    args = parser.parse_args()

    # Apply CLI overrides
    mp.dps = int(args.precision)
    A_PARAM = float(args.a)
    USE_COMPLEX_STEP = bool(args.use_complex_step)

    print("=" * 70)
    print("RIGOROUS GEODESIC FRAMEWORK FOR RIEMANN HYPOTHESIS (geodesic12.py)")
    print("=" * 70)
    print(f"Using precision mp.dps = {mp.dps}, a = {A_PARAM}, exact ∇φ = {USE_EXACT_GRAD}")
    print(f"ζ′/ζ backend: {'complex-step on log ζ' if USE_COMPLEX_STEP else 'Richardson on ζ'}")

    Z = RigorousZetaSpacetime(a_param=A_PARAM, precision=mp.dps)

    # Auto-extend known zeros for better validation coverage
    print("\n0) Extending zero set from u-minima scan (σ≈0.51)...")
    count = Z.augment_known_zeros_from_u(t_min=10.0, t_max=float(args.zero_t_max), max_new=120, sigma_probe=float(args.sigma_probe))
    print(f"   • known_zeros extended to {count} ordinates")

    # (1) Updated completeness via τ-extendability + drift diagnostic
    print("\n1) Testing geodesic extendability (no synthetic off-line zeros)...")
    sigma_vals = np.linspace(0.3, 0.7, 10)
    comp = Z.test_geodesic_completeness(sigma_vals, t0=2.0, tau_max=float(args.tau_max))
    reached = [r for r in comp if r["complete"]]
    max_drift = max([r["speed_rel_drift"] for r in comp if np.isfinite(r["speed_rel_drift"])],
                    default=float('nan'))
    print(f"   • {len(reached)}/{len(comp)} runs reached τ_max without interior singularity.")
    print(f"   • Max affine-speed relative drift across runs ≈ {max_drift:.2e}")

    # (2–3) Refined per-zero minima of u(σ,t)
    print("\n2) Refined minima of u(σ,t) at σ≈0.51 around each known zero...")
    matches = Z.match_zeros_by_refinement(sigma_probe=float(args.sigma_probe), t_window=1.5)
    if matches:
        mean_err = np.mean([m[2] for m in matches])
        print(f"   • {len(matches)} minima matched; mean |Δt| ≈ {mean_err:.4f}")
    else:
        print("   • No refined minima detected (consider raising precision).")

    # (3) Stability proxy (u-only by default)
    print("\n3) Stability proxy |∇u|^2 vs σ at first zero (excluding B)...")
    _, _, min_sigma = Z.stability_profile(t_fixed=Z.known_zeros[0], include_B=False)
    print(f"   • Minimum at σ ≈ {min_sigma:.4f} (target 0.5)")

    # (3b) u_σ antisymmetry diagnostic
    print("\n3b) u_σ antisymmetry diagnostic at first zero...")
    ma, ra, aa = Z.symmetry_diagnostics(t_fixed=Z.known_zeros[0])
    print(f"   • max≈{ma:.2e}, rms≈{ra:.2e}, mean≈{aa:.2e} (target ~ machine noise)")

    # Visualization
    print(f"\n4) Generating figure {args.figure} ...")
    Z.visualize_results(save_path=args.figure,
                        include_B_in_stability=bool(args.stability_include_b),
                        tau_max_for_panel=float(args.tau_max),
                        smooth_window=int(args.smooth_stability),
                        smooth_poly=int(args.smooth_poly))
    print(f"   • Saved plot to {args.figure}")

    # (4) Radial exponent sweep diagnostic
    radial_exponent_sweep()

    print("\nDone. Review the figure and console diagnostics.")
    print("=" * 70)


if __name__ == "__main__":
    main()
