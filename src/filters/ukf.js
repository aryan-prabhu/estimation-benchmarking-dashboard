/**
 * Unscented Kalman Filter (UKF) — the sigma-point approach.
 *
 * Key idea (Julier & Uhlmann): instead of linearizing the model, sample the
 * *distribution*. Pick a small, deterministic set of "sigma points" that
 * exactly capture the mean and covariance of the current Gaussian, push each
 * point through the full nonlinear function, and recover the transformed
 * mean and covariance from the weighted outputs. This "unscented transform"
 * matches the true posterior moments to 2nd order (3rd for symmetric
 * nonlinearities), versus 1st order for the EKF — and it never needs a
 * Jacobian.
 *
 * For an n-dimensional state there are 2n+1 sigma points. Here n = 1, so we
 * use 3 points:
 *
 *   χ⁰ = x̂
 *   χ¹ = x̂ + √((n+λ) P)
 *   χ² = x̂ − √((n+λ) P)
 *
 * with scaling λ = α²(n+κ) − n and weights
 *
 *   W_m⁰ = λ/(n+λ)                 (mean weight, center point)
 *   W_c⁰ = W_m⁰ + (1 − α² + β)     (covariance weight, center point)
 *   W_mⁱ = W_cⁱ = 1/(2(n+λ))       (all other points)
 *
 * Parameter intuition:
 *   α — spread of the sigma points around the mean. Tiny α (1e-3) hugs the
 *       mean and behaves EKF-like; for strongly nonlinear systems like this
 *       one a wide spread (α = 1) probes the nonlinearity better.
 *   β — prior knowledge of the distribution's shape; β = 2 is optimal if
 *       the true distribution is Gaussian.
 *   κ — secondary scaling; κ = 3 − n is the classic heuristic that matches
 *       some 4th-order moment terms for a Gaussian.
 *
 * Importantly for THIS system: h(x) = x²/20 is symmetric, so the EKF's
 * H = h'(0) = 0 blind spot doesn't occur — the spread sigma points see the
 * curvature of h even when the mean sits at zero.
 */

import { DEFAULTS, f, h } from '../simulation/model.js';

/**
 * Run the UKF over a measurement sequence (scalar state, n = 1).
 *
 * @param {number[]} measurements  z_0 .. z_{N-1}
 * @param {object}   params        { Q, R, P0, xInit, alpha, beta, kappa }
 * @returns {{ estimates: number[], covariances: number[] }}
 */
export function unscentedKalmanFilter(measurements, params = {}) {
  const { Q, R, P0 } = { ...DEFAULTS, ...params };
  const xInit = params.xInit ?? 0;

  const n = 1; // state dimension
  const alpha = params.alpha ?? 1.0;
  const beta = params.beta ?? 2.0;
  const kappa = params.kappa ?? 3 - n;

  const lambda = alpha * alpha * (n + kappa) - n;
  const c = n + lambda; // sigma-point scaling (n+λ)

  // Weights (shared by every step — they depend only on n, α, β, κ).
  const Wm = [lambda / c, 1 / (2 * c), 1 / (2 * c)]; // for means
  const Wc = [lambda / c + (1 - alpha * alpha + beta), 1 / (2 * c), 1 / (2 * c)]; // for covariances

  /** The 2n+1 = 3 sigma points of N(mean, cov). √((n+λ)P) is the scalar
   *  "matrix square root" — in 1-D, just sqrt. */
  function sigmaPoints(mean, cov) {
    const s = Math.sqrt(c * cov);
    return [mean, mean + s, mean - s];
  }

  let x = xInit; // posterior mean
  let P = P0; // posterior variance

  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    // ---- Predict (time update) ----------------------------------------
    if (k > 0) {
      // 1. Draw sigma points from the current posterior N(x, P).
      const chi = sigmaPoints(x, P);
      // 2. Propagate each point through the FULL nonlinear dynamics.
      const chiPred = chi.map((p) => f(p, k - 1));
      // 3. Predicted mean: weighted average of transformed points.
      //    x̂⁻ = Σ W_mⁱ χⁱ
      x = chiPred.reduce((acc, p, i) => acc + Wm[i] * p, 0);
      // 4. Predicted covariance: weighted spread plus process noise.
      //    P⁻ = Σ W_cⁱ (χⁱ − x̂⁻)² + Q
      P = chiPred.reduce((acc, p, i) => acc + Wc[i] * (p - x) * (p - x), 0) + Q;
    }

    // ---- Update (measurement update) ----------------------------------
    // Re-draw sigma points from the *predicted* distribution. (Re-drawing
    // captures the process noise just added to P; reusing the propagated
    // points would not.)
    const chi = sigmaPoints(x, P);
    // Push them through the measurement function.
    const Z = chi.map((p) => h(p));
    // Predicted measurement: ẑ = Σ W_mⁱ Zⁱ. Because the points straddle the
    // mean, ẑ ≈ E[x²]/20 = (x̂² + P)/20 — it correctly includes the variance
    // contribution that the EKF's first-order ẑ = x̂²/20 misses.
    const zPred = Z.reduce((acc, z, i) => acc + Wm[i] * z, 0);
    // Innovation variance S = Σ W_cⁱ (Zⁱ − ẑ)² + R.
    const S = Z.reduce((acc, z, i) => acc + Wc[i] * (z - zPred) * (z - zPred), 0) + R;
    // State–measurement cross-covariance P_xz = Σ W_cⁱ (χⁱ − x̂)(Zⁱ − ẑ).
    // This plays the role P Hᵀ played in the (E)KF.
    const Pxz = chi.reduce((acc, p, i) => acc + Wc[i] * (p - x) * (Z[i] - zPred), 0);
    // Gain, mean, covariance — same structure as every Kalman-type filter:
    //   K = P_xz S⁻¹,  x̂ ← x̂ + K(z − ẑ),  P ← P − K S Kᵀ
    const K = Pxz / S;
    x = x + K * (measurements[k] - zPred);
    P = P - K * S * K;

    estimates.push(x);
    covariances.push(P);
  }

  return { estimates, covariances };
}
