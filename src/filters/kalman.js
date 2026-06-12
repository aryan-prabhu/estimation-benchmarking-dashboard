/**
 * Linear Kalman Filter (KF) — the mismatched baseline.
 *
 * The Kalman filter is the *optimal* (minimum mean-square-error) estimator,
 * but only when the system is linear-Gaussian:
 *
 *   x_{k+1} = A x_k + B u_k + w_k
 *   z_k     = H x_k + v_k
 *
 * Our benchmark system is NOT linear, so a pure KF cannot be applied
 * directly. The only honest way to use one is to linearize the model ONCE
 * around a fixed nominal point and never update that linearization (if we
 * re-linearized around the current estimate every step, we would have built
 * an EKF). This is sometimes called a "linearized Kalman filter".
 *
 * Linearization choices, made up front and held fixed:
 *
 *  - Dynamics: keep the linear part of f. d/dx [0.5x + 25x/(1+x²)] at x = 0
 *    is 0.5 + 25 = 25.5 — but the slope at the nominal point xNom is more
 *    representative of where the state actually lives, so we use
 *    A = f'(xNom). The 8·cos(1.2k) term is linear in nothing but known
 *    exactly, so it enters as a deterministic input u_k (like B·u_k).
 *
 *  - Measurement: h(x) = x²/20, so h'(x) = x/10. We fix H = xNom/10 and
 *    predict measurements with the first-order expansion
 *    ẑ = h(xNom) + H (x̂ − xNom).
 *
 * Expect this filter to do poorly: the state swings across ±20 and through
 * zero, so any single linearization point is wrong most of the time. That
 * failure is the pedagogical point — it motivates the EKF/UKF/PF.
 */

import { DEFAULTS, h } from '../simulation/model.js';

/** Analytic derivative of the drift: d/dx [0.5x + 25x/(1+x²)]. */
function fPrime(x) {
  const d = 1 + x * x;
  return 0.5 + (25 * (1 - x * x)) / (d * d);
}

/**
 * Run the linearized KF over a measurement sequence.
 *
 * @param {number[]} measurements  z_0 .. z_{N-1}
 * @param {object}   params        { Q, R, x0, P0, xNom }
 * @returns {{ estimates: number[], covariances: number[] }}
 */
export function kalmanFilter(measurements, params = {}) {
  const { Q, R, P0 } = { ...DEFAULTS, ...params };
  // The filter does not know the true x0; it starts at a prior mean of 0
  // with variance P0 (an honest statement of ignorance).
  const xInit = params.xInit ?? 0;
  // Fixed nominal linearization point. xNom = 5 sits in the middle of the
  // state's typical positive excursion; H = xNom/10 = 0.5.
  const xNom = params.xNom ?? 5;

  const A = fPrime(xNom); // fixed dynamics slope
  const H = xNom / 10; // fixed measurement slope h'(xNom)

  let x = xInit; // posterior mean  x̂_{k|k}
  let P = P0; // posterior variance P_{k|k}

  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    // ---- Predict (time update) ----------------------------------------
    // Skip at k = 0: the first measurement observes the initial state, so
    // we update the prior directly.
    if (k > 0) {
      // x̂_{k|k-1} = A x̂ + u_{k-1}, with the known cosine forcing as input.
      // (Everything is scalar here, so Aᵀ = A and the algebra collapses.)
      const u = 8 * Math.cos(1.2 * (k - 1));
      x = A * x + u;
      // P_{k|k-1} = A P Aᵀ + Q — uncertainty grows through the dynamics
      // and the process noise is added.
      P = A * P * A + Q;
    }

    // ---- Update (measurement update) ----------------------------------
    // Predicted measurement from the FIXED first-order expansion of h
    // around xNom: ẑ = h(xNom) + H (x̂ − xNom).
    const zPred = h(xNom) + H * (x - xNom);
    // Innovation y = z − ẑ: the part of the measurement the model didn't
    // anticipate.
    const y = measurements[k] - zPred;
    // Innovation variance S = H P Hᵀ + R.
    const S = H * P * H + R;
    // Kalman gain K = P Hᵀ S⁻¹: how much to trust the innovation, trading
    // prior confidence (P) against measurement noise (R).
    const K = (P * H) / S;
    // Posterior mean and variance:
    //   x̂_{k|k} = x̂_{k|k-1} + K y
    //   P_{k|k} = (1 − K H) P_{k|k-1}
    x = x + K * y;
    P = (1 - K * H) * P;

    estimates.push(x);
    covariances.push(P);
  }

  return { estimates, covariances };
}
