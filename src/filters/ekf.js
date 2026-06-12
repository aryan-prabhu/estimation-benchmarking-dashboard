/**
 * Extended Kalman Filter (EKF) with numerical Jacobians.
 *
 * The EKF handles nonlinear models by re-linearizing at every step around
 * the *current estimate* — a first-order Taylor expansion:
 *
 *   f(x) ≈ f(x̂) + F (x − x̂),   F = ∂f/∂x |_{x̂}
 *   h(x) ≈ h(x̂) + H (x − x̂),   H = ∂h/∂x |_{x̂}
 *
 * The mean is propagated through the *exact* nonlinear functions; only the
 * covariance uses the Jacobians. This makes the EKF a "best effort" Gaussian
 * approximation: it is no longer optimal, and it can diverge when the
 * linearization is poor (high curvature relative to the local uncertainty).
 *
 * On this benchmark the EKF has a specific Achilles' heel: h(x) = x²/20 has
 * h'(0) = 0. Whenever the estimate passes near zero the filter computes a
 * near-zero gain and effectively ignores the measurement — exactly when the
 * sign ambiguity makes the measurement most needed.
 *
 * The Jacobians are computed NUMERICALLY (central differences) rather than
 * analytically. For a scalar state that is overkill, but it mirrors real
 * practice where f comes from a complex simulator with no closed-form
 * derivative.
 */

import { DEFAULTS, f, h } from '../simulation/model.js';

/**
 * Central-difference derivative: g'(x) ≈ [g(x+ε) − g(x−ε)] / 2ε.
 * Central differences are O(ε²)-accurate vs O(ε) for forward differences.
 * ε = √(machine epsilon) · max(|x|, 1) balances truncation error (wants
 * small ε) against floating-point cancellation (wants large ε).
 */
function numericalJacobian(g, x) {
  const eps = Math.sqrt(Number.EPSILON) * Math.max(Math.abs(x), 1);
  return (g(x + eps) - g(x - eps)) / (2 * eps);
}

/**
 * Run the EKF over a measurement sequence.
 *
 * @param {number[]} measurements  z_0 .. z_{N-1}
 * @param {object}   params        { Q, R, P0, xInit }
 * @returns {{ estimates: number[], covariances: number[] }}
 */
export function extendedKalmanFilter(measurements, params = {}) {
  const { Q, R, P0 } = { ...DEFAULTS, ...params };
  const xInit = params.xInit ?? 0;

  let x = xInit; // posterior mean  x̂_{k|k}
  let P = P0; // posterior variance P_{k|k}

  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    // ---- Predict (time update) ----------------------------------------
    if (k > 0) {
      // Jacobian of the dynamics AT THE CURRENT ESTIMATE — this is the
      // step that distinguishes the EKF from the fixed-point linearized KF.
      const F = numericalJacobian((xi) => f(xi, k - 1), x);
      // Mean goes through the exact nonlinearity:
      //   x̂_{k|k-1} = f(x̂_{k-1|k-1}, k-1)
      x = f(x, k - 1);
      // Covariance goes through the linearization:
      //   P_{k|k-1} = F P Fᵀ + Q
      P = F * P * F + Q;
    }

    // ---- Update (measurement update) ----------------------------------
    // Measurement Jacobian at the predicted mean. Note H → 0 as x̂ → 0:
    // the EKF's blind spot on this system.
    const H = numericalJacobian(h, x);
    // Innovation uses the exact h, not its linearization:
    const y = measurements[k] - h(x);
    // Innovation variance and gain, same algebra as the linear KF but with
    // the local Jacobian standing in for the (nonexistent) global H:
    const S = H * P * H + R;
    const K = (P * H) / S;
    x = x + K * y;
    P = (1 - K * H) * P;

    estimates.push(x);
    covariances.push(P);
  }

  return { estimates, covariances };
}
