/**
 * The benchmark system: the Univariate Nonlinear Growth Model (UNGM).
 *
 * This is a classic stress test for nonlinear filters (used by Gordon et al.
 * 1993 to introduce the particle filter, and by virtually every UKF/PF paper
 * since). It is hard for two reasons:
 *
 *  1. The dynamics 25x/(1+x²) are strongly nonlinear and push the state
 *     back and forth across zero.
 *
 *  2. The measurement z = x²/20 is *even* in x — a measurement cannot tell
 *     +x from -x. The true posterior is therefore often bimodal, which a
 *     Gaussian filter (KF/EKF/UKF) fundamentally cannot represent. This is
 *     where the particle filter earns its keep.
 *
 * State transition (k is the time index, passed in because of the cos term):
 *   x_{k+1} = 0.5 x_k + 25 x_k / (1 + x_k²) + 8 cos(1.2 k) + w_k,  w_k ~ N(0, Q)
 *
 * Observation:
 *   z_k = x_k² / 20 + v_k,  v_k ~ N(0, R)
 */

import { gaussianRng } from './random.js';

/** Deterministic part of the state transition, x_{k+1} = f(x_k, k). */
export function f(x, k) {
  return 0.5 * x + (25 * x) / (1 + x * x) + 8 * Math.cos(1.2 * k);
}

/** Deterministic part of the observation, z_k = h(x_k). */
export function h(x) {
  return (x * x) / 20;
}

export const DEFAULTS = {
  Q: 1.0, // process noise variance
  R: 1.0, // measurement noise variance
  steps: 50, // number of timesteps
  x0: 0.1, // true initial state
  P0: 2.0, // filters' initial covariance (their uncertainty about x0)
  seed: 42,
};

/**
 * Simulate one realization of the system.
 *
 * Returns the ground-truth trajectory and the noisy measurements the filters
 * will see. Convention: trueStates[k] is x_k for k = 0..steps-1, and
 * measurements[k] = h(x_k) + v_k is the measurement of that same state.
 * (x_0 itself is measured; the recursion generates x_1..x_{steps-1}.)
 */
export function simulate(params = {}) {
  const { Q, R, steps, x0, seed } = { ...DEFAULTS, ...params };
  const randn = gaussianRng(seed);

  const trueStates = [x0];
  for (let k = 0; k < steps - 1; k++) {
    // w_k ~ N(0, Q): scale a standard normal by the standard deviation √Q.
    trueStates.push(f(trueStates[k], k) + Math.sqrt(Q) * randn());
  }

  const measurements = trueStates.map((x) => h(x) + Math.sqrt(R) * randn());

  return { trueStates, measurements };
}
