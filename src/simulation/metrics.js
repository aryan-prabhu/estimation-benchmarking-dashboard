/**
 * Performance metrics for comparing filters against ground truth.
 */

/**
 * Root mean squared error between a filter's estimates and the true states:
 *
 *   RMSE = sqrt( (1/N) Σ_k (x̂_k − x_k)² )
 *
 * Lower is better. On the UNGM benchmark expect roughly:
 * PF < UKF ≤ EKF << linear KF.
 */
export function rmse(estimates, trueStates) {
  const n = Math.min(estimates.length, trueStates.length);
  let sumSq = 0;
  for (let k = 0; k < n; k++) {
    const e = estimates[k] - trueStates[k];
    sumSq += e * e;
  }
  return Math.sqrt(sumSq / n);
}
