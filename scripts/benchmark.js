// Headless sanity check: run all four filters on one simulated trajectory
// and print RMSE. Usage: node scripts/benchmark.js [seed]
import { simulate } from '../src/simulation/model.js';
import { rmse } from '../src/simulation/metrics.js';
import { kalmanFilter } from '../src/filters/kalman.js';
import { extendedKalmanFilter } from '../src/filters/ekf.js';
import { unscentedKalmanFilter } from '../src/filters/ukf.js';
import { particleFilter } from '../src/filters/particleFilter.js';

const seed = Number(process.argv[2] ?? 42);
const { trueStates, measurements } = simulate({ seed });

const filters = {
  'Kalman Filter (linearized)': kalmanFilter,
  'Extended Kalman Filter': extendedKalmanFilter,
  'Unscented Kalman Filter': unscentedKalmanFilter,
  'Particle Filter (500)': particleFilter,
};

for (const [name, run] of Object.entries(filters)) {
  const { estimates, covariances } = run(measurements);
  const bad = estimates.some((e) => !Number.isFinite(e)) || covariances.some((c) => !Number.isFinite(c) || c < 0);
  console.log(
    `${name.padEnd(28)} RMSE = ${rmse(estimates, trueStates).toFixed(3)}${bad ? '  ⚠ non-finite or negative covariance!' : ''}`
  );
}
