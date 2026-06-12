/**
 * Filter registry — one place the UI can iterate over.
 * Each entry has the same signature: (measurements, params) =>
 * { estimates, covariances }.
 */
import { kalmanFilter } from './kalman.js';
import { extendedKalmanFilter } from './ekf.js';
import { unscentedKalmanFilter } from './ukf.js';
import { particleFilter } from './particleFilter.js';

export const FILTERS = [
  { id: 'kf', name: 'Kalman Filter (linearized)', run: kalmanFilter },
  { id: 'ekf', name: 'Extended Kalman Filter', run: extendedKalmanFilter },
  { id: 'ukf', name: 'Unscented Kalman Filter', run: unscentedKalmanFilter },
  { id: 'pf', name: 'Particle Filter (500)', run: particleFilter },
];
