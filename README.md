# State Estimation Benchmarking Dashboard

An interactive React dashboard that benchmarks four Bayesian state estimation
filters on the **Univariate Nonlinear Growth Model (UNGM)** — the classic
stress test introduced by Gordon, Salmond & Smith (1993) to motivate the
particle filter. All filters are implemented from scratch in JavaScript with
the math commented at each step.

## System

State transition (k is the time index):

```
x_{k+1} = 0.5·x_k + 25·x_k/(1 + x_k²) + 8·cos(1.2k) + w_k,   w_k ~ N(0, Q)
```

Observation:

```
z_k = x_k²/20 + v_k,   v_k ~ N(0, R)
```

50 timesteps; Q, R, and the noise seed are configurable in the UI. The squared
measurement cannot distinguish +x from −x, so the true posterior is often
bimodal — which Gaussian filters fundamentally cannot represent.

## Filters

| Filter | Approach |
|---|---|
| **KF** (linearized) | Linear Kalman filter, linearized once around a fixed nominal point x = 5 — the deliberately mismatched baseline. |
| **EKF** | Extended Kalman filter; re-linearizes each step using central-difference numerical Jacobians. |
| **UKF** | Unscented Kalman filter; scaled sigma-point transform (α=1, β=2, κ=0), no Jacobians needed. |
| **PF** | Particle filter; 500 particles, SIR/bootstrap with log-weights and systematic resampling. |

## Results (Q = 1, R = 1, seed = 42)

| Filter | RMSE |
|---|---|
| Kalman Filter (linearized) | 9.720 |
| Extended Kalman Filter | 6.846 |
| Unscented Kalman Filter | 7.310 |
| **Particle Filter (500)** | **3.762** |

The PF wins decisively because it is the only filter that can represent the
bimodal ±x posterior induced by the quadratic measurement.

## Running

```sh
npm install
npm run dev
```

Then open http://localhost:5173/. A headless benchmark is also available:
`node scripts/benchmark.js [seed]`.

## Structure

Two equivalent implementations:

- **`src/StateEstimationDashboard.jsx`** — standalone single-file artifact
  (only `react` + `recharts` as dependencies); currently the app entry point.
- **`src/App.jsx`** + `src/filters/`, `src/simulation/`, `src/components/` —
  the modular version, with each filter as a separate documented module.

---

*Part of an AI engineering portfolio targeting estimation theory research.*

