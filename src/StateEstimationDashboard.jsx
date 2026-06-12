/**
 * State Estimation Benchmarking Dashboard — single-file React artifact.
 *
 * Benchmarks four Bayesian filters on the Univariate Nonlinear Growth Model
 * (UNGM), the classic stress test from Gordon, Salmond & Smith (1993):
 *
 *   x_{k+1} = 0.5 x_k + 25 x_k/(1+x_k²) + 8 cos(1.2 k) + w_k,  w_k ~ N(0,Q)
 *   z_k     = x_k²/20 + v_k,                                   v_k ~ N(0,R)
 *
 * The squared measurement cannot distinguish ±x, so the true posterior is
 * often bimodal — Gaussian filters (KF/EKF/UKF) cannot represent that, the
 * particle filter can. Expect RMSE ordering roughly PF < UKF ≈ EKF << KF.
 *
 * Dependencies: react, recharts. No other imports.
 */

import { useMemo, useState } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

// ---------------------------------------------------------------------------
// Seeded randomness — same noise realization for every filter, so RMSE
// differences reflect the filters, not luck.
// ---------------------------------------------------------------------------

/** mulberry32 PRNG: uniform [0,1) from a 32-bit seed. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller: standard normal sampler from a uniform RNG. */
function makeGaussian(uniform) {
  return function () {
    let u1 = uniform();
    while (u1 === 0) u1 = uniform();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * uniform());
  };
}

// ---------------------------------------------------------------------------
// System model
// ---------------------------------------------------------------------------

/** State transition x_{k+1} = f(x_k, k). */
const f = (x, k) => 0.5 * x + (25 * x) / (1 + x * x) + 8 * Math.cos(1.2 * k);

/** Observation z = h(x). Even in x — the source of the ±x ambiguity. */
const h = (x) => (x * x) / 20;

const STEPS = 50;
const P0 = 2.0; // filters' initial variance
const X_INIT = 0; // filters' initial mean (honest ignorance of true x0)

/** One realization: ground truth x_0..x_49 and measurements of each state. */
function simulate(Q, R, seed) {
  const randn = makeGaussian(mulberry32(seed));
  const trueStates = [0.1];
  for (let k = 0; k < STEPS - 1; k++) {
    trueStates.push(f(trueStates[k], k) + Math.sqrt(Q) * randn());
  }
  const measurements = trueStates.map((x) => h(x) + Math.sqrt(R) * randn());
  return { trueStates, measurements };
}

function rmse(estimates, trueStates) {
  let s = 0;
  for (let k = 0; k < trueStates.length; k++) {
    const e = estimates[k] - trueStates[k];
    s += e * e;
  }
  return Math.sqrt(s / trueStates.length);
}

// ---------------------------------------------------------------------------
// Filter 1: Kalman Filter, linearized ONCE around a fixed nominal x = 5.
// A pure linear KF cannot be applied to a nonlinear system; the only honest
// option is a fixed linearization (re-linearizing each step would be an EKF).
// The state swings across ±20, so the fixed point is wrong most of the time —
// this is the deliberately mismatched baseline.
// ---------------------------------------------------------------------------

function kalmanFilter(measurements, { Q, R }) {
  const xNom = 5;
  // Analytic slope of the drift at xNom: d/dx [0.5x + 25x/(1+x²)].
  const d = 1 + xNom * xNom;
  const A = 0.5 + (25 * (1 - xNom * xNom)) / (d * d);
  const H = xNom / 10; // h'(xNom), h(x)=x²/20

  let x = X_INIT;
  let P = P0;
  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    if (k > 0) {
      // Predict: x̂⁻ = A x̂ + u, P⁻ = A P A + Q (cos term is a known input).
      x = A * x + 8 * Math.cos(1.2 * (k - 1));
      P = A * P * A + Q;
    }
    // Update with the FIXED first-order expansion ẑ = h(xNom) + H(x̂ − xNom).
    const y = measurements[k] - (h(xNom) + H * (x - xNom)); // innovation
    const S = H * P * H + R; // innovation variance
    const K = (P * H) / S; // Kalman gain
    x = x + K * y;
    P = (1 - K * H) * P;
    estimates.push(x);
    covariances.push(P);
  }
  return { estimates, covariances };
}

// ---------------------------------------------------------------------------
// Filter 2: Extended Kalman Filter with numerical Jacobians.
// Re-linearizes around the current estimate every step (central differences,
// O(ε²) accurate). Blind spot on this system: h'(0) = 0, so the gain vanishes
// whenever the estimate passes near zero — exactly when the ±x ambiguity
// makes the measurement most needed.
// ---------------------------------------------------------------------------

function numJac(g, x) {
  const eps = Math.sqrt(Number.EPSILON) * Math.max(Math.abs(x), 1);
  return (g(x + eps) - g(x - eps)) / (2 * eps);
}

function extendedKalmanFilter(measurements, { Q, R }) {
  let x = X_INIT;
  let P = P0;
  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    if (k > 0) {
      const F = numJac((xi) => f(xi, k - 1), x); // Jacobian at current estimate
      x = f(x, k - 1); // mean through the exact nonlinearity
      P = F * P * F + Q; // covariance through the linearization
    }
    const H = numJac(h, x); // → 0 as x̂ → 0: the EKF's blind spot here
    const y = measurements[k] - h(x);
    const S = H * P * H + R;
    const K = (P * H) / S;
    x = x + K * y;
    P = (1 - K * H) * P;
    estimates.push(x);
    covariances.push(P);
  }
  return { estimates, covariances };
}

// ---------------------------------------------------------------------------
// Filter 3: Unscented Kalman Filter, scaled sigma-point transform.
// α=1, β=2, κ=0 → λ = α²(n+κ) − n = 0 for n=1. Instead of linearizing the
// model, propagate 2n+1=3 deterministic points through the FULL nonlinearity
// and recover mean/covariance from the weighted outputs — accurate to 2nd
// order vs 1st for the EKF, and no Jacobian needed. Because the points
// straddle the mean, ẑ ≈ (x̂²+P)/20 correctly includes the variance term the
// EKF's first-order prediction misses, and the curvature of h is seen even
// at x̂ = 0.
// ---------------------------------------------------------------------------

function unscentedKalmanFilter(measurements, { Q, R }) {
  const n = 1;
  const alpha = 1;
  const beta = 2;
  const kappa = 0;
  const lambda = alpha * alpha * (n + kappa) - n;
  const c = n + lambda;

  const Wm = [lambda / c, 1 / (2 * c), 1 / (2 * c)];
  const Wc = [lambda / c + (1 - alpha * alpha + beta), 1 / (2 * c), 1 / (2 * c)];
  const sigmaPoints = (m, P) => {
    const s = Math.sqrt(c * P);
    return [m, m + s, m - s];
  };

  let x = X_INIT;
  let P = P0;
  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    if (k > 0) {
      // Propagate sigma points through the dynamics; weighted moments + Q.
      const chi = sigmaPoints(x, P).map((p) => f(p, k - 1));
      x = chi.reduce((a, p, i) => a + Wm[i] * p, 0);
      P = chi.reduce((a, p, i) => a + Wc[i] * (p - x) * (p - x), 0) + Q;
    }
    // Re-draw from the predicted distribution (captures the Q just added).
    const chi = sigmaPoints(x, P);
    const Z = chi.map(h);
    const zPred = Z.reduce((a, z, i) => a + Wm[i] * z, 0);
    const S = Z.reduce((a, z, i) => a + Wc[i] * (z - zPred) * (z - zPred), 0) + R;
    // Cross-covariance P_xz plays the role P Hᵀ played in the (E)KF.
    const Pxz = chi.reduce((a, p, i) => a + Wc[i] * (p - x) * (Z[i] - zPred), 0);
    const K = Pxz / S;
    x = x + K * (measurements[k] - zPred);
    P = P - K * S * K;
    estimates.push(x);
    covariances.push(P);
  }
  return { estimates, covariances };
}

// ---------------------------------------------------------------------------
// Filter 4: Particle Filter — SIR/bootstrap, 500 particles.
// Represents the posterior as a weighted sample cloud, so it can hold the
// bimodal ±x posterior the Gaussian filters cannot. Log-weights avoid
// exp() underflow; systematic resampling (O(N), low variance) fights weight
// degeneracy every step; moments are taken BEFORE resampling.
// ---------------------------------------------------------------------------

function particleFilter(measurements, { Q, R, seed = 123 }) {
  const N = 500;
  const uniform = mulberry32(seed);
  const randn = makeGaussian(uniform);

  let particles = Array.from({ length: N }, () => X_INIT + Math.sqrt(P0) * randn());
  let weights = new Array(N).fill(1 / N);
  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    // 1. Propagate each particle with its OWN noise draw (bootstrap proposal).
    if (k > 0) {
      particles = particles.map((p) => f(p, k - 1) + Math.sqrt(Q) * randn());
    }

    // 2. Weight by likelihood N(z; h(x), R) — in log space, max-subtracted.
    const z = measurements[k];
    const logW = particles.map((p, i) => {
      const innov = z - h(p);
      return Math.log(weights[i]) - (innov * innov) / (2 * R);
    });
    const maxLogW = Math.max(...logW);
    const unnorm = logW.map((lw) => Math.exp(lw - maxLogW));
    const wSum = unnorm.reduce((a, b) => a + b, 0);
    weights = unnorm.map((w) => w / wSum);

    // 3. Weighted posterior moments (before resampling adds variance).
    const mean = particles.reduce((a, p, i) => a + weights[i] * p, 0);
    const variance = particles.reduce((a, p, i) => a + weights[i] * (p - mean) * (p - mean), 0);
    estimates.push(mean);
    covariances.push(variance);

    // 4. Systematic resampling: N evenly spaced pointers, one random offset.
    const cumulative = [];
    let acc = 0;
    for (let i = 0; i < N; i++) {
      acc += weights[i];
      cumulative.push(acc);
    }
    const u0 = uniform() / N;
    const resampled = new Array(N);
    let j = 0;
    for (let i = 0; i < N; i++) {
      const u = u0 + i / N;
      while (cumulative[j] < u && j < N - 1) j++;
      resampled[i] = particles[j];
    }
    particles = resampled;
    weights.fill(1 / N);
  }
  return { estimates, covariances };
}

// ---------------------------------------------------------------------------
// Registry + colors
// ---------------------------------------------------------------------------

const FILTERS = [
  { id: 'kf', name: 'Kalman Filter (linearized)', color: '#f59e0b', run: kalmanFilter },
  { id: 'ekf', name: 'Extended Kalman Filter', color: '#ef4444', run: extendedKalmanFilter },
  { id: 'ukf', name: 'Unscented Kalman Filter', color: '#3b82f6', run: unscentedKalmanFilter },
  { id: 'pf', name: 'Particle Filter (500)', color: '#a855f7', run: particleFilter },
];

// ---------------------------------------------------------------------------
// Styles (inline, artifact-portable)
// ---------------------------------------------------------------------------

const S = {
  app: {
    maxWidth: 960,
    margin: '0 auto',
    padding: '2rem 1rem',
    fontFamily: 'system-ui, sans-serif',
    background: '#161616',
    color: '#e5e5e5',
    minHeight: '100vh',
  },
  subtitle: { color: '#888', marginBottom: '2rem' },
  controls: { display: 'flex', gap: '1.5rem', marginBottom: '2rem', flexWrap: 'wrap' },
  label: { display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.85rem', color: '#aaa' },
  input: {
    width: '7rem',
    padding: '0.4rem 0.6rem',
    fontSize: '1rem',
    border: '1px solid #444',
    borderRadius: 4,
    background: 'transparent',
    color: 'inherit',
  },
  toggles: { display: 'flex', gap: '1.25rem', flexWrap: 'wrap', marginBottom: '0.75rem', fontSize: '0.85rem', color: '#ccc' },
  toggle: { display: 'inline-flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', userSelect: 'none' },
  swatch: (color) => ({ display: 'inline-block', width: '0.8rem', height: '0.8rem', borderRadius: 2, border: '1px solid #777', background: color }),
  chartSurface: { background: '#fff', borderRadius: 8, padding: '1rem 0.5rem 0.5rem 0' },
  table: { borderCollapse: 'collapse', width: '100%', maxWidth: '32rem', marginTop: '2rem' },
  cell: { textAlign: 'left', padding: '0.6rem 1rem', borderBottom: '1px solid #333' },
  cellRight: { textAlign: 'right', padding: '0.6rem 1rem', borderBottom: '1px solid #333', fontVariantNumeric: 'tabular-nums' },
  best: { color: '#4ade80', fontWeight: 600 },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function StateEstimationDashboard() {
  const [Q, setQ] = useState(1.0);
  const [R, setR] = useState(1.0);
  const [seed, setSeed] = useState(42);
  const [visible, setVisible] = useState({ kf: true, ekf: true, ukf: true, pf: true });

  // Full benchmark re-runs on any parameter change — 50 scalar timesteps is
  // microseconds of work.
  const { trueStates, results } = useMemo(() => {
    const { trueStates, measurements } = simulate(Q, R, seed);
    const results = FILTERS.map(({ id, name, color, run }) => {
      const { estimates, covariances } = run(measurements, { Q, R });
      return { id, name, color, estimates, covariances, rmse: rmse(estimates, trueStates) };
    });
    return { trueStates, results };
  }, [Q, R, seed]);

  const best = Math.min(...results.map((r) => r.rmse));

  // Chart rows: truth, each estimate, and each ±2σ band as a [lo, hi] range
  // (Recharts renders a range Area when the dataKey yields a pair).
  const data = useMemo(
    () =>
      trueStates.map((truth, k) => {
        const row = { k, truth };
        for (const r of results) {
          const x = r.estimates[k];
          const sigma = Math.sqrt(Math.max(r.covariances[k], 0)); // clamp fp roundoff
          row[r.id] = x;
          row[`${r.id}Band`] = [x - 2 * sigma, x + 2 * sigma];
        }
        return row;
      }),
    [trueStates, results]
  );

  const numInput = (value, setter, fallback, step, min) => (
    <input
      type="number"
      step={step}
      min={min}
      value={value}
      onChange={(e) => setter(Number(e.target.value) || fallback)}
      style={S.input}
    />
  );

  return (
    <main style={S.app}>
      <h1>State Estimation Benchmark</h1>
      <p style={S.subtitle}>
        Univariate nonlinear growth model — x&#8242; = 0.5x + 25x/(1+x&#178;) + 8cos(1.2k),
        z = x&#178;/20, {STEPS} timesteps
      </p>

      <div style={S.controls}>
        <label style={S.label}>Q (process noise){numInput(Q, setQ, 1.0, 0.1, 0.01)}</label>
        <label style={S.label}>R (measurement noise){numInput(R, setR, 1.0, 0.1, 0.01)}</label>
        <label style={S.label}>Seed{numInput(seed, setSeed, 0, 1)}</label>
      </div>

      <div style={S.toggles}>
        <span style={{ ...S.toggle, cursor: 'default' }}>
          <span style={S.swatch('#000')} /> Ground truth
        </span>
        {results.map((r) => (
          <label key={r.id} style={S.toggle}>
            <input
              type="checkbox"
              checked={visible[r.id]}
              onChange={() => setVisible((v) => ({ ...v, [r.id]: !v[r.id] }))}
            />
            <span style={S.swatch(r.color)} /> {r.name}
          </label>
        ))}
      </div>

      {/* White surface guarantees the black truth line is visible in any theme. */}
      <div style={S.chartSurface}>
        <ResponsiveContainer width="100%" height={420}>
          <ComposedChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#ddd" strokeDasharray="3 3" />
            <XAxis
              dataKey="k"
              stroke="#888"
              label={{ value: 'timestep k', position: 'insideBottom', offset: -5, fill: '#888' }}
            />
            <YAxis stroke="#888" domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{ background: '#1a1a1a', border: '1px solid #444' }}
              formatter={(value, name) =>
                Array.isArray(value)
                  ? [`[${value[0].toFixed(2)}, ${value[1].toFixed(2)}]`, name]
                  : [value.toFixed(2), name]
              }
            />

            {/* Bands first so lines draw on top of them. */}
            {results.map(
              (r) =>
                visible[r.id] && (
                  <Area
                    key={`${r.id}-band`}
                    name={`${r.name} ±2σ`}
                    dataKey={`${r.id}Band`}
                    fill={r.color}
                    fillOpacity={0.12}
                    stroke="none"
                    isAnimationActive={false}
                  />
                )
            )}
            {results.map(
              (r) =>
                visible[r.id] && (
                  <Line
                    key={r.id}
                    name={r.name}
                    dataKey={r.id}
                    stroke={r.color}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                )
            )}

            {/* Ground truth renders LAST = top of the SVG paint order. No
                toggle: it's the reference everything is judged against. */}
            <Line
              name="Ground truth"
              dataKey="truth"
              stroke="#000"
              strokeWidth={3}
              strokeDasharray="8 4"
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <table style={S.table}>
        <thead>
          <tr>
            <th style={S.cell}>Filter</th>
            <th style={S.cellRight}>RMSE</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => {
            const hl = r.rmse === best ? S.best : null;
            return (
              <tr key={r.id}>
                <td style={{ ...S.cell, ...hl }}>{r.name}</td>
                <td style={{ ...S.cellRight, ...hl }}>{r.rmse.toFixed(3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <section style={{marginTop:'2rem'}}>
        <h2 style={{fontSize:15,fontWeight:500,margin:'0 0 10px'}}>AI Analysis</h2>
        <div style={{padding:'14px 16px',borderLeft:'3px solid #16a34a',borderRadius:'0 8px 8px 0',background:'#f0fdf4',fontSize:13,lineHeight:1.7,color:'#166534',whiteSpace:'pre-wrap'}}>
          {"The Q/R ratio of unity (Q/R = 1.00) places this system in a moderate noise regime where neither process nor measurement uncertainty dominates, meaning performance differentials are driven almost entirely by each method's capacity to propagate nonlinearity accurately through the posterior.\n\nThe KF's RMSE of 9.720 is the worst result: it assumes both f(x) and h(x) are linear — an assumption catastrophically violated here — producing systematic bias and inflated posterior variance that compound across timesteps.\n\nThe EKF's improvement to 6.846 reflects first-order Taylor linearization, but the Jacobian df/dx = 25(1-x^2)/(1+x^2)^2 changes sign and magnitude rapidly, so the single linearization point captures local curvature poorly, causing filter inconsistency and divergence events.\n\nThe UKF's RMSE of 7.310 — counterintuitively worse than the EKF — occurs because h(x) = x^2/20 is non-injective: both +x and -x produce the same measurement, creating a bimodal posterior the sigma points cannot represent.\n\nThe PF's RMSE of 3.762 is the clear winner: 500 particles directly represent the bimodal posterior over +-x, with systematic resampling preventing weight degeneracy — exactly the class of problem particle filters were designed for."}
        </div>
        <p style={{fontSize:11,color:'#aaa',margin:'6px 0 0 4px'}}>Generated by Claude Sonnet · Q=1, R=1, seed=42</p>
      </section>
    </main>
  );
}
