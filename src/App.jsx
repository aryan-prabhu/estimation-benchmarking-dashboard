import { useMemo, useState } from 'react';
import { simulate, DEFAULTS } from './simulation/model.js';
import { rmse } from './simulation/metrics.js';
import { FILTERS } from './filters/index.js';
import TrajectoryChart from './components/TrajectoryChart.jsx';
import AnalysisPanel from './components/AnalysisPanel.jsx';
import './App.css';

export default function App() {
  const [Q, setQ] = useState(DEFAULTS.Q);
  const [R, setR] = useState(DEFAULTS.R);
  const [seed, setSeed] = useState(DEFAULTS.seed);

  // Re-run the whole benchmark whenever a parameter changes. 50 steps of
  // scalar filtering is microseconds of work, so useMemo is plenty.
  const { trueStates, results } = useMemo(() => {
    const { trueStates, measurements } = simulate({ Q, R, seed });
    const results = FILTERS.map(({ id, name, run }) => {
      const { estimates, covariances } = run(measurements, { Q, R });
      return { id, name, estimates, covariances, rmse: rmse(estimates, trueStates) };
    });
    return { trueStates, results };
  }, [Q, R, seed]);

  const best = Math.min(...results.map((r) => r.rmse));

  return (
    <main className="app">
      <h1>State Estimation Benchmark</h1>
      <p className="subtitle">
        Univariate nonlinear growth model — x&#8242; = 0.5x + 25x/(1+x&#178;) +
        8cos(1.2k), z = x&#178;/20, {DEFAULTS.steps} timesteps
      </p>

      <div className="controls">
        <label>
          Q (process noise)
          <input
            type="number"
            step="0.1"
            min="0.01"
            value={Q}
            onChange={(e) => setQ(Number(e.target.value) || DEFAULTS.Q)}
          />
        </label>
        <label>
          R (measurement noise)
          <input
            type="number"
            step="0.1"
            min="0.01"
            value={R}
            onChange={(e) => setR(Number(e.target.value) || DEFAULTS.R)}
          />
        </label>
        <label>
          Seed
          <input
            type="number"
            step="1"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
          />
        </label>
      </div>

      <TrajectoryChart trueStates={trueStates} results={results} />

      <table className="results">
        <thead>
          <tr>
            <th>Filter</th>
            <th>RMSE</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.id} className={r.rmse === best ? 'best' : ''}>
              <td>{r.name}</td>
              <td>{r.rmse.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <AnalysisPanel Q={Q} R={R} seed={seed} results={results} />
    </main>
  );
}
