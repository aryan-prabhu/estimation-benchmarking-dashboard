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

export const FILTER_COLORS = {
  kf: '#f59e0b', // amber
  ekf: '#ef4444', // red
  ukf: '#3b82f6', // blue
  pf: '#a855f7', // purple
};

const TRUTH_COLOR = '#000000';

/**
 * Trajectory plot: ground truth + each filter's estimate with a shaded
 * ±2σ credible band (estimate ± 2√P). For a Gaussian posterior that band
 * should contain the true state ~95% of the time — watching the truth
 * escape a filter's band is the quickest way to spot overconfidence.
 *
 * @param {number[]} trueStates
 * @param {Array<{id, name, estimates, covariances}>} results
 */
export default function TrajectoryChart({ trueStates, results }) {
  // Which filters are drawn. Four overlapping bands get busy, so each
  // filter can be toggled off to study the others.
  const [visible, setVisible] = useState(() =>
    Object.fromEntries(results.map((r) => [r.id, true]))
  );

  // One row per timestep. Recharts renders a *range* Area when its dataKey
  // yields a [low, high] pair — that gives the band in a single series.
  const data = useMemo(
    () =>
      trueStates.map((truth, k) => {
        const row = { k, truth };
        for (const r of results) {
          const x = r.estimates[k];
          // Clamp: covariance can dip epsilon-negative from floating-point
          // roundoff, and sqrt(NaN) would break the band.
          const sigma = Math.sqrt(Math.max(r.covariances[k], 0));
          row[r.id] = x;
          row[`${r.id}Band`] = [x - 2 * sigma, x + 2 * sigma];
        }
        return row;
      }),
    [trueStates, results]
  );

  const toggle = (id) => setVisible((v) => ({ ...v, [id]: !v[id] }));

  return (
    <div className="chart-block">
      <div className="chart-toggles">
        <span className="toggle truth-toggle">
          <span className="swatch" style={{ background: TRUTH_COLOR }} />
          Ground truth
        </span>
        {results.map((r) => (
          <label key={r.id} className="toggle">
            <input
              type="checkbox"
              checked={visible[r.id]}
              onChange={() => toggle(r.id)}
            />
            <span className="swatch" style={{ background: FILTER_COLORS[r.id] }} />
            {r.name}
          </label>
        ))}
      </div>

      {/* White surface guarantees the black truth line is visible whether
          the page renders in light or dark mode. */}
      <div className="chart-surface">
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

          {/* Bands first so every line draws on top of every band. */}
          {results.map(
            (r) =>
              visible[r.id] && (
                <Area
                  key={`${r.id}-band`}
                  name={`${r.name} ±2σ`}
                  dataKey={`${r.id}Band`}
                  fill={FILTER_COLORS[r.id]}
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
                  stroke={FILTER_COLORS[r.id]}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              )
          )}

          {/* Ground truth renders LAST so it sits above every band and
              filter line (SVG paints in document order — no z-index).
              It has no toggle: it's the reference everything is judged
              against, so it must always be on screen. */}
          <Line
            name="Ground truth"
            dataKey="truth"
            stroke={TRUTH_COLOR}
            strokeWidth={3}
            strokeDasharray="8 4"
            dot={false}
            isAnimationActive={false}
          />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
