import { useState } from 'react';

const SYSTEM_PROMPT =
  'You are an estimation theory expert. A student has run a benchmark comparing four ' +
  'filters on a univariate nonlinear system. Analyse the results in 4-6 sentences. ' +
  'Explain WHY each filter performed as it did — reference noise regime, nonlinearity, ' +
  'Jacobian approximation error, sigma point accuracy, and particle degeneracy where ' +
  'relevant. Be specific, use engineering language, do not be generic.';

/**
 * Claude-powered analysis of the current benchmark run. Builds the user
 * message from live simulation state and calls the Anthropic Messages API
 * directly from the browser.
 */
export default function AnalysisPanel({ Q, R, seed, results }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyse = async () => {
    setLoading(true);
    setError(null);
    setAnalysis(null);

    const rmseOf = (id) => results.find((r) => r.id === id).rmse.toFixed(3);
    const userMessage =
      `System: x' = 0.5x + 25x/(1+x²) + 8cos(1.2k), z = x²/20. ` +
      `Q=${Q}, R=${R}, seed=${seed}. ` +
      `Results — KF RMSE: ${rmseOf('kf')}, EKF RMSE: ${rmseOf('ekf')}, ` +
      `UKF RMSE: ${rmseOf('ukf')}, PF(500) RMSE: ${rmseOf('pf')}. ` +
      `Q/R ratio: ${(Q / R).toFixed(2)}.`;

    try {
      const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'anthropic-version': '2023-06-01',
          // Required for direct browser (CORS) access to the API.
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-6',
          max_tokens: 400,
          system: SYSTEM_PROMPT,
          messages: [{ role: 'user', content: userMessage }],
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error?.message ?? `API error ${res.status}`);
      }

      const data = await res.json();
      const text = data.content
        .filter((block) => block.type === 'text')
        .map((block) => block.text)
        .join('');
      setAnalysis(text);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="analysis">
      <h2>AI Analysis</h2>
      <button className="analyse-btn" onClick={analyse} disabled={loading}>
        {loading ? (
          <>
            <span className="spinner" aria-hidden="true" /> Analysing…
          </>
        ) : (
          'Analyse with AI'
        )}
      </button>

      {error && <div className="analysis-card error">Error: {error}</div>}
      {analysis && <div className="analysis-card">{analysis}</div>}
    </section>
  );
}
