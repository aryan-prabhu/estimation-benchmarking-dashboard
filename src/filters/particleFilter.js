/**
 * Particle Filter — Sequential Importance Resampling (SIR), the "bootstrap
 * filter" of Gordon, Salmond & Smith (1993), introduced on exactly this
 * benchmark system.
 *
 * All the Kalman-family filters above force the posterior to be Gaussian.
 * The particle filter drops that assumption entirely: it represents the
 * posterior p(x_k | z_{1:k}) by a cloud of N weighted samples,
 *
 *   p(x_k | z_{1:k}) ≈ Σᵢ wᵢ δ(x − xᵢ)
 *
 * which can be bimodal, skewed — anything. That matters here because the
 * measurement x²/20 can't distinguish ±x, so the true posterior really IS
 * bimodal at times, and the PF is the only filter in this benchmark that can
 * represent it.
 *
 * Each step of the SIR/bootstrap filter:
 *
 *   1. PROPAGATE (importance sampling): draw each particle's next state from
 *      the transition prior, xᵢ ~ p(x_k | xᵢ_{k-1}) = N(f(xᵢ), Q). Using the
 *      prior as the proposal makes the weight update collapse to step 2.
 *   2. WEIGHT: wᵢ ∝ wᵢ · p(z_k | xᵢ) — the measurement likelihood
 *      N(z_k; h(xᵢ), R). Particles that explain the measurement survive.
 *   3. NORMALIZE: Σ wᵢ = 1.
 *   4. RESAMPLE: redraw N particles in proportion to their weights and reset
 *      all weights to 1/N. This kills near-zero-weight particles and clones
 *      good ones, fighting *weight degeneracy* (without it, after a few steps
 *      one particle carries all the weight and the rest are wasted).
 *
 * We resample every step (the bootstrap convention). A common refinement is
 * to resample only when the effective sample size N_eff = 1/Σwᵢ² drops below
 * a threshold (e.g. N/2), trading degeneracy against *sample
 * impoverishment* (loss of diversity from repeated cloning).
 */

import { DEFAULTS, f, h } from '../simulation/model.js';
import { mulberry32, makeGaussian } from '../simulation/random.js';

/**
 * Run the SIR particle filter over a measurement sequence.
 *
 * @param {number[]} measurements  z_0 .. z_{N-1}
 * @param {object}   params        { Q, R, P0, xInit, numParticles, seed }
 * @returns {{ estimates: number[], covariances: number[] }}
 */
export function particleFilter(measurements, params = {}) {
  const { Q, R, P0 } = { ...DEFAULTS, ...params };
  const xInit = params.xInit ?? 0;
  const N = params.numParticles ?? 500;
  // The PF is the one stochastic filter here, so it gets its own seeded RNG
  // (independent of the simulation's) to stay reproducible.
  const uniform = mulberry32(params.seed ?? 123);
  const randn = makeGaussian(uniform);

  // Initialize the cloud from the prior N(xInit, P0), uniform weights.
  let particles = Array.from({ length: N }, () => xInit + Math.sqrt(P0) * randn());
  let weights = new Array(N).fill(1 / N);

  const estimates = [];
  const covariances = [];

  for (let k = 0; k < measurements.length; k++) {
    // ---- 1. Propagate through the dynamics (skip at k=0: the first
    // measurement observes the initial state, which the prior cloud already
    // represents). Each particle gets its OWN process-noise draw — that
    // diversity is what lets the cloud cover multiple hypotheses.
    if (k > 0) {
      particles = particles.map((p) => f(p, k - 1) + Math.sqrt(Q) * randn());
    }

    // ---- 2. Weight by the measurement likelihood.
    // p(z | x) = N(z; h(x), R) ∝ exp(−(z − h(x))² / 2R).
    // Work with log-weights and subtract the max before exponentiating —
    // otherwise exp() underflows to 0 for all particles when the cloud is
    // far from the measurement, and normalization divides by zero.
    const z = measurements[k];
    const logW = particles.map((p, i) => {
      const innov = z - h(p);
      return Math.log(weights[i]) - (innov * innov) / (2 * R);
    });
    const maxLogW = Math.max(...logW);
    let unnorm = logW.map((lw) => Math.exp(lw - maxLogW));

    // ---- 3. Normalize so the weights form a probability distribution.
    const wSum = unnorm.reduce((a, b) => a + b, 0);
    weights = unnorm.map((w) => w / wSum);

    // ---- Estimate BEFORE resampling (resampling adds variance, so the
    // weighted moments are the better estimate).
    // Posterior mean: x̂ = Σ wᵢ xᵢ.
    const mean = particles.reduce((acc, p, i) => acc + weights[i] * p, 0);
    // Posterior variance: P = Σ wᵢ (xᵢ − x̂)². NOTE: when the posterior is
    // bimodal (±x ambiguity), this mean can sit between the modes where
    // there is no probability mass — a limitation of summarizing any
    // posterior by one number, not of the PF itself.
    const variance = particles.reduce(
      (acc, p, i) => acc + weights[i] * (p - mean) * (p - mean),
      0
    );
    estimates.push(mean);
    covariances.push(variance);

    // ---- 4. Systematic resampling.
    // Lay the cumulative weights on [0,1) and sweep N evenly spaced pointers
    // u_j = (u0 + j)/N with one random offset u0 ~ U[0,1). Compared to
    // drawing N independent uniforms (multinomial resampling), this has
    // lower variance and costs O(N) instead of O(N log N).
    const cumulative = new Array(N);
    let acc = 0;
    for (let i = 0; i < N; i++) {
      acc += weights[i];
      cumulative[i] = acc;
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
    weights.fill(1 / N); // resampled cloud is equally weighted again
  }

  return { estimates, covariances };
}
