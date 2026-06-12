/**
 * Seeded random number generation.
 *
 * A fixed seed makes every benchmark run reproducible: all filters see the
 * exact same noise realization, so RMSE differences reflect the filters
 * themselves, not luck of the draw.
 */

/**
 * mulberry32 — a small, fast 32-bit PRNG. Returns a function producing
 * uniform samples in [0, 1).
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Wrap a uniform RNG into a standard-normal sampler via the Box–Muller
 * transform:
 *
 *   z = sqrt(-2 ln u1) * cos(2π u2)  ~  N(0, 1)
 *
 * To sample N(0, σ²), multiply the result by σ = sqrt(variance).
 */
export function makeGaussian(uniform) {
  return function gaussian() {
    // Guard against ln(0).
    let u1 = uniform();
    while (u1 === 0) u1 = uniform();
    const u2 = uniform();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

/** Convenience: seeded N(0,1) sampler. */
export function gaussianRng(seed) {
  return makeGaussian(mulberry32(seed));
}
