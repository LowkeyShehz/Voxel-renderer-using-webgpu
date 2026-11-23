/** Small statistics helpers shared by the benchmark code. All ignore non-numeric entries. */

const nums = (values) => values.filter((x) => typeof x === 'number' && Number.isFinite(x));

export function mean(values) {
  const v = nums(values);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

/** Linear-interpolated percentile, p in [0, 100]. */
export function percentile(values, p) {
  const v = nums(values).sort((a, b) => a - b);
  if (v.length === 0) return null;
  const pos = ((v.length - 1) * p) / 100;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

export const median = (values) => percentile(values, 50);

/** Per-pass median over results whose passMs is { passName: ms }; null if none were timed. */
export function medianPassMs(results) {
  const names = Object.keys(results.find((r) => r.passMs)?.passMs ?? {});
  if (names.length === 0) return null;
  return Object.fromEntries(names.map((n) => [n, median(results.map((r) => r.passMs?.[n]))]));
}

export function max(values) {
  const v = nums(values);
  return v.length ? Math.max(...v) : null;
}

/**
 * Summarizes a list of frame times (ms).
 *
 * `hitches` counts frames slower than twice the median frame time. That is relative
 * to the display's refresh rate, so it means the same thing on 60 Hz and 144 Hz
 * screens; `over33` is the absolute "visibly dropped below 30 fps" count.
 */
export function summarizeFrames(frameTimes) {
  const med = median(frameTimes);
  return {
    frames: frameTimes.length,
    totalMs: frameTimes.reduce((a, b) => a + b, 0),
    avg: mean(frameTimes),
    median: med,
    p95: percentile(frameTimes, 95),
    p99: percentile(frameTimes, 99),
    max: max(frameTimes),
    hitches: med === null ? 0 : frameTimes.filter((t) => t > 2 * med).length,
    over33: frameTimes.filter((t) => t > 33.3).length,
  };
}
