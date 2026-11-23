import { median } from './stats.js';

/**
 * Metrics reported per (grid, mode) configuration of a sweep. Each is read from one
 * raw run; the summary gives the median over repeats and the spread (min..max).
 *   initialMs  - full meshing of the whole grid from fresh terrain (chunk times summed)
 *   editMs     - median meshing time per edit (summed over the chunks an edit touched)
 *   latencyMs  - median time from an edit to its last chunk being shown
 *   frameP99 / frameMax / hitches / over33 - frame times during the edit run
 *   upKB / downKB - average CPU->GPU / GPU->CPU kilobytes per edit
 *   gpuMs and pass_<name> - median GPU time per edit, total and per compute pass
 */
export const SWEEP_METRICS = [
  ['initialMs', (r) => r.initial.totalMs],
  ['editMs', (r) => r.edit.editMs.median],
  ['latencyMs', (r) => r.edit.latencyMs?.median],
  ['frameP99', (r) => r.edit.frames.p99],
  ['frameMax', (r) => r.edit.frames.max],
  ['hitches', (r) => r.edit.frames.hitches],
  ['over33', (r) => r.edit.frames.over33],
  ['upKB', (r) => (r.edit.bytesUpPerEdit ?? 0) / 1024],
  ['downKB', (r) => (r.edit.bytesDownPerEdit ?? 0) / 1024],
  ['gpuMs', (r) => r.edit.gpuMs],
];

const PASSES = ['mask', 'merge', 'finalize'];

function spread(values) {
  const v = values.filter((x) => typeof x === 'number' && Number.isFinite(x));
  if (v.length === 0) return null;
  return { median: median(v), min: Math.min(...v), max: Math.max(...v), n: v.length };
}

/**
 * Groups raw sweep runs by (gridSize, mode) and reduces each metric to
 * { median, min, max, n } over the repeats. Runs whose frame coverage was too low
 * (rendering paused) are counted in `unreliable` and still included, so the
 * reader can decide; the app warns about them.
 *
 * @param {Array<{gridSize:number, mode:string, repeat:number, initial:object, edit:object}>} runs
 */
export function summarizeSweep(runs, { minCoverage = 0.8 } = {}) {
  const groups = new Map();
  for (const r of runs) {
    const key = `${r.gridSize}|${r.mode}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  return [...groups.values()].map((rs) => {
    const metrics = {};
    for (const [name, read] of SWEEP_METRICS) metrics[name] = spread(rs.map(read));
    for (const p of PASSES) metrics[`pass_${p}`] = spread(rs.map((r) => r.edit.passMs?.[p]));
    return {
      gridSize: rs[0].gridSize,
      chunks: rs[0].gridSize ** 2,
      mode: rs[0].mode,
      repeats: rs.length,
      unreliable: rs.filter((r) => r.edit.frames.coverage < minCoverage).length,
      metrics,
    };
  });
}
