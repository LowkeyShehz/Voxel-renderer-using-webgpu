import { runEditBenchmark, measureIdleFrames } from './editBenchmark';
import { summarizeSweep } from './sweepSummary.js';

export const SWEEP_GRIDS = [1, 2, 3, 4];
export const SWEEP_REPEATS = 3;

/** Rough wall-clock estimate in seconds, for the UI. */
export function estimateSweepSeconds({ grids, modes, repeats, edits, intervalMs }) {
  const perEditRun = (edits * intervalMs) / 1000 + 2.5; // + settle and tail
  return Math.round(grids.length * modes.length * repeats * perEditRun + grids.reduce((a, g) => a + g * g, 0) * 0.5 + 3);
}

/**
 * Benchmark sweep over world sizes and meshing modes. For every grid size, every
 * mode and every repeat it records
 *   (a) `initial`: full meshing of the whole grid from fresh terrain
 *       (MeshController.measure: chunk times summed, per-pass GPU time summed), and
 *   (b) `edit`: the seeded edit benchmark (runEditBenchmark: frame times, edit ms,
 *       bytes up/down, per-pass GPU time per edit).
 * Repeats of one configuration run back to back. The world is rebuilt once per
 * grid size (not timed).
 *
 * Refuses to start, and stops, if the tab is hidden: browsers pause rendering in
 * background tabs, which would make the frame times meaningless.
 *
 * @param {object} opts
 * @param {import('../meshing/MeshController').MeshController} opts.controller
 * @param {(gridSize: number) => Promise<void>} opts.setGridSize  rebuilds the world
 * @returns {Promise<{ idle: object, runs: object[], summary: object[] }>}
 */
export async function runSweep({
  controller, recorder, setGridSize, grids = SWEEP_GRIDS, modes, repeats = SWEEP_REPEATS,
  edits, intervalMs, radius, seed, onProgress, onRun,
}) {
  const assertVisible = () => {
    if (document.visibilityState !== 'visible') {
      throw new Error('Sweep stopped: the tab was hidden, which pauses rendering and invalidates frame times.');
    }
  };
  assertVisible();
  const idle = await measureIdleFrames(recorder);
  const runs = [];
  const total = grids.length * modes.length * repeats;

  for (const gridSize of grids) {
    onProgress?.(`building ${gridSize}×${gridSize} world…`);
    await setGridSize(gridSize);
    for (const mode of modes) {
      for (let repeat = 1; repeat <= repeats; repeat++) {
        assertVisible();
        const label = `${gridSize}×${gridSize} · ${mode} · run ${repeat}/${repeats} (${runs.length + 1}/${total})`;

        onProgress?.(`${label}: initial meshing`);
        controller.resetTerrain();
        const full = await controller.measure(mode);
        controller.show(full, mode);
        const { perChunk, ...initial } = full;

        const edit = await runEditBenchmark({
          controller, recorder, mode, edits, intervalMs, radius, seed,
          onProgress: (done, n) => onProgress?.(`${label}: edit ${done}/${n}`),
        });

        const run = { gridSize, mode, repeat, initial: { ...initial, perChunkMs: perChunk.map((r) => r.totalMs) }, edit };
        runs.push(run);
        onRun?.(run, runs);
      }
    }
  }
  return { idle, runs, summary: summarizeSweep(runs) };
}
