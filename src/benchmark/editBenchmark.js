import { CHUNK_SIZE } from '../voxel/ChunkData';
import { FrameRecorder } from './FrameRecorder';
import { mean, median, percentile, max, summarizeFrames, medianPassMs } from './stats';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Small seeded PRNG so every mode sees the exact same sequence of explosions. */
export function mulberry32(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Explosion centres spread over the world. `extent` is the world size in voxels per
 * axis; for one chunk ([96, 96, 96], the default) the sequence is exactly the one
 * the Phase 4 benchmark used.
 */
export function explosionSequence({ seed, count, radius, extent = [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE] }) {
  const rnd = mulberry32(seed);
  const margin = Math.ceil(radius / 2);
  const pick = (axis) => margin + Math.floor(rnd() * (extent[axis] - 2 * margin));
  return Array.from({ length: count }, () => ({ x: pick(0), y: pick(1), z: pick(2) }));
}

/**
 * Summarizes a recorded window and adds `coverage`: recorded frame time divided by the
 * wall-clock length of the window. Close to 1 when frames were delivered continuously;
 * well below 1 means rendering was paused (background tab, throttling) and the
 * frame statistics cannot be trusted.
 */
function summarizeWindow(frameTimes, wallMs) {
  const summary = summarizeFrames(frameTimes);
  return { ...summary, wallMs, coverage: wallMs > 0 ? summary.totalMs / wallMs : 0 };
}

/** Frame times with no edits happening: the display/vsync floor to compare against. */
export async function measureIdleFrames(recorder, durationMs = 2000) {
  const t0 = performance.now();
  recorder.begin();
  await sleep(durationMs);
  const frames = recorder.end();
  return summarizeWindow(frames, performance.now() - t0);
}

/**
 * Frame-time stability under repeated edits.
 *
 * Resets the terrain, then applies `edits` explosions at a fixed cadence while the
 * render loop keeps running, recording every frame time. Explosions are applied
 * and remeshed exactly as an interactive edit would be (every chunk the explosion
 * touched, plus neighbours whose shared border changed, one chunk per frame), so
 * CPU meshing shows up as long frames and the GPU paths show whatever they cost.
 *
 * editMs   = meshing time summed over the chunks one edit remeshed
 * latencyMs = edit to last chunk shown, including the frames between chunks
 *
 * @param {object} opts
 * @param {import('../meshing/MeshController').MeshController} opts.controller
 * @param {FrameRecorder} opts.recorder  must be ticked by the render loop
 */
export async function runEditBenchmark({ controller, recorder, mode, edits, intervalMs, radius, seed, onProgress }) {
  const world = controller.world;
  const sequence = explosionSequence({ seed, count: edits, radius, extent: world.bounds.max });

  // Start every mode from identical terrain and a fully meshed, uploaded world.
  controller.resetTerrain();
  await controller.remesh(mode);
  await sleep(1000); // let the one-off full upload / first draw settle before recording

  const perEdit = [];
  let removedTotal = 0;

  recorder.begin();
  const start = performance.now();
  for (let i = 0; i < sequence.length; i++) {
    const wait = start + (i + 1) * intervalMs - performance.now();
    if (wait > 0) await sleep(wait);

    const e = sequence[i];
    removedTotal += controller.explode(e.x, e.y, e.z, radius);
    perEdit.push(await controller.remeshDirty(mode));
    onProgress?.(i + 1, sequence.length);
  }
  await sleep(500); // include the frames that display the last edit
  const frames = recorder.end();
  const wallMs = performance.now() - start;

  return {
    mode,
    gridSize: world.gridSize,
    chunks: world.chunks.length,
    edits,
    intervalMs,
    radius,
    seed,
    voxelsRemoved: removedTotal,
    frames: summarizeWindow(frames, wallMs),
    editMs: {
      median: median(perEdit.map((r) => r.totalMs)),
      mean: mean(perEdit.map((r) => r.totalMs)),
      p95: percentile(perEdit.map((r) => r.totalMs), 95),
      max: max(perEdit.map((r) => r.totalMs)),
    },
    latencyMs: {
      median: median(perEdit.map((r) => r.wallMs)),
      max: max(perEdit.map((r) => r.wallMs)),
    },
    chunksPerEdit: mean(perEdit.map((r) => r.chunks)),
    gpuMs: median(perEdit.map((r) => r.gpuMs)),
    passMs: medianPassMs(perEdit),
    bytesUpPerEdit: mean(perEdit.map((r) => r.bytesUp)),
    bytesDownPerEdit: mean(perEdit.map((r) => r.bytesDown)),
  };
}
