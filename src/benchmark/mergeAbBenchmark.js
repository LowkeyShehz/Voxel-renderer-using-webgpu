import { ChunkData } from '../voxel/ChunkData.js';
import { TerrainGenerator } from '../voxel/TerrainGenerator.js';
import { median, percentile } from './stats.js';

export const MERGE_AB_ROUNDS = 55;
export const MERGE_AB_WARMUP = 5;
const IMPLS = ['legacy', 'bitrun'];
const PASSES = ['mask', 'merge', 'finalize'];

const spread = (values) => ({
  median: median(values),
  p25: percentile(values, 25),
  p75: percentile(values, 75),
  min: values.length ? Math.min(...values) : null,
  max: values.length ? Math.max(...values) : null,
  n: values.length,
});

/**
 * Reduces A/B samples to per-implementation pass times and the merge speed-up.
 *   speedup       - median legacy merge / median bit-run merge (what Table VII reports)
 *   pairedSpeedup - median over rounds of legacy merge / bit-run merge in the same
 *                   round; robust to clock changes that last longer than one round
 *
 * @param {Array<{round:number, impl:string, passMs:Object<string, number>, totalMs:number}>} samples
 */
export function summarizeMergeAb(samples) {
  const impls = Object.fromEntries(IMPLS.map((impl) => {
    const mine = samples.filter((s) => s.impl === impl);
    const passes = Object.fromEntries(PASSES.map((p) => [p, spread(mine.map((s) => s.passMs[p]))]));
    return [impl, { ...passes, total: spread(mine.map((s) => s.totalMs)) }];
  }));

  const byRound = new Map();
  for (const s of samples) {
    if (!byRound.has(s.round)) byRound.set(s.round, {});
    byRound.get(s.round)[s.impl] = s.passMs.merge;
  }
  const ratios = [...byRound.values()].filter((r) => r.legacy > 0 && r.bitrun > 0).map((r) => r.legacy / r.bitrun);

  const legacyMerge = impls.legacy.merge.median;
  const bitrunMerge = impls.bitrun.merge.median;
  return {
    rounds: byRound.size,
    impls,
    speedup: legacyMerge && bitrunMerge ? legacyMerge / bitrunMerge : null,
    pairedSpeedup: ratios.length ? median(ratios) : null,
  };
}

const yieldToBrowser = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Times the legacy and bit-run merge passes alternately in one session, so GPU clock
 * changes affect both equally (paper, Section V.B, RQ4). Each round meshes the same
 * default-terrain chunk once with each merge, with the order swapped every round so
 * neither always runs first. The first `warmup` rounds are discarded. Pass times come
 * from timestamp queries; the world on screen is untouched.
 *
 * Before timing, both merges are checked to produce the same quad count.
 *
 * @param {import('../meshing/gpuMeshPipeline').GpuMesher} mesher
 */
export async function runMergeAb(mesher, { rounds = MERGE_AB_ROUNDS, warmup = MERGE_AB_WARMUP, onProgress } = {}) {
  if (!mesher.timestamps) throw new Error('Merge A/B needs the timestamp-query feature, which this GPU/browser does not expose.');
  const assertVisible = () => {
    if (document.visibilityState !== 'visible') throw new Error('Merge A/B stopped: keep the tab visible while it runs.');
  };
  assertVisible();

  const chunk = new ChunkData();
  new TerrainGenerator().generate(chunk);
  const target = mesher.createReadbackTarget();
  const samples = [];
  const quadCounts = {};
  try {
    for (const impl of IMPLS) {
      mesher.mesh(chunk, target, { mergeImpl: impl });
      const counters = await mesher.readCounters(target);
      if (counters.overflowed) throw new Error(`${impl} merge overflowed the output buffer`);
      quadCounts[impl] = counters.quadCount;
    }
    if (quadCounts.legacy !== quadCounts.bitrun) {
      throw new Error(`Merges disagree: legacy ${quadCounts.legacy} quads, bit-run ${quadCounts.bitrun}. Run the correctness check.`);
    }

    for (let i = 0; i < warmup + rounds; i++) {
      assertVisible();
      const order = i % 2 === 0 ? IMPLS : [...IMPLS].reverse();
      for (const impl of order) {
        const timing = await mesher.mesh(chunk, target, { profile: true, mergeImpl: impl }).gpuMs;
        if (i >= warmup) samples.push({ round: i - warmup, impl, first: impl === order[0], ...timing });
      }
      onProgress?.(i < warmup ? `warm-up ${i + 1}/${warmup}` : `round ${i - warmup + 1}/${rounds}`);
      await yieldToBrowser();
    }
  } finally {
    mesher.destroyTarget(target);
    mesher.releaseChunk(chunk);
  }

  const report = {
    kind: 'merge-ab',
    rounds,
    warmup,
    mergeWorkgroupSize: mesher.mergeWorkgroupSize,
    quadCounts,
    summary: summarizeMergeAb(samples),
    samples,
  };
  console.log('Merge A/B:', JSON.stringify({ ...report, samples: undefined }, null, 2));
  return report;
}
