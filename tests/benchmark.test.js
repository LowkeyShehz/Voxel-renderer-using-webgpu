// Benchmark bookkeeping: the merge A/B summary and the export-to-markdown script.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { summarizeMergeAb } from '../src/benchmark/mergeAbBenchmark.js';
import { explosionSequence } from '../src/benchmark/editBenchmark.js';

const sample = (round, impl, merge) => ({ round, impl, passMs: { mask: 0.05, merge, finalize: 0.01 }, totalMs: merge + 0.06 });

test('merge A/B summary: medians per pass and both speed-ups', () => {
  // Round 2 has a clock dip that slows both merges; the paired ratio ignores it.
  const samples = [
    sample(0, 'legacy', 12), sample(0, 'bitrun', 4),
    sample(1, 'bitrun', 4), sample(1, 'legacy', 12),
    sample(2, 'legacy', 24), sample(2, 'bitrun', 8),
  ];
  const s = summarizeMergeAb(samples);
  assert.equal(s.rounds, 3);
  assert.equal(s.impls.legacy.merge.median, 12);
  assert.equal(s.impls.bitrun.merge.median, 4);
  assert.equal(s.impls.bitrun.mask.median, 0.05);
  assert.equal(s.speedup, 3);
  assert.equal(s.pairedSpeedup, 3);
});

test('edit benchmark: the 1x1 explosion sequence is unchanged from Phase 4', () => {
  // Seed 1337, radius 24 is the sequence behind Tables V and VI; changing it would
  // make new edit-benchmark runs incomparable with the published ones.
  const seq = explosionSequence({ seed: 1337, count: 40, radius: 24 });
  assert.equal(seq.length, 40);
  for (const e of seq) for (const v of [e.x, e.y, e.z]) assert.ok(v >= 12 && v < 84);
  assert.deepEqual(seq, explosionSequence({ seed: 1337, count: 40, radius: 24 }));
});

test('benchmarks-to-md: merge A/B export becomes a table with the speed-up', () => {
  const dir = mkdtempSync(join(tmpdir(), 'merge-ab-'));
  const file = join(dir, 'merge-ab-test.json');
  const samples = [sample(0, 'legacy', 13.8), sample(0, 'bitrun', 5.1), sample(1, 'bitrun', 5.0), sample(1, 'legacy', 13.9)];
  writeFileSync(file, JSON.stringify({
    kind: 'merge-ab', rounds: 2, warmup: 0, mergeWorkgroupSize: 8, quadCounts: { legacy: 50112, bitrun: 50112 },
    summary: summarizeMergeAb(samples), samples, meta: { date: 'test' },
  }));
  const md = execFileSync(process.execPath, ['scripts/benchmarks-to-md.mjs', file], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
  assert.match(md, /\| legacy \| 0\.050 .* \| 13\.850 /);
  assert.match(md, /\| bitrun \| 0\.050 .* \| 5\.050 /);
  assert.match(md, /Speed-up of the merge pass: \*\*2\.74×\*\*/);
  assert.match(readFileSync(join(dir, 'merge-ab-test.csv'), 'utf8'), /^"impl","mask_median"/);
});

test('benchmarks-to-md: mesh-runs export gives time and per-chunk memory tables', () => {
  const dir = mkdtempSync(join(tmpdir(), 'mesh-runs-'));
  const file = join(dir, 'mesh-runs-test.json');
  const MiB = 1048576;
  writeFileSync(file, JSON.stringify({
    kind: 'mesh-runs', meta: { date: 'test' },
    runs: [{
      mode: 'gpu-direct', gridSize: 2, chunks: 4, runs: 20, warmup: 2, quads: 204826,
      totalMs: { median: 12.5, min: 11.9, max: 14 }, gpuMs: 9.6, readbackMs: 0, passMs: { mask: 0.12, merge: 9.4, finalize: 0.01 },
      bytesUp: 0, bytesDown: 128,
      memory: { directOutputBytes: 4 * 6 * MiB, directMeshes: 4, readbackBytes: 6 * MiB, voxelBytes: 4 * 110592, voxelChunks: 4, scratchBytes: 670464 },
      heapMB: 150,
    }],
  }));
  const md = execFileSync(process.execPath, ['scripts/benchmarks-to-md.mjs', file], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
  assert.match(md, /\| GPU direct \| 2×2 \| 20 \| 12\.50 \| 11\.90 \|/);
  assert.match(md, /\| GPU direct \| 2×2 \| 24\.00 \| 4 \| 6\.00 \| 6\.00 \| 0\.42 \| 4 \| 0\.64 \| 150 \|/);
});
