#!/usr/bin/env node
/**
 * Turns benchmark exports from the app into markdown tables (stdout) and CSV files.
 *
 *   node scripts/benchmarks-to-md.mjs benchmarks/sweep-*.json
 *   node scripts/benchmarks-to-md.mjs --csv-dir out/ benchmarks/*.json > tables.md
 *
 * Understands three export kinds:
 *   - sweep JSON ("Export sweep JSON"): per (grid, mode) median and [min-max] over
 *     the repeats; CSV has one row per (grid, mode) with median/min/max columns.
 *   - edit-benchmark JSON ("Export JSON"): one row per mode, as in the app's table.
 *   - merge A/B JSON ("Export A/B JSON"): legacy vs bit-run pass times and speed-up.
 *   - mesh JSON ("Export mesh JSON"): every Generate Mesh / Benchmark ×N run with its GPU memory.
 * The CSV is written next to each input (same name, .csv) unless --csv-dir is given.
 * No dependencies; reads the summary logic from src/benchmark/sweepSummary.js so the
 * numbers match the app.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { summarizeSweep, SWEEP_METRICS } from '../src/benchmark/sweepSummary.js';

const MODE_LABELS = { cpu: 'CPU greedy', 'gpu-readback': 'GPU + readback', 'gpu-direct': 'GPU direct' };
const label = (m) => MODE_LABELS[m] ?? m;
const num = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '–');
const cellOf = (s, d = 2) => (s ? `${num(s.median, d)} [${num(s.min, d)}–${num(s.max, d)}]` : '–');
const csvCell = (v) => (typeof v === 'number' ? String(+v.toFixed(4)) : v == null ? '' : `"${String(v).replace(/"/g, '""')}"`);

function table(header, rows) {
  const line = (cells) => `| ${cells.join(' | ')} |`;
  return [line(header), line(header.map((_, i) => (i < 2 ? '---' : '---:'))), ...rows.map(line)].join('\n');
}

function metaBlock(meta = {}) {
  const gpu = meta.gpu ? [meta.gpu.vendor, meta.gpu.architecture, meta.gpu.description].filter(Boolean).join(' / ') : 'unknown';
  return [
    `- Date: ${meta.date ?? 'unknown'}`,
    `- GPU (adapter info): ${gpu}${meta.gpu?.timestampQuery ? ', timestamp-query' : ''}`,
    `- Browser: ${meta.userAgent ?? 'unknown'}`,
    `- CPU threads: ${meta.hardwareConcurrency ?? 'unknown'} · display: ${meta.display ? `${meta.display.viewport?.join('×')} @ ${meta.display.devicePixelRatio}x` : 'unknown'}`,
    `- Merge pass: ${meta.mergeImpl ?? 'n/a'}${meta.mergeImpl === 'bitrun' ? ` (workgroup ${meta.mergeWorkgroupSize})` : ''}`,
    meta.edits != null && `- Edits: ${meta.edits} explosions r=${meta.radius} every ${meta.intervalMs} ms, seed ${meta.seed}${meta.repeats ? ` · ${meta.repeats} runs per configuration` : ''}`,
  ].filter(Boolean).join('\n');
}

function sweepToMarkdown(json, name) {
  const summary = json.summary?.length ? json.summary : summarizeSweep(json.runs ?? []);
  const sorted = [...summary].sort((a, b) => a.gridSize - b.gridSize || a.mode.localeCompare(b.mode));
  const key = (s) => [`${s.gridSize}×${s.gridSize} (${s.chunks})`, label(s.mode)];
  const out = [`## Sweep: ${name}`, '', metaBlock(json.meta), '', 'Median over runs, [min–max] in brackets.', ''];

  out.push('### Initial full meshing of the grid (ms, chunk times summed)', '');
  out.push(table(['grid (chunks)', 'mode', 'initial ms'], sorted.map((s) => [...key(s), cellOf(s.metrics.initialMs)])), '');

  out.push('### Edit benchmark', '');
  out.push(table(
    ['grid (chunks)', 'mode', 'edit ms', 'latency ms', 'p99 frame', 'max frame', 'hitches', '>33 ms', '↑ KB/edit', '↓ KB/edit'],
    sorted.map((s) => [
      ...key(s), cellOf(s.metrics.editMs), cellOf(s.metrics.latencyMs), cellOf(s.metrics.frameP99), cellOf(s.metrics.frameMax),
      cellOf(s.metrics.hitches, 0), cellOf(s.metrics.over33, 0), cellOf(s.metrics.upKB, 1), cellOf(s.metrics.downKB, 1),
    ])
  ), '');

  const gpuRows = sorted.filter((s) => s.metrics.pass_merge);
  if (gpuRows.length) {
    out.push('### GPU time per edit by pass (ms, timestamp queries)', '');
    out.push(table(
      ['grid (chunks)', 'mode', 'mask', 'merge', 'finalize', 'total'],
      gpuRows.map((s) => [...key(s), cellOf(s.metrics.pass_mask, 3), cellOf(s.metrics.pass_merge, 3), cellOf(s.metrics.pass_finalize, 3), cellOf(s.metrics.gpuMs, 3)])
    ), '');
  }
  const unreliable = sorted.filter((s) => s.unreliable);
  if (unreliable.length) {
    out.push(`> ⚠️ ${unreliable.map((s) => `${s.gridSize}×${s.gridSize} ${label(s.mode)}: ${s.unreliable} run(s)`).join('; ')} had paused rendering (frame coverage < 80%); re-run those.`, '');
  }

  const names = [...SWEEP_METRICS.map(([n]) => n), 'pass_mask', 'pass_merge', 'pass_finalize'];
  const header = ['gridSize', 'chunks', 'mode', 'repeats', 'unreliable', ...names.flatMap((n) => [`${n}_median`, `${n}_min`, `${n}_max`])];
  const rows = sorted.map((s) => [
    s.gridSize, s.chunks, s.mode, s.repeats, s.unreliable,
    ...names.flatMap((n) => [s.metrics[n]?.median, s.metrics[n]?.min, s.metrics[n]?.max]),
  ]);
  return { markdown: out.join('\n'), csv: [header, ...rows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n' };
}

function editBenchToMarkdown(json, name) {
  const rows = (json.results ?? []).map((r) => [
    label(r.mode), num(r.frames.avg), num(r.frames.p95), num(r.frames.p99), num(r.frames.max), r.frames.hitches, r.frames.over33,
    `${Math.round(r.frames.coverage * 100)}%`, num(r.editMs.median), num(r.editMs.max), num(r.bytesUpPerEdit / 1024, 1), num(r.bytesDownPerEdit / 1024, 1),
    num(r.gpuMs, 3), r.passMs ? Object.entries(r.passMs).map(([k, v]) => `${k} ${num(v, 3)}`).join(', ') : '–',
  ]);
  const header = ['mode', 'avg', 'p95', 'p99', 'max', 'hitches', '>33 ms', 'coverage', 'edit ms (median)', 'edit ms (max)', '↑ KB/edit', '↓ KB/edit', 'GPU ms', 'passes'];
  const md = [
    `## Edit benchmark: ${name}`, '', metaBlock(json.meta),
    `- World: ${json.meta?.gridSize ?? 1}×${json.meta?.gridSize ?? 1}`, '',
    json.idle ? `Idle frame time: avg ${num(json.idle.avg)} ms, max ${num(json.idle.max)} ms.` : '', '',
    table(header, rows), '',
  ].join('\n');
  const csvRows = (json.results ?? []).map((r) => [
    r.mode, r.gridSize ?? json.meta?.gridSize ?? 1, r.frames.avg, r.frames.p95, r.frames.p99, r.frames.max, r.frames.hitches, r.frames.over33,
    r.frames.coverage, r.editMs.median, r.editMs.max, r.bytesUpPerEdit, r.bytesDownPerEdit, r.gpuMs,
    r.passMs?.mask, r.passMs?.merge, r.passMs?.finalize,
  ]);
  const csvHeader = ['mode', 'gridSize', 'frameAvg', 'frameP95', 'frameP99', 'frameMax', 'hitches', 'over33', 'coverage',
    'editMsMedian', 'editMsMax', 'bytesUpPerEdit', 'bytesDownPerEdit', 'gpuMs', 'passMask', 'passMerge', 'passFinalize'];
  return { markdown: md, csv: [csvHeader, ...csvRows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n' };
}

function mergeAbToMarkdown(json, name) {
  const { impls, speedup, pairedSpeedup, rounds } = json.summary;
  const q = (s) => `${num(s.median, 3)} [${num(s.p25, 3)}–${num(s.p75, 3)}]`;
  const rows = ['legacy', 'bitrun'].map((impl) => [impl, q(impls[impl].mask), q(impls[impl].merge), q(impls[impl].finalize), q(impls[impl].total)]);
  const md = [
    `## Merge A/B: ${name}`, '', metaBlock({ ...json.meta, mergeImpl: 'legacy vs bitrun', mergeWorkgroupSize: json.mergeWorkgroupSize }),
    `- ${rounds} alternating rounds after ${json.warmup} warm-up, ${json.quadCounts?.bitrun?.toLocaleString?.() ?? '?'} quads per mesh`, '',
    'GPU ms per pass, median [p25–p75].', '',
    table(['merge pass', 'mask', 'merge', 'finalize', 'total'], rows),
    '', `Speed-up of the merge pass: **${num(speedup)}×** (median of per-round ratios ${num(pairedSpeedup)}×).`, '',
  ].join('\n');
  const header = ['impl', ...['mask', 'merge', 'finalize', 'total'].flatMap((p) => [`${p}_median`, `${p}_p25`, `${p}_p75`, `${p}_min`, `${p}_max`])];
  const csvRows = ['legacy', 'bitrun'].map((impl) => [impl, ...['mask', 'merge', 'finalize', 'total'].flatMap((p) => {
    const s = impls[impl][p];
    return [s.median, s.p25, s.p75, s.min, s.max];
  })]);
  return { markdown: md, csv: [header, ...csvRows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n' };
}

function meshRunsToMarkdown(json, name) {
  const MB = (b) => (typeof b === 'number' ? (b / 1048576).toFixed(2) : '–');
  const runs = json.runs ?? [];
  const header = ['mode', 'grid', 'runs', 'median ms', 'best ms', 'GPU ms', 'mask', 'merge', 'finalize', 'readback ms', 'quads', '↑ KB', '↓ KB'];
  const rows = runs.map((r) => [
    label(r.mode), `${r.gridSize}×${r.gridSize}`, r.runs, num(r.totalMs.median), num(r.totalMs.min), num(r.gpuMs), num(r.passMs?.mask, 3),
    num(r.passMs?.merge, 3), num(r.passMs?.finalize, 3), num(r.readbackMs), r.quads?.toLocaleString?.() ?? '–', num(r.bytesUp / 1024, 1), num(r.bytesDown / 1024, 1),
  ]);
  const memHeader = ['mode', 'grid', 'zero-copy MB', 'chunks', 'MB per chunk', 'readback MB', 'voxels MB', 'voxel chunks', 'scratch MB', 'JS heap MB'];
  const memRows = runs.map((r) => {
    const m = r.memory ?? {};
    return [label(r.mode), `${r.gridSize}×${r.gridSize}`, MB(m.directOutputBytes), m.directMeshes ?? '–',
      m.directMeshes ? MB(m.directOutputBytes / m.directMeshes) : '–', MB(m.readbackBytes), MB(m.voxelBytes), m.voxelChunks ?? '–',
      MB(m.scratchBytes), num(r.heapMB, 0)];
  });
  const md = [
    `## Mesh runs: ${name}`, '', metaBlock(json.meta), '',
    'Full-world meshing; times are the median of the runs after warm-up (Benchmark ×N) or a single run.', '',
    table(header, rows), '', '### GPU memory after each run (MiB)', '', table(memHeader, memRows), '',
  ].join('\n');
  const csvHeader = ['mode', 'gridSize', 'chunks', 'runs', 'medianMs', 'minMs', 'maxMs', 'gpuMs', 'passMask', 'passMerge', 'passFinalize',
    'readbackMs', 'quads', 'bytesUp', 'bytesDown', 'directOutputBytes', 'directMeshes', 'readbackBytes', 'voxelBytes', 'voxelChunks', 'scratchBytes', 'heapMB'];
  const csvRows = runs.map((r) => [r.mode, r.gridSize, r.chunks, r.runs, r.totalMs.median, r.totalMs.min, r.totalMs.max, r.gpuMs,
    r.passMs?.mask, r.passMs?.merge, r.passMs?.finalize, r.readbackMs, r.quads, r.bytesUp, r.bytesDown,
    r.memory?.directOutputBytes, r.memory?.directMeshes, r.memory?.readbackBytes, r.memory?.voxelBytes, r.memory?.voxelChunks, r.memory?.scratchBytes, r.heapMB]);
  return { markdown: md, csv: [csvHeader, ...csvRows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n' };
}

const args = process.argv.slice(2);
let csvDir = null;
const files = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--csv-dir') csvDir = args[++i];
  else files.push(args[i]);
}
if (files.length === 0) {
  console.error('usage: node scripts/benchmarks-to-md.mjs [--csv-dir DIR] export.json [...]');
  process.exit(1);
}
if (csvDir) mkdirSync(csvDir, { recursive: true });

for (const file of files) {
  const json = JSON.parse(readFileSync(file, 'utf8'));
  const name = basename(file);
  const isSweep = json.kind === 'sweep' || Array.isArray(json.runs);
  const convert = json.kind === 'merge-ab' ? mergeAbToMarkdown : json.kind === 'mesh-runs' ? meshRunsToMarkdown
    : isSweep ? sweepToMarkdown : editBenchToMarkdown;
  const { markdown, csv } = convert(json, name);
  process.stdout.write(markdown + '\n');
  const csvPath = join(csvDir ?? dirname(file), name.replace(/\.json$/i, '') + '.csv');
  writeFileSync(csvPath, csv);
  console.error(`wrote ${csvPath}`);
}
