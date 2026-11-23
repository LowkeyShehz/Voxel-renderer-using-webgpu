#!/usr/bin/env node
/**
 * Turns benchmark exports from the app into markdown tables (stdout) and CSV files.
 *
 *   node scripts/benchmarks-to-md.mjs benchmarks/sweep-*.json
 *   node scripts/benchmarks-to-md.mjs --csv-dir out/ benchmarks/*.json > tables.md
 *
 * Understands both export kinds:
 *   - sweep JSON ("Export sweep JSON"): per (grid, mode) median and [min-max] over
 *     the repeats; CSV has one row per (grid, mode) with median/min/max columns.
 *   - edit-benchmark JSON ("Export JSON"): one row per mode, as in the app's table.
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
    `- Edits: ${meta.edits} explosions r=${meta.radius} every ${meta.intervalMs} ms, seed ${meta.seed}${meta.repeats ? ` · ${meta.repeats} runs per configuration` : ''}`,
  ].join('\n');
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
  const { markdown, csv } = isSweep ? sweepToMarkdown(json, name) : editBenchToMarkdown(json, name);
  process.stdout.write(markdown + '\n');
  const csvPath = join(csvDir ?? dirname(file), name.replace(/\.json$/i, '') + '.csv');
  writeFileSync(csvPath, csv);
  console.error(`wrote ${csvPath}`);
}
