import React from 'react';
import { MODES } from '../meshing/MeshController';

export const fmt = (v, digits = 2) => (typeof v === 'number' ? v.toFixed(digits) : '–');
export const fmtMs = (v) => (typeof v === 'number' ? `${v.toFixed(2)} ms` : 'n/a');
export const fmtKB = (bytes) => (typeof bytes === 'number' ? `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB` : '–');

const muted = { fontSize: '11px', color: '#95a5a6' };

/** Mesh stats for a full (re)mesh: single run or median of a benchmark series. */
export function MeshStats({ stats, timestamps }) {
  if (!stats) return null;
  const shown = stats.runs > 1 ? stats.median : stats.last;
  const gpuMode = stats.mode !== 'cpu';

  return (
    <div style={{ marginTop: '15px' }}>
      <div style={{ ...muted, marginBottom: '6px' }}>
        {MODES[stats.mode].label} ·{' '}
        {stats.runs > 1 ? `median of ${stats.runs} runs (after ${stats.warmup} warm-up)` : 'single run'}
        {stats.mergeImpl && ` · merge: ${stats.mergeImpl}`}
      </div>
      <div>⏱️ Total Time: <strong>{fmtMs(shown.totalMs)}</strong></div>
      {stats.runs > 1 && <div>🏁 Best Run: <strong>{fmtMs(stats.minTotalMs)}</strong></div>}
      {gpuMode && (
        <div>⚡ GPU Compute: <strong>{timestamps ? fmtMs(shown.gpuMs) : 'n/a (no timestamp-query)'}</strong></div>
      )}
      {gpuMode && timestamps && shown.passMs && (
        <div style={muted}>
          &nbsp;&nbsp;{Object.entries(shown.passMs).map(([name, ms]) => `${name} ${fmt(ms)}`).join(' · ')} ms
        </div>
      )}
      <div>
        📤 Readback:{' '}
        <strong style={{ color: stats.mode === 'gpu-direct' ? '#2ecc71' : 'white' }}>
          {stats.mode === 'gpu-direct' ? 'none (zero-copy)' : fmtMs(shown.readbackMs)}
        </strong>
      </div>
      <div>
        🔁 CPU↔GPU transfer: <strong>↑ {fmtKB(stats.last.bytesUp)}</strong> <strong>↓ {fmtKB(stats.last.bytesDown)}</strong>
      </div>
      <div>📉 Naive Faces (No Merge): <strong>{stats.naiveFaces != null ? stats.naiveFaces.toLocaleString() : 'run CPU once to compute'}</strong></div>
      <div>🚀 Greedy Quads (Merged): <strong style={{ color: '#2ecc71' }}>{stats.quads?.toLocaleString() ?? '–'}</strong></div>
      {stats.overflowed && <div style={{ color: '#ff6b6b' }}>⚠️ Output buffer overflowed — mesh is truncated</div>}
      <div style={{ ...muted, marginTop: '6px' }}>CPU time excludes Three's geometry upload on first draw.</div>
    </div>
  );
}

/** Result of the GPU-vs-CPU correctness check (one line per merge implementation). */
export function CorrectnessResult({ report }) {
  if (!report) return null;
  return (
    <div style={{ marginTop: '8px', fontSize: '12px' }}>
      <div style={{ color: report.pass ? '#2ecc71' : '#ff6b6b' }}>
        {report.pass ? '✅' : '❌'} CPU: <strong>{report.cpuQuads.toLocaleString()}</strong> quads (default terrain)
      </div>
      {report.results.map((r) => (
        <div key={r.impl} style={{ color: r.countMatch && r.setsEqual ? 'white' : '#ff6b6b' }}>
          &nbsp;&nbsp;GPU {r.impl}: <strong>{r.gpuQuads.toLocaleString()}</strong> quads ·{' '}
          {r.setsEqual ? 'same quad set' : `quad sets differ (${r.onlyInCpu.length}+ only on CPU, ${r.onlyInGpu.length}+ only on GPU)`}
          {r.overflowed && ' · overflowed'}
        </div>
      ))}
      <div style={muted}>Details in the console.</div>
    </div>
  );
}

/** One line describing the last interactive edit and what it cost. */
export function EditInfo({ info }) {
  if (!info) return null;
  return (
    <div style={{ marginTop: '10px', fontSize: '12px' }}>
      <div>
        ✏️ {info.label}: <strong>{info.voxels.toLocaleString()}</strong> voxel{info.voxels === 1 ? '' : 's'} · remesh{' '}
        <strong>{fmtMs(info.totalMs)}</strong> <span style={muted}>({MODES[info.mode].label})</span>
      </div>
      <div style={muted}>
        transfer ↑ {fmtKB(info.bytesUp)} ↓ {fmtKB(info.bytesDown)}
        {info.quadCount != null && ` · ${info.quadCount.toLocaleString()} quads`}
      </div>
    </div>
  );
}

// Below this, rendering was paused for part of the run and frame stats are unreliable.
const MIN_COVERAGE = 0.8;

const cell = { padding: '3px 8px', textAlign: 'right', whiteSpace: 'nowrap' };

/** Results table for the scripted edit benchmark. */
export function BenchmarkResults({ idle, results }) {
  if (!idle && results.length === 0) return null;
  const fmtFrames = (f) => (
    <>
      <td style={cell}>{fmt(f.avg)}</td>
      <td style={cell}>{fmt(f.p99)}</td>
      <td style={cell}>{fmt(f.max)}</td>
      <td style={cell}>{f.hitches}</td>
      <td style={cell}>{f.over33}</td>
      <td style={{ ...cell, color: f.coverage < MIN_COVERAGE ? '#ff6b6b' : '#95a5a6' }}>{Math.round(f.coverage * 100)}%</td>
    </>
  );

  const unreliable = [idle, ...results.map((r) => r.frames)].some((f) => f && f.coverage < MIN_COVERAGE);

  return (
    <>
    {unreliable && (
      <div style={{ color: '#ff6b6b', fontSize: '12px', marginTop: '6px' }}>
        ⚠️ Rendering was paused for part of a run (coverage &lt; {MIN_COVERAGE * 100}%), so those frame times are unreliable. Keep the tab visible and re-run.
      </div>
    )}
    <table style={{ borderCollapse: 'collapse', fontSize: '12px', marginTop: '8px' }}>
      <thead>
        <tr style={{ color: '#95a5a6', borderBottom: '1px solid #444' }}>
          <th style={{ ...cell, textAlign: 'left' }}>mode</th>
          <th style={cell}>avg</th>
          <th style={cell}>p99</th>
          <th style={cell}>max</th>
          <th style={cell} title="frames slower than 2x the median">hitches</th>
          <th style={cell} title="frames slower than 33.3 ms">&gt;33ms</th>
          <th style={cell} title="recorded frame time / wall-clock time; low = rendering was paused">cov.</th>
          <th style={cell} title="median remesh time per edit">edit ms</th>
          <th style={cell} title="average CPU->GPU bytes per edit">↑/edit</th>
          <th style={cell} title="average GPU->CPU bytes per edit">↓/edit</th>
        </tr>
      </thead>
      <tbody>
        {idle && (
          <tr style={{ color: '#95a5a6' }}>
            <td style={{ ...cell, textAlign: 'left' }}>idle (no edits)</td>
            {fmtFrames(idle)}
            <td style={cell}>–</td><td style={cell}>–</td><td style={cell}>–</td>
          </tr>
        )}
        {results.map((r) => (
          <tr key={r.mode} style={{ borderTop: '1px solid #333' }}>
            <td style={{ ...cell, textAlign: 'left' }}>{MODES[r.mode].label}</td>
            {fmtFrames(r.frames)}
            <td style={cell}>{fmt(r.editMs.median)}</td>
            <td style={cell}>{fmtKB(r.bytesUpPerEdit)}</td>
            <td style={cell}>{fmtKB(r.bytesDownPerEdit)}</td>
          </tr>
        ))}
      </tbody>
    </table>
    </>
  );
}
