import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CHUNK_SIZE } from './src/voxel/ChunkData';
import { GpuMesher, mergeOptionsFromUrl } from './src/meshing/gpuMeshPipeline';
import { runCorrectnessCheck } from './src/meshing/correctnessCheck';
import { MeshController, MODES } from './src/meshing/MeshController';
import { DirectRenderBridge } from './src/render/DirectRenderBridge';
import { getGpuDevice, getGpuInfo } from './src/gpu/gpuContext';
import { VoxelEditor } from './src/interaction/VoxelEditor';
import { FrameRecorder } from './src/benchmark/FrameRecorder';
import { runEditBenchmark, measureIdleFrames } from './src/benchmark/editBenchmark';
import { median, medianPassMs } from './src/benchmark/stats';
import { MeshStats, EditInfo, BenchmarkResults, CorrectnessResult, fmt } from './src/ui/panels';

/**
 * VOXEL ENGINE PROTOTYPE - WebGPU Compute Shader Meshing
 *
 * Phase 4: the chunk can be edited (click to remove/place blocks, explosions) and
 * every edit is remeshed in the selected mode, so the three modes can be compared
 * on what actually matters for the thesis: frame-time stability while editing.
 */

const WARMUP_RUNS = 2;
const BENCH_RUNS = 20;

// Scripted edit benchmark parameters (same for every mode so results are comparable)
const EDIT_BENCH = { edits: 40, intervalMs: 250, seed: 1337 };

const panel = {
  background: 'rgba(0,0,0,0.8)', color: 'white', padding: '16px 20px',
  borderRadius: '8px', fontFamily: 'monospace',
};
const button = (color, busy, outline = false) => ({
  padding: '8px 14px', background: outline ? 'transparent' : color, color: 'white',
  border: `1px solid ${color}`, borderRadius: '4px', cursor: busy ? 'wait' : 'pointer',
  opacity: busy ? 0.6 : 1,
});
const section = { marginTop: '14px', paddingTop: '12px', borderTop: '1px solid #333' };

// === REACT COMPONENT ===
export default function VoxelEnginePrototype() {
  const mountRef = useRef(null);
  const [mode, setMode] = useState('cpu');
  const [tool, setTool] = useState('blocks'); // 'blocks' | 'explosion'
  const [radius, setRadius] = useState(8);
  const [stats, setStats] = useState(null);
  const [editInfo, setEditInfo] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  // gpu: { ready, direct, timestamps, note }
  const [gpu, setGpu] = useState({ ready: false, direct: false, timestamps: false, note: 'Initializing…' });
  const [available, setAvailable] = useState({ cpu: true });
  const [bench, setBench] = useState({ running: false, progress: null, idle: null, results: [] });
  const [frameInfo, setFrameInfo] = useState(null);
  const [recorder] = useState(() => new FrameRecorder());
  const [check, setCheck] = useState(null); // correctness check report

  // Event handlers registered once (pointer events, render loop) need current values.
  const latest = useRef({});
  latest.current = { mode, tool, radius, benchRunning: bench.running };

  // Everything that lives outside React's render cycle.
  const ctx = useRef({
    controller: null,
    chunk: null, // survives remounts (StrictMode) so edits are not lost
    mesher: null,
    hideHover: null,
  });

  const modeColor = `#${MODES[mode].color.toString(16).padStart(6, '0')}`;
  const locked = busy || bench.running;

  // ---- scene / renderer / controller lifecycle ----------------------------
  useEffect(() => {
    let cancelled = false;
    let frameId = 0;
    let renderer = null;
    let controls = null;
    let editor = null;
    const mount = mountRef.current;
    const c = ctx.current;

    (async () => {
      // One device shared by compute and rendering: this is what makes zero-copy possible.
      let device = null;
      try {
        device = await getGpuDevice();
      } catch (err) {
        console.warn('WebGPU unavailable:', err.message);
        if (!cancelled) setGpu({ ready: true, direct: false, timestamps: false, note: err.message });
      }
      if (cancelled) return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1a2e);

      const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(CHUNK_SIZE * 1.5, CHUNK_SIZE * 1.5, CHUNK_SIZE * 1.5);

      renderer = new THREE.WebGPURenderer(device ? { device, antialias: true } : { antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      mount.appendChild(renderer.domElement);

      const light = new THREE.DirectionalLight(0xffffff, 1);
      light.position.set(50, 50, 50);
      scene.add(light);
      scene.add(new THREE.AmbientLight(0x404040));

      const gridHelper = new THREE.GridHelper(CHUNK_SIZE, CHUNK_SIZE);
      gridHelper.position.set(CHUNK_SIZE / 2, 0, CHUNK_SIZE / 2);
      scene.add(gridHelper);

      // Outline of the voxel under the cursor (scaled up to preview an explosion's reach)
      const highlight = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1.04, 1.04, 1.04)),
        new THREE.LineBasicMaterial({ color: 0xffffff })
      );
      highlight.visible = false;
      scene.add(highlight);
      c.hideHover = () => { highlight.visible = false; };

      await renderer.init();
      if (cancelled) return;

      const bridge = new DirectRenderBridge(renderer);
      c.mesher = null;
      if (device) {
        c.mesher = await GpuMesher.create(mergeOptionsFromUrl());
        if (cancelled) return;
        if (!bridge.isWebGPU) console.warn('THREE.WebGPURenderer is on the WebGL2 backend; zero-copy disabled.');
        setGpu({
          ready: true,
          direct: bridge.isWebGPU,
          timestamps: c.mesher.timestamps,
          mergeImpl: c.mesher.mergeImpl,
          mergeWorkgroupSize: c.mesher.mergeWorkgroupSize,
          note: bridge.isWebGPU ? '' : 'Renderer fell back to WebGL2, so zero-copy rendering is unavailable.',
        });
      }

      const controller = new MeshController({ scene, mesher: c.mesher, bridge: c.mesher ? bridge : null, chunk: c.chunk });
      c.chunk = controller.chunk;
      c.controller = controller;
      setAvailable(Object.fromEntries(Object.keys(MODES).map((m) => [m, controller.isAvailable(m)])));

      controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2);
      controls.enableDamping = true;
      controls.update();

      editor = new VoxelEditor({
        domElement: renderer.domElement,
        camera,
        getChunk: () => controller.chunk,
        isEnabled: () => !latest.current.benchRunning,
        onHover: (hit) => {
          highlight.visible = !!hit;
          if (!hit) return;
          const { tool: t, radius: r } = latest.current;
          highlight.position.set(hit.x + 0.5, hit.y + 0.5, hit.z + 0.5);
          highlight.scale.setScalar(t === 'explosion' ? 2 * r + 1 : 1);
        },
        onPick: ({ button, hit }) => {
          const { tool: t, radius: r } = latest.current;
          if (t === 'explosion') {
            if (button === 0) applyEdit(`Explosion r=${r}`, (ctrl) => ctrl.explode(hit.x, hit.y, hit.z, r));
          } else if (button === 0) {
            applyEdit('Removed block', (ctrl) => (ctrl.chunk.set(hit.x, hit.y, hit.z, false) ? 1 : 0));
          } else if (button === 2) {
            const [nx, ny, nz] = hit.normal;
            if (nx || ny || nz) {
              applyEdit('Placed block', (ctrl) => (ctrl.chunk.set(hit.x + nx, hit.y + ny, hit.z + nz, true) ? 1 : 0));
            }
          }
        },
      });

      const animate = (now) => {
        frameId = requestAnimationFrame(animate);
        recorder.tick(now);
        controls.update();
        renderer.render(scene, camera);
      };
      frameId = requestAnimationFrame(animate);
    })().catch((err) => {
      console.error(err);
      if (!cancelled) setError(err.message);
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
      editor?.dispose();
      controls?.dispose();
      c.controller?.dispose();
      c.controller = null;
      c.mesher = null;
      c.hideHover = null;
      if (renderer) {
        renderer.dispose();
        renderer.domElement.remove();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Live frame-time readout (rolling window), refreshed twice a second.
  useEffect(() => {
    const id = setInterval(() => setFrameInfo(recorder.recent()), 500);
    return () => clearInterval(id);
  }, [recorder]);

  // ---- editing ------------------------------------------------------------

  // Applies an edit to the chunk right away (so the next raycast sees it) and
  // remeshes in the current mode; rapid edits are coalesced into one remesh.
  const applyEdit = useCallback((label, mutate) => {
    const ctrl = ctx.current.controller;
    const { mode: currentMode, benchRunning } = latest.current;
    if (!ctrl || benchRunning) return;

    const voxels = mutate(ctrl);
    if (!voxels) return;
    ctx.current.hideHover?.();

    ctrl.remeshCoalesced(currentMode)
      .then((r) => setEditInfo({
        label, voxels, mode: currentMode,
        totalMs: r.totalMs, bytesUp: r.bytesUp, bytesDown: r.bytesDown, quadCount: r.quadCount,
      }))
      .catch((err) => { console.error(err); setError(err.message); });
  }, []);

  const randomExplosion = () => {
    const margin = Math.ceil(radius / 2);
    const pick = () => margin + Math.floor(Math.random() * (CHUNK_SIZE - 2 * margin));
    const [x, y, z] = [pick(), pick(), pick()];
    applyEdit(`Explosion r=${radius}`, (ctrl) => ctrl.explode(x, y, z, radius));
  };

  const resetTerrain = async () => {
    const ctrl = ctx.current.controller;
    if (!ctrl) return;
    setError(null);
    setBusy(true);
    try {
      ctrl.resetTerrain();
      const r = await ctrl.remesh(mode);
      setEditInfo({ label: 'Reset terrain', voxels: 0, mode, totalMs: r.totalMs, bytesUp: r.bytesUp, bytesDown: r.bytesDown, quadCount: r.quadCount });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  // ---- static mesh benchmark (whole chunk, no edits) ----------------------
  const execute = async (runs) => {
    const ctrl = ctx.current.controller;
    if (!ctrl) return;
    setError(null);
    setBusy(true);
    try {
      const total = runs > 1 ? runs + WARMUP_RUNS : 1;
      const samples = [];
      let last = null;

      for (let i = 0; i < total; i++) {
        if (last?.geometry) last.geometry.dispose(); // only the final run's geometry is shown
        last = await ctrl.measure(mode);
        if (i >= total - runs) samples.push(last);
      }
      ctrl.show(last, mode);

      const med = (key) => median(samples.map((s) => s[key]));
      setStats({
        mode, runs, warmup: WARMUP_RUNS,
        mergeImpl: mode === 'cpu' ? null : ctx.current.mesher?.mergeImpl,
        quads: last.quadCount,
        overflowed: !!last.overflowed,
        naiveFaces: ctrl.naiveFaces,
        last: {
          totalMs: last.totalMs, gpuMs: last.gpuMs, passMs: last.passMs, readbackMs: last.readbackMs,
          bytesUp: last.bytesUp, bytesDown: last.bytesDown,
        },
        median: {
          totalMs: med('totalMs'), gpuMs: med('gpuMs'), readbackMs: med('readbackMs'),
          passMs: medianPassMs(samples),
        },
        minTotalMs: Math.min(...samples.map((s) => s.totalMs)),
      });
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setBusy(false);
    }
  };

  // ---- GPU vs CPU correctness check (default terrain, separate chunk) --------
  const runCheck = async () => {
    const mesher = ctx.current.mesher;
    if (!mesher) return;
    setError(null);
    setBusy(true);
    try {
      setCheck(await runCorrectnessCheck(mesher));
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  // ---- scripted edit benchmark (frame-time stability) ---------------------
  const runBenchmark = async () => {
    const ctrl = ctx.current.controller;
    if (!ctrl) return;
    if (document.visibilityState !== 'visible') {
      setError('Keep this tab visible: the browser pauses rendering in background tabs, which would invalidate the frame times.');
      return;
    }
    setError(null);
    setBench({ running: true, progress: 'measuring idle frame time…', idle: null, results: [] });
    const results = [];
    let idle = null;
    try {
      idle = await measureIdleFrames(recorder);
      setBench((b) => ({ ...b, idle }));

      const modes = Object.keys(MODES).filter((m) => ctrl.isAvailable(m));
      for (const m of modes) {
        const result = await runEditBenchmark({
          controller: ctrl, recorder, mode: m, radius, ...EDIT_BENCH,
          onProgress: (done, total) =>
            setBench((b) => ({ ...b, progress: `${MODES[m].label}: edit ${done}/${total}` })),
        });
        results.push(result);
        setBench((b) => ({ ...b, results: [...results] }));
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      // Leave a clean, meshed chunk on screen.
      try {
        ctrl.resetTerrain();
        await ctrl.remesh(mode);
      } catch (err) {
        console.error(err);
      }
      setBench((b) => ({ ...b, running: false, progress: null, idle, results }));
    }
  };

  const exportBenchmark = () => {
    const payload = {
      meta: {
        date: new Date().toISOString(),
        userAgent: navigator.userAgent,
        gpu: getGpuInfo(),
        chunkSize: CHUNK_SIZE,
        mergeImpl: gpu.mergeImpl ?? null,
        mergeWorkgroupSize: gpu.mergeWorkgroupSize ?? null,
        ...EDIT_BENCH,
        radius: bench.results[0]?.radius ?? radius,
        display: { devicePixelRatio: window.devicePixelRatio, viewport: [window.innerWidth, window.innerHeight] },
      },
      idle: bench.idle,
      results: bench.results,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `edit-benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const modeDisabled = (key) => !available[key];

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <div ref={mountRef} />

      {/* Controls + stats */}
      <div style={{ ...panel, position: 'absolute', top: 20, left: 20, width: 340, maxHeight: 'calc(100vh - 72px)', overflowY: 'auto' }}>
        <h2 style={{ marginTop: 0 }}>🎮 Voxel Engine Prototype</h2>
        <div style={{ marginBottom: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {Object.entries(MODES).map(([key, { label }]) => (
            <label key={key} style={{ opacity: modeDisabled(key) ? 0.5 : 1 }}>
              <input
                type="radio" value={key} checked={mode === key}
                disabled={locked || modeDisabled(key)}
                onChange={() => setMode(key)}
              /> {label}
            </label>
          ))}
        </div>
        {gpu.note && <div style={{ color: '#f39c12', fontSize: '12px', marginBottom: '10px' }}>⚠️ {gpu.note}</div>}

        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={() => execute(1)} disabled={locked} style={button(modeColor, locked)}>
            Generate Mesh
          </button>
          <button onClick={() => execute(BENCH_RUNS)} disabled={locked} style={button(modeColor, locked, true)}>
            Benchmark ×{BENCH_RUNS}
          </button>
        </div>

        {available['gpu-readback'] && (
          <div style={{ marginTop: '8px' }}>
            <button onClick={runCheck} disabled={locked} style={button('#16a085', locked, true)}>
              Run correctness check
            </button>
            <span style={{ fontSize: '11px', color: '#95a5a6', marginLeft: '8px' }}>
              merge: {gpu.mergeImpl}{gpu.mergeImpl === 'bitrun' ? ` (wg ${gpu.mergeWorkgroupSize})` : ''}
            </span>
          </div>
        )}
        <CorrectnessResult report={check} />

        {error && <div style={{ color: '#ff6b6b', marginTop: '10px' }}>❌ {error}</div>}
        <MeshStats stats={stats} timestamps={gpu.timestamps} />

        {/* Editing */}
        <div style={section}>
          <div style={{ marginBottom: '6px' }}><strong>✏️ Edit terrain</strong></div>
          <label style={{ display: 'block' }}>
            <input type="radio" checked={tool === 'blocks'} onChange={() => setTool('blocks')} disabled={bench.running} />{' '}
            Blocks <span style={{ color: '#95a5a6' }}>(click: remove · right-click: place)</span>
          </label>
          <label style={{ display: 'block' }}>
            <input type="radio" checked={tool === 'explosion'} onChange={() => setTool('explosion')} disabled={bench.running} />{' '}
            Explosion <span style={{ color: '#95a5a6' }}>(click)</span>
          </label>
          <label style={{ display: 'block', marginTop: '6px' }}>
            Blast radius: <strong>{radius}</strong>
            <input
              type="range" min="3" max="24" value={radius}
              onChange={(e) => setRadius(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </label>
          <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
            <button onClick={randomExplosion} disabled={bench.running} style={button('#e67e22', bench.running)}>
              💥 Random explosion
            </button>
            <button onClick={resetTerrain} disabled={locked} style={button('#7f8c8d', locked, true)}>
              Reset terrain
            </button>
          </div>
          <div style={{ fontSize: '11px', color: '#95a5a6', marginTop: '6px' }}>
            Drag to orbit · right-drag to pan · scroll to zoom
          </div>
          <EditInfo info={editInfo} />
        </div>

        {/* Scripted benchmark */}
        <div style={section}>
          <div style={{ marginBottom: '6px' }}><strong>📈 Edit benchmark</strong></div>
          <div style={{ fontSize: '12px', color: '#95a5a6', marginBottom: '8px' }}>
            {EDIT_BENCH.edits} seeded explosions (r={radius}) every {EDIT_BENCH.intervalMs} ms in each available mode,
            recording frame times. Takes about {Math.round((EDIT_BENCH.edits * EDIT_BENCH.intervalMs / 1000 + 4) * Math.max(1, Object.values(available).filter(Boolean).length) + 2)} s.
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={runBenchmark} disabled={locked} style={button('#9b59b6', locked)}>
              {bench.running ? 'Running…' : 'Run edit benchmark'}
            </button>
            <button onClick={exportBenchmark} disabled={bench.running || bench.results.length === 0} style={button('#9b59b6', bench.running, true)}>
              Export JSON
            </button>
          </div>
          {bench.progress && <div style={{ marginTop: '8px', fontSize: '12px', color: '#f39c12' }}>⏳ {bench.progress}</div>}
        </div>
      </div>

      {/* Live frame time */}
      <div style={{ ...panel, padding: '8px 12px', position: 'absolute', top: 20, right: 20, fontSize: '12px' }}>
        {frameInfo
          ? <>🖥️ frame <strong>{fmt(frameInfo.avg, 1)} ms</strong> avg · <strong style={{ color: frameInfo.max > 33 ? '#ff6b6b' : 'white' }}>{fmt(frameInfo.max, 1)} ms</strong> worst</>
          : '🖥️ frame –'}
      </div>

      {/* Benchmark results */}
      {(bench.idle || bench.results.length > 0) && (
        <div style={{ ...panel, position: 'absolute', bottom: 20, right: 20, maxWidth: 'calc(100vw - 420px)', overflowX: 'auto' }}>
          <strong>Frame times during edits (ms)</strong>
          <BenchmarkResults idle={bench.idle} results={bench.results} />
          <div style={{ fontSize: '11px', color: '#95a5a6', marginTop: '6px' }}>
            hitches = frames slower than 2× the median · edit ms = median remesh time · ↑/↓ = CPU↔GPU bytes per edit
          </div>
        </div>
      )}
    </div>
  );
}
