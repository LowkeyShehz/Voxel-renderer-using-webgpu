import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CHUNK_SIZE } from './src/voxel/ChunkData';
import { MAX_GRID_SIZE } from './src/voxel/ChunkManager.js';
import { GpuMesher, mergeOptionsFromUrl } from './src/meshing/gpuMeshPipeline';
import { runCorrectnessCheck } from './src/meshing/correctnessCheck';
import { MeshController, MODES, createWorld, withoutMesh } from './src/meshing/MeshController';
import { DirectRenderBridge } from './src/render/DirectRenderBridge';
import { FreeFlyControls } from './src/render/FreeFlyControls';
import { getGpuDevice, getGpuInfo } from './src/gpu/gpuContext';
import { VoxelEditor } from './src/interaction/VoxelEditor';
import { FrameRecorder } from './src/benchmark/FrameRecorder';
import { runEditBenchmark, measureIdleFrames } from './src/benchmark/editBenchmark';
import { runSweep, estimateSweepSeconds, SWEEP_REPEATS } from './src/benchmark/sweepBenchmark';
import { summarizeSweep } from './src/benchmark/sweepSummary.js';
import { runMergeAb, MERGE_AB_ROUNDS } from './src/benchmark/mergeAbBenchmark.js';
import { median, medianPassMs } from './src/benchmark/stats';
import { MeshStats, EditInfo, BenchmarkResults, CorrectnessResult, MergeAbResult, WorldInfo, SweepResults, fmt } from './src/ui/panels';

/**
 * VOXEL ENGINE PROTOTYPE - WebGPU Compute Shader Meshing
 *
 * Phase 5: an N x N grid of 96^3 chunks. Every chunk is meshed on its own (with its
 * neighbours' borders, so shared faces are culled) and drawn at its world offset;
 * edits remesh only the chunks they touch, one per frame. The three modes can be
 * compared on frame-time stability while editing, at several world sizes.
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

// Falls back to a timeout: requestAnimationFrame never fires in a hidden tab, which
// would otherwise leave the UI stuck in its busy state.
const nextFrame = (timeoutMs = 100) =>
  new Promise((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    requestAnimationFrame(() => {
      clearTimeout(timer);
      resolve();
    });
  });

// === REACT COMPONENT ===
export default function VoxelEnginePrototype() {
  const mountRef = useRef(null);
  const [mode, setMode] = useState('cpu');
  const [tool, setTool] = useState('blocks'); // 'blocks' | 'explosion'
  const [radius, setRadius] = useState(8);
  const [gridSize, setGridSize] = useState(1);
  const [cameraMode, setCameraMode] = useState('orbit'); // 'orbit' | 'fly'
  const [pointerLocked, setPointerLocked] = useState(false);
  const [stats, setStats] = useState(null);
  const [editInfo, setEditInfo] = useState(null);
  const [worldInfo, setWorldInfo] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  // gpu: { ready, direct, timestamps, note }
  const [gpu, setGpu] = useState({ ready: false, direct: false, timestamps: false, note: 'Initializing…' });
  const [available, setAvailable] = useState({ cpu: true });
  const [bench, setBench] = useState({ running: false, progress: null, idle: null, results: [] });
  const [frameInfo, setFrameInfo] = useState(null);
  const [recorder] = useState(() => new FrameRecorder());
  const [check, setCheck] = useState(null); // correctness check report
  const [mergeAb, setMergeAb] = useState({ running: false, progress: null, report: null }); // legacy vs bit-run merge timing
  const [sweepMin, setSweepMin] = useState(1); // sweep grids sweepMin x sweepMin .. sweepMax x sweepMax
  const [sweepMax, setSweepMax] = useState(4);
  const [sweep, setSweep] = useState({ running: false, progress: null, idle: null, runs: [], summary: [], meta: null });

  // Event handlers registered once (pointer events, render loop) need current values.
  const latest = useRef({});
  latest.current = { mode, tool, radius, benchRunning: bench.running || sweep.running };

  // Everything that lives outside React's render cycle.
  const ctx = useRef({
    controller: null,
    world: null, // survives remounts (StrictMode) so edits are not lost
    mesher: null,
    hideHover: null,
    frameWorld: null,
    orbit: null,
    fly: null,
  });

  const modeColor = `#${MODES[mode].color.toString(16).padStart(6, '0')}`;
  const locked = busy || bench.running || sweep.running || mergeAb.running;

  /** Refreshes the world summary (chunks, quads, GPU memory). */
  const updateWorldInfo = useCallback(() => {
    const ctrl = ctx.current.controller;
    if (!ctrl) return;
    setWorldInfo({
      gridSize: ctrl.world.gridSize,
      chunks: ctrl.world.chunks.length,
      quads: ctrl.shownQuadCount(),
      memory: ctrl.gpuMemory(),
    });
  }, []);

  // ---- scene / renderer / controller lifecycle ----------------------------
  useEffect(() => {
    let cancelled = false;
    let frameId = 0;
    let renderer = null;
    let orbit = null;
    let fly = null;
    let editor = null;
    const mount = mountRef.current;
    const c = ctx.current;
    const onLockChange = () => setPointerLocked(!!document.pointerLockElement);
    document.addEventListener('pointerlockchange', onLockChange);

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

      const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 4000);

      renderer = new THREE.WebGPURenderer(device ? { device, antialias: true } : { antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      mount.appendChild(renderer.domElement);

      const light = new THREE.DirectionalLight(0xffffff, 1);
      light.position.set(50, 50, 50);
      scene.add(light);
      scene.add(new THREE.AmbientLight(0x404040));

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

      const controller = new MeshController({ scene, mesher: c.mesher, bridge: c.mesher ? bridge : null, world: c.world });
      c.world = controller.world;
      c.controller = controller;
      setAvailable(Object.fromEntries(Object.keys(MODES).map((m) => [m, controller.isAvailable(m)])));
      setGridSize(controller.world.gridSize);

      orbit = new OrbitControls(camera, renderer.domElement);
      orbit.enableDamping = true;
      fly = new FreeFlyControls(camera, renderer.domElement);
      c.orbit = orbit;
      c.fly = fly;

      // Grid floor + camera framing for the current world size.
      let gridHelper = null;
      c.frameWorld = (world) => {
        const [sx, sy, sz] = world.bounds.max;
        if (gridHelper) {
          scene.remove(gridHelper);
          gridHelper.dispose();
        }
        gridHelper = new THREE.GridHelper(sx, sx / 4);
        gridHelper.position.set(sx / 2, 0, sz / 2);
        scene.add(gridHelper);
        const reach = Math.max(sx, sz);
        camera.position.set(sx / 2 + reach, sy * 1.5 + reach * 0.5, sz / 2 + reach);
        orbit.target.set(sx / 2, sy / 2, sz / 2);
        camera.lookAt(orbit.target);
        orbit.update();
      };
      c.frameWorld(controller.world);
      updateWorldInfo();

      editor = new VoxelEditor({
        domElement: renderer.domElement,
        camera,
        getWorld: () => controller.world,
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
            applyEdit('Removed block', (ctrl) => (ctrl.setVoxel(hit.x, hit.y, hit.z, false) ? 1 : 0));
          } else if (button === 2) {
            const [nx, ny, nz] = hit.normal;
            if (nx || ny || nz) {
              applyEdit('Placed block', (ctrl) => (ctrl.setVoxel(hit.x + nx, hit.y + ny, hit.z + nz, true) ? 1 : 0));
            }
          }
        },
      });

      let last = performance.now();
      const animate = (now) => {
        frameId = requestAnimationFrame(animate);
        recorder.tick(now);
        const dt = (now - last) / 1000;
        last = now;
        if (orbit.enabled) orbit.update();
        fly.update(dt);
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
      document.removeEventListener('pointerlockchange', onLockChange);
      editor?.dispose();
      orbit?.dispose();
      fly?.dispose();
      c.controller?.dispose();
      c.controller = null;
      c.mesher = null;
      c.hideHover = null;
      c.frameWorld = null;
      c.orbit = null;
      c.fly = null;
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

  // Orbit vs free-fly camera.
  useEffect(() => {
    const { orbit, fly } = ctx.current;
    if (!orbit || !fly) return;
    orbit.enabled = cameraMode === 'orbit';
    fly.enabled = cameraMode === 'fly';
    if (cameraMode === 'orbit') {
      if (document.pointerLockElement) document.exitPointerLock();
      // Orbit around the point the fly camera was looking at.
      const cam = orbit.object;
      const dir = new THREE.Vector3();
      cam.getWorldDirection(dir);
      orbit.target.copy(cam.position).addScaledVector(dir, CHUNK_SIZE);
      orbit.update();
    }
  }, [cameraMode]);

  // ---- world size ---------------------------------------------------------

  /** Builds a new gridSize x gridSize world and meshes it in `meshMode` (null = don't). */
  const changeWorld = useCallback(async (size, meshMode) => {
    const c = ctx.current;
    const ctrl = c.controller;
    if (!ctrl) return null;
    setGridSize(size);
    if (ctrl.world.gridSize !== size) {
      await nextFrame(); // let React paint the busy state before the (blocking) generation
      ctrl.setWorld(createWorld(size));
      c.world = ctrl.world;
      c.frameWorld?.(ctrl.world);
      setStats(null);
      setEditInfo(null);
    }
    const result = meshMode ? await ctrl.remesh(meshMode) : null;
    updateWorldInfo();
    return result;
  }, [updateWorldInfo]);

  const onGridSize = async (size) => {
    setError(null);
    setBusy(true);
    try {
      await changeWorld(size, mode);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setGridSize(ctx.current.controller?.world.gridSize ?? size);
      updateWorldInfo();
      setBusy(false);
    }
  };

  // ---- editing ------------------------------------------------------------

  // Applies an edit to the world right away (so the next raycast sees it) and
  // remeshes the affected chunks in the current mode; rapid edits coalesce.
  const applyEdit = useCallback((label, mutate) => {
    const ctrl = ctx.current.controller;
    const { mode: currentMode, benchRunning } = latest.current;
    if (!ctrl || benchRunning) return;

    const voxels = mutate(ctrl);
    if (!voxels) return;
    ctx.current.hideHover?.();

    ctrl.remeshDirty(currentMode)
      .then((r) => {
        setEditInfo({
          label, voxels, mode: currentMode, chunks: r.chunks,
          totalMs: r.totalMs, wallMs: r.wallMs, bytesUp: r.bytesUp, bytesDown: r.bytesDown, quadCount: r.quadCount,
        });
        updateWorldInfo();
      })
      .catch((err) => { console.error(err); setError(err.message); });
  }, [updateWorldInfo]);

  const randomExplosion = () => {
    const ctrl = ctx.current.controller;
    if (!ctrl) return;
    const margin = Math.ceil(radius / 2);
    const pick = (axis) => margin + Math.floor(Math.random() * (ctrl.world.bounds.max[axis] - 2 * margin));
    const [x, y, z] = [pick(0), pick(1), pick(2)];
    applyEdit(`Explosion r=${radius}`, (c) => c.explode(x, y, z, radius));
  };

  const resetTerrain = async () => {
    const ctrl = ctx.current.controller;
    if (!ctrl) return;
    setError(null);
    setBusy(true);
    try {
      ctrl.resetTerrain();
      const r = await ctrl.remesh(mode);
      setEditInfo({
        label: 'Reset terrain', voxels: 0, mode, chunks: r.chunks,
        totalMs: r.totalMs, bytesUp: r.bytesUp, bytesDown: r.bytesDown, quadCount: r.quadCount,
      });
      updateWorldInfo();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  // ---- static mesh benchmark (whole world, no edits) ----------------------
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
        if (last) ctrl.discard(last); // only the final run's geometry is shown
        last = await ctrl.measure(mode);
        // Keep only the numbers: holding every run's geometry exhausts memory on big worlds.
        if (i >= total - runs) samples.push({ ...last, perChunk: last.perChunk.map(withoutMesh) });
      }
      ctrl.show(last, mode);

      const med = (key) => median(samples.map((s) => s[key]));
      setStats({
        mode, runs, warmup: WARMUP_RUNS,
        mergeImpl: mode === 'cpu' ? null : ctx.current.mesher?.mergeImpl,
        chunks: last.chunks,
        quads: last.quadCount,
        overflowed: !!last.overflowed,
        naiveFaces: last.naiveFaces,
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
      updateWorldInfo();
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setBusy(false);
    }
  };

  // ---- GPU vs CPU correctness check (own chunks, the world is untouched) ----
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

  // ---- legacy vs bit-run merge, timed alternately (paper Table VII) --------
  const runMergeAbBenchmark = async () => {
    const mesher = ctx.current.mesher;
    if (!mesher) return;
    setError(null);
    setMergeAb({ running: true, progress: 'starting…', report: null });
    let report = null;
    try {
      report = await runMergeAb(mesher, { onProgress: (progress) => setMergeAb((s) => ({ ...s, progress })) });
      report.meta = {
        date: new Date().toISOString(),
        userAgent: navigator.userAgent,
        gpu: getGpuInfo(),
        hardwareConcurrency: navigator.hardwareConcurrency ?? null,
        chunkSize: CHUNK_SIZE,
        display: { devicePixelRatio: window.devicePixelRatio, viewport: [window.innerWidth, window.innerHeight] },
      };
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setMergeAb({ running: false, progress: null, report });
    }
  };

  const exportMergeAb = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(mergeAb.report, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `merge-ab-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
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
      // Leave a clean, meshed world on screen.
      try {
        ctrl.resetTerrain();
        await ctrl.remesh(mode);
        updateWorldInfo();
      } catch (err) {
        console.error(err);
      }
      setBench((b) => ({ ...b, running: false, progress: null, idle, results }));
    }
  };

  // ---- sweep: grid sizes x modes x repeats ---------------------------------
  const sweepModes = Object.keys(MODES).filter((m) => available[m]);
  const sweepGrids = Array.from({ length: Math.max(0, sweepMax - sweepMin + 1) }, (_, i) => sweepMin + i);

  const runSweepBenchmark = async () => {
    const ctrl = ctx.current.controller;
    if (!ctrl) return;
    if (document.visibilityState !== 'visible') {
      setError('Keep this tab visible: the browser pauses rendering in background tabs, which would invalidate the frame times.');
      return;
    }
    setError(null);
    const meta = {
      date: new Date().toISOString(),
      userAgent: navigator.userAgent,
      gpu: getGpuInfo(),
      hardwareConcurrency: navigator.hardwareConcurrency ?? null,
      deviceMemoryGB: navigator.deviceMemory ?? null,
      display: { devicePixelRatio: window.devicePixelRatio, viewport: [window.innerWidth, window.innerHeight] },
      chunkSize: CHUNK_SIZE,
      mergeImpl: gpu.mergeImpl ?? null,
      mergeWorkgroupSize: gpu.mergeWorkgroupSize ?? null,
      grids: sweepGrids,
      modes: sweepModes,
      repeats: SWEEP_REPEATS,
      ...EDIT_BENCH,
      radius,
    };
    setSweep({ running: true, progress: 'measuring idle frame time…', idle: null, runs: [], summary: [], meta });
    let result = null;
    try {
      result = await runSweep({
        controller: ctrl, recorder, grids: sweepGrids, modes: sweepModes, repeats: SWEEP_REPEATS, radius, ...EDIT_BENCH,
        setGridSize: (n) => changeWorld(n, null),
        onProgress: (progress) => setSweep((s) => ({ ...s, progress })),
        onRun: (_, runs) => setSweep((s) => ({ ...s, runs: [...runs], summary: summarizeSweep(runs) })),
      });
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      try {
        ctrl.resetTerrain();
        await ctrl.remesh(mode);
      } catch (err) {
        console.error(err);
        setError(err.message);
      }
      setGridSize(ctrl.world.gridSize);
      updateWorldInfo();
      setSweep((s) => ({ ...s, ...(result ?? {}), running: false, progress: null }));
    }
  };

  const exportSweep = () => {
    const payload = { kind: 'sweep', meta: sweep.meta, idle: sweep.idle, summary: sweep.summary, runs: sweep.runs };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `sweep-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportBenchmark = () => {
    const payload = {
      meta: {
        date: new Date().toISOString(),
        userAgent: navigator.userAgent,
        gpu: getGpuInfo(),
        chunkSize: CHUNK_SIZE,
        gridSize: bench.results[0]?.gridSize ?? gridSize,
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

      {pointerLocked && (
        <div style={{
          position: 'absolute', left: '50%', top: '50%', width: 14, height: 14, marginLeft: -7, marginTop: -7,
          border: '2px solid white', borderRadius: '50%', pointerEvents: 'none', mixBlendMode: 'difference',
        }} />
      )}

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

        <label style={{ display: 'block', marginBottom: '10px' }}>
          World:{' '}
          <select value={gridSize} disabled={locked} onChange={(e) => onGridSize(Number(e.target.value))}>
            {Array.from({ length: MAX_GRID_SIZE }, (_, i) => i + 1).map((n) => (
              <option key={n} value={n}>{n}×{n} chunks</option>
            ))}
          </select>
          <span style={{ fontSize: '11px', color: '#95a5a6' }}> (meshes in the selected mode)</span>
        </label>

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
        {available['gpu-readback'] && (
          <div style={{ marginTop: '8px', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button onClick={runMergeAbBenchmark} disabled={locked || !gpu.timestamps} style={button('#8e44ad', locked, true)}
              title={gpu.timestamps ? '' : 'needs timestamp-query'}>
              {mergeAb.running ? 'Running…' : `Merge A/B ×${MERGE_AB_ROUNDS}`}
            </button>
            <button onClick={exportMergeAb} disabled={!mergeAb.report} style={button('#8e44ad', false, true)}>
              Export A/B JSON
            </button>
          </div>
        )}
        {mergeAb.progress && <div style={{ marginTop: '6px', fontSize: '12px', color: '#f39c12' }}>⏳ {mergeAb.progress}</div>}
        <MergeAbResult report={mergeAb.report} />

        {error && <div style={{ color: '#ff6b6b', marginTop: '10px' }}>❌ {error}</div>}
        <WorldInfo info={worldInfo} />
        <MeshStats stats={stats} timestamps={gpu.timestamps} />

        {/* Camera */}
        <div style={section}>
          <div style={{ marginBottom: '6px' }}><strong>🎥 Camera</strong></div>
          <label style={{ marginRight: '12px' }}>
            <input type="radio" checked={cameraMode === 'orbit'} onChange={() => setCameraMode('orbit')} /> Orbit
          </label>
          <label>
            <input type="radio" checked={cameraMode === 'fly'} onChange={() => setCameraMode('fly')} /> Free-fly
          </label>
          {cameraMode === 'fly' && (
            <div style={{ marginTop: '6px' }}>
              <button onClick={() => ctx.current.fly?.lockPointer()} disabled={pointerLocked} style={button('#34495e', false, true)}>
                {pointerLocked ? 'Mouse locked (Esc to release)' : 'Lock mouse'}
              </button>
            </div>
          )}
          <div style={{ fontSize: '11px', color: '#95a5a6', marginTop: '6px' }}>
            {cameraMode === 'orbit'
              ? 'Drag to orbit · right-drag to pan · scroll to zoom'
              : 'WASD move · Space/Shift up/down · drag (or lock mouse) to look · scroll = speed · clicks edit at the cursor (crosshair when locked)'}
          </div>
        </div>

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
          <EditInfo info={editInfo} />
        </div>

        {/* Scripted benchmark */}
        <div style={section}>
          <div style={{ marginBottom: '6px' }}><strong>📈 Edit benchmark</strong></div>
          <div style={{ fontSize: '12px', color: '#95a5a6', marginBottom: '8px' }}>
            {EDIT_BENCH.edits} seeded explosions (r={radius}) every {EDIT_BENCH.intervalMs} ms over the current
            {' '}{gridSize}×{gridSize} world in each available mode, recording frame times.
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

        {/* Sweep */}
        <div style={section}>
          <div style={{ marginBottom: '6px' }}><strong>🧪 Sweep benchmark</strong></div>
          <div style={{ fontSize: '12px', color: '#95a5a6', marginBottom: '8px' }}>
            For each world from{' '}
            <select value={sweepMin} disabled={locked} onChange={(e) => setSweepMin(Number(e.target.value))}>
              {Array.from({ length: sweepMax }, (_, i) => i + 1).map((n) => <option key={n} value={n}>{n}×{n}</option>)}
            </select>
            {' '}to{' '}
            <select value={sweepMax} disabled={locked} onChange={(e) => {
              const n = Number(e.target.value);
              setSweepMax(n);
              setSweepMin((m) => Math.min(m, n));
            }}>
              {Array.from({ length: MAX_GRID_SIZE }, (_, i) => i + 1).map((n) => <option key={n} value={n}>{n}×{n}</option>)}
            </select>
            {' '}and each available mode, {SWEEP_REPEATS} runs of: full meshing of the grid + the edit benchmark above.
            About {Math.ceil(estimateSweepSeconds({ grids: sweepGrids, modes: sweepModes, repeats: SWEEP_REPEATS, ...EDIT_BENCH }) / 60)} min; keep the tab visible.
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={runSweepBenchmark} disabled={locked} style={button('#2980b9', locked)}>
              {sweep.running ? 'Running…' : 'Run sweep'}
            </button>
            <button onClick={exportSweep} disabled={sweep.running || sweep.runs.length === 0} style={button('#2980b9', sweep.running, true)}>
              Export sweep JSON
            </button>
          </div>
          {sweep.progress && <div style={{ marginTop: '8px', fontSize: '12px', color: '#f39c12' }}>⏳ {sweep.progress}</div>}
        </div>
      </div>

      {/* Live frame time */}
      <div style={{ ...panel, padding: '8px 12px', position: 'absolute', top: 20, right: 20, fontSize: '12px' }}>
        {frameInfo
          ? <>🖥️ frame <strong>{fmt(frameInfo.avg, 1)} ms</strong> avg · <strong style={{ color: frameInfo.max > 33 ? '#ff6b6b' : 'white' }}>{fmt(frameInfo.max, 1)} ms</strong> worst</>
          : '🖥️ frame –'}
      </div>

      {/* Sweep results */}
      {sweep.summary.length > 0 && !(bench.idle || bench.results.length > 0) && (
        <div style={{ ...panel, position: 'absolute', bottom: 20, right: 20, maxWidth: 'calc(100vw - 420px)', maxHeight: '50vh', overflow: 'auto' }}>
          <strong>Sweep (median over runs, [min–max])</strong>
          <SweepResults summary={sweep.summary} />
        </div>
      )}

      {/* Benchmark results */}
      {(bench.idle || bench.results.length > 0) && (
        <div style={{ ...panel, position: 'absolute', bottom: 20, right: 20, maxWidth: 'calc(100vw - 420px)', overflowX: 'auto' }}>
          <strong>Frame times during edits (ms)</strong>
          <BenchmarkResults idle={bench.idle} results={bench.results} />
          <div style={{ fontSize: '11px', color: '#95a5a6', marginTop: '6px' }}>
            hitches = frames slower than 2× the median · edit ms = median meshing time per edit (all chunks it touched) · ↑/↓ = CPU↔GPU bytes per edit
          </div>
        </div>
      )}
    </div>
  );
}
