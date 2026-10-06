import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { ChunkData, CHUNK_SIZE } from './src/voxel/ChunkData';
import { TerrainGenerator } from './src/voxel/TerrainGenerator';
import { cpuGreedyMeshGeneration } from './src/meshing/cpuGreedyMesh';
import { GpuMesher } from './src/meshing/gpuMeshPipeline';
import { DirectRenderBridge } from './src/render/DirectRenderBridge';
import { getGpuDevice } from './src/gpu/gpuContext';

/**
 * VOXEL ENGINE PROTOTYPE - WebGPU Compute Shader Meshing
 *
 * Three meshing modes over the same chunk, for the CPU-vs-GPU benchmark:
 *   cpu          - CPU greedy meshing, geometry built on the CPU
 *   gpu-readback - GPU greedy meshing, vertices copied back to the CPU (mapAsync)
 *   gpu-direct   - GPU greedy meshing, vertices stay on the GPU and are drawn with
 *                  drawIndirect() (zero-copy)
 */

const MODES = {
  cpu: { label: 'CPU Greedy Meshing', color: 0x3498db },
  'gpu-readback': { label: 'GPU + Readback', color: 0xe74c3c },
  'gpu-direct': { label: 'GPU Direct (zero-copy)', color: 0x2ecc71 },
};

const WARMUP_RUNS = 2;
const BENCH_RUNS = 20;

const median = (values) => {
  const v = values.filter((x) => typeof x === 'number').sort((a, b) => a - b);
  if (v.length === 0) return null;
  const mid = v.length >> 1;
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
};

const fmt = (ms) => (typeof ms === 'number' ? `${ms.toFixed(2)} ms` : 'n/a');

// === REACT COMPONENT ===
export default function VoxelEnginePrototype() {
  const mountRef = useRef(null);
  const [mode, setMode] = useState('cpu');
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  // gpu: { ready, direct, timestamps, note }
  const [gpu, setGpu] = useState({ ready: false, direct: false, timestamps: false, note: 'Initializing…' });

  // Everything that lives outside React's render cycle.
  const ctx = useRef({
    scene: null,
    renderer: null,
    bridge: null,
    mesher: null,
    chunk: null,
    readbackTarget: null,
    direct: null, // { mesh, target, dispose }
    naiveFaces: null,
  });
  const materials = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(MODES).map(([key, { color }]) => [
          key,
          new THREE.MeshStandardMaterial({ color, flatShading: true }),
        ])
      ),
    []
  );

  useEffect(() => {
    let cancelled = false;
    let frameId = 0;
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
      camera.lookAt(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2);

      const renderer = new THREE.WebGPURenderer(device ? { device, antialias: true } : { antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      mount.appendChild(renderer.domElement);
      c.renderer = renderer;

      const light = new THREE.DirectionalLight(0xffffff, 1);
      light.position.set(50, 50, 50);
      scene.add(light);
      scene.add(new THREE.AmbientLight(0x404040));

      const gridHelper = new THREE.GridHelper(CHUNK_SIZE, CHUNK_SIZE);
      gridHelper.position.set(CHUNK_SIZE / 2, 0, CHUNK_SIZE / 2);
      scene.add(gridHelper);

      c.scene = scene;

      await renderer.init();
      if (cancelled) return;

      const bridge = new DirectRenderBridge(renderer);
      c.bridge = bridge;

      if (device) {
        c.mesher = await GpuMesher.create();
        if (cancelled) return;
        const direct = bridge.isWebGPU;
        if (!direct) console.warn('THREE.WebGPURenderer is on the WebGL2 backend; zero-copy disabled.');
        setGpu({
          ready: true,
          direct,
          timestamps: c.mesher.timestamps,
          note: direct ? '' : 'Renderer fell back to WebGL2, so zero-copy rendering is unavailable.',
        });
      }

      const animate = () => {
        frameId = requestAnimationFrame(animate);
        const radius = CHUNK_SIZE * 1.5;
        camera.position.x = Math.cos(Date.now() * 0.0001) * radius + CHUNK_SIZE / 2;
        camera.position.z = Math.sin(Date.now() * 0.0001) * radius + CHUNK_SIZE / 2;
        camera.lookAt(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2);
        renderer.renderAsync(scene, camera); // WebGPURenderer requires renderAsync
      };
      animate();
    })().catch((err) => {
      console.error(err);
      if (!cancelled) setError(err.message);
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
      c.direct?.dispose();
      c.direct = null;
      if (c.renderer) {
        c.renderer.dispose();
        c.renderer.domElement.remove();
        c.renderer = null;
      }
      c.scene = null;
      c.bridge = null;
      c.mesher = null;
      c.readbackTarget = null;
    };
  }, []);

  const getChunk = () => {
    const c = ctx.current;
    if (!c.chunk) {
      c.chunk = new ChunkData();
      new TerrainGenerator().generate(c.chunk);
    }
    return c.chunk;
  };

  // Runs one meshing pass in the given mode and returns a uniform result shape.
  const runOnce = async (runMode) => {
    const c = ctx.current;
    const chunk = getChunk();

    if (runMode === 'cpu') {
      const r = cpuGreedyMeshGeneration(chunk);
      c.naiveFaces = r.naiveFaceCount;
      return {
        geometry: r.geometry,
        quadCount: r.quadCount,
        totalMs: r.time,
        submitMs: null,
        gpuMs: null,
        readbackMs: 0,
      };
    }

    if (runMode === 'gpu-readback') {
      c.readbackTarget ??= c.mesher.createReadbackTarget();
      return c.mesher.meshWithReadback(chunk, c.readbackTarget);
    }

    // gpu-direct: upload occupancy, compute, done. Vertices never leave the GPU.
    c.direct ??= c.bridge.createDirectMesh(materials['gpu-direct']);
    const t0 = performance.now();
    const { submitMs, gpuMs } = c.mesher.mesh(chunk, c.direct.target, { profile: true });
    await c.mesher.whenIdle(); // waits for completion; reads nothing back
    const totalMs = performance.now() - t0;

    // Stats only (32 bytes), outside the timed region.
    const counters = await c.mesher.readCounters(c.direct.target);
    return {
      mesh: c.direct.mesh,
      quadCount: counters.quadCount,
      overflowed: counters.overflowed,
      totalMs,
      submitMs,
      gpuMs: await gpuMs,
      readbackMs: 0,
    };
  };

  const showResult = (result, shownMode) => {
    const c = ctx.current;
    if (!c.scene) return;

    const old = c.scene.getObjectByName('voxelMesh');
    if (old) {
      c.scene.remove(old);
      if (old !== c.direct?.mesh) old.geometry.dispose();
    }

    let mesh = result.mesh;
    if (!mesh) {
      mesh = new THREE.Mesh(result.geometry, materials[shownMode]);
      mesh.name = 'voxelMesh';
    }
    c.scene.add(mesh);
  };

  const execute = async (runs) => {
    setError(null);
    setBusy(true);
    try {
      const total = runs > 1 ? runs + WARMUP_RUNS : 1;
      const samples = [];
      let last = null;

      for (let i = 0; i < total; i++) {
        if (last?.geometry) last.geometry.dispose(); // only the final run's geometry is shown
        last = await runOnce(mode);
        if (i >= total - runs) samples.push(last);
      }

      showResult(last, mode);

      setStats({
        mode,
        runs,
        quads: last.quadCount,
        overflowed: !!last.overflowed,
        naiveFaces: ctx.current.naiveFaces,
        last: { totalMs: last.totalMs, submitMs: last.submitMs, gpuMs: last.gpuMs, readbackMs: last.readbackMs },
        median: {
          totalMs: median(samples.map((s) => s.totalMs)),
          gpuMs: median(samples.map((s) => s.gpuMs)),
          readbackMs: median(samples.map((s) => s.readbackMs)),
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

  const modeDisabled = (key) => {
    if (key === 'cpu') return false;
    if (key === 'gpu-direct') return !gpu.direct;
    return !gpu.ready || !ctx.current.mesher;
  };

  const modeColor = `#${MODES[mode].color.toString(16).padStart(6, '0')}`;
  const shown = stats && (stats.runs > 1 ? stats.median : stats.last);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <div ref={mountRef} />
      <div style={{
        position: 'absolute', top: 20, left: 20,
        background: 'rgba(0,0,0,0.8)', color: 'white', padding: '20px',
        borderRadius: '8px', fontFamily: 'monospace', minWidth: '320px',
      }}>
        <h2 style={{ marginTop: 0 }}>🎮 Voxel Engine Prototype</h2>
        <div style={{ marginBottom: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {Object.entries(MODES).map(([key, { label }]) => (
            <label key={key} style={{ opacity: modeDisabled(key) ? 0.5 : 1 }}>
              <input
                type="radio" value={key} checked={mode === key}
                disabled={busy || modeDisabled(key)}
                onChange={() => setMode(key)}
              /> {label}
            </label>
          ))}
        </div>
        {gpu.note && <div style={{ color: '#f39c12', fontSize: '12px', marginBottom: '10px' }}>⚠️ {gpu.note}</div>}

        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={() => execute(1)} disabled={busy} style={{
            padding: '10px 16px', background: modeColor,
            color: 'white', border: 'none', borderRadius: '4px', cursor: busy ? 'wait' : 'pointer',
          }}>
            Generate Mesh
          </button>
          <button onClick={() => execute(BENCH_RUNS)} disabled={busy} style={{
            padding: '10px 16px', background: 'transparent',
            color: 'white', border: `1px solid ${modeColor}`, borderRadius: '4px', cursor: busy ? 'wait' : 'pointer',
          }}>
            Benchmark ×{BENCH_RUNS}
          </button>
        </div>

        {error && <div style={{ color: '#ff6b6b', marginTop: '10px' }}>❌ {error}</div>}

        {stats && (
          <div style={{ marginTop: '15px' }}>
            <div style={{ fontSize: '11px', color: '#95a5a6', marginBottom: '6px' }}>
              {MODES[stats.mode].label} · {stats.runs > 1 ? `median of ${stats.runs} runs (after ${WARMUP_RUNS} warm-up)` : 'single run'}
            </div>
            <div>⏱️ Total Time: <strong>{fmt(shown.totalMs)}</strong></div>
            {stats.runs > 1 && <div>🏁 Best Run: <strong>{fmt(stats.minTotalMs)}</strong></div>}
            {stats.mode !== 'cpu' && (
              <div>⚡ GPU Compute: <strong>{gpu.timestamps ? fmt(shown.gpuMs) : 'n/a (no timestamp-query)'}</strong></div>
            )}
            <div>
              📤 Readback:{' '}
              <strong style={{ color: stats.mode === 'gpu-direct' ? '#2ecc71' : 'white' }}>
                {stats.mode === 'gpu-direct' ? 'none (zero-copy)' : fmt(shown.readbackMs)}
              </strong>
            </div>
            <div>📉 Naive Faces (No Merge): <strong>{stats.naiveFaces != null ? stats.naiveFaces.toLocaleString() : 'run CPU once to compute'}</strong></div>
            <div>🚀 Greedy Quads (Merged): <strong style={{ color: '#2ecc71' }}>{stats.quads.toLocaleString()}</strong></div>
            {stats.overflowed && (
              <div style={{ color: '#ff6b6b' }}>⚠️ Output buffer overflowed — mesh is truncated</div>
            )}
            <div style={{ fontSize: '11px', color: '#95a5a6', marginTop: '6px' }}>
              CPU time excludes Three's geometry upload on first draw.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
