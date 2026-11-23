import * as THREE from 'three';
import { CHUNK_SIZE } from '../voxel/ChunkData';
import { ChunkManager } from '../voxel/ChunkManager.js';
import { TerrainGenerator } from '../voxel/TerrainGenerator';
import { cpuGreedyMeshGeneration } from './cpuGreedyMesh';
import { growCapacity, MAX_CAPACITY, POSITION_STRIDE_BYTES } from './gpuMeshPipeline';

/**
 * Three meshing modes over the same world, for the CPU-vs-GPU benchmark:
 *   cpu          - CPU greedy meshing, geometry built on the CPU
 *   gpu-readback - GPU greedy meshing, vertices copied back to the CPU (mapAsync)
 *   gpu-direct   - GPU greedy meshing, vertices stay on the GPU and are drawn with
 *                  drawIndirect() (zero-copy)
 */
export const MODES = {
  cpu: { label: 'CPU Greedy Meshing', color: 0x3498db },
  'gpu-readback': { label: 'GPU + Readback', color: 0xe74c3c },
  'gpu-direct': { label: 'GPU Direct (zero-copy)', color: 0x2ecc71 },
};

const VOXEL_BUFFER_BYTES = (CHUNK_SIZE ** 3) / 8;
const COUNTER_BYTES = 32;

/** Resolves on the next animation frame, so queued work lets a frame render in between. */
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

export function createWorld(gridSize = 1) {
  return new ChunkManager({ gridSize, generator: new TerrainGenerator() });
}

/**
 * Adds per-chunk results into one result of the same shape. Times, bytes and
 * quads are summed (chunks are meshed one after another); passMs is summed per
 * pass; quadCount is null if any chunk's count is unknown.
 */
function combine(results) {
  const sum = (key) => {
    const v = results.map((r) => r[key]).filter((x) => typeof x === 'number');
    return v.length ? v.reduce((a, b) => a + b, 0) : null;
  };
  let passMs = null;
  for (const r of results) {
    if (!r.passMs) continue;
    passMs ??= {};
    for (const [name, ms] of Object.entries(r.passMs)) passMs[name] = (passMs[name] ?? 0) + ms;
  }
  return {
    chunks: results.length,
    quadCount: results.every((r) => typeof r.quadCount === 'number') ? sum('quadCount') : null,
    naiveFaces: results.every((r) => typeof r.naiveFaces === 'number') ? sum('naiveFaces') : null,
    totalMs: sum('totalMs') ?? 0,
    submitMs: sum('submitMs'),
    gpuMs: sum('gpuMs'),
    passMs,
    readbackMs: sum('readbackMs') ?? 0,
    bytesUp: sum('bytesUp') ?? 0,
    bytesDown: sum('bytesDown') ?? 0,
    overflowed: results.some((r) => r.overflowed),
    grew: results.filter((r) => r.grew).length,
    perChunk: results,
  };
}

/**
 * Owns the world (a ChunkManager) and whatever is shown for each chunk, and knows
 * how to (re)mesh chunks in each mode. The UI, the editor and the benchmarks all go
 * through this so they measure exactly the same code path.
 *
 * Per-chunk results have the shape
 *   { chunk, quadCount|null, naiveFaces|null, totalMs, submitMs|null, gpuMs|null,
 *     passMs|null, readbackMs, bytesUp, bytesDown, overflowed, grew, geometry? | mesh? }
 *   totalMs   - meshing time for this chunk (CPU: generation; GPU: upload + compute
 *               to queue completion; readback: up to vertices on the CPU)
 *   bytesUp   - bytes sent CPU -> GPU (occupancy delta + neighbour borders, or geometry)
 *   bytesDown - bytes copied GPU -> CPU (vertex readback; 32 counter bytes in direct
 *               mode, read after the timed region to detect buffer overflow)
 * and multi-chunk results are combine()d into the same shape plus `chunks`,
 * `perChunk` and, for queued remeshes, `wallMs` (start to finish including the
 * frames rendered between chunks).
 */
export class MeshController {
  /**
   * @param {object} opts
   * @param {THREE.Scene} opts.scene
   * @param {import('./gpuMeshPipeline').GpuMesher|null} opts.mesher   null when WebGPU is unavailable
   * @param {import('../render/DirectRenderBridge').DirectRenderBridge|null} opts.bridge
   * @param {ChunkManager} [opts.world]  reused across controller instances so edits survive a remount
   */
  constructor({ scene, mesher, bridge, world }) {
    this.scene = scene;
    this.mesher = mesher;
    this.bridge = bridge;
    this.world = world ?? createWorld(1);

    this._shown = new Map(); // chunk -> THREE.Mesh in the scene
    this._shownQuads = new Map(); // chunk -> quad count of the shown mesh (null if unknown)
    this._direct = new Map(); // chunk -> { mesh, target, gpuBytes, dispose }
    this._gpuChunks = new Set(); // chunks with a GPU occupancy mirror
    this._readbackTarget = null;
    this._inflight = null;

    this.materials = Object.fromEntries(
      Object.entries(MODES).map(([key, { color }]) => [
        key,
        new THREE.MeshStandardMaterial({ color, flatShading: true }),
      ])
    );
  }

  isAvailable(mode) {
    if (mode === 'cpu') return true;
    if (mode === 'gpu-readback') return !!this.mesher;
    return !!this.mesher && !!this.bridge?.isWebGPU;
  }

  /** Replaces the world (e.g. a new grid size); frees everything owned for the old one. */
  setWorld(world) {
    this._releaseWorld();
    this.world = world;
  }

  // ---- editing -------------------------------------------------------------

  /** Removes a sphere of voxels (world coordinates); returns how many were removed. */
  explode(x, y, z, radius) {
    return this.world.carveSphere(x, y, z, radius).removed;
  }

  setVoxel(x, y, z, solid) {
    return this.world.set(x, y, z, solid);
  }

  /** Restores the original terrain (every chunk becomes dirty). */
  resetTerrain() {
    this.world.resetTerrain();
  }

  // ---- memory --------------------------------------------------------------

  /** GPU memory held for meshing, in bytes (excludes Three's own scene buffers). */
  gpuMemory() {
    let output = 0;
    for (const d of this._direct.values()) output += d.gpuBytes;
    if (this._readbackTarget) output += this._readbackTarget.capacity * POSITION_STRIDE_BYTES + COUNTER_BYTES;
    return {
      outputBytes: output,
      voxelBytes: this._gpuChunks.size * VOXEL_BUFFER_BYTES,
      scratchBytes: this.mesher?.scratchBytes ?? 0,
    };
  }

  // ---- meshing -------------------------------------------------------------

  /** Meshes one chunk in `mode` (with its neighbours) without touching the scene. */
  async meshChunk(chunk, mode) {
    if (!this.isAvailable(mode)) throw new Error(`${MODES[mode].label} is not available on this device.`);
    const neighbours = this.world.neighbours(chunk);

    if (mode === 'cpu') {
      const r = cpuGreedyMeshGeneration(chunk, neighbours);
      const g = r.geometry;
      const geometryBytes =
        g.attributes.position.array.byteLength +
        g.attributes.normal.array.byteLength +
        g.index.array.byteLength;
      return {
        chunk,
        geometry: g,
        quadCount: r.quadCount,
        naiveFaces: r.naiveFaceCount,
        totalMs: r.time,
        submitMs: null,
        gpuMs: null,
        passMs: null,
        readbackMs: 0,
        bytesUp: geometryBytes, // Three uploads the freshly built geometry on the next draw
        bytesDown: 0,
        overflowed: false,
        grew: false,
      };
    }

    this._gpuChunks.add(chunk);
    return mode === 'gpu-readback' ? this._meshReadback(chunk, neighbours) : this._meshDirect(chunk, neighbours);
  }

  async _meshReadback(chunk, neighbours, grew = false) {
    this._readbackTarget ??= this.mesher.createReadbackTarget();
    const target = this._readbackTarget;
    const r = await this.mesher.meshWithReadback(chunk, target, { neighbours });
    if (r.overflowed && target.capacity < MAX_CAPACITY) {
      r.geometry.dispose();
      this.mesher.destroyTarget(target);
      this._readbackTarget = this.mesher.createReadbackTarget(growCapacity(r.requestedVertices));
      return this._meshReadback(chunk, neighbours, true);
    }
    return {
      ...r,
      chunk,
      naiveFaces: null,
      // occupancy delta (+ borders) up, vertices down, then Three uploads those vertices again
      bytesUp: r.uploadedBytes + r.readbackBytes,
      bytesDown: r.readbackBytes,
      grew,
    };
  }

  // gpu-direct: upload occupancy delta, compute, done. Vertices never leave the GPU.
  async _meshDirect(chunk, neighbours, grew = false) {
    let direct = this._direct.get(chunk);
    if (!direct) {
      direct = this.bridge.createDirectMesh(this.materials['gpu-direct']);
      this._direct.set(chunk, direct);
    }
    const t0 = performance.now();
    const { submitMs, uploadedBytes, gpuMs } = this.mesher.mesh(chunk, direct.target, { profile: true, neighbours });
    await this.mesher.whenIdle(); // waits for completion; reads nothing back
    const totalMs = performance.now() - t0;
    const timing = await gpuMs;

    // Counters (32 bytes), outside the timed region: the quad count, and whether the
    // output buffer was too small. On overflow the chunk gets a bigger buffer and is
    // meshed again; the shown mesh is truncated only until that finishes.
    const counters = await this.mesher.readCounters(direct.target);
    if (counters.overflowed && direct.target.capacity < MAX_CAPACITY) {
      this._replaceDirect(chunk, growCapacity(counters.requestedVertices));
      return this._meshDirect(chunk, neighbours, true);
    }
    return {
      chunk,
      mesh: direct.mesh,
      quadCount: counters.quadCount,
      naiveFaces: null,
      overflowed: counters.overflowed,
      grew,
      totalMs,
      submitMs,
      gpuMs: timing?.totalMs ?? null,
      passMs: timing?.passMs ?? null,
      readbackMs: 0,
      bytesUp: uploadedBytes,
      bytesDown: COUNTER_BYTES,
    };
  }

  _replaceDirect(chunk, capacity) {
    const old = this._direct.get(chunk);
    if (old) {
      if (this._shown.get(chunk) === old.mesh) {
        this.scene.remove(old.mesh);
        this._shown.delete(chunk);
      }
      old.dispose();
    }
    this._direct.set(chunk, this.bridge.createDirectMesh(this.materials['gpu-direct'], capacity));
  }

  /** Puts a chunk result into the scene at the chunk's world position. */
  showChunk(result, mode) {
    const { chunk } = result;
    const old = this._shown.get(chunk);
    let mesh = result.mesh;
    if (!mesh) {
      mesh = new THREE.Mesh(result.geometry, this.materials[mode]);
      mesh.name = 'voxelMesh';
    }
    if (old && old !== mesh) {
      this.scene.remove(old);
      if (![...this._direct.values()].some((d) => d.mesh === old)) old.geometry.dispose();
    }
    const [ox, oy, oz] = this.world.origin(chunk);
    mesh.position.set(ox, oy, oz);
    if (old !== mesh) this.scene.add(mesh);
    this._shown.set(chunk, mesh);
    this._shownQuads.set(chunk, result.quadCount);
  }

  /** Total quads currently shown, or null if some chunk is unmeshed or its count unknown. */
  shownQuadCount() {
    let total = 0;
    for (const chunk of this.world.chunks) {
      const n = this._shownQuads.get(chunk);
      if (typeof n !== 'number') return null;
      total += n;
    }
    return total;
  }

  /** Shows a combined result (every chunk in it). */
  show(result, mode) {
    for (const r of result.perChunk) this.showChunk(r, mode);
  }

  /** Disposes a combined result's CPU geometries without showing them (benchmark runs). */
  discard(result) {
    for (const r of result.perChunk) r.geometry?.dispose();
  }

  /**
   * Meshes every chunk of the world, one after another, without touching the scene.
   * Clears the dirty flags of everything it meshed.
   */
  async measure(mode) {
    const results = [];
    for (const chunk of this.world.chunks) {
      this.world.clearDirty(chunk);
      results.push(await this.meshChunk(chunk, mode));
    }
    return combine(results);
  }

  /** measure() + show(): a full remesh of the whole world. */
  async remesh(mode) {
    const result = await this.measure(mode);
    this.show(result, mode);
    return result;
  }

  /**
   * Remeshes the chunks marked dirty by edits, one chunk per frame so several dirty
   * chunks do not land in a single frame. Edits that arrive meanwhile mark more
   * chunks dirty and are picked up by the same loop, so rapid clicks coalesce.
   * Every caller's promise resolves with the combined result of everything this
   * loop remeshed, with `wallMs` from start to finish (including the frames between
   * chunks).
   */
  remeshDirty(mode) {
    if (this._inflight) return this._inflight;
    const run = (async () => {
      const t0 = performance.now();
      const results = [];
      let chunk;
      while ((chunk = this.world.dirtyChunks[0])) {
        if (results.length > 0) await nextFrame();
        this.world.clearDirty(chunk); // before meshing, so edits during it re-mark it
        const r = await this.meshChunk(chunk, mode);
        this.showChunk(r, mode);
        results.push(r);
      }
      return { ...combine(results), wallMs: performance.now() - t0 };
    })();
    // Cleared by identity once settled. (With nothing dirty the body finishes before
    // `run` is assigned, so clearing from inside it would leave a stale promise.)
    this._inflight = run;
    const clear = () => {
      if (this._inflight === run) this._inflight = null;
    };
    run.then(clear, clear);
    return run;
  }

  _releaseWorld() {
    for (const mesh of this._shown.values()) {
      this.scene.remove(mesh);
      if (![...this._direct.values()].some((d) => d.mesh === mesh)) mesh.geometry.dispose();
    }
    this._shown.clear();
    this._shownQuads.clear();
    for (const d of this._direct.values()) d.dispose();
    this._direct.clear();
    for (const chunk of this._gpuChunks) this.mesher?.releaseChunk(chunk);
    this._gpuChunks.clear();
  }

  dispose() {
    this._releaseWorld();
    if (this._readbackTarget) this.mesher?.destroyTarget(this._readbackTarget);
    this._readbackTarget = null;
    for (const m of Object.values(this.materials)) m.dispose();
  }
}
