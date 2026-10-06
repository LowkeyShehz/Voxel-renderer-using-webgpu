import * as THREE from 'three';
import { ChunkData } from '../voxel/ChunkData';
import { TerrainGenerator } from '../voxel/TerrainGenerator';
import { cpuGreedyMeshGeneration } from './cpuGreedyMesh';

/**
 * Three meshing modes over the same chunk, for the CPU-vs-GPU benchmark:
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

/**
 * Owns the chunk and whatever is currently shown for it, and knows how to
 * (re)mesh it in each mode. The UI, the editor and the benchmarks all go through
 * this so they measure exactly the same code path.
 *
 * Every measure()/remesh() result has the same shape:
 *   { quadCount|null, totalMs, submitMs|null, gpuMs|null, passMs|null, readbackMs,
 *     bytesUp, bytesDown, overflowed?, geometry? | mesh? }
 *   bytesUp   - bytes sent CPU -> GPU for this remesh (occupancy delta, or geometry upload)
 *   bytesDown - bytes copied GPU -> CPU (vertex readback)
 */
export class MeshController {
  /**
   * @param {object} opts
   * @param {THREE.Scene} opts.scene
   * @param {import('./gpuMeshPipeline').GpuMesher|null} opts.mesher   null when WebGPU is unavailable
   * @param {import('../render/DirectRenderBridge').DirectRenderBridge|null} opts.bridge
   * @param {ChunkData} [opts.chunk]  reused across controller instances so edits survive a remount
   */
  constructor({ scene, mesher, bridge, chunk }) {
    this.scene = scene;
    this.mesher = mesher;
    this.bridge = bridge;
    this.generator = new TerrainGenerator();
    this.chunk = chunk ?? this._newChunk();
    this.naiveFaces = null;

    this._direct = null; // { mesh, target, dispose }
    this._readbackTarget = null;
    this._inflight = null;
    this._wantRemesh = false;

    this.materials = Object.fromEntries(
      Object.entries(MODES).map(([key, { color }]) => [
        key,
        new THREE.MeshStandardMaterial({ color, flatShading: true }),
      ])
    );
  }

  _newChunk() {
    const chunk = new ChunkData();
    this.generator.generate(chunk);
    return chunk;
  }

  isAvailable(mode) {
    if (mode === 'cpu') return true;
    if (mode === 'gpu-readback') return !!this.mesher;
    return !!this.mesher && !!this.bridge?.isWebGPU;
  }

  // ---- editing -------------------------------------------------------------

  /** Removes a sphere of voxels; returns how many were removed. */
  explode(x, y, z, radius) {
    return this.chunk.carveSphere(x, y, z, radius);
  }

  /** Restores the original terrain (all words marked dirty for the GPU mirror). */
  resetTerrain() {
    this.chunk.clear();
    this.generator.generate(this.chunk);
  }

  // ---- meshing -------------------------------------------------------------

  /**
   * Meshes the chunk in `mode` without touching the scene.
   * @param {{withCounters?: boolean}} [opts] withCounters=false skips the 32-byte
   *   counter read that supplies the quad count in direct mode (keeps benchmarks lean).
   */
  async measure(mode, { withCounters = true } = {}) {
    if (!this.isAvailable(mode)) throw new Error(`${MODES[mode].label} is not available on this device.`);

    if (mode === 'cpu') {
      const r = cpuGreedyMeshGeneration(this.chunk);
      this.naiveFaces = r.naiveFaceCount;
      const g = r.geometry;
      const geometryBytes =
        g.attributes.position.array.byteLength +
        g.attributes.normal.array.byteLength +
        g.index.array.byteLength;
      return {
        geometry: g,
        quadCount: r.quadCount,
        totalMs: r.time,
        submitMs: null,
        gpuMs: null,
        passMs: null,
        readbackMs: 0,
        bytesUp: geometryBytes, // Three uploads the freshly built geometry on the next draw
        bytesDown: 0,
      };
    }

    if (mode === 'gpu-readback') {
      this._readbackTarget ??= this.mesher.createReadbackTarget();
      const r = await this.mesher.meshWithReadback(this.chunk, this._readbackTarget);
      return {
        ...r,
        // occupancy delta up, vertices down, then Three uploads those vertices again
        bytesUp: r.uploadedBytes + r.readbackBytes,
        bytesDown: r.readbackBytes,
      };
    }

    // gpu-direct: upload occupancy delta, compute, done. Vertices never leave the GPU.
    this._direct ??= this.bridge.createDirectMesh(this.materials['gpu-direct']);
    const t0 = performance.now();
    const { submitMs, uploadedBytes, gpuMs } = this.mesher.mesh(this.chunk, this._direct.target, { profile: true });
    await this.mesher.whenIdle(); // waits for completion; reads nothing back
    const totalMs = performance.now() - t0;
    const timing = await gpuMs;

    // Stats only (32 bytes), outside the timed region.
    const counters = withCounters ? await this.mesher.readCounters(this._direct.target) : null;
    return {
      mesh: this._direct.mesh,
      quadCount: counters?.quadCount ?? null,
      overflowed: counters?.overflowed ?? false,
      totalMs,
      submitMs,
      gpuMs: timing?.totalMs ?? null,
      passMs: timing?.passMs ?? null,
      readbackMs: 0,
      bytesUp: uploadedBytes,
      bytesDown: 0,
    };
  }

  /** Puts a measured result into the scene, replacing whatever was shown. */
  show(result, mode) {
    const old = this.scene.getObjectByName('voxelMesh');
    if (old) {
      this.scene.remove(old);
      if (old !== this._direct?.mesh) old.geometry.dispose();
    }

    let mesh = result.mesh;
    if (!mesh) {
      mesh = new THREE.Mesh(result.geometry, this.materials[mode]);
      mesh.name = 'voxelMesh';
    }
    this.scene.add(mesh);
  }

  async remesh(mode, opts) {
    const result = await this.measure(mode, opts);
    this.show(result, mode);
    return result;
  }

  /**
   * remesh() for interactive edits: if more edits arrive while a remesh is running,
   * they are folded into one more pass instead of queuing one remesh per click.
   * Every caller's promise resolves with the result of the final pass.
   */
  remeshCoalesced(mode, opts) {
    this._wantRemesh = true;
    if (!this._inflight) {
      this._inflight = (async () => {
        let result;
        try {
          while (this._wantRemesh) {
            this._wantRemesh = false;
            result = await this.remesh(mode, opts);
          }
        } finally {
          this._inflight = null;
        }
        return result;
      })();
    }
    return this._inflight;
  }

  dispose() {
    this._direct?.dispose();
    this._direct = null;
    this._readbackTarget = null;
    for (const m of Object.values(this.materials)) m.dispose();
  }
}
