import { ChunkData, CHUNK_SIZE } from './ChunkData.js';

/** Face order used everywhere: +X -X +Y -Y +Z -Z (same as the mesher's face ids). */
export const FACE_DIRS = [
  [1, 0, 0], [-1, 0, 0],
  [0, 1, 0], [0, -1, 0],
  [0, 0, 1], [0, 0, -1],
];

export const MAX_GRID_SIZE = 6;

/**
 * An N x N grid of 96^3 chunks at y = 0. Chunk (cx, 0, cz) covers world voxels
 * [cx*96, cx*96+96) x [0, 96) x [cz*96, cz*96+96).
 *
 * Besides storage, it tracks which chunks need a remesh. A chunk's mesh depends on
 * its own voxels and on the layer of each neighbour that touches it, so an edit on a
 * border voxel also marks that neighbour. (Voxel *upload* tracking stays per chunk,
 * in ChunkData's dirty word range.)
 */
export class ChunkManager {
  /**
   * @param {object} opts
   * @param {number} [opts.gridSize=1]
   * @param {{ generate(chunk: ChunkData): void }} [opts.generator]  fills each chunk
   */
  constructor({ gridSize = 1, generator = null } = {}) {
    if (!Number.isInteger(gridSize) || gridSize < 1 || gridSize > MAX_GRID_SIZE) {
      throw new Error(`gridSize must be 1..${MAX_GRID_SIZE}`);
    }
    this.gridSize = gridSize;
    this.generator = generator;
    /** @type {ChunkData[]} */
    this.chunks = [];
    this._byKey = new Map();
    for (let cz = 0; cz < gridSize; cz++) {
      for (let cx = 0; cx < gridSize; cx++) {
        const chunk = new ChunkData(cx, 0, cz);
        generator?.generate(chunk);
        this.chunks.push(chunk);
        this._byKey.set(chunk.id, chunk);
      }
    }
    /** World-space voxel bounds, [min, max) per axis. */
    this.bounds = { min: [0, 0, 0], max: [gridSize * CHUNK_SIZE, CHUNK_SIZE, gridSize * CHUNK_SIZE] };
    this._meshDirty = new Set(this.chunks);
  }

  getChunk(cx, cy, cz) {
    return this._byKey.get(`${cx},${cy},${cz}`) ?? null;
  }

  /** The 6 neighbouring chunks in FACE_DIRS order (null where there is none). */
  neighbours(chunk) {
    return FACE_DIRS.map(([dx, dy, dz]) => this.getChunk(chunk.x + dx, chunk.y + dy, chunk.z + dz));
  }

  /** World voxel origin of a chunk. */
  origin(chunk) {
    return [chunk.x * CHUNK_SIZE, chunk.y * CHUNK_SIZE, chunk.z * CHUNK_SIZE];
  }

  _locate(wx, wy, wz) {
    const cx = Math.floor(wx / CHUNK_SIZE);
    const cy = Math.floor(wy / CHUNK_SIZE);
    const cz = Math.floor(wz / CHUNK_SIZE);
    const chunk = this.getChunk(cx, cy, cz);
    if (!chunk) return null;
    return { chunk, x: wx - cx * CHUNK_SIZE, y: wy - cy * CHUNK_SIZE, z: wz - cz * CHUNK_SIZE };
  }

  /** Occupancy at a world voxel; 0 outside the grid. */
  get(wx, wy, wz) {
    const at = this._locate(wx, wy, wz);
    return at ? at.chunk.get(at.x, at.y, at.z) : 0;
  }

  /**
   * Sets a world voxel and marks the chunk (and any neighbour sharing that voxel's
   * border) for remeshing. @returns {boolean} true if the voxel changed
   */
  set(wx, wy, wz, isSolid) {
    const at = this._locate(wx, wy, wz);
    if (!at || !at.chunk.set(at.x, at.y, at.z, isSolid)) return false;
    this._markAround(at.chunk, [at.x, at.y, at.z], [at.x, at.y, at.z]);
    return true;
  }

  /**
   * Removes every solid voxel within `radius` of world voxel (cx, cy, cz), across as
   * many chunks as the sphere touches.
   * @returns {{ removed: number, chunks: ChunkData[] }} chunks = those whose voxels changed
   */
  carveSphere(cx, cy, cz, radius) {
    let removed = 0;
    const changed = [];
    for (const chunk of this.chunks) {
      const [ox, oy, oz] = this.origin(chunk);
      const lo = [cx - radius - ox, cy - radius - oy, cz - radius - oz].map(Math.ceil);
      const hi = [cx + radius - ox, cy + radius - oy, cz + radius - oz].map(Math.floor);
      if (hi.some((h) => h < 0) || lo.some((l) => l > CHUNK_SIZE - 1)) continue;

      const n = chunk.carveSphere(cx - ox, cy - oy, cz - oz, radius);
      if (n === 0) continue;
      removed += n;
      changed.push(chunk);
      // The carve was clamped to this chunk, so its local box is the clamped sphere box.
      const clamp = (v) => Math.min(CHUNK_SIZE - 1, Math.max(0, v));
      this._markAround(chunk, lo.map(clamp), hi.map(clamp));
    }
    return { removed, chunks: changed };
  }

  /**
   * Marks `chunk` dirty, plus each neighbour whose shared border plane intersects
   * the edited local box [lo, hi] (inclusive). May over-mark (the box is a bound),
   * never under-marks.
   */
  _markAround(chunk, lo, hi) {
    this._meshDirty.add(chunk);
    const last = CHUNK_SIZE - 1;
    FACE_DIRS.forEach(([dx, dy, dz], face) => {
      const axis = face >> 1;
      const positive = (face & 1) === 0;
      const touches = positive ? hi[axis] >= last : lo[axis] <= 0;
      if (!touches) return;
      const n = this.getChunk(chunk.x + dx, chunk.y + dy, chunk.z + dz);
      if (n) this._meshDirty.add(n);
    });
  }

  markAllDirty() {
    for (const c of this.chunks) this._meshDirty.add(c);
  }

  /** Chunks waiting for a remesh, in grid order. */
  get dirtyChunks() {
    return this.chunks.filter((c) => this._meshDirty.has(c));
  }

  isDirty(chunk) {
    return this._meshDirty.has(chunk);
  }

  clearDirty(chunk) {
    this._meshDirty.delete(chunk);
  }

  /** Regenerates every chunk from the generator (all chunks become dirty). */
  resetTerrain() {
    for (const chunk of this.chunks) {
      chunk.clear();
      this.generator?.generate(chunk);
    }
    this.markAllDirty();
  }
}
