export const CHUNK_SIZE = 96;
const VOXEL_COUNT = CHUNK_SIZE ** 3;
const WORD_COUNT = Math.ceil(VOXEL_COUNT / 32);

/**
 * Wraps a Uint32Array for bit-packed voxel storage.
 * 32 binary voxels (0=air, 1=solid) are packed into each u32 word.
 *
 * Edits are tracked as a dirty range of word indices, so a GPU mirror of the chunk
 * can be updated by uploading only the words that changed since its last sync.
 */
export class ChunkData {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.id = `${x},${y},${z}`;
    // Initialize bit-packed storage (all zeros/air by default)
    this.data = new Uint32Array(WORD_COUNT);

    // Dirty word range since the last consumeDirtyRange(): [start, end). Empty when start >= end.
    this._dirtyStart = Infinity;
    this._dirtyEnd = -1;
  }

  /**
   * Helper to get flat index from 3D local coordinates
   */
  _getFlatIndex(x, y, z) {
    if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) {
      return -1;
    }
    return x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
  }

  _markDirty(wordIdx) {
    if (wordIdx < this._dirtyStart) this._dirtyStart = wordIdx;
    if (wordIdx + 1 > this._dirtyEnd) this._dirtyEnd = wordIdx + 1;
  }

  /**
   * Check if voxel is solid at (x, y, z)
   */
  get(x, y, z) {
    const idx = this._getFlatIndex(x, y, z);
    if (idx === -1) return 0;

    const wordIdx = idx >>> 5; // idx / 32
    const bitOffset = idx & 31; // idx % 32
    return (this.data[wordIdx] & (1 << bitOffset)) !== 0 ? 1 : 0;
  }

  /**
   * Set voxel occupancy at (x, y, z).
   * @returns {boolean} true if the voxel actually changed
   */
  set(x, y, z, isSolid) {
    const idx = this._getFlatIndex(x, y, z);
    if (idx === -1) return false;

    const wordIdx = idx >>> 5;
    const bit = 1 << (idx & 31);
    const wasSolid = (this.data[wordIdx] & bit) !== 0;
    if (wasSolid === !!isSolid) return false;

    if (isSolid) {
      this.data[wordIdx] |= bit;
    } else {
      this.data[wordIdx] &= ~bit;
    }
    this._markDirty(wordIdx);
    return true;
  }

  /**
   * Removes every solid voxel within `radius` of voxel (cx, cy, cz).
   * @returns {number} how many voxels were removed
   */
  carveSphere(cx, cy, cz, radius) {
    const r2 = radius * radius;
    let removed = 0;

    const zMin = Math.max(0, Math.ceil(cz - radius));
    const zMax = Math.min(CHUNK_SIZE - 1, Math.floor(cz + radius));
    for (let z = zMin; z <= zMax; z++) {
      const dz = z - cz;
      const yMin = Math.max(0, Math.ceil(cy - radius));
      const yMax = Math.min(CHUNK_SIZE - 1, Math.floor(cy + radius));
      for (let y = yMin; y <= yMax; y++) {
        const dy = y - cy;
        const rem = r2 - dz * dz - dy * dy;
        if (rem < 0) continue;
        const half = Math.sqrt(rem);
        const xMin = Math.max(0, Math.ceil(cx - half));
        const xMax = Math.min(CHUNK_SIZE - 1, Math.floor(cx + half));
        for (let x = xMin; x <= xMax; x++) {
          if (this.set(x, y, z, false)) removed++;
        }
      }
    }
    return removed;
  }

  /** Empties the chunk and marks everything dirty. */
  clear() {
    this.data.fill(0);
    this.markAllDirty();
  }

  markAllDirty() {
    this._dirtyStart = 0;
    this._dirtyEnd = WORD_COUNT;
  }

  /**
   * Returns the word range [start, end) modified since the last call, or null if
   * nothing changed, and resets tracking. Intended to have a single consumer (the
   * GPU mirror); CPU-side readers always see `data` directly.
   */
  consumeDirtyRange() {
    if (this._dirtyStart >= this._dirtyEnd) return null;
    const range = { start: this._dirtyStart, end: this._dirtyEnd };
    this._dirtyStart = Infinity;
    this._dirtyEnd = -1;
    return range;
  }

  /**
   * Returns the raw underlying Uint32Array buffer.
   * This is what gets sent directly to the GPU.
   */
  getBuffer() {
    return this.data;
  }
}
