export const CHUNK_SIZE = 96;
const VOXEL_COUNT = CHUNK_SIZE ** 3;
const WORD_COUNT = Math.ceil(VOXEL_COUNT / 32);

/**
 * Wraps a Uint32Array for bit-packed voxel storage.
 * 32 binary voxels (0=air, 1=solid) are packed into each u32 word.
 */
export class ChunkData {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.id = `${x},${y},${z}`;
    // Initialize bit-packed storage (all zeros/air by default)
    this.data = new Uint32Array(WORD_COUNT);
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
   * Set voxel occupancy at (x, y, z)
   */
  set(x, y, z, isSolid) {
    const idx = this._getFlatIndex(x, y, z);
    if (idx === -1) return;

    const wordIdx = idx >>> 5;
    const bitOffset = idx & 31;
    
    if (isSolid) {
      this.data[wordIdx] |= (1 << bitOffset);
    } else {
      this.data[wordIdx] &= ~(1 << bitOffset);
    }
  }

  /**
   * Returns the raw underlying Uint32Array buffer.
   * This is what gets sent directly to the GPU.
   */
  getBuffer() {
    return this.data;
  }
}
