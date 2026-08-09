import { ChunkData, CHUNK_SIZE } from './ChunkData';

/**
 * Handles procedural generation of voxel data for a chunk.
 */
export class TerrainGenerator {
  constructor(options = {}) {
    this.scale = options.scale || 0.15;
    this.threshold = options.threshold || 0;
  }

  /**
   * Populates a ChunkData instance with procedural noise.
   * @param {ChunkData} chunk 
   */
  generate(chunk) {
    const startX = chunk.x * CHUNK_SIZE;
    const startY = chunk.y * CHUNK_SIZE;
    const startZ = chunk.z * CHUNK_SIZE;

    // We iterate in flat index order for performance
    const VOXEL_COUNT = CHUNK_SIZE ** 3;
    
    for (let i = 0; i < VOXEL_COUNT; i++) {
      const lx = i % CHUNK_SIZE;
      const ly = Math.floor(i / CHUNK_SIZE) % CHUNK_SIZE;
      const lz = Math.floor(i / (CHUNK_SIZE * CHUNK_SIZE));

      const wx = startX + lx;
      const wy = startY + ly;
      const wz = startZ + lz;

      // Complex noise pattern (High Density)
      // Creates caves and mountains
      const noise = Math.sin(wx * this.scale) + Math.sin(wy * this.scale) + Math.sin(wz * this.scale);

      if (noise > this.threshold) {
        chunk.set(lx, ly, lz, true);
      }
    }
  }
}
