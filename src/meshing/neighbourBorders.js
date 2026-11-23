import { CHUNK_SIZE } from '../voxel/ChunkData.js';

const N = CHUNK_SIZE;
const WORDS_PER_LAYER = (N * N) / 32; // 288: one 96x96 layer, also one mask slice
export const BORDER_WORDS = 6 * WORDS_PER_LAYER;

/**
 * Packs the layer of each neighbour that touches a chunk into 6 bit planes, the
 * data mask_main needs to decide whether a border face is hidden.
 *
 * Plane `face` (FACE_DIRS order: +X -X +Y -Y +Z -Z) holds the neighbour's touching
 * layer, indexed with the same (u, v) as that face's mask slice: bit u + v * 96 at
 * word face * 288 + (bit >> 5), where (u, v) = (y, z) for X, (x, z) for Y, (x, y) for Z.
 * A missing neighbour leaves its plane zero (air), which is the old behaviour.
 *
 * @param {Array<import('../voxel/ChunkData.js').ChunkData|null>} neighbours  6 entries
 * @param {Uint32Array} [out]
 */
export function buildBorderPlanes(neighbours, out = new Uint32Array(BORDER_WORDS)) {
  out.fill(0);
  neighbours.forEach((n, face) => {
    if (!n) return;
    const d = n.data;
    const base = face * WORDS_PER_LAYER;
    switch (face) {
      case 0: // +X neighbour, its x = 0 column
      case 1: { // -X neighbour, its x = 95 column
        const x = face === 0 ? 0 : N - 1;
        for (let z = 0; z < N; z++) {
          for (let y = 0; y < N; y++) {
            const i = x + y * N + z * N * N;
            if (d[i >>> 5] & (1 << (i & 31))) {
              const bit = y + z * N;
              out[base + (bit >>> 5)] |= 1 << (bit & 31);
            }
          }
        }
        break;
      }
      case 2: // +Y neighbour, its y = 0 rows
      case 3: { // -Y neighbour, its y = 95 rows
        // Row (y, z) is 96 bits = 3 whole words starting at z * 288 + y * 3.
        const y = face === 2 ? 0 : N - 1;
        for (let z = 0; z < N; z++) {
          const src = z * WORDS_PER_LAYER + y * 3;
          out.set(d.subarray(src, src + 3), base + z * 3);
        }
        break;
      }
      default: { // +Z neighbour (z = 0) or -Z neighbour (z = 95): a contiguous layer
        const z = face === 4 ? 0 : N - 1;
        out.set(d.subarray(z * WORDS_PER_LAYER, (z + 1) * WORDS_PER_LAYER), base);
      }
    }
  });
  return out;
}

/** Bit (u, v) of border plane `face`, as mask_main reads it. */
export function borderBit(planes, face, u, v) {
  const bit = u + v * N;
  return (planes[face * WORDS_PER_LAYER + (bit >>> 5)] >>> (bit & 31)) & 1;
}
