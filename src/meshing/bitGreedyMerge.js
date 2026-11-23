import { CHUNK_SIZE } from '../voxel/ChunkData.js';
import { borderBit } from './neighbourBorders.js';

/**
 * JS reference for the GPU "bit-run" greedy merge (merge_bitrun_main in
 * gpuMeshShader.js). The WGSL is a line-for-line port of mergeSliceBitRuns(), so
 * this file is what the unit tests check against cpuGreedyMesh: same quad count and
 * same rectangles. Keep the two in sync when changing either.
 *
 * Mask layout (same as the GPU mask buffer, see DATA_FORMAT.md):
 *   word index = face * CHUNK_SIZE * WORDS_PER_SLICE + slice * WORDS_PER_SLICE + (bit >> 5)
 *   bit        = u + v * CHUNK_SIZE
 * With CHUNK_SIZE = 96 every row v is exactly ROW_WORDS = 3 whole words, so a row
 * can be scanned and tested with word operations instead of cell by cell.
 */
export const WORDS_PER_SLICE = (CHUNK_SIZE * CHUNK_SIZE) / 32;
export const ROW_WORDS = CHUNK_SIZE / 32;

// The merge tests a row as exactly three words (m0, m1, m2 below and in the WGSL).
if (ROW_WORDS !== 3) {
  throw new Error('bit-run merge assumes CHUNK_SIZE = 96 (3 words per row)');
}

/** countTrailingZeros() for a u32, 32 for 0 (same as WGSL). */
export function ctz(x) {
  x >>>= 0;
  return x === 0 ? 32 : 31 - Math.clz32(x & -x);
}

/**
 * Bits of row word k (k = 0..ROW_WORDS-1) that fall inside [u0, u1).
 * Written the way the WGSL needs it: the early return avoids u32 underflow, and
 * both shifts stay below 32.
 */
export function rangeMask(k, u0, u1) {
  if (u1 <= k * 32 || u0 >= k * 32 + 32) return 0;
  const lo = Math.max(u0, k * 32) - k * 32; // 0..31
  const hi = Math.min(u1, k * 32 + 32) - k * 32; // lo+1..32
  return ((0xffffffff >>> (32 - hi)) & (0xffffffff << lo)) >>> 0;
}

/** Length of the run of set bits starting at u in the row whose first word is rowBase. */
function runLength(mask, rowBase, u) {
  let w = 0;
  let k = u >> 5;
  let b = u & 31;
  while (k < ROW_WORDS) {
    // Shifting zero-fills the top, so the inverted word always ends the run by bit 32 - b.
    const run = ctz(~(mask[rowBase + k] >>> b));
    w += run;
    if (run < 32 - b) break;
    k++;
    b = 0;
  }
  return w;
}

/**
 * Greedy-merges one slice mask in place (it is consumed) and calls
 * emit(u, v, w, h) for each quad, in the same order as the cell-by-cell CPU scan:
 * row by row, lowest u first. A quad takes the run of set bits starting at its
 * first cell, then grows down while the next row has that whole run set.
 *
 * @param {Uint32Array} mask   WORDS_PER_SLICE words starting at `offset`
 */
export function mergeSliceBitRuns(mask, offset, emit) {
  for (let v = 0; v < CHUNK_SIZE; v++) {
    const rowBase = offset + v * ROW_WORDS;
    for (let k = 0; k < ROW_WORDS; k++) {
      // Each quad clears its own cells, so this loop ends once the word is empty.
      while (mask[rowBase + k] !== 0) {
        const u = k * 32 + ctz(mask[rowBase + k]);
        const w = runLength(mask, rowBase, u);
        const m0 = rangeMask(0, u, u + w);
        const m1 = rangeMask(1, u, u + w);
        const m2 = rangeMask(2, u, u + w);

        let h = 1;
        while (v + h < CHUNK_SIZE) {
          const r = rowBase + h * ROW_WORDS;
          if ((mask[r] & m0) >>> 0 !== m0 || (mask[r + 1] & m1) >>> 0 !== m1 || (mask[r + 2] & m2) >>> 0 !== m2) break;
          h++;
        }

        for (let dv = 0; dv < h; dv++) {
          const r = rowBase + dv * ROW_WORDS;
          mask[r] &= ~m0;
          mask[r + 1] &= ~m1;
          mask[r + 2] &= ~m2;
        }

        emit(u, v, w, h);
      }
    }
  }
}

/**
 * JS port of mask_main: per-face exposure masks for a chunk. A neighbour voxel
 * outside the chunk is read from the border planes (see neighbourBorders.js), or is
 * air when `borders` is null.
 * @param {import('../voxel/ChunkData.js').ChunkData} chunk
 * @param {Uint32Array|null} [borders]  from buildBorderPlanes()
 */
export function buildFaceMasks(chunk, borders = null) {
  const N = CHUNK_SIZE;
  const masks = new Uint32Array(6 * N * WORDS_PER_SLICE);
  const set = (face, slice, u, v) => {
    const bit = u + v * N;
    masks[face * N * WORDS_PER_SLICE + slice * WORDS_PER_SLICE + (bit >> 5)] |= 1 << (bit & 31);
  };
  // Same as the WGSL neighbourVoxel(): inside the chunk read it, outside use the plane.
  const nb = (x, y, z, face, u, v) => {
    if (x >= 0 && x < N && y >= 0 && y < N && z >= 0 && z < N) return chunk.get(x, y, z);
    return borders ? borderBit(borders, face, u, v) : 0;
  };
  for (let z = 0; z < N; z++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (!chunk.get(x, y, z)) continue;
        if (!nb(x + 1, y, z, 0, y, z)) set(0, x, y, z);
        if (!nb(x - 1, y, z, 1, y, z)) set(1, x, y, z);
        if (!nb(x, y + 1, z, 2, x, z)) set(2, y, x, z);
        if (!nb(x, y - 1, z, 3, x, z)) set(3, y, x, z);
        if (!nb(x, y, z + 1, 4, x, y)) set(4, z, x, y);
        if (!nb(x, y, z - 1, 5, x, y)) set(5, z, x, y);
      }
    }
  }
  return masks;
}

/**
 * The 4 corners of a quad, in the same order the shaders (and cpuGreedyMesh) emit
 * them. Face ids: 0..5 = +X -X +Y -Y +Z -Z; (u, v) are (y, z) for X faces,
 * (x, z) for Y faces and (x, y) for Z faces.
 */
export function quadCorners(face, s, u, v, w, h) {
  switch (face) {
    case 0: return [[s + 1, u, v], [s + 1, u + w, v], [s + 1, u + w, v + h], [s + 1, u, v + h]];
    case 1: return [[s, u, v + h], [s, u + w, v + h], [s, u + w, v], [s, u, v]];
    case 2: return [[u, s + 1, v], [u, s + 1, v + h], [u + w, s + 1, v + h], [u + w, s + 1, v]];
    case 3: return [[u, s, v + h], [u, s, v], [u + w, s, v], [u + w, s, v + h]];
    case 4: return [[u + w, v, s + 1], [u + w, v + h, s + 1], [u, v + h, s + 1], [u, v, s + 1]];
    default: return [[u, v, s], [u, v + h, s], [u + w, v + h, s], [u + w, v, s]];
  }
}

/**
 * Meshes a whole chunk with the bit-run merge (`borders` as in buildFaceMasks).
 * @returns {Array<{face:number, slice:number, u:number, v:number, w:number, h:number}>}
 */
export function bitGreedyMesh(chunk, borders = null) {
  const masks = buildFaceMasks(chunk, borders);
  const quads = [];
  for (let face = 0; face < 6; face++) {
    for (let slice = 0; slice < CHUNK_SIZE; slice++) {
      const offset = (face * CHUNK_SIZE + slice) * WORDS_PER_SLICE;
      mergeSliceBitRuns(masks, offset, (u, v, w, h) => quads.push({ face, slice, u, v, w, h }));
    }
  }
  return quads;
}
