import { CHUNK_SIZE } from '../voxel/ChunkData.js';

/**
 * Casts a ray through a chunk's voxel grid (Amanatides & Woo DDA) and returns the
 * first solid voxel it hits.
 *
 * This works on the CPU-side occupancy data, not on rendered geometry. That matters
 * because in zero-copy mode the mesh only exists on the GPU, so THREE.Raycaster has
 * nothing to intersect.
 *
 * The chunk occupies [0, CHUNK_SIZE)^3 in the same space as `origin`/`direction`.
 *
 * @param {ChunkData} chunk
 * @param {{x:number,y:number,z:number}} origin
 * @param {{x:number,y:number,z:number}} direction  need not be normalized
 * @param {number} [maxDistance]
 * @returns {{x:number,y:number,z:number,normal:[number,number,number],distance:number}|null}
 *   `normal` is the face of the voxel the ray entered through (points back toward
 *   the ray); [0,0,0] if the ray started inside a solid voxel.
 */
export function raycastVoxels(chunk, origin, direction, maxDistance = Infinity) {
  const S = CHUNK_SIZE;
  const len = Math.hypot(direction.x, direction.y, direction.z);
  if (len === 0) return null;
  const o = [origin.x, origin.y, origin.z];
  const d = [direction.x / len, direction.y / len, direction.z / len];

  // 1. Clip the ray against the chunk's bounding box (slab test).
  let tEnter = 0;
  let tExit = maxDistance;
  let entryAxis = -1;
  let entrySign = 0;
  for (let a = 0; a < 3; a++) {
    if (d[a] === 0) {
      if (o[a] < 0 || o[a] >= S) return null;
      continue;
    }
    const t1 = (0 - o[a]) / d[a];
    const t2 = (S - o[a]) / d[a];
    const near = Math.min(t1, t2);
    const far = Math.max(t1, t2);
    if (near > tEnter) {
      tEnter = near;
      entryAxis = a;
      entrySign = d[a] > 0 ? -1 : 1; // entering through the min face has a -axis normal
    }
    tExit = Math.min(tExit, far);
    if (tEnter > tExit) return null;
  }

  // 2. Start just inside the box so the first voxel index is unambiguous.
  const t0 = tEnter + 1e-6;
  const p = [o[0] + d[0] * t0, o[1] + d[1] * t0, o[2] + d[2] * t0];
  const cell = [0, 0, 0];
  const step = [0, 0, 0];
  const tDelta = [Infinity, Infinity, Infinity];
  const tNext = [Infinity, Infinity, Infinity];
  for (let a = 0; a < 3; a++) {
    cell[a] = Math.min(S - 1, Math.max(0, Math.floor(p[a])));
    if (d[a] > 0) {
      step[a] = 1;
      tDelta[a] = 1 / d[a];
      tNext[a] = t0 + (cell[a] + 1 - p[a]) / d[a];
    } else if (d[a] < 0) {
      step[a] = -1;
      tDelta[a] = -1 / d[a];
      tNext[a] = t0 + (p[a] - cell[a]) / -d[a];
    }
  }

  const normal = [0, 0, 0];
  if (entryAxis >= 0) normal[entryAxis] = entrySign;

  // 3. Walk voxel by voxel until we hit something or leave the chunk.
  let distance = t0;
  for (let i = 0; i < 3 * S + 3; i++) {
    if (chunk.get(cell[0], cell[1], cell[2])) {
      return { x: cell[0], y: cell[1], z: cell[2], normal: [...normal], distance };
    }

    // Advance along whichever axis boundary is closest.
    let a = 0;
    if (tNext[1] < tNext[a]) a = 1;
    if (tNext[2] < tNext[a]) a = 2;

    distance = tNext[a];
    if (distance > tExit || distance > maxDistance) return null;

    cell[a] += step[a];
    if (cell[a] < 0 || cell[a] >= S) return null;
    tNext[a] += tDelta[a];

    normal[0] = normal[1] = normal[2] = 0;
    normal[a] = -step[a];
  }
  return null;
}
