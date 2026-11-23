import { CHUNK_SIZE } from '../voxel/ChunkData.js';

/**
 * Casts a ray through a voxel grid (Amanatides & Woo DDA) and returns the first
 * solid voxel it hits. The grid is a single chunk or a whole ChunkManager world.
 *
 * This works on the CPU-side occupancy data, not on rendered geometry. That matters
 * because in zero-copy mode the mesh only exists on the GPU, so THREE.Raycaster has
 * nothing to intersect.
 *
 * The grid occupies `grid.bounds` ([min, max) per axis, default [0, CHUNK_SIZE)^3 for a
 * chunk) in the same space as `origin`/`direction`; coordinates are world voxels.
 *
 * @param {{ get(x:number,y:number,z:number): number, bounds?: {min:number[], max:number[]} }} grid
 * @param {{x:number,y:number,z:number}} origin
 * @param {{x:number,y:number,z:number}} direction  need not be normalized
 * @param {number} [maxDistance]
 * @returns {{x:number,y:number,z:number,normal:[number,number,number],distance:number}|null}
 *   `normal` is the face of the voxel the ray entered through (points back toward
 *   the ray); [0,0,0] if the ray started inside a solid voxel.
 */
export function raycastVoxels(grid, origin, direction, maxDistance = Infinity) {
  const lo = grid.bounds?.min ?? [0, 0, 0];
  const hi = grid.bounds?.max ?? [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE];
  const len = Math.hypot(direction.x, direction.y, direction.z);
  if (len === 0) return null;
  const o = [origin.x, origin.y, origin.z];
  const d = [direction.x / len, direction.y / len, direction.z / len];

  // 1. Clip the ray against the grid's bounding box (slab test).
  let tEnter = 0;
  let tExit = maxDistance;
  let entryAxis = -1;
  let entrySign = 0;
  for (let a = 0; a < 3; a++) {
    if (d[a] === 0) {
      if (o[a] < lo[a] || o[a] >= hi[a]) return null;
      continue;
    }
    const t1 = (lo[a] - o[a]) / d[a];
    const t2 = (hi[a] - o[a]) / d[a];
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
    cell[a] = Math.min(hi[a] - 1, Math.max(lo[a], Math.floor(p[a])));
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

  // 3. Walk voxel by voxel until we hit something or leave the grid.
  let distance = t0;
  const maxSteps = hi[0] - lo[0] + hi[1] - lo[1] + hi[2] - lo[2] + 3;
  for (let i = 0; i < maxSteps; i++) {
    if (grid.get(cell[0], cell[1], cell[2])) {
      return { x: cell[0], y: cell[1], z: cell[2], normal: [...normal], distance };
    }

    // Advance along whichever axis boundary is closest.
    let a = 0;
    if (tNext[1] < tNext[a]) a = 1;
    if (tNext[2] < tNext[a]) a = 2;

    distance = tNext[a];
    if (distance > tExit || distance > maxDistance) return null;

    cell[a] += step[a];
    if (cell[a] < lo[a] || cell[a] >= hi[a]) return null;
    tNext[a] += tDelta[a];

    normal[0] = normal[1] = normal[2] = 0;
    normal[a] = -step[a];
  }
  return null;
}
