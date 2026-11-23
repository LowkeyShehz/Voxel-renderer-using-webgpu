/**
 * Order-independent comparison of two meshes as sets of quads. A quad's key is its
 * 4 corners in emission order, so winding/orientation differences count as a
 * mismatch too. Used by the in-app correctness check and by the unit tests.
 */

const key = (c) => c.map((p) => p.join(',')).join(' ');

/** Keys from cpuGreedyMesh's geometry: 4 vertices (xyz) per quad. */
export function quadKeysFromCpuPositions(positions) {
  const keys = [];
  for (let i = 0; i < positions.length; i += 12) {
    const c = [];
    for (let j = 0; j < 4; j++) c.push([positions[i + 3 * j], positions[i + 3 * j + 1], positions[i + 3 * j + 2]]);
    keys.push(key(c));
  }
  return keys;
}

/**
 * Keys from the GPU output: 6 vec4 vertices per quad laid out as triangles
 * (c0 c1 c2) (c0 c2 c3), w = face id.
 */
export function quadKeysFromGpuVec4(data, vertexCount) {
  const keys = [];
  const at = (v) => [data[4 * v], data[4 * v + 1], data[4 * v + 2]];
  for (let v = 0; v + 6 <= vertexCount; v += 6) {
    keys.push(key([at(v), at(v + 1), at(v + 2), at(v + 5)]));
  }
  return keys;
}

/** Keys from a list of quad corner arrays (see quadCorners in bitGreedyMerge.js). */
export function quadKeysFromCorners(cornerList) {
  return cornerList.map(key);
}

/**
 * @returns {{ equal: boolean, countA: number, countB: number, onlyInA: string[], onlyInB: string[] }}
 *   onlyInA/onlyInB hold up to `limit` example keys present in one set but not the other.
 */
export function compareQuadKeys(a, b, limit = 5) {
  const counts = new Map();
  for (const k of a) counts.set(k, (counts.get(k) ?? 0) + 1);
  for (const k of b) counts.set(k, (counts.get(k) ?? 0) - 1);
  const onlyInA = [];
  const onlyInB = [];
  for (const [k, n] of counts) {
    if (n > 0 && onlyInA.length < limit) onlyInA.push(k);
    if (n < 0 && onlyInB.length < limit) onlyInB.push(k);
  }
  const equal = a.length === b.length && [...counts.values()].every((n) => n === 0);
  return { equal, countA: a.length, countB: b.length, onlyInA, onlyInB };
}
