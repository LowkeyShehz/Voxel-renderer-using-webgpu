import * as THREE from 'three/webgpu';
import { CHUNK_SIZE } from '../voxel/ChunkData';

/**
 * @param {import('../voxel/ChunkData').ChunkData} chunk
 * @param {Array<import('../voxel/ChunkData').ChunkData|null>} [neighbours]  6 chunks in
 *   face order +X -X +Y -Y +Z -Z (see ChunkManager.neighbours). A face on the chunk
 *   border is only emitted if the neighbouring voxel is air; missing neighbours = air.
 */
export function cpuGreedyMeshGeneration(chunk, neighbours = null) {
  const start = performance.now();
  
  const vertices = [];
  const indices = [];
  let naiveFaceCount = 0;

  // Mask is a 2D array representing a slice
  const mask = new Uint8Array(CHUNK_SIZE * CHUNK_SIZE);

  // Helper to add a quad
  const emitQuad = (corners, normal) => {
    const baseIdx = vertices.length / 6;
    for (const [ox, oy, oz] of corners) {
      vertices.push(ox, oy, oz, normal[0], normal[1], normal[2]);
    }
    indices.push(
      baseIdx, baseIdx + 1, baseIdx + 2,
      baseIdx, baseIdx + 2, baseIdx + 3
    );
  };

  // 6 Face Definitions mapping to 3 axes and 2 directions
  // axis: 0=X, 1=Y, 2=Z
  // dir: 1 (positive), -1 (negative)
  // u, v: the axes for the 2D slice (0, 1, 2)
  const faces = [
    { 
      axis: 0, dir: 1, normal: [1, 0, 0], u: 1, v: 2,
      // +X face at x+1
      getCorners: (x, y, z, w, h) => [
        [x + 1, y, z], [x + 1, y + w, z], [x + 1, y + w, z + h], [x + 1, y, z + h]
      ]
    },
    { 
      axis: 0, dir: -1, normal: [-1, 0, 0], u: 1, v: 2,
      // -X face at x
      getCorners: (x, y, z, w, h) => [
        [x, y, z + h], [x, y + w, z + h], [x, y + w, z], [x, y, z]
      ]
    },
    { 
      axis: 1, dir: 1, normal: [0, 1, 0], u: 0, v: 2,
      // +Y face at y+1
      getCorners: (x, y, z, w, h) => [
        [x, y + 1, z], [x, y + 1, z + h], [x + w, y + 1, z + h], [x + w, y + 1, z]
      ]
    },
    { 
      axis: 1, dir: -1, normal: [0, -1, 0], u: 0, v: 2,
      // -Y face at y
      getCorners: (x, y, z, w, h) => [
        [x, y, z + h], [x, y, z], [x + w, y, z], [x + w, y, z + h]
      ]
    },
    { 
      axis: 2, dir: 1, normal: [0, 0, 1], u: 0, v: 1,
      // +Z face at z+1
      getCorners: (x, y, z, w, h) => [
        [x + w, y, z + 1], [x + w, y + h, z + 1], [x, y + h, z + 1], [x, y, z + 1]
      ]
    },
    { 
      axis: 2, dir: -1, normal: [0, 0, -1], u: 0, v: 1,
      // -Z face at z
      getCorners: (x, y, z, w, h) => [
        [x, y, z], [x, y + h, z], [x + w, y + h, z], [x + w, y, z]
      ]
    }
  ];

  const S = CHUNK_SIZE;
  const getVoxel = (x, y, z) => {
    if (x >= 0 && x < S && y >= 0 && y < S && z >= 0 && z < S) return chunk.get(x, y, z);
    if (!neighbours) return 0;
    // Face neighbours only ever step one voxel out along one axis.
    if (x >= S) return neighbours[0]?.get(x - S, y, z) ?? 0;
    if (x < 0) return neighbours[1]?.get(x + S, y, z) ?? 0;
    if (y >= S) return neighbours[2]?.get(x, y - S, z) ?? 0;
    if (y < 0) return neighbours[3]?.get(x, y + S, z) ?? 0;
    if (z >= S) return neighbours[4]?.get(x, y, z - S) ?? 0;
    return neighbours[5]?.get(x, y, z + S) ?? 0;
  };

  for (const face of faces) {
    const { axis, dir, normal, u, v, getCorners } = face;

    const d_axis = [0, 0, 0];
    d_axis[axis] = dir;

    // Iterate through slices along the main axis
    for (let slice = 0; slice < CHUNK_SIZE; slice++) {
      
      // 1. Build Mask for this slice
      mask.fill(0);
      let n = 0;
      for (let j = 0; j < CHUNK_SIZE; j++) {
        for (let i = 0; i < CHUNK_SIZE; i++) {
          const coords = [0, 0, 0];
          coords[axis] = slice;
          coords[u] = i;
          coords[v] = j;

          const current = getVoxel(coords[0], coords[1], coords[2]);
          if (current) {
            const neighbor = getVoxel(coords[0] + d_axis[0], coords[1] + d_axis[1], coords[2] + d_axis[2]);
            if (!neighbor) {
              mask[n] = 1;
              naiveFaceCount++;
            }
          }
          n++;
        }
      }

      // 2. Greedy merge the mask
      n = 0;
      for (let j = 0; j < CHUNK_SIZE; j++) {
        for (let i = 0; i < CHUNK_SIZE; i++) {
          if (mask[n]) {
            // Compute width
            let w = 1;
            while (i + w < CHUNK_SIZE && mask[n + w]) {
              w++;
            }

            // Compute height
            let h = 1;
            let done = false;
            while (j + h < CHUNK_SIZE) {
              for (let k = 0; k < w; k++) {
                if (!mask[n + k + h * CHUNK_SIZE]) {
                  done = true;
                  break;
                }
              }
              if (done) break;
              h++;
            }

            // Clear mask
            for (let dh = 0; dh < h; dh++) {
              for (let dw = 0; dw < w; dw++) {
                mask[n + dw + dh * CHUNK_SIZE] = 0;
              }
            }

            // Emit quad
            const coords = [0, 0, 0];
            coords[axis] = slice;
            coords[u] = i;
            coords[v] = j;

            const corners = getCorners(coords[0], coords[1], coords[2], w, h);
            emitQuad(corners, normal);
          }
          n++;
        }
      }
    }
  }

  const elapsed = performance.now() - start;

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(vertices.length / 2);
  const normals = new Float32Array(vertices.length / 2);

  for (let i = 0; i < vertices.length / 6; i++) {
    positions[i * 3] = vertices[i * 6];
    positions[i * 3 + 1] = vertices[i * 6 + 1];
    positions[i * 3 + 2] = vertices[i * 6 + 2];
    normals[i * 3] = vertices[i * 6 + 3];
    normals[i * 3 + 1] = vertices[i * 6 + 4];
    normals[i * 3 + 2] = vertices[i * 6 + 5];
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(indices);

  return { geometry, time: elapsed, quadCount: indices.length / 6, naiveFaceCount };
}
