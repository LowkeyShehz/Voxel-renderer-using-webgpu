import { Mesher } from './Mesher';
import * as THREE from 'three/webgpu';

/**
 * CPU-based meshing implementation.
 * Iterates over voxels one by one.
 */
export class CPUMesher extends Mesher {
    async mesh(chunk) {
        const CHUNK_SIZE = 32; // Hardcode for now, will be dynamic later
        const { data } = chunk;
        const positions = [];
        const normals = [];
        const indices = [];

        // Helper to check standard voxel data
        const getVoxel = (x, y, z) => {
            if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) return 0;
            const i = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
            return data[i];
        };

        let vertexCount = 0;

        for (let i = 0; i < data.length; i++) {
            const x = i % CHUNK_SIZE;
            const y = Math.floor(i / CHUNK_SIZE) % CHUNK_SIZE;
            const z = Math.floor(i / (CHUNK_SIZE * CHUNK_SIZE));

            if (!data[i]) continue;

            const faces = [
                { dir: [1, 0, 0], corners: [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], normal: [1, 0, 0] }, // Right
                { dir: [-1, 0, 0], corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], normal: [-1, 0, 0] }, // Left
                { dir: [0, 1, 0], corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]], normal: [0, 1, 0] }, // Top
                { dir: [0, -1, 0], corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], normal: [0, -1, 0] }, // Bottom
                { dir: [0, 0, 1], corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], normal: [0, 0, 1] }, // Front
                { dir: [0, 0, -1], corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]], normal: [0, 0, -1] }, // Back
            ];

            for (const face of faces) {
                const [nx, ny, nz] = face.dir;
                const neighbor = getVoxel(x + nx, y + ny, z + nz);

                if (!neighbor) {
                    // Add vertices
                    for (const corner of face.corners) {
                        positions.push(x + corner[0], y + corner[1], z + corner[2]);
                        normals.push(...face.normal);
                    }
                    // Add indices (2 triangles)
                    indices.push(vertexCount, vertexCount + 1, vertexCount + 2);
                    indices.push(vertexCount, vertexCount + 2, vertexCount + 3);
                    vertexCount += 4;
                }
            }
        }

        return {
            positions: new Float32Array(positions),
            normals: new Float32Array(normals),
            indices: new Uint16Array(indices)
        };
    }
}
