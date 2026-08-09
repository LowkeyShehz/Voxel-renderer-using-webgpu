import * as THREE from 'three';

export const CHUNK_SIZE = 32;
const VOXEL_COUNT = CHUNK_SIZE ** 3;

/**
 * Represents a single chunk of voxel data.
 * Handles data generation, mesh storage, and state.
 */
export class Chunk {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.id = `${x},${y},${z}`;
        this.data = new Uint8Array(VOXEL_COUNT);
        this.isDirty = true; // Needs remeshing
        this.mesh = null; // Three.js mesh
    }

    generateTerrain() {
        // Determine world position of chunk origin
        const startX = this.x * CHUNK_SIZE;
        const startY = this.y * CHUNK_SIZE;
        const startZ = this.z * CHUNK_SIZE;

        // Simple Noise Generation
        for (let i = 0; i < VOXEL_COUNT; i++) {
            const lx = i % CHUNK_SIZE;
            const ly = Math.floor(i / CHUNK_SIZE) % CHUNK_SIZE;
            const lz = Math.floor(i / (CHUNK_SIZE * CHUNK_SIZE));

            const wx = startX + lx;
            const wy = startY + ly;
            const wz = startZ + lz;

            // Base terrain height
            const height = Math.sin(wx * 0.05) * Math.cos(wz * 0.05) * 10 + 16;

            // Simple distinct layers
            if (wy < height) {
                this.data[i] = 1; // Solid
            } else {
                this.data[i] = 0; // Air
            }
        }
    }

    dispose() {
        if (this.mesh) {
            this.mesh.geometry.dispose();
            // Material is usually shared, so don't dispose it unless unique
        }
    }
}
