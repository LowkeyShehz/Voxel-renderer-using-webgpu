
import * as THREE from 'three';
import { World } from './World';
import { CPUMesher } from './meshing/CPUMesher';
import { GPUMesher } from './meshing/GPUMesher';

/**
 * Acts as the bridge between the Game Loop and the World state.
 * Handles:
 * 1. Player position tracking
 * 2. Calling World.update()
 * 3. Triggering meshing for dirty chunks
 */
export class ChunkManager {
    constructor(scene) {
        this.world = new World(scene);
        this.scene = scene;
        this.cpuMesher = new CPUMesher();
        this.gpuMesher = new GPUMesher();
        this.mesher = this.cpuMesher; // Default
        this.mode = 'cpu';
        this.isDirty = false;
    }

    setMode(mode) {
        if (mode === 'gpu') {
            this.mesher = this.gpuMesher;
            // Init GPU if needed
            if (!this.gpuMesher.device) {
                this.gpuMesher.init().then(success => {
                    if (!success) {
                        console.error("Failed to init GPU mesher, falling back to CPU");
                        this.setMode('cpu');
                    }
                });
            }
        } else {
            this.mesher = this.cpuMesher;
        }
        this.mode = mode;

        // Mark all chunks dirty to re-mesh with new mode
        for (const chunk of this.world.chunks.values()) {
            chunk.isDirty = true;
        }
    }

    update(playerPos) {
        this.world.update(playerPos);

        // Check for dirty chunks and mesh them
        // (In a real game, this would be throttled/queued)
        for (const chunk of this.world.chunks.values()) {
            if (chunk.isDirty) {
                this.meshChunk(chunk);
                chunk.isDirty = false;
            }
        }
    }

    async meshChunk(chunk) {
        try {
            const data = await this.mesher.mesh(chunk);
            if (data) {
                this.updateChunkMesh(chunk, data);
            }
        } catch (e) {
            console.error("Meshing failed for chunk", chunk.id, e);
        }
    }

    updateChunkMesh(chunk, { positions, normals, indices }) {
        // Dispose old mesh
        if (chunk.mesh) {
            this.scene.remove(chunk.mesh);
            chunk.dispose();
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        if (indices) {
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        }

        const material = new THREE.MeshStandardMaterial({
            color: this.mode === 'cpu' ? 0x3498db : 0xe74c3c,
            flatShading: true,
            roughness: 0.8,
        });

        const mesh = new THREE.Mesh(geometry, material);
        chunk.mesh = mesh;
        this.scene.add(mesh);
    }
}
