import { Chunk, CHUNK_SIZE } from './Chunk';

/**
 * Manages the infinite world of chunks.
 * Handles loading/unloading based on player position.
 */
export class World {
    constructor(scene) {
        this.scene = scene;
        this.chunks = new Map(); // "x,y,z" -> Chunk
        this.renderDistance = 2; // Radius in chunks
    }

    update(playerPos) {
        // Current chunk coordinates of the player
        const cx = Math.floor(playerPos.x / CHUNK_SIZE);
        const cz = Math.floor(playerPos.z / CHUNK_SIZE);

        // Identify chunks that should be loaded
        const visibleChunks = new Set();
        for (let x = cx - this.renderDistance; x <= cx + this.renderDistance; x++) {
            for (let z = cz - this.renderDistance; z <= cz + this.renderDistance; z++) {
                // For now, only 1 chunk high (y=0)
                const id = `${x},0,${z}`;
                visibleChunks.add(id);

                if (!this.chunks.has(id)) {
                    this.loadChunk(x, 0, z);
                }
            }
        }

        // Unload chunks that are too far
        for (const [id, chunk] of this.chunks) {
            if (!visibleChunks.has(id)) {
                this.unloadChunk(id);
            }
        }
    }

    loadChunk(x, y, z) {
        const chunk = new Chunk(x, y, z);
        chunk.generateTerrain();
        this.chunks.set(chunk.id, chunk);
        // Note: Meshing happens in a separate step or callback
        // For now, this is just state management
        console.log(`Loaded chunk ${chunk.id}`);
    }

    unloadChunk(id) {
        const chunk = this.chunks.get(id);
        if (chunk) {
            if (chunk.mesh) {
                this.scene.remove(chunk.mesh);
                chunk.dispose();
            }
            this.chunks.delete(id);
            console.log(`Unloaded chunk ${id}`);
        }
    }
}
