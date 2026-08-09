/**
 * Base class/interface for meshing strategies.
 */
export class Mesher {
    constructor() {
        if (this.constructor === Mesher) {
            throw new Error("Abstract classes can't be instantiated.");
        }
    }

    /**
     * Generates mesh data for a chunk.
     * @param {Chunk} chunk 
     * @returns {Promise<{positions, normals, indices}>}
     */
    async mesh(chunk) {
        throw new Error("Method 'mesh()' must be implemented.");
    }
}
