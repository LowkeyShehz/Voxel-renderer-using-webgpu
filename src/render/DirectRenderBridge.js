import * as THREE from 'three/webgpu';
import { CHUNK_SIZE } from '../voxel/ChunkData';
import { DEFAULT_CAPACITY, INDIRECT_WORDS, POSITION_STRIDE_BYTES } from '../meshing/gpuMeshPipeline';

// Mirrors AttributeType in three/src/renderers/common/Constants.js (not part of
// Three's public API, so duplicated here). Verified against three@0.185.1.
const ATTRIBUTE_TYPE_STORAGE = 3;
const ATTRIBUTE_TYPE_INDIRECT = 4;

/**
 * DirectRenderBridge
 *
 * Creates a THREE.Mesh whose vertex buffer and draw arguments are written by the
 * compute mesher and consumed by the renderer without ever visiting the CPU:
 *
 *   compute pass --writes--> positions (STORAGE | VERTEX)   --> vertex fetch
 *                 \--writes-> indirect  (STORAGE | INDIRECT) --> drawIndirect()
 *
 * Three allocates the GPUBuffers, and we hand those same buffers to our compute
 * pipeline. This relies on Three internals (`renderer._attributes`,
 * `renderer.backend.get(attr).buffer`), so everything that touches them lives in
 * this file and the Three version is pinned.
 *
 * Requires the renderer and the mesher to share one GPUDevice (see gpuContext.js).
 */
export class DirectRenderBridge {
  constructor(renderer) {
    this.renderer = renderer;
  }

  /** True when Three is running on its WebGPU backend (not the WebGL2 fallback). */
  get isWebGPU() {
    return this.renderer.backend?.isWebGPUBackend === true;
  }

  /**
   * @param {THREE.Material} material
   * @param {number} [capacity]  vertex capacity of the position buffer (multiple of 6)
   * @returns {{ mesh: THREE.Mesh, target: {positions: GPUBuffer, indirect: GPUBuffer, capacity: number},
   *             gpuBytes: number, dispose: () => void }}
   */
  createDirectMesh(material, capacity = DEFAULT_CAPACITY) {
    if (!this.isWebGPU) {
      throw new Error(
        'Direct rendering needs the WebGPU backend, but THREE.WebGPURenderer fell back to WebGL2.'
      );
    }

    // vec4 per vertex (xyz + face id). Three pads vec3 storage attributes to vec4
    // anyway, so declaring 4 up front keeps the layout under our control.
    const positions = new THREE.StorageBufferAttribute(capacity, 4);

    // Initial draw args [0 vertices, 1 instance, 0, 0]: valid to draw (nothing)
    // before the first compute pass has run.
    const indirect = new THREE.IndirectStorageBufferAttribute(INDIRECT_WORDS, 1);
    indirect.array[1] = 1;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', positions);
    geometry.setIndirect(indirect);
    // The vertices only exist on the GPU, so give Three the bounds instead of letting
    // it compute them from the CPU array: every vertex lies in the chunk's local cube.
    // This also lets each chunk be frustum-culled.
    geometry.boundingBox = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE));
    geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new THREE.Sphere());

    // Allocate the GPU buffers now (instead of on first render) so compute can bind
    // them. Going through Attributes.update() also records that the buffers exist,
    // so Three will not allocate (and orphan) a second copy on the first draw.
    const attributes = this.renderer._attributes;
    if (!attributes) {
      throw new Error('Renderer is not initialized yet; await renderer.init() first.');
    }
    attributes.update(positions, ATTRIBUTE_TYPE_STORAGE);
    attributes.update(indirect, ATTRIBUTE_TYPE_INDIRECT);

    const target = {
      capacity,
      positions: this.renderer.backend.get(positions)?.buffer,
      indirect: this.renderer.backend.get(indirect)?.buffer,
    };
    if (!target.positions || !target.indirect) {
      throw new Error('Could not obtain GPUBuffers from Three (internal API changed?).');
    }

    // Drop the CPU shadow array. Three needs it only to size and fill the GPU buffer
    // at creation; afterwards it reads `array` again only on a version bump (we never
    // set needsUpdate: compute writes the buffer) or for wireframe index generation
    // (not used). Verified against three@0.185.1 (WebGPUAttributeUtils.createAttribute,
    // Geometries.getWireframeIndex). Keeping the same typed-array type keeps the vertex
    // format Three derives from it (float32x4). Saves capacity * 16 bytes of CPU
    // memory per chunk (6 MB at the default capacity; it was 8 MB per chunk before).
    positions.array = new Float32Array(4);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'voxelMesh';

    return {
      mesh,
      target,
      gpuBytes: capacity * POSITION_STRIDE_BYTES + INDIRECT_WORDS * 4,
      dispose: () => geometry.dispose(),
    };
  }
}
