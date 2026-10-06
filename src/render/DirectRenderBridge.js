import * as THREE from 'three/webgpu';
import { MAX_VERTICES, INDIRECT_WORDS } from '../meshing/gpuMeshPipeline';

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
   * @returns {{ mesh: THREE.Mesh, target: {positions: GPUBuffer, indirect: GPUBuffer}, dispose: () => void }}
   */
  createDirectMesh(material) {
    if (!this.isWebGPU) {
      throw new Error(
        'Direct rendering needs the WebGPU backend, but THREE.WebGPURenderer fell back to WebGL2.'
      );
    }

    // vec4 per vertex (xyz + face id). Three pads vec3 storage attributes to vec4
    // anyway, so declaring 4 up front keeps the layout under our control.
    const positions = new THREE.StorageBufferAttribute(MAX_VERTICES, 4);

    // Initial draw args [0 vertices, 1 instance, 0, 0]: valid to draw (nothing)
    // before the first compute pass has run.
    const indirect = new THREE.IndirectStorageBufferAttribute(INDIRECT_WORDS, 1);
    indirect.array[1] = 1;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', positions);
    geometry.setIndirect(indirect);

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
      positions: this.renderer.backend.get(positions)?.buffer,
      indirect: this.renderer.backend.get(indirect)?.buffer,
    };
    if (!target.positions || !target.indirect) {
      throw new Error('Could not obtain GPUBuffers from Three (internal API changed?).');
    }

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'voxelMesh';
    // The CPU-side position array is all zeros, so Three would compute a degenerate
    // bounding sphere and frustum-cull the mesh. The real extent only exists on the GPU.
    mesh.frustumCulled = false;

    return { mesh, target, dispose: () => geometry.dispose() };
  }
}
