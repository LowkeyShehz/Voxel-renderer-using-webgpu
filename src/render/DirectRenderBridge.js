import * as THREE from 'three';

/**
 * DirectRenderBridge
 * Bridges the gap between raw WebGPU compute buffers and Three.js WebGPURenderer
 * by extracting the underlying GPUBuffer from a THREE.BufferAttribute.
 */
export class DirectRenderBridge {
  constructor(renderer) {
    this.renderer = renderer;
    this.device = null;
  }

  async init() {
    // WebGPURenderer initializes its backend asynchronously
    await this.renderer.init();
    this.device = this.renderer.backend.device || this.renderer.backend.get('device'); // Try different properties based on R185 internals
    
    // In R185, renderer.backend.device might not be exposed directly on the backend object?
    // Wait, WebGPUBackend has `device` property.
    if (!this.device) {
      // Fallback: request our own device but this breaks zero-copy if contexts mismatch.
      // R185's WebGPUBackend exposes .device publicly.
      console.warn("Could not find renderer.backend.device, trying to extract via other means.");
    }
  }

  /**
   * Forces Three.js to initialize the GPU buffer for an attribute,
   * then returns the underlying WebGPU GPUBuffer.
   */
  getGPUBuffer(attribute) {
    const backend = this.renderer.backend;
    
    // Force the attribute to be created on the GPU
    if (backend.attributeUtils && backend.attributeUtils.createAttribute) {
      // R185 internal API to create buffer if it doesn't exist
      backend.attributeUtils.createAttribute(attribute, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    }
    
    // Retrieve the buffer
    const backendAttr = backend.get(attribute);
    if (!backendAttr || !backendAttr.bufferGPU) {
      // R185 uses bufferGPU inside the attribute wrapper
      // Let's check buffer as well
      const buf = backendAttr && (backendAttr.bufferGPU || backendAttr.buffer);
      if (!buf) {
          throw new Error("Could not extract GPUBuffer from BufferAttribute. Ensure WebGPURenderer is fully initialized.");
      }
      return buf;
    }
    return backendAttr.bufferGPU;
  }
}
