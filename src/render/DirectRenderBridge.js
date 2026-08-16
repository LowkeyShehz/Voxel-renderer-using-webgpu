import * as THREE from 'three/webgpu';

/**
 * DirectRenderBridge
 * Bridges the gap between raw WebGPU compute buffers and Three.js WebGPURenderer
 * by extracting the underlying GPUBuffer from a THREE.BufferAttribute.
 */
export class DirectRenderBridge {
  constructor(renderer) {
    this.renderer = renderer;
    this.device = null;
    this._initialized = false;
  }

  async init() {
    if (this._initialized) return;

    // WebGPURenderer initializes its backend asynchronously
    // Only call init if the renderer hasn't been initialized yet
    if (!this.renderer.backend) {
      await this.renderer.init();
    }

    // R185's WebGPUBackend exposes .device publicly
    const backend = this.renderer.backend;
    this.device = backend.device;

    if (!this.device) {
      // Fallback: try to get device from renderer info
      console.warn("Could not find renderer.backend.device, requesting standalone device.");
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) {
        throw new Error("WebGPU is not supported on this device/browser (adapter is null).");
      }
      this.device = await adapter.requestDevice();
      console.warn("WARNING: Using standalone device - zero-copy will NOT work across contexts!");
    }

    this._initialized = true;
    console.log("DirectRenderBridge: initialized with device", this.device);
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
    if (!backendAttr) {
      throw new Error("Could not extract GPUBuffer from BufferAttribute. Ensure WebGPURenderer is fully initialized.");
    }

    const buf = backendAttr.bufferGPU || backendAttr.buffer;
    if (!buf) {
      throw new Error("GPUBuffer not found on backend attribute object.");
    }
    return buf;
  }
}
