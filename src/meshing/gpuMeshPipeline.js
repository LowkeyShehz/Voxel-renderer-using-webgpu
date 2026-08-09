import * as THREE from 'three';
import { CHUNK_SIZE } from '../voxel/ChunkData';
import { gpuMeshShader } from './gpuMeshShader';

let cachedDevice = null;
let cachedPipelineMask = null;
let cachedPipelineMerge = null;
let cachedMaskBuffer = null;
let cachedBindGroupLayout = null;

export async function gpuGreedyMeshGeneration(chunk, bridge) {
  const start = performance.now();

  const device = bridge.device;
  if (!device) {
    throw new Error('WebGPU device not initialized on DirectRenderBridge');
  }

  // 1. Setup pipelines if not cached
  if (!cachedPipelineMask || device !== cachedDevice) {
    cachedDevice = device;
    const shaderModule = device.createShaderModule({ code: gpuMeshShader });
    
    cachedPipelineMask = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'mask_main' }
    });
    
    cachedPipelineMerge = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'merge_main' }
    });
    
    cachedBindGroupLayout = cachedPipelineMask.getBindGroupLayout(0);
  }

  // 2. Prepare Input Buffers
  const maxVertices = Math.min(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 6 * 6, 3500000);
  
  const voxelDataU32 = chunk.getBuffer();
  const voxelBuffer = device.createBuffer({
    size: voxelDataU32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(voxelBuffer, 0, voxelDataU32);

  const MASK_BUFFER_SIZE = 6 * CHUNK_SIZE * 288 * 4; // bytes
  if (!cachedMaskBuffer || device !== cachedDevice) {
    cachedMaskBuffer = device.createBuffer({
      size: MASK_BUFFER_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  // 3. Prepare Compute Output Buffers
  const positionsComputeBuffer = device.createBuffer({
    size: maxVertices * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const normalsComputeBuffer = device.createBuffer({
    size: maxVertices * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const counterBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]));

  // 4. Bind Groups
  const bindGroup = device.createBindGroup({
    layout: cachedBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: voxelBuffer } },
      { binding: 1, resource: { buffer: cachedMaskBuffer } },
      { binding: 2, resource: { buffer: positionsComputeBuffer } },
      { binding: 3, resource: { buffer: normalsComputeBuffer } },
      { binding: 4, resource: { buffer: counterBuffer } },
    ]
  });

  // 5. Dispatch Compute Passes
  const commandEncoder = device.createCommandEncoder();

  // Clear mask buffer
  commandEncoder.clearBuffer(cachedMaskBuffer);

  const passEncoder = commandEncoder.beginComputePass();
  
  // Pass A
  passEncoder.setPipeline(cachedPipelineMask);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(
    Math.ceil(CHUNK_SIZE / 4),
    Math.ceil(CHUNK_SIZE / 4),
    Math.ceil(CHUNK_SIZE / 4)
  );

  // Pass B
  passEncoder.setPipeline(cachedPipelineMerge);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(1, 1, 1);
  passEncoder.end();

  // Read counter to determine vertex count
  const counterReadBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
  
  device.queue.submit([commandEncoder.finish()]);

  // Wait for GPU execution to read back just the 4-byte counter
  const computeEnd = performance.now();
  await counterReadBuffer.mapAsync(GPUMapMode.READ);
  const vertexCount = new Uint32Array(counterReadBuffer.getMappedRange())[0];
  counterReadBuffer.unmap();
  const readbackEnd = performance.now();
  const readbackTime = readbackEnd - computeEnd;

  const geometry = new THREE.BufferGeometry();
  
  if (vertexCount > 0) {
    // Phase 3: TRUE ZERO-COPY DIRECT RENDERING
    // Allocate Three.js attributes
    const positionAttr = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    const normalAttr = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    
    geometry.setAttribute('position', positionAttr);
    geometry.setAttribute('normal', normalAttr);
    
    // Extract the underlying WebGPU buffers allocated by Three.js
    const threePositionGPUBuffer = bridge.getGPUBuffer(positionAttr);
    const threeNormalGPUBuffer = bridge.getGPUBuffer(normalAttr);
    
    // Dispatch a copy from our compute output directly to Three.js buffers entirely on the GPU!
    const copyEncoder = device.createCommandEncoder();
    const byteSize = vertexCount * 3 * 4;
    copyEncoder.copyBufferToBuffer(positionsComputeBuffer, 0, threePositionGPUBuffer, 0, byteSize);
    copyEncoder.copyBufferToBuffer(normalsComputeBuffer, 0, threeNormalGPUBuffer, 0, byteSize);
    device.queue.submit([copyEncoder.finish()]);
  }

  // Cleanup compute-only buffers
  voxelBuffer.destroy();
  positionsComputeBuffer.destroy();
  normalsComputeBuffer.destroy();
  counterBuffer.destroy();
  counterReadBuffer.destroy();

  const time = performance.now() - start;
  return { geometry, time, quadCount: vertexCount / 6, readbackTime };
}
