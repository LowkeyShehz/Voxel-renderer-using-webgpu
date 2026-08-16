import * as THREE from 'three/webgpu';
import { CHUNK_SIZE } from '../voxel/ChunkData';
import { createGpuMeshShader } from './gpuMeshShader';

let cachedDevice = null;
let cachedPipelineMask = null;
let cachedPipelineMerge = null;
let cachedMaskBuffer = null;
let cachedPositionsBuffer = null;
let cachedNormalsBuffer = null;
let cachedCounterBuffer = null;

// Right-size the output buffers. Greedy meshing on a 96^3 chunk typically
// produces ~50-60K quads = ~300-360K vertices. 500K vertices gives headroom.
const MAX_VERTICES = 500000;

async function getWebGPUDevice() {
  if (cachedDevice) return cachedDevice;
  
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    throw new Error('WebGPU is not supported on this device/browser.');
  }
  cachedDevice = await adapter.requestDevice();
  return cachedDevice;
}

function ensureOutputBuffers(device) {
  if (cachedPositionsBuffer) return;

  cachedPositionsBuffer = device.createBuffer({
    size: MAX_VERTICES * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  cachedNormalsBuffer = device.createBuffer({
    size: MAX_VERTICES * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  cachedCounterBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

export async function gpuGreedyMeshGeneration(chunk) {
  const start = performance.now();

  const device = await getWebGPUDevice();

  // 1. Setup pipelines if not cached
  if (!cachedPipelineMask) {
    const shaderCode = createGpuMeshShader(CHUNK_SIZE);
    const shaderModule = device.createShaderModule({ code: shaderCode });
    
    cachedPipelineMask = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'mask_main' }
    });
    
    cachedPipelineMerge = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'merge_main' }
    });
  }

  // 2. Prepare Input Buffers
  const voxelDataU32 = chunk.getBuffer();
  const voxelBuffer = device.createBuffer({
    size: voxelDataU32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(voxelBuffer, 0, voxelDataU32);

  const WORDS_PER_SLICE = (CHUNK_SIZE * CHUNK_SIZE) / 32;
  const MASK_BUFFER_SIZE = 6 * CHUNK_SIZE * WORDS_PER_SLICE * 4;
  if (!cachedMaskBuffer) {
    cachedMaskBuffer = device.createBuffer({
      size: MASK_BUFFER_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  // 3. Reuse cached output buffers (no 84MB allocation per call!)
  ensureOutputBuffers(device);

  // Reset counter to [0, 1, 0, 0]
  device.queue.writeBuffer(cachedCounterBuffer, 0, new Uint32Array([0, 1, 0, 0]));

  // 4. Bind Groups
  const bindGroupMask = device.createBindGroup({
    layout: cachedPipelineMask.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: voxelBuffer } },
      { binding: 1, resource: { buffer: cachedMaskBuffer } },
    ]
  });

  const bindGroupMerge = device.createBindGroup({
    layout: cachedPipelineMerge.getBindGroupLayout(0),
    entries: [
      { binding: 1, resource: { buffer: cachedMaskBuffer } },
      { binding: 2, resource: { buffer: cachedPositionsBuffer } },
      { binding: 3, resource: { buffer: cachedNormalsBuffer } },
      { binding: 4, resource: { buffer: cachedCounterBuffer } },
    ]
  });

  // 5. Dispatch Compute Passes
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.clearBuffer(cachedMaskBuffer);

  const passEncoder = commandEncoder.beginComputePass();
  
  // Pass A: Mask generation
  passEncoder.setPipeline(cachedPipelineMask);
  passEncoder.setBindGroup(0, bindGroupMask);
  passEncoder.dispatchWorkgroups(
    Math.ceil(CHUNK_SIZE / 4),
    Math.ceil(CHUNK_SIZE / 4),
    Math.ceil(CHUNK_SIZE / 4)
  );

  // Pass B: Greedy merging
  passEncoder.setPipeline(cachedPipelineMerge);
  passEncoder.setBindGroup(0, bindGroupMerge);
  passEncoder.dispatchWorkgroups(1, Math.ceil(CHUNK_SIZE / 16), 1);
  passEncoder.end();

  // 6. Read back counter (in same command buffer as compute)
  const counterReadBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(cachedCounterBuffer, 0, counterReadBuffer, 0, 4);
  
  device.queue.submit([commandEncoder.finish()]);

  // Wait for GPU compute + counter readback
  await counterReadBuffer.mapAsync(GPUMapMode.READ);
  const gpuDone = performance.now();
  const gpuComputeTime = gpuDone - start;

  const vertexCount = new Uint32Array(counterReadBuffer.getMappedRange())[0];
  counterReadBuffer.unmap();
  counterReadBuffer.destroy();

  const geometry = new THREE.BufferGeometry();

  if (vertexCount > 0) {
    const byteSize = vertexCount * 3 * 4;
    
    // Create readback buffers sized to actual vertex count (not max)
    const posReadBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const normReadBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(cachedPositionsBuffer, 0, posReadBuffer, 0, byteSize);
    copyEncoder.copyBufferToBuffer(cachedNormalsBuffer, 0, normReadBuffer, 0, byteSize);
    device.queue.submit([copyEncoder.finish()]);

    // Parallel readback — both maps happen concurrently
    await Promise.all([
      posReadBuffer.mapAsync(GPUMapMode.READ),
      normReadBuffer.mapAsync(GPUMapMode.READ),
    ]);

    const positions = new Float32Array(posReadBuffer.getMappedRange()).slice();
    const normals = new Float32Array(normReadBuffer.getMappedRange()).slice();

    posReadBuffer.unmap();
    normReadBuffer.unmap();
    posReadBuffer.destroy();
    normReadBuffer.destroy();

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  }

  const readbackEnd = performance.now();
  const readbackTime = readbackEnd - gpuDone;

  // Cleanup only the per-call voxel buffer
  voxelBuffer.destroy();

  const time = performance.now() - start;
  return { 
    geometry, 
    time, 
    quadCount: vertexCount / 6, 
    gpuComputeTime: gpuComputeTime.toFixed(2),
    readbackTime 
  };
}
