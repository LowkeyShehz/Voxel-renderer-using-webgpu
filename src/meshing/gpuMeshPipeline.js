import * as THREE from 'three/webgpu';
import { CHUNK_SIZE } from '../voxel/ChunkData';
import { createGpuMeshShader } from './gpuMeshShader';
import { getGpuDevice, hasTimestampQuery } from '../gpu/gpuContext';

// Right-size the output buffers. Greedy meshing on a 96^3 chunk typically
// produces ~50-60K quads = ~300-360K vertices. 500K vertices gives headroom.
// The shader derives its write cap from the same value.
export const MAX_VERTICES = 500000;

// 6 vertices of vec4f per quad (xyz position, w = face id)
export const POSITION_STRIDE_BYTES = 16;

// [vertexCount, instanceCount, firstVertex, firstInstance, requestedVertices, pad x3]
// The first 4 words are the drawIndirect() argument layout.
export const INDIRECT_WORDS = 8;
const INDIRECT_BYTES = INDIRECT_WORDS * 4;
const INDIRECT_INITIAL = new Uint32Array(INDIRECT_WORDS);
INDIRECT_INITIAL[1] = 1; // instanceCount

const WORDS_PER_SLICE = (CHUNK_SIZE * CHUNK_SIZE) / 32;
const MASK_BUFFER_SIZE = 6 * CHUNK_SIZE * WORDS_PER_SLICE * 4;

/**
 * A mesh target is the pair of GPU buffers the compute passes write into:
 *   { positions: GPUBuffer (STORAGE), indirect: GPUBuffer (STORAGE | COPY_DST) }
 * For the readback path we own these buffers; for the direct path they are the
 * buffers backing a Three.js geometry (see DirectRenderBridge).
 */
export class GpuMesher {
  static async create() {
    const device = await getGpuDevice();
    const mesher = new GpuMesher(device);
    await mesher._createPipelines();
    return mesher;
  }

  constructor(device) {
    this.device = device;
    this.timestamps = hasTimestampQuery(device);
    this._voxelBuffers = new WeakMap(); // ChunkData -> persistent GPUBuffer
    this._maskBuffer = device.createBuffer({
      size: MASK_BUFFER_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  // Pipelines are built up front so the first measured run is not a compile.
  async _createPipelines() {
    const module = this.device.createShaderModule({
      code: createGpuMeshShader(CHUNK_SIZE, MAX_VERTICES),
    });
    const make = (entryPoint) =>
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } });

    [this._maskPipeline, this._mergePipeline, this._finalizePipeline] = await Promise.all([
      make('mask_main'),
      make('merge_main'),
      make('finalize_main'),
    ]);
  }

  /** Buffers for the CPU-readback path. Caller owns them. */
  createReadbackTarget() {
    return {
      positions: this.device.createBuffer({
        size: MAX_VERTICES * POSITION_STRIDE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      indirect: this.device.createBuffer({
        size: INDIRECT_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    };
  }

  _voxelBufferFor(chunk) {
    let buffer = this._voxelBuffers.get(chunk);
    if (!buffer) {
      buffer = this.device.createBuffer({
        size: chunk.getBuffer().byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._voxelBuffers.set(chunk, buffer);
    }
    return buffer;
  }

  /**
   * Upload the chunk's occupancy bits and record + submit the three compute passes.
   * Nothing is read back here: the vertices stay in `target.positions` and the draw
   * arguments in `target.indirect`.
   *
   * @returns {{ submitMs: number, gpuMs: Promise<number|null> }}
   *   submitMs - CPU time spent encoding and submitting.
   *   gpuMs    - GPU execution time from timestamp queries (null if unsupported
   *              or `profile` is false). Reads back 16 bytes of timestamps only.
   */
  mesh(chunk, target, { profile = false } = {}) {
    const device = this.device;
    const t0 = performance.now();

    const voxelBuffer = this._voxelBufferFor(chunk);
    device.queue.writeBuffer(voxelBuffer, 0, chunk.getBuffer());
    device.queue.writeBuffer(target.indirect, 0, INDIRECT_INITIAL);

    const bind = (pipeline, entries) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    const maskGroup = bind(this._maskPipeline, [
      { binding: 0, resource: { buffer: voxelBuffer } },
      { binding: 1, resource: { buffer: this._maskBuffer } },
    ]);
    const mergeGroup = bind(this._mergePipeline, [
      { binding: 1, resource: { buffer: this._maskBuffer } },
      { binding: 2, resource: { buffer: target.positions } },
      { binding: 4, resource: { buffer: target.indirect } },
    ]);
    const finalizeGroup = bind(this._finalizePipeline, [
      { binding: 4, resource: { buffer: target.indirect } },
    ]);

    const timed = profile && this.timestamps;
    let querySet = null;
    const passDescriptor = {};
    if (timed) {
      querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      passDescriptor.timestampWrites = {
        querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      };
    }

    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(this._maskBuffer);

    const pass = encoder.beginComputePass(passDescriptor);

    // Pass A: mask generation
    pass.setPipeline(this._maskPipeline);
    pass.setBindGroup(0, maskGroup);
    pass.dispatchWorkgroups(
      Math.ceil(CHUNK_SIZE / 4),
      Math.ceil(CHUNK_SIZE / 4),
      Math.ceil(CHUNK_SIZE / 4)
    );

    // Pass B: greedy merging
    pass.setPipeline(this._mergePipeline);
    pass.setBindGroup(0, mergeGroup);
    pass.dispatchWorkgroups(1, Math.ceil(CHUNK_SIZE / 16), 1);

    // Pass C: clamp the vertex counter
    pass.setPipeline(this._finalizePipeline);
    pass.setBindGroup(0, finalizeGroup);
    pass.dispatchWorkgroups(1);

    pass.end();

    let resolveBuffer = null;
    let timestampRead = null;
    if (timed) {
      resolveBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      timestampRead = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, timestampRead, 0, 16);
    }

    device.queue.submit([encoder.finish()]);
    const submitMs = performance.now() - t0;

    const gpuMs = timed
      ? timestampRead.mapAsync(GPUMapMode.READ).then(() => {
          const t = new BigUint64Array(timestampRead.getMappedRange());
          const ms = Number(t[1] - t[0]) / 1e6;
          timestampRead.unmap();
          timestampRead.destroy();
          resolveBuffer.destroy();
          querySet.destroy();
          return ms;
        })
      : Promise.resolve(null);

    return { submitMs, gpuMs };
  }

  /** Resolves when everything submitted so far has finished executing on the GPU. No data is read back. */
  whenIdle() {
    return this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Reads the two counters from a target's indirect buffer (32 bytes).
   * For stats only: vertex data is never touched.
   */
  async readCounters(target) {
    const device = this.device;
    const staging = device.createBuffer({
      size: INDIRECT_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(target.indirect, 0, staging, 0, INDIRECT_BYTES);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();

    const vertexCount = words[0];
    const requestedVertices = words[4];
    return {
      vertexCount,
      quadCount: vertexCount / 6,
      overflowed: requestedVertices > vertexCount,
    };
  }

  /**
   * Legacy path: mesh, then copy the vertices back to the CPU and build a regular
   * THREE.BufferGeometry. Kept as the "GPU + readback" benchmark mode so the cost
   * of the readback can be compared against the zero-copy path.
   */
  async meshWithReadback(chunk, target) {
    const device = this.device;
    const start = performance.now();

    const { submitMs, gpuMs } = this.mesh(chunk, target, { profile: true });

    // Mapping the counter waits for the compute work to finish.
    const counters = await this.readCounters(target);
    const waitMs = performance.now() - start;

    const geometry = new THREE.BufferGeometry();
    let readbackMs = 0;

    if (counters.vertexCount > 0) {
      const readbackStart = performance.now();
      const byteSize = counters.vertexCount * POSITION_STRIDE_BYTES;

      const posRead = device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(target.positions, 0, posRead, 0, byteSize);
      device.queue.submit([encoder.finish()]);

      await posRead.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(posRead.getMappedRange()).slice();
      posRead.unmap();
      posRead.destroy();

      // Interleaved view over the vec4 data: position = xyz, w (face id) is skipped.
      const interleaved = new THREE.InterleavedBuffer(data, 4);
      geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));

      readbackMs = performance.now() - readbackStart;
    }

    const totalMs = performance.now() - start;
    return {
      geometry,
      quadCount: counters.quadCount,
      overflowed: counters.overflowed,
      submitMs,
      gpuMs: await gpuMs, // already resolved by now; awaited after totalMs so it is not counted
      waitMs, // upload + compute + queue wait, up to the counter becoming readable
      readbackMs,
      totalMs,
    };
  }
}
