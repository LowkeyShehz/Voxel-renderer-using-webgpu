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

/** Merge-pass implementations (`?mergeImpl=` in the URL picks the default). */
export const MERGE_IMPLS = ['bitrun', 'legacy'];
export const DEFAULT_MERGE_WG = 8;

/**
 * Reads `?mergeImpl=legacy|bitrun` and `?mergeWG=<threads>` from the page URL so the
 * merge passes can be A/B tested without a rebuild.
 */
export function mergeOptionsFromUrl(search = globalThis.location?.search ?? '') {
  const params = new URLSearchParams(search);
  const impl = params.get('mergeImpl');
  const wg = Number(params.get('mergeWG'));
  return {
    mergeImpl: MERGE_IMPLS.includes(impl) ? impl : 'bitrun',
    mergeWorkgroupSize: Number.isInteger(wg) && wg > 0 ? wg : DEFAULT_MERGE_WG,
  };
}

/**
 * A mesh target is the pair of GPU buffers the compute passes write into:
 *   { positions: GPUBuffer (STORAGE), indirect: GPUBuffer (STORAGE | COPY_DST) }
 * For the readback path we own these buffers; for the direct path they are the
 * buffers backing a Three.js geometry (see DirectRenderBridge).
 */
export class GpuMesher {
  /**
   * @param {{ mergeImpl?: 'bitrun'|'legacy', mergeWorkgroupSize?: number }} [opts]
   *   mergeImpl is the default for mesh(); both merge pipelines are always built so
   *   the correctness check can run each of them.
   */
  static async create({ mergeImpl = 'bitrun', mergeWorkgroupSize = DEFAULT_MERGE_WG } = {}) {
    const device = await getGpuDevice();
    const mesher = new GpuMesher(device, { mergeImpl, mergeWorkgroupSize });
    await mesher._createPipelines();
    return mesher;
  }

  constructor(device, { mergeImpl = 'bitrun', mergeWorkgroupSize = DEFAULT_MERGE_WG } = {}) {
    if (!MERGE_IMPLS.includes(mergeImpl)) throw new Error(`Unknown mergeImpl "${mergeImpl}"`);
    this.device = device;
    this.mergeImpl = mergeImpl;
    this.mergeWorkgroupSize = mergeWorkgroupSize;
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
      code: createGpuMeshShader(CHUNK_SIZE, MAX_VERTICES, { mergeWorkgroupSize: this.mergeWorkgroupSize }),
    });
    const make = (entryPoint) =>
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } });

    let legacy, bitrun;
    [this._maskPipeline, legacy, bitrun, this._finalizePipeline] = await Promise.all([
      make('mask_main'),
      make('merge_main'),
      make('merge_bitrun_main'),
      make('finalize_main'),
    ]);
    this._merge = {
      legacy: { pipeline: legacy, workgroups: [1, Math.ceil(CHUNK_SIZE / 16), 1] },
      bitrun: { pipeline: bitrun, workgroups: [Math.ceil((6 * CHUNK_SIZE) / this.mergeWorkgroupSize), 1, 1] },
    };
  }

  /** Frees the GPU mirror of a chunk that will not be meshed again. */
  releaseChunk(chunk) {
    this._voxelBuffers.get(chunk)?.destroy();
    this._voxelBuffers.delete(chunk);
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

  /**
   * Keeps one persistent GPU mirror of the chunk's occupancy bits and brings it up
   * to date by uploading only the words edited since the last sync (the whole chunk
   * the first time). Assumes this mesher is the only consumer of the chunk's dirty
   * range, so edits made while another meshing mode is active are still picked up.
   *
   * @returns {{ buffer: GPUBuffer, uploadedBytes: number }}
   */
  _syncVoxelBuffer(chunk) {
    const data = chunk.getBuffer();
    let buffer = this._voxelBuffers.get(chunk);

    if (!buffer) {
      buffer = this.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._voxelBuffers.set(chunk, buffer);
      this.device.queue.writeBuffer(buffer, 0, data);
      chunk.consumeDirtyRange(); // the full upload supersedes any pending edits
      return { buffer, uploadedBytes: data.byteLength };
    }

    const range = chunk.consumeDirtyRange();
    if (!range) return { buffer, uploadedBytes: 0 };

    // dataOffset and size are in elements (u32 words) for typed-array sources
    this.device.queue.writeBuffer(buffer, range.start * 4, data, range.start, range.end - range.start);
    return { buffer, uploadedBytes: (range.end - range.start) * 4 };
  }

  /**
   * Upload the chunk's occupancy bits and record + submit the three compute passes.
   * Nothing is read back here: the vertices stay in `target.positions` and the draw
   * arguments in `target.indirect`.
   *
   * @returns {{ submitMs: number, uploadedBytes: number, gpuMs: Promise<{totalMs: number, passMs: Object<string, number>}|null> }}
   *   submitMs      - CPU time spent encoding and submitting.
   *   uploadedBytes - occupancy bytes sent CPU -> GPU for this call (dirty range only).
   *   gpuMs         - GPU execution time from timestamp queries, total and per pass
   *                   keyed by pass name { mask, merge, finalize } (null if
   *                   unsupported or `profile` is false). Reads back 16 bytes of
   *                   timestamps per pass only.
   */
  mesh(chunk, target, { profile = false, mergeImpl = this.mergeImpl } = {}) {
    const device = this.device;
    const t0 = performance.now();
    const merge = this._merge[mergeImpl];
    if (!merge) throw new Error(`Unknown mergeImpl "${mergeImpl}"`);

    const { buffer: voxelBuffer, uploadedBytes } = this._syncVoxelBuffer(chunk);
    device.queue.writeBuffer(target.indirect, 0, INDIRECT_INITIAL);

    const bind = (pipeline, entries) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    const maskGroup = bind(this._maskPipeline, [
      { binding: 0, resource: { buffer: voxelBuffer } },
      { binding: 1, resource: { buffer: this._maskBuffer } },
    ]);
    const mergeGroup = bind(merge.pipeline, [
      { binding: 1, resource: { buffer: this._maskBuffer } },
      { binding: 2, resource: { buffer: target.positions } },
      { binding: 4, resource: { buffer: target.indirect } },
    ]);
    const finalizeGroup = bind(this._finalizePipeline, [
      { binding: 4, resource: { buffer: target.indirect } },
    ]);

    const steps = [
      {
        name: 'mask',
        pipeline: this._maskPipeline,
        group: maskGroup,
        workgroups: [Math.ceil(CHUNK_SIZE / 4), Math.ceil(CHUNK_SIZE / 4), Math.ceil(CHUNK_SIZE / 4)],
      },
      { name: 'merge', pipeline: merge.pipeline, group: mergeGroup, workgroups: merge.workgroups },
      { name: 'finalize', pipeline: this._finalizePipeline, group: finalizeGroup, workgroups: [1, 1, 1] },
    ];
    const record = (pass, step) => {
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.group);
      pass.dispatchWorkgroups(...step.workgroups);
    };

    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(this._maskBuffer);

    const timed = profile && this.timestamps;
    let querySet = null;
    let resolveBuffer = null;
    let timestampRead = null;

    if (timed) {
      // One compute pass per step so each gets its own begin/end timestamps.
      // Passes still execute in order with storage writes visible to the next one.
      const queryCount = steps.length * 2;
      const bytes = queryCount * 8;
      querySet = device.createQuerySet({ type: 'timestamp', count: queryCount });
      steps.forEach((step, i) => {
        const pass = encoder.beginComputePass({
          timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 },
        });
        record(pass, step);
        pass.end();
      });
      resolveBuffer = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      timestampRead = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.resolveQuerySet(querySet, 0, queryCount, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, timestampRead, 0, bytes);
    } else {
      const pass = encoder.beginComputePass();
      steps.forEach((step) => record(pass, step));
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
    const submitMs = performance.now() - t0;

    const gpuMs = timed
      ? timestampRead.mapAsync(GPUMapMode.READ).then(() => {
          const t = new BigUint64Array(timestampRead.getMappedRange());
          const passMs = Object.fromEntries(steps.map((step, i) => [step.name, Number(t[2 * i + 1] - t[2 * i]) / 1e6]));
          timestampRead.unmap();
          timestampRead.destroy();
          resolveBuffer.destroy();
          querySet.destroy();
          return { totalMs: Object.values(passMs).reduce((a, b) => a + b, 0), passMs };
        })
      : Promise.resolve(null);

    return { submitMs, uploadedBytes, gpuMs };
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

  /** Copies the first `vertexCount` vec4 vertices of a target back to the CPU. */
  async readPositions(target, vertexCount) {
    const byteSize = vertexCount * POSITION_STRIDE_BYTES;
    if (byteSize === 0) return new Float32Array(0);
    const posRead = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(target.positions, 0, posRead, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await posRead.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(posRead.getMappedRange()).slice();
    posRead.unmap();
    posRead.destroy();
    return data;
  }

  /**
   * Legacy path: mesh, then copy the vertices back to the CPU and build a regular
   * THREE.BufferGeometry. Kept as the "GPU + readback" benchmark mode so the cost
   * of the readback can be compared against the zero-copy path.
   */
  async meshWithReadback(chunk, target) {
    const device = this.device;
    const start = performance.now();

    const { submitMs, uploadedBytes, gpuMs } = this.mesh(chunk, target, { profile: true });

    // Mapping the counter waits for the compute work to finish.
    const counters = await this.readCounters(target);
    const waitMs = performance.now() - start;

    const geometry = new THREE.BufferGeometry();
    let readbackMs = 0;
    let readbackBytes = 0;

    if (counters.vertexCount > 0) {
      const readbackStart = performance.now();
      const byteSize = counters.vertexCount * POSITION_STRIDE_BYTES;
      const data = await this.readPositions(target, counters.vertexCount);

      // Interleaved view over the vec4 data: position = xyz, w (face id) is skipped.
      const interleaved = new THREE.InterleavedBuffer(data, 4);
      geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));

      readbackMs = performance.now() - readbackStart;
      readbackBytes = byteSize;
    }

    const totalMs = performance.now() - start;
    const gpuTiming = await gpuMs; // already resolved by now; awaited after totalMs so it is not counted
    return {
      geometry,
      quadCount: counters.quadCount,
      overflowed: counters.overflowed,
      submitMs,
      uploadedBytes,
      gpuMs: gpuTiming?.totalMs ?? null,
      passMs: gpuTiming?.passMs ?? null,
      waitMs, // upload + compute + queue wait, up to the counter becoming readable
      readbackMs,
      readbackBytes,
      totalMs,
    };
  }
}
