import { Mesher } from './Mesher';
import * as THREE from 'three/webgpu';
import { CHUNK_SIZE } from '../Chunk';
import { createGpuMeshShader } from '../../meshing/gpuMeshShader';

/**
 * GPU-based meshing implementation.
 * Uses WebGPU Compute Shaders.
 */
export class GPUMesher extends Mesher {
    constructor(bridge) {
        super();
        this.bridge = bridge;
        this.device = null;
        this.pipelineMask = null;
        this.pipelineMerge = null;
        this.bindGroupLayout = null;
        this.maskBuffer = null;
        this.isSupported = !!navigator.gpu;
    }

    async init() {
        if (!this.isSupported) {
            console.warn("WebGPU not supported");
            return false;
        }

        if (!this.bridge) {
            console.error("No DirectRenderBridge provided to GPUMesher");
            return false;
        }

        if (this.device) return true;

        try {
            await this.bridge.init();
            this.device = this.bridge.device;
            if (!this.device) throw new Error("Bridge has no device");
            console.log("GPUMesher: Device initialized via Bridge");

            this.createPipeline();
            return true;
        } catch (e) {
            console.error("GPUMesher init failed:", e);
            this.isSupported = false;
            return false;
        }
    }

    createPipeline() {
        const shaderModule = this.device.createShaderModule({ code: createGpuMeshShader(CHUNK_SIZE) });

        this.pipelineMask = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'mask_main' }
        });

        this.pipelineMerge = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'merge_main' }
        });

        this.bindGroupLayout = this.pipelineMask.getBindGroupLayout(0);
    }

    async mesh(chunk) {
        if (!this.device) {
            const success = await this.init();
            if (!success) return null;
        }

        const startCompute = performance.now();

        const voxelData = chunk.data; // Uint8Array, 1 byte per voxel
        // Expand to Uint32Array (1 voxel per u32) for WGSL storage buffer alignment
        const voxelDataU32 = new Uint32Array(voxelData.length);
        for (let i = 0; i < voxelData.length; i++) {
            voxelDataU32[i] = voxelData[i];
        }

        const voxelBuffer = this.device.createBuffer({
            size: voxelDataU32.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(voxelBuffer, 0, voxelDataU32);

        // Mask buffer (6 faces * CHUNK_SIZE slices * WORDS_PER_SLICE u32 words)
        const WORDS_PER_SLICE = (CHUNK_SIZE * CHUNK_SIZE) / 32;
        const MASK_BUFFER_SIZE = 6 * CHUNK_SIZE * WORDS_PER_SLICE * 4; // bytes
        if (!this.maskBuffer) {
            this.maskBuffer = this.device.createBuffer({
                size: MASK_BUFFER_SIZE,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }

        // Output Buffers
        const maxVertices = Math.min(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 6 * 6, 3500000);
        const positionsComputeBuffer = this.device.createBuffer({
            size: maxVertices * 3 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const normalsComputeBuffer = this.device.createBuffer({
            size: maxVertices * 3 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const counterBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]));

        // Bind Group
        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: voxelBuffer } },
                { binding: 1, resource: { buffer: this.maskBuffer } },
                { binding: 2, resource: { buffer: positionsComputeBuffer } },
                { binding: 3, resource: { buffer: normalsComputeBuffer } },
                { binding: 4, resource: { buffer: counterBuffer } },
            ]
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.clearBuffer(this.maskBuffer);

        const passEncoder = commandEncoder.beginComputePass();

        // Pass A
        passEncoder.setPipeline(this.pipelineMask);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            Math.ceil(CHUNK_SIZE / 4),
            Math.ceil(CHUNK_SIZE / 4),
            Math.ceil(CHUNK_SIZE / 4)
        );

        // Pass B
        passEncoder.setPipeline(this.pipelineMerge);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(1, 1, 1);
        passEncoder.end();

        // Read counter
        const counterReadBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);

        this.device.queue.submit([commandEncoder.finish()]);

        const computeEnd = performance.now();

        await counterReadBuffer.mapAsync(GPUMapMode.READ);
        const vertexCount = new Uint32Array(counterReadBuffer.getMappedRange())[0];
        counterReadBuffer.unmap();

        const readbackEnd = performance.now();

        const geometry = new THREE.BufferGeometry();

        if (vertexCount > 0) {
            // ZERO-COPY DIRECT RENDERING
            const positionAttr = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
            const normalAttr = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);

            geometry.setAttribute('position', positionAttr);
            geometry.setAttribute('normal', normalAttr);

            const threePositionGPUBuffer = this.bridge.getGPUBuffer(positionAttr);
            const threeNormalGPUBuffer = this.bridge.getGPUBuffer(normalAttr);

            const copyEncoder = this.device.createCommandEncoder();
            const byteSize = vertexCount * 3 * 4;
            copyEncoder.copyBufferToBuffer(positionsComputeBuffer, 0, threePositionGPUBuffer, 0, byteSize);
            copyEncoder.copyBufferToBuffer(normalsComputeBuffer, 0, threeNormalGPUBuffer, 0, byteSize);
            this.device.queue.submit([copyEncoder.finish()]);
        }

        // Cleanup
        voxelBuffer.destroy();
        positionsComputeBuffer.destroy();
        normalsComputeBuffer.destroy();
        counterBuffer.destroy();
        counterReadBuffer.destroy();

        return {
            geometry,
            stats: {
                gpuTime: computeEnd - startCompute,
                readbackTime: readbackEnd - computeEnd
            }
        };
    }
}
