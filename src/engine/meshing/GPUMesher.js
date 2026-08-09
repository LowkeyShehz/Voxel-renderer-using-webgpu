import { Mesher } from './Mesher';
import * as THREE from 'three';
import { CHUNK_SIZE } from '../Chunk'; // Import consistent size

/**
 * GPU-based meshing implementation.
 * Uses WebGPU Compute Shaders.
 */
export class GPUMesher extends Mesher {
    constructor() {
        super();
        this.device = null;
        this.pipeline = null;
        this.stagingBuffer = null;
        this.isSupported = !!navigator.gpu;
    }

    async init() {
        if (!this.isSupported) {
            console.warn("WebGPU not supported");
            return false;
        }

        if (this.device) return true; // Already init

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error("No WebGPU adapter");
            this.device = await adapter.requestDevice();
            console.log("GPUMesher: Device initialized");
            return true;
        } catch (e) {
            console.error("GPUMesher init failed:", e);
            this.isSupported = false;
            return false;
        }
    }

    async mesh(chunk) {
        if (!this.device) {
            const success = await this.init();
            if (!success) return null;
        }

        const voxelData = chunk.data;
        const VOXEL_COUNT = CHUNK_SIZE ** 3;

        // 1. Upload Data
        // Convert to Uint32Array for WGSL alignment
        const voxelDataU32 = new Uint32Array(voxelData);
        const voxelBuffer = this.device.createBuffer({
            size: voxelDataU32.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(voxelBuffer, 0, voxelDataU32);

        // 2. Prepare Output Buffers
        // Clamp to ~3.5M vertices (Safeguard)
        const maxVertices = Math.min(VOXEL_COUNT * 6 * 6, 3500000);
        const floatsPerVertex = 8; // vec3 + pad + vec3 + pad
        const vertexBufferSize = maxVertices * floatsPerVertex * 4;

        const vertexBuffer = this.device.createBuffer({
            size: vertexBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const counterBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]));

        // 3. Pipeline
        if (!this.pipeline) {
            this.createPipeline();
        }

        // 4. Bind Group
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: voxelBuffer } },
                { binding: 1, resource: { buffer: vertexBuffer } },
                { binding: 2, resource: { buffer: counterBuffer } },
            ],
        });

        // 5. Dispatch
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            Math.ceil(CHUNK_SIZE / 4),
            Math.ceil(CHUNK_SIZE / 4),
            Math.ceil(CHUNK_SIZE / 4)
        );
        passEncoder.end();

        // 6. Readback Counter
        const counterReadBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);

        // Submit Pass 1
        this.device.queue.submit([commandEncoder.finish()]);

        // Map Counter
        await counterReadBuffer.mapAsync(GPUMapMode.READ);
        const vertexCount = new Uint32Array(counterReadBuffer.getMappedRange())[0];
        counterReadBuffer.unmap();

        if (vertexCount === 0) return null;

        // 7. Readback Vertices (Optimized)
        const byteSize = vertexCount * floatsPerVertex * 4;

        // Resize staging buffer if needed
        if (!this.stagingBuffer || this.stagingBuffer.size < byteSize) {
            if (this.stagingBuffer) this.stagingBuffer.destroy();
            // Allocate for max potential to avoid thrashing, or just byteSize?
            // Let's stick to the VoxelEnginePrototype strategy: allocate max if needed 
            // logic there was: if cachedStagingBuffer.size < requiredSize
            // Here, let's just allocate what we need plus a bit of buffer? 
            // Actually, let's replicate the safety max allocation logic
            const safeMax = maxVertices * floatsPerVertex * 4;
            this.stagingBuffer = this.device.createBuffer({
                size: safeMax,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
        }

        if (this.stagingBuffer.mapState === 'mapped') {
            this.stagingBuffer.unmap();
        }

        const readEncoder = this.device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(vertexBuffer, 0, this.stagingBuffer, 0, byteSize);
        this.device.queue.submit([readEncoder.finish()]);

        await this.stagingBuffer.mapAsync(GPUMapMode.READ, 0, byteSize);
        // Copy to JS Array
        const rawData = new Float32Array(this.stagingBuffer.getMappedRange(0, byteSize));
        const vertexDataCopy = new Float32Array(rawData);
        this.stagingBuffer.unmap();

        // Cleanup temp buffers
        // (Note: voxelBuffer, vertexBuffer, counterBuffer, counterReadBuffer are collected by GC/WebGPU)
        // In a real engine, we'd pool these too!

        // 8. Geometry Construction
        // We unpack the padding here?
        // Or can we just give Three.js the stride? Three.js BufferAttribute has itemSize...
        // But we have interleaved padding (vec3, float, vec3, float).
        // Three.js InterleavedBufferAttribute can handle this!
        // That would be faster (zero copy unpacking).
        // For now, let's stick to the copy loop for safety/compatibility with CPUMesher output format.

        const positions = new Float32Array(vertexCount * 3);
        const normals = new Float32Array(vertexCount * 3);

        for (let i = 0; i < vertexCount; i++) {
            const base = i * 8;
            positions[i * 3] = vertexDataCopy[base];
            positions[i * 3 + 1] = vertexDataCopy[base + 1];
            positions[i * 3 + 2] = vertexDataCopy[base + 2];

            normals[i * 3] = vertexDataCopy[base + 4];
            normals[i * 3 + 1] = vertexDataCopy[base + 5];
            normals[i * 3 + 2] = vertexDataCopy[base + 6];
        }

        return {
            positions,
            normals,
            indices: null // Non-indexed geometry
        };
    }

    createPipeline() {
        // Shader Code
        // Note: Using dynamic CHUNK_SIZE injection
        const shaderCode = `
            struct Voxel { data: array<u32> }
            struct Vertex { position: vec3f, normal: vec3f }
            struct Vertices { data: array<Vertex> }
            struct Counter { value: atomic<u32> }

            @group(0) @binding(0) var<storage, read> voxels: Voxel;
            @group(0) @binding(1) var<storage, read_write> vertices: Vertices;
            @group(0) @binding(2) var<storage, read_write> counter: Counter;

            const CHUNK_SIZE: u32 = ${CHUNK_SIZE}u;
            const LIMIT: u32 = 3500000u; // Safety clamp

            fn getVoxel(x: i32, y: i32, z: i32) -> u32 {
                if (x < 0 || x >= i32(CHUNK_SIZE) || y < 0 || y >= i32(CHUNK_SIZE) || z < 0 || z >= i32(CHUNK_SIZE)) {
                    return 0u;
                }
                let idx = u32(x) + u32(y) * CHUNK_SIZE + u32(z) * CHUNK_SIZE * CHUNK_SIZE;
                return voxels.data[idx];
            }

            fn emitFace(x: f32, y: f32, z: f32, normal: vec3f, corners: array<vec3f, 4>) {
                let baseIdx = atomicAdd(&counter.value, 6u);
                if (baseIdx + 6u >= LIMIT) { return; }

                vertices.data[baseIdx + 0u].position = vec3f(x, y, z) + corners[0];
                vertices.data[baseIdx + 0u].normal = normal;
                vertices.data[baseIdx + 1u].position = vec3f(x, y, z) + corners[1];
                vertices.data[baseIdx + 1u].normal = normal;
                vertices.data[baseIdx + 2u].position = vec3f(x, y, z) + corners[2];
                vertices.data[baseIdx + 2u].normal = normal;

                vertices.data[baseIdx + 3u].position = vec3f(x, y, z) + corners[0];
                vertices.data[baseIdx + 3u].normal = normal;
                vertices.data[baseIdx + 4u].position = vec3f(x, y, z) + corners[2];
                vertices.data[baseIdx + 4u].normal = normal;
                vertices.data[baseIdx + 5u].position = vec3f(x, y, z) + corners[3];
                vertices.data[baseIdx + 5u].normal = normal;
            }

            @compute @workgroup_size(4, 4, 4)
            fn main(@builtin(global_invocation_id) global_id: vec3u) {
                let x = i32(global_id.x);
                let y = i32(global_id.y);
                let z = i32(global_id.z);
                
                if (x >= i32(CHUNK_SIZE) || y >= i32(CHUNK_SIZE) || z >= i32(CHUNK_SIZE)) { return; }
                
                let voxel = getVoxel(x, y, z);
                if (voxel == 0u) { return; }

                let fx = f32(x); let fy = f32(y); let fz = f32(z);

                if (getVoxel(x + 1, y, z) == 0u) { emitFace(fx, fy, fz, vec3f(1,0,0), array<vec3f, 4>(vec3f(1,0,0), vec3f(1,1,0), vec3f(1,1,1), vec3f(1,0,1))); }
                if (getVoxel(x - 1, y, z) == 0u) { emitFace(fx, fy, fz, vec3f(-1,0,0), array<vec3f, 4>(vec3f(0,0,1), vec3f(0,1,1), vec3f(0,1,0), vec3f(0,0,0))); }
                if (getVoxel(x, y + 1, z) == 0u) { emitFace(fx, fy, fz, vec3f(0,1,0), array<vec3f, 4>(vec3f(0,1,0), vec3f(0,1,1), vec3f(1,1,1), vec3f(1,1,0))); }
                if (getVoxel(x, y - 1, z) == 0u) { emitFace(fx, fy, fz, vec3f(0,-1,0), array<vec3f, 4>(vec3f(0,0,1), vec3f(0,0,0), vec3f(1,0,0), vec3f(1,0,1))); }
                if (getVoxel(x, y, z + 1) == 0u) { emitFace(fx, fy, fz, vec3f(0,0,1), array<vec3f, 4>(vec3f(1,0,1), vec3f(1,1,1), vec3f(0,1,1), vec3f(0,0,1))); }
                if (getVoxel(x, y, z - 1) == 0u) { emitFace(fx, fy, fz, vec3f(0,0,-1), array<vec3f, 4>(vec3f(0,0,0), vec3f(0,1,0), vec3f(1,1,0), vec3f(1,0,0))); }
            }
        `;

        const module = this.device.createShaderModule({ code: shaderCode });
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });
    }
}
