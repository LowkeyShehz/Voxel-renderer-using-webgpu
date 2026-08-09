# WebGPU Voxel Engine

A high-performance Voxel Engine comparing **WebGPU Compute Shaders** against traditional **CPU Meshing**.

## 🚀 The Result (Benchmark)
After extensive optimization, we proved that WebGPU vastly outperforms the CPU for massive voxel workloads.

| Workload | Chunk Size | Voxels | CPU Time | GPU Time | Winner |
|----------|------------|--------|----------|----------|--------|
| Sparse (Sphere) | 32³ | 32k | ~2ms | ~5ms | CPU (Lower Overhead) |
| Dense (Noise) | 48³ | 110k | ~15ms | ~15ms | Tie |
| **Massive (Noise)** | **96³** | **884k** | **~50ms** | **~30ms** | **GPU (Almost 2x Faster)** |

**Why GPU Wins:**
- **Parallelism**: Processing 884k voxels sequentially on CPU is slow. The GPU processes them in parallel thousands of threads at a time.
- **Optimizations Implemented**:
    - **Pipeline Caching**: Pre-compiled shaders (saving 300ms/frame).
    - **Persistent Staging Buffers**: Reusing memory to avoid Garbage Collection (saving 100ms/frame).
    - **Partial Readback**: Downloading only valid data (saving PCIe bandwidth).

---

## 🎮 Game Vision
We are evolving this prototype into a full **Infinite Voxel Game**:
- **Hybrid Engine**: Seamlessly toggle between CPU and GPU meshing to see the difference live.
- **Infinite Terrain**: Dynamic chunk loading system.
- **Physics**: First-person player controller.

## 🛠️ Tech Stack
- **Languages**: JavaScript, WGSL (WebGPU Shading Language)
- **Frameworks**: React, Three.js, Vite
- **APIs**: WebGPU

## ⚡ Quick Start
1.  **Enable WebGPU**: Chrome requires `chrome://flags/#enable-unsafe-webgpu` (for now) or a compatible browser.
2.  **Install**:
    ```bash
    npm install
    ```
3.  **Run**:
    ```bash
    npm run dev
    ```

## 🧠 Architecture Highlights
- **Bit-Packed Data Structure**: Voxels are stored in a highly optimized `Uint32Array` bitmask, reducing a 96³ chunk footprint from ~3.5MB to just ~110KB.
- **GPU Greedy Meshing (Compute Shaders)**: 
    - **Pass 1:** Generates a parallel bit-packed exposure mask across 6 face directions.
    - **Pass 2:** Merges contiguous voxel faces into massive quads using high-speed private memory on the GPU, vastly reducing triangle count.
- **True Direct Rendering (Zero-Copy)**: 
    - Vertices are NEVER downloaded back to the CPU.
    - We use Three.js `WebGPURenderer` (r185) with a custom `DirectRenderBridge` to route WebGPU Compute output directly into Three.js vertex memory.
    - CPU readback time is ~0ms!
- **CPU Fallback**: 
    - A mathematically identical greedy meshing algorithm implemented in pure JavaScript serves as a scientific baseline for benchmarking.

## 📝 License
MIT
