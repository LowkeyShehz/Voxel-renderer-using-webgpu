# WebGPU Voxel Engine

A research prototype that compares **CPU greedy meshing** with **GPU greedy meshing in
WebGPU compute shaders** for editable voxel terrain, built with React, Three.js r185
(`WebGPURenderer`) and Vite. It is the implementation for the project *Voxel-Based 3D
Rendering Optimization Using GPU Parallel Computing for Real-Time Visualization*.

What it does today:

- An **N×N world of 96³ chunks** (1×1 up to 6×6, one layer high) with sine-noise terrain.
- **Greedy meshing three ways**, switchable at runtime (all produce the same quads):
  | Mode | Meshing | Getting vertices to the screen |
  |------|---------|--------------------------------|
  | CPU greedy | JavaScript | geometry built on the CPU, uploaded by Three |
  | GPU + readback | WGSL compute | vertices copied back (`mapAsync`), re-uploaded by Three |
  | GPU direct (zero-copy) | WGSL compute | compute writes Three's vertex buffer, drawn with `drawIndirect` |
- **Editing**: click to remove/place blocks or set off explosions (radius 3–24). Only the
  chunks an edit touches are remeshed (plus neighbours whose shared border changed), one
  chunk per frame.
- **Benchmarks** built in: full-mesh timing, a seeded edit benchmark (frame-time
  stability), a sweep over world sizes and modes, per-pass GPU timings, a GPU-vs-CPU
  correctness check, JSON export.

Measured results (RX 5600 XT) are in [BENCHMARKS.md](BENCHMARKS.md).

## Requirements

- A browser with WebGPU: current Chrome or Edge on Windows, macOS or Linux. A
  discrete GPU is recommended for benchmarking; integrated GPUs work.
- `timestamp-query` for per-pass GPU times (Chrome exposes it on most desktop GPUs; the
  app shows "n/a" otherwise).
- Node.js 18+ for the dev server and the scripts.
- Without WebGPU the app still runs in **CPU mode only** (the GPU modes are disabled).

## Run

```bash
npm install
npm run dev          # http://localhost:3000
npm run build        # production build in dist/
```

URL options for A/B tests:

- `?mergeImpl=legacy` uses the original merge pass instead of the bit-run pass (default).
- `?mergeWG=<1..14>` sets the bit-run merge workgroup size (default 8).

Controls: **Orbit** (drag, right-drag to pan, scroll) or **Free-fly** (WASD, Space/Shift
up/down, drag or *Lock mouse* to look, wheel changes speed). Click to edit at the cursor,
or at the crosshair while the mouse is locked.

## Benchmark

See [BENCHMARKS.md → How to run](BENCHMARKS.md#how-to-run) and [TESTING.md](TESTING.md).
In short: keep the tab in the foreground, use **Benchmark ×20** for full-chunk meshing,
**Run edit benchmark** for frame times under edits, **Run sweep** for world sizes × modes,
export the JSON, then

```bash
node scripts/benchmarks-to-md.mjs benchmarks/sweep-<date>.json
```

prints markdown tables and writes a CSV next to the JSON.

## Architecture

```
src/
  voxel/ChunkData.js           96³ chunk, 1 bit per voxel (27,648 u32 words = 108 KB), dirty word range
  voxel/ChunkManager.js        N×N grid, world get/set/carveSphere, remesh-dirty tracking
  voxel/TerrainGenerator.js    sine-noise terrain in world coordinates
  meshing/cpuGreedyMesh.js     CPU greedy mesher (baseline), neighbour-aware
  meshing/gpuMeshShader.js     WGSL: mask, merge (bit-run and legacy), finalize
  meshing/gpuMeshPipeline.js   GpuMesher: buffers, dirty-range upload, dispatch, timestamps
  meshing/bitGreedyMerge.js    JS reference of the GPU mask + bit-run merge (unit-tested)
  meshing/neighbourBorders.js  neighbour layers packed for the GPU mask pass
  meshing/MeshController.js    the three modes per chunk, queued remesh, buffer growth
  meshing/correctnessCheck.js  GPU vs CPU quad-set comparison (1 chunk and a 2×2 world)
  render/DirectRenderBridge.js zero-copy: Three-owned storage/indirect buffers for compute
  render/FreeFlyControls.js    free-fly camera
  gpu/gpuContext.js            one GPUDevice shared by compute and Three
  interaction/                 DDA raycast over voxel data (not the mesh), editor
  benchmark/                   frame recorder, edit benchmark, sweep, statistics
  ui/panels.jsx                stats and result panels
VoxelEnginePrototype.jsx       app composition (scene, controls, UI)
scripts/benchmarks-to-md.mjs   benchmark JSON -> markdown + CSV
```

**GPU pipeline per chunk** (all in one command buffer, nothing read back in direct mode):

1. **Upload** only the occupancy words changed since the last mesh (dirty range), plus
   6.75 KB of neighbour border planes when the chunk has neighbours.
2. **mask** (one thread per voxel): for each of the 6 face directions, set a bit in a
   96×96 slice mask where a solid voxel faces air. Voxels across a chunk border are
   read from the neighbour planes, so faces between two solid chunks are not drawn.
3. **merge** (bit-run, default; one thread per slice, 72 workgroups of 8): scan each row
   (3 words) with `countTrailingZeros`, take the run of set bits as the quad width, grow
   the quad down while the next row has the whole run set, clear it, append 6 vertices
   with an atomic counter. Same quads as the CPU rule (unit-tested and checked on GPU).
4. **finalize**: clamp the vertex count to the buffer capacity; the count sits in the
   same buffer as the `drawIndirect` arguments, so the CPU never needs to know it.

Output is `vec4f` positions (w = face id; normals come from flat shading) in a
per-chunk buffer, 6 MB by default, grown and re-meshed if a chunk overflows it. In
direct mode that buffer belongs to a Three.js `StorageBufferAttribute`; its CPU copy is
released after allocation. Details: [DATA_FORMAT.md](DATA_FORMAT.md).

Picking uses a DDA raycast over the occupancy bits, so it works the same in every mode,
including zero-copy where no CPU geometry exists.

## Limits (by design, for now)

- One chunk layer (y = 0 only), fixed world size chosen in the UI; no streaming, LOD or
  frustum-based meshing. Chunks are frustum-culled at draw time.
- Binary voxels (solid/air), one material per mode; no textures or lighting beyond flat
  shading.
- The direct path relies on Three.js internals (`renderer._attributes`,
  `backend.get(attr).buffer`), verified against three@0.185.1; keep `three` pinned.

## Development

- The pure-logic modules (`bitGreedyMerge.js`, `neighbourBorders.js`, `ChunkManager.js`,
  `raycastVoxels.js`, `sweepSummary.js`) use explicit `.js` import extensions so Node can
  load them; they were unit-tested in Node during development against
  `cpuGreedyMesh.js` (the tests are not part of the repository).
- GPU-side checks: **Run correctness check** in the app; see [TESTING.md](TESTING.md).
- `.gitattributes` enforces LF line endings.

## License

MIT
