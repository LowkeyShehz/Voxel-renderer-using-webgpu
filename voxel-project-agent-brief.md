# Voxel-Based 3D Rendering Optimization Using GPU Parallel Computing
### Full Project Context, Architecture Brief & Two-Person Task Division
### For use as a system/context prompt for an AI coding agent working on the GitHub repo

**Repo:** `LowkeyShehz/Voxel-renderer-using-webgpu`
**Team:** Shehzan (core/compute/backend) & Abdul Waris (frontend + supporting backend)
**Institution:** AITM Bhatkal, VTU — Guide: Mr. Sarfraz Mirzanaik

---

## 0. Read This First — Instructions for the AI Agent

You are acting as an expert graphics-programming collaborator on a WebGPU/WGSL/Three.js voxel engine. This document is the single source of truth for:

- What the academic project requires (from the approved project PPT/abstract)
- What the current prototype **actually** does today (verified against the live code, not the README's marketing description)
- What is left to build, broken into phases
- Who on the two-person team owns which piece

Do not assume features exist just because the README claims them. Section 2 below documents a real discrepancy between the README and the code — treat the code as ground truth.

When implementing any task, always:
1. State which phase/task number you're working on.
2. Confirm the file(s) you're touching.
3. Preserve the CPU-baseline mode (`mode === 'cpu'`) as a working control group — it's required for benchmarking (Objective 5 in the PPT).
4. Keep WGSL and JS/React changes in separate, reviewable commits.

---

## 1. Project Abstract & Objectives (from approved project PPT)

**Title:** Voxel-Based 3D Rendering Optimization Using GPU Parallel Computing for Real-Time Visualization in Computer Graphics Applications.

**Problem:** Traditional voxel engines use CPU-based greedy meshing. Dynamic terrain updates (block placement/removal, explosions) force repeated CPU→GPU buffer uploads and sequential CPU processing, causing frame-time spikes and stutter.

**Proposed solution:** Move mesh generation entirely onto the GPU using WebGPU compute shaders (WGSL), so the CPU only ever transmits raw voxel occupancy data, and rendering reads directly from GPU-resident buffers.

**Objectives (verbatim intent from PPT):**
1. Implement a GPU-accelerated greedy meshing system using WebGPU + Three.js.
2. Enhance real-time rendering performance and reduce CPU bottlenecks during dynamic terrain modification.
3. Transfer only minimal voxel occupancy data between CPU and GPU.
4. Use GPU parallelism for face detection and mesh generation, reducing sync overhead and improving frame stability.
5. Benchmark GPU-based performance against traditional CPU-based systems.

**Target architecture (Existing → Proposed):**

| | Existing (CPU) Pipeline | Proposed (GPU) Pipeline |
|---|---|---|
| Step 1 | CPU stores voxels, computes occupancy/neighbors | CPU stores voxels, sends only raw occupancy to GPU |
| Step 2 | CPU does sequential greedy meshing → vertex/index buffers | WGSL compute shader does parallel face detection + mask generation + **quad merging** on GPU |
| Step 3 | Full buffers re-uploaded to GPU every update | Scene renders directly from GPU-resident buffer; no re-upload, no CPU roundtrip |

---

## 2. Current Prototype State — Verified Against Actual Code

The repo's README overstates what's implemented. Here is what `VoxelEnginePrototype.jsx` (594 lines, main and only substantial file) actually does, confirmed by reading the source directly:

**What exists and works:**
- A single 96×96×96 voxel chunk (`CHUNK_SIZE = 96`), generated once from a sine-noise function — not the "32³" the README text mentions in places.
- Two meshing code paths, switchable via radio button:
  - **CPU path** (`cpuMeshGeneration`): triple-nested loop, checks 6 neighbors per solid voxel, emits one quad per exposed face. Naive face culling only — **no face merging of any kind.**
  - **GPU path** (`gpuMeshGeneration`): uploads voxel data as a storage buffer, dispatches a WGSL compute shader (`@workgroup_size(4,4,4)`) that does the same per-voxel, per-face exposure check in parallel, using `atomicAdd` on a counter buffer to reserve output slots — this correctly avoids race conditions on the shared vertex buffer.
- A CPU-vs-GPU timer comparison UI (`stats.time`, `stats.faces`) — this is a genuinely useful head start on Objective 5 (benchmarking).
- Device/pipeline caching across runs (`cachedDevice`, `cachedPipeline`) to avoid re-init overhead — good practice, keep it.

**What the README claims but the code does NOT do (do not assume these are done):**
- ❌ **"GPU generates AND renders mesh (no CPU roundtrip)"** — false as written. After the compute pass, the code does `counterReadBuffer.mapAsync()` and `stagingBuffer.mapAsync()` to **copy the entire vertex buffer back to the CPU**, unpacks it into typed arrays, builds a `THREE.BufferGeometry`, and renders it with a standard `THREE.WebGLRenderer`. There is no WebGPU-native render path at all — WebGPU is used purely as a compute co-processor, then the result is round-tripped back through the CPU into WebGL. **Step 3 of the target architecture ("Direct Render") is not implemented.**
- ❌ **Greedy meshing / quad merging** — not implemented on either the CPU or GPU path. Both simply emit one quad per exposed voxel face (`emitFace` in WGSL, the `faces` loop in JS). This is the single feature most directly named in your objectives ("mask generation and quad merging") and it does not exist yet, anywhere in the codebase.
- ❌ **Dynamic terrain modification** — the prototype is a one-shot "click to generate" demo. There is no block add/remove, no partial/incremental chunk regeneration, no notion of "an explosion happened, remesh only the affected region."
- ❌ **Chunking / multi-chunk world** — only a single hardcoded chunk exists. No chunk manager, no streaming, no frustum culling, no LOD.

**Bottom line:** the prototype currently proves "raw per-face GPU generation is faster than CPU per-face generation," which is a fine proof-of-concept starting point, but the actual required deliverables — GPU-side greedy meshing, a true zero-copy render path, and dynamic update handling — are still ahead, not partially built.

---

## 3. Corrected Priority Order

Unlike a generic roadmap, this ordering reflects what's *actually* missing, weighted by (a) how central it is to your stated objectives and (b) technical dependency order:

1. **GPU-side greedy meshing (WGSL)** — highest priority. It's explicitly named in your objectives and is currently 0% implemented on both paths.
2. **True direct/zero-copy rendering** — needed to honestly claim "eliminates CPU-GPU synchronization," which is central to your abstract's problem statement. Two viable approaches (pick one, discussed in Phase 3 below): Three.js `WebGPURenderer` integration, or manual `drawIndirect()`.
3. **Dynamic terrain modification + partial remeshing** — this is what your abstract's motivating example (explosions, block edits) actually requires; without it you can't demonstrate the frame-time-spike problem you're solving.
4. **Multi-chunk system + benchmarking rigor** — needed for realistic-scale results in your final report.

---

## 4. Technology Stack

- **Graphics API:** WebGPU
- **Shader language:** WGSL (compute shaders for meshing; eventually a WebGPU render pipeline)
- **Rendering framework:** Three.js (currently `WebGLRenderer`; Phase 3 evaluates migrating to `WebGPURenderer`)
- **Build tooling:** Vite + React (already set up in repo)
- **Language:** JavaScript/JSX (consider TypeScript for the compute/data-layer modules going forward, optional)

---

## 5. Two-Person Task Division

**Shehzan — Core / Compute / Backend (data structures, WGSL, performance)**
**Abdul Waris — Frontend / UI / Interaction, plus supporting backend where noted**

The split below is organized by phase. Tasks are labeled `[SZ]`, `[AW]`, or `[SHARED]`. Within a phase, `[SZ]` tasks generally must land before `[AW]` tasks that consume their output — this is noted explicitly as a dependency.

### Phase 1 — Data Layer Cleanup & Chunk Abstraction
*Goal: make the voxel data structure chunk-aware and reusable, instead of one hardcoded 96³ blob baked into the component.*

- `[SZ]` Extract voxel storage into a standalone module (e.g. `voxel/ChunkData.js`): a class/struct wrapping a flat `Uint8Array` (or `Uint16Array` if you want block-type IDs later) with `get(x,y,z)`, `set(x,y,z,value)`, bounds checks, and a fixed `CHUNK_SIZE` constant pulled out of the component file.
- `[SZ]` Define the binary/packing format for occupancy data that will be sent to the GPU (currently expanded to `Uint32Array` per voxel — wasteful; consider bit-packing occupancy into `u32` bitmasks, 32 voxels per word, which also sets up the greedy-meshing mask step in Phase 2).
- `[AW]` Pull the noise-based terrain generator (`generateVoxelData`) out into its own module (`voxel/TerrainGenerator.js`) with adjustable parameters (scale, seed, density) exposed as function args — this will let the UI later expose terrain controls.
- `[SHARED]` Agree on and document the occupancy data format (bit-packed vs byte-per-voxel) in a short `DATA_FORMAT.md` before Phase 2 starts, since both the WGSL shader and the UI/debug tools need to agree on it.

**Definition of done:** voxel data generation and storage are decoupled from the React component; both CPU and GPU meshing paths read from the same `ChunkData` abstraction without behavior change (this phase should be a refactor with no visible functional change — verify old CPU/GPU timings still work before moving on).

---

### Phase 2 — GPU Greedy Meshing (the core deliverable)
*Goal: implement mask generation + quad merging in WGSL, replacing the current "one quad per exposed face" approach.*

- `[SZ]` Implement the binary greedy meshing algorithm in WGSL. Standard approach (based on the well-known Minecraft-meshing technique — see reference in Section 8):
  1. For each of the 6 face directions, and for each axis-aligned slice through the chunk, build a 2D "exposure mask" (which cells in this slice have an exposed face in this direction).
  2. Run a greedy rectangle-merging pass over each 2D mask to combine adjacent exposed cells into the largest possible rectangles (quads).
  3. Emit one quad per merged rectangle instead of one quad per voxel face.
- `[SZ]` This is naturally a **two-pass compute pipeline**: Pass A builds mask buffers per slice (storage buffer of booleans/bits), Pass B reads the masks and emits merged quads with atomic counter allocation (reuse the existing `atomicAdd` pattern from the current shader).
- `[SZ]` Decide and document: do you merge on GPU per-slice in parallel (harder, more "correct" for the project's GPU-parallelism objective) or do a lighter hybrid (CPU-side merge parameters, GPU emits) — **recommendation: keep it GPU-side**, since "GPU parallel computing for mask generation and quad merging" is explicitly named in your objectives and is likely to be evaluated directly.
- `[AW]` Build a **face-count / triangle-count comparison readout** in the UI: show side-by-side "naive face count" vs "greedy-merged quad count" for the same chunk, so the merging benefit is visually demonstrable in your demo and report (this is good report material — a bar or number comparison, doesn't need to be fancy).
- `[SHARED]` Validate correctness: render the greedy-meshed output and visually confirm there are no gaps, no overlapping quads, and no incorrect normals versus the naive per-face version. Write a small manual test chunk (e.g. a single 4×4×4 solid cube) where you can hand-verify the expected merged quad count.

**Definition of done:** GPU compute shader produces significantly fewer quads than the naive per-face approach for the same voxel data, with no visual artifacts, and the CPU baseline is updated to also do greedy meshing (so your Objective 5 benchmark compares "CPU greedy vs GPU greedy," not "CPU naive vs GPU greedy," which would be an unfair/invalid comparison for your report).

---

### Phase 3 — True Direct Rendering (Zero-Copy)
*Goal: eliminate the CPU roundtrip that currently exists after every GPU compute pass.*

- `[SZ]` Evaluate and choose one of two approaches (document the decision and reasoning in the report — this is a legitimate design-tradeoff discussion for your final submission):
  - **Option A — Three.js `WebGPURenderer` / TSL path:** migrate rendering from `WebGLRenderer` to Three.js's WebGPU renderer, and feed it a `THREE.StorageBufferAttribute` backed by the compute shader's output buffer directly, avoiding the `mapAsync`/CPU-copy step entirely.
  - **Option B — Manual `drawIndirect()`:** keep a raw WebGPU render pipeline outside Three.js for the voxel mesh specifically, using `GPURenderPassEncoder.drawIndirect()` with an indirect-args buffer written by the compute shader's atomic counter, so the GPU decides the draw vertex count without any CPU readback.
  - *Recommendation:* Option A is likely easier to integrate incrementally with your existing Three.js scene/camera/lighting setup; Option B is the more "pure" WebGPU implementation and may map more cleanly to how your PPT describes Step 3, but requires bypassing Three.js's render loop for this mesh.
- `[SZ]` Remove the `counterReadBuffer.mapAsync()` / `stagingBuffer.mapAsync()` CPU-readback path once the chosen direct-render approach is working — this is the code change that actually earns the "eliminates repeated uploads and CPU-GPU synchronization" claim in your abstract.
- `[AW]` Update the benchmarking UI to add a new timing metric distinguishing "GPU compute time" from "GPU→CPU readback time" (currently these are lumped into one `elapsed` measurement) — once the direct-render path lands, the readback number should drop to ~0, which is a great before/after chart for your report.
- `[SHARED]` Confirm frame rate / GPU timing stays stable when this mesh is combined with the rest of the Three.js scene (lighting, grid helper, orbiting camera) — integration between a raw WebGPU buffer and the rest of a WebGL/Three.js scene can be fiddly; budget real debugging time here.

**Definition of done:** vertex data generated by the compute shader never returns to CPU memory before being rendered; the "readback time" benchmark metric is at (or near) zero.

---

### Phase 4 — Dynamic Terrain Modification
*Goal: demonstrate the actual problem your abstract describes — that dynamic edits cause CPU bottlenecks in traditional engines, and that your system avoids them.*

- `[SZ]` Implement partial/incremental remeshing: when a voxel changes, only the affected chunk (or affected region within a chunk) should be re-sent to the GPU and re-meshed — not the whole chunk from scratch, and definitely not a full CPU rebuild.
- `[SZ]` Add a dirty-region tracking mechanism (e.g. track a bounding box of changed voxels since last remesh) so the compute dispatch can be scoped down for small edits.
- `[AW]` Build the **interactive editing UI**: raycasting from the camera/mouse into the voxel grid (Three.js raycaster against the current mesh or against a simple bounding-box grid) to let a user click to add/remove a voxel.
- `[AW]` Add an "explosion" test mode — a UI button that removes a spherical/cubic region of voxels at once, simulating the PPT's explicit example scenario, and displays the resulting remesh time.
- `[SHARED]` Extend the benchmark harness to specifically measure **frame-time stability during edits** (not just single-shot generation time) — e.g. average and max frame time over N consecutive edits, for both the CPU-baseline path and the GPU path. This is the metric that most directly proves/disproves your core thesis.

**Definition of done:** a user can click to add/remove voxels (and trigger an "explosion") and see the mesh update in real time, with measured frame-time data showing the GPU path staying stable where the CPU baseline visibly spikes.

---

### Phase 5 — Multi-Chunk System, Benchmarking Polish & Report Data
*Goal: scale beyond a single chunk and produce the final data/figures for your report.*

- `[SZ]` Implement a basic chunk manager: multiple `ChunkData` instances laid out in a grid, each independently meshed/remeshed.
- `[SZ]` (Stretch/Future Scope per PPT) Basic LOD or frustum culling — only required if time allows; your PPT explicitly lists this under "Future Scope," not "Current Scope," so don't let this block your deadline.
- `[AW]` Camera/navigation controls suitable for multiple chunks (free-fly or orbit controls covering a larger world, replacing the current fixed auto-orbit camera).
- `[SHARED]` Run the final benchmark suite: CPU-greedy vs GPU-greedy, across a range of chunk counts/sizes, capturing generation time, edit/remesh time, and frame-time stability. Export this as charts/tables for the final report — this directly satisfies Objective 5.

**Definition of done:** the demo runs a multi-chunk world with interactive editing, and you have a clean benchmark dataset comparing CPU vs GPU across all four objectives from Section 1.

---

## 6. Suggested File Structure (target state)

```
voxel-engine-webgpu/
├── src/
│   ├── voxel/
│   │   ├── ChunkData.js          [SZ] data structure from Phase 1
│   │   ├── TerrainGenerator.js   [AW] from Phase 1
│   │   ├── ChunkManager.js       [SZ] from Phase 5
│   ├── meshing/
│   │   ├── cpuGreedyMesh.js      [SZ] CPU baseline, updated in Phase 2
│   │   ├── gpuMeshShader.wgsl.js [SZ] WGSL source, Phase 2 mask+merge logic
│   │   ├── gpuMeshPipeline.js    [SZ] buffer setup, dispatch, Phase 3 direct-render wiring
│   ├── render/
│   │   ├── SceneSetup.js         [AW] camera, lights, controls
│   │   ├── DirectRenderBridge.js [SZ] Phase 3 WebGPU/Three.js bridge
│   ├── interaction/
│   │   ├── VoxelEditor.js        [AW] Phase 4 raycasting/editing
│   ├── ui/
│   │   ├── BenchmarkPanel.jsx    [AW] timing UI, Phase 2/3/4 metrics
│   │   ├── Controls.jsx          [AW] mode switch, terrain params
│   ├── VoxelEnginePrototype.jsx  # slimmed down to composition only
├── DATA_FORMAT.md                 [SHARED] Phase 1 output
├── BENCHMARKS.md                  [SHARED] Phase 5 results log
```

---

## 7. Git Workflow Suggestion

- Branch per phase/task (e.g. `phase2-greedy-meshing`, `phase4-editing-ui`), merged via PR even if it's just the two of you — gives you a clean commit history to reference/screenshot for your report.
- Keep WGSL shader strings in dedicated `.wgsl` files (imported as raw strings via Vite's `?raw` import) rather than inline template literals once Phase 2 grows the shader significantly — much easier to review as a diff.
- Since Shehzan owns most of the compute/data layer and Abdul owns most of the UI/interaction layer, most day-to-day work should avoid merge conflicts naturally — the main integration points needing coordination are: the `DATA_FORMAT.md` contract (Phase 1), the benchmark data shape consumed by the UI (Phases 2–5), and the raycasting/editing hookup into the mesh (Phase 4).

---

## 8. Reference Material

- Greedy meshing algorithm reference: https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
- WebGPU fundamentals: https://webgpufundamentals.org/
- WGSL spec: https://www.w3.org/TR/WGSL/
- Three.js WebGPURenderer docs (check current Three.js version's examples for `WebGPURenderer` + `StorageBufferAttribute` / TSL compute nodes)

---

## 9. Definition of Done (Whole Project)

The implementation satisfies the PPT's objectives when:
1. Mesh generation uses GPU-side mask generation and quad merging (not per-face emission) — Phase 2.
2. The CPU only ever transmits raw voxel occupancy; no CPU-side vertex list generation — Phase 1/2.
3. Rendering reads directly from GPU-resident buffers with no per-frame CPU roundtrip — Phase 3.
4. Dynamic edits (including a simulated "explosion") trigger only a partial GPU remesh, and measured frame-time stays stable where the CPU baseline visibly spikes — Phase 4.
5. You have a documented CPU-vs-GPU benchmark dataset (generation time, edit time, frame stability) suitable for the final report — Phase 5.
