# Frontend & UI Implementation Plan (Abdul Waris)

We have successfully completed **Phase 1, Phase 2, and Phase 3**. The voxel engine now generates geometry exclusively on the GPU and renders it directly without transferring the massive vertex arrays back to the CPU (Zero-Copy).

As per the `voxel-project-agent-brief.md`, you are now up! Here is your implementation plan for **Phase 4 and Phase 5**.

## Phase 4: Dynamic Terrain Modification

The goal of this phase is to prove our academic thesis: *traditional CPU meshing stutters when terrain changes, while our GPU mesher handles dynamic updates smoothly.*

### Task 1: Raycasting & Interactive Editing
We need the user to be able to click on the mesh to add or remove blocks.
1. Add an event listener to the canvas in `VoxelEnginePrototype.jsx` for mouse clicks.
2. Use a `THREE.Raycaster` to intersect the mouse position with `sceneRef.current.mesh`.
3. When an intersection occurs, use `intersect.point` and `intersect.face.normal` to determine the exact voxel coordinate `(x, y, z)`.
   - To **remove** a block: step slightly inside the mesh along the inverse normal.
   - To **add** a block: step slightly outside the mesh along the normal.
4. Update the data by calling `chunk.set(x, y, z, 0)` (remove) or `chunk.set(x, y, z, 1)` (add).

### Task 2: "Explosion" Test Mode
The presentation specifically mentions explosions as a stress test.
1. Add an "Explode!" button to the UI panel.
2. When clicked, cast a ray to find the intersection point on the terrain.
3. Run a loop to `chunk.set(x, y, z, 0)` for all voxels within a sphere (e.g., radius = 5 blocks) centered on that intersection point.

### Task 3: Trigger Remesh & Measure
1. After any edit (single block or explosion), immediately trigger the `gpuGreedyMeshGeneration()` pipeline (or CPU pipeline if in CPU mode).
2. Measure the exact frame-time it took to regenerate and render the mesh, and update the UI `stats`. You should visually see the CPU mode stutter heavily on an explosion, while the GPU mode remains smooth!

## Phase 5: Navigation & Multi-Chunk Polish

### Task 4: Camera Controls
1. The current camera just orbits automatically (`Math.cos(...)` loop).
2. Replace the `animate` camera movement with `THREE.OrbitControls` or `THREE.PointerLockControls` (imported from `three/addons/controls/OrbitControls.js`).
3. Allow the user to fly around the terrain freely.

### Task 5: Benchmark Export (Final Polish)
1. Update the Benchmark UI panel to track the **Average Edit Time** and **Max Edit Time** over the last 10 edits.
2. This is the exact data we need to put into the final project report/PPT to satisfy Objective 5!

---

**Note on Git & Code Organization:**
The Three.js renderer is now `WebGPURenderer`. It uses an asynchronous render loop (`renderer.renderAsync(scene, camera)`), so keep that in mind when updating the animate function for your camera controls!
