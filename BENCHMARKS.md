# Benchmarks

Results log for the CPU-vs-GPU comparison (Objective 5). Raw exports from the app's
**Export JSON** button can be pasted or linked under each section.

**Test machine:** discrete GPU PC (fill in: GPU model, driver, browser + version, display refresh rate)
**Chunk:** 96³ (884,736 voxels), sine-noise terrain, 50,112 greedy quads / 116,820 naive faces

---

## Phase 3: full-chunk meshing (zero-copy vs readback vs CPU)

Measured with **Benchmark ×20** (median of 20 runs after 2 warm-up runs), code at `45b1a6a`.

| Mode | Total (median) | Best run | GPU compute | Readback |
|------|---------------:|---------:|------------:|---------:|
| CPU greedy meshing | 101.87 ms | 96.86 ms | n/a | 0 ms |
| GPU + readback (`mapAsync`) | 20.78 ms | 19.46 ms | 10.03 ms | 9.19 ms |
| **GPU direct (zero-copy)** | **7.77 ms** | **6.70 ms** | 6.33 ms | none |

- GPU direct is **13.1×** faster than the CPU baseline and **2.7×** faster than the
  readback path.
- Readback alone costs 9.19 ms, which is more than the GPU compute in direct mode.
  Removing it is what the zero-copy path buys.
- "Total" definitions: CPU = mesh generation only (excludes Three's geometry upload on
  first draw, which favours the CPU). GPU readback = occupancy upload + compute +
  vertex readback. GPU direct = occupancy upload + compute, measured to queue completion
  (`onSubmittedWorkDone`), no data returned.

---

## Phase 4: frame-time stability under edits

Not yet recorded. To produce this section:

1. `npm run dev`, open the app in a **visible** tab (background tabs pause rendering).
2. Click **Run edit benchmark**. It applies 40 seeded explosions (radius 8, one every
   250 ms) in each available mode, from identical starting terrain, and records every
   frame time. An idle (no edits) run provides the display's baseline.
3. Check the `cov.` column is close to 100%; the app warns if rendering was paused.
4. **Export JSON**, then paste the table below.

Columns: `avg`/`p99`/`max` = frame time in ms; `hitches` = frames slower than 2× the
median; `>33ms` = frames below 30 fps; `edit ms` = median remesh time per edit;
`↑/↓ per edit` = CPU→GPU / GPU→CPU bytes per edit.

| Mode | avg | p99 | max | hitches | >33ms | edit ms | ↑ per edit | ↓ per edit |
|------|----:|----:|----:|--------:|------:|--------:|-----------:|-----------:|
| idle | | | | | | n/a | n/a | n/a |
| CPU greedy | | | | | | | | |
| GPU + readback | | | | | | | | |
| GPU direct | | | | | | | | |

### Whether partial (slice-scoped) remeshing is needed

Phase 4a remeshes the whole chunk on the GPU after each edit and uploads only the
changed voxel words. The per-pass GPU timings (mask / merge / finalize, shown in the
stats panel when `timestamp-query` is available) decide whether to build slice-scoped
remeshing (4b): if a full-chunk remesh fits comfortably inside a frame budget, it is
not worth the added complexity.

| Pass | GPU time |
|------|---------:|
| mask | |
| merge | |
| finalize | |
