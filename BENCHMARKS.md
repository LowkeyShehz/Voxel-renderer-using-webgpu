# Benchmarks

Results log for the CPU-vs-GPU comparison (Objective 5). Raw exports from the app's
**Export JSON** button can be pasted or linked under each section.

**Test machine:** AMD Radeon RX 5600 XT (RADV NAVI10, RDNA 1), Linux, Chrome 152, 1920×999 @ 60 Hz (`timestamp-query` available)
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

Raw data: [`benchmarks/edit-benchmark-2026-10-06-rx5600xt.json`](benchmarks/edit-benchmark-2026-10-06-rx5600xt.json)
(code at `d377c70`/`6d91dea`).

**Setup:** 40 seeded explosions, **radius 24** (the slider maximum), one every 250 ms,
same starting terrain and same sequence for every mode (344,515 voxels removed in
total). Frame times are rAF-to-rAF at 60 Hz; the idle row is the display floor.
Frame-time coverage was 99.97–100.1% in every run, so no run was affected by paused
rendering.

| Mode | avg | p95 | p99 | max | hitches | >33ms | frames delivered |
|------|----:|----:|----:|----:|--------:|------:|-----------------:|
| idle (no edits) | 16.67 | 16.67 | 16.67 | 16.69 | 0 | 0 | 120 / 2.0 s |
| CPU greedy | 23.21 | 83.33 | 100.00 | 150.00 | **40** | **40** | 456 / 10.6 s |
| GPU + readback | 16.67 | 16.67 | 16.67 | 16.67 | 0 | 0 | 632 / 10.5 s |
| **GPU direct** | 16.67 | 16.67 | 16.67 | 16.68 | 0 | 0 | 631 / 10.5 s |

Per-edit cost:

| Mode | edit ms (median) | edit ms (max) | CPU→GPU per edit | GPU→CPU per edit | GPU compute |
|------|-----------------:|--------------:|-----------------:|-----------------:|------------:|
| CPU greedy | 89.74 | 153.64 | 4,122 KB | 0 | n/a |
| GPU + readback | 22.43 | 23.87 | 3,333 KB | 3,298 KB | 13.86 ms |
| **GPU direct** | **14.67** | 16.53 | **34.8 KB** | **0** | 13.90 ms |

### What this shows

- **CPU meshing produces one visible hitch per edit** (40 of 40 edits), up to 150 ms
  (9 dropped frames), and loses 176 of ~630 frames over the run. Both GPU modes held
  the 60 Hz display floor exactly: no frame slower than 16.7 ms.
- **Zero-copy does not change frame stability vs readback at this scale.** The
  readback path stays just as smooth: `mapAsync` is asynchronous, so the main thread
  is not blocked waiting (inferred; the frame data shows no stall). The measurable
  benefits of zero-copy are elsewhere:
  - **Edit latency:** 14.7 ms vs 22.4 ms (35% lower).
  - **Bandwidth:** 34.8 KB up / 0 down per edit, vs 3.3 MB up + 3.3 MB down for
    readback (about 190× less total traffic) and 4.1 MB up for the CPU path (118× less upload).
  - The GPU→CPU readback and re-upload of vertices disappears entirely.
- Frame times saturate at the display's refresh interval, so they cannot show GPU
  *headroom*. The GPU compute column does: see below.

### Partial (slice-scoped) remeshing: is it needed?

Per-pass GPU time for a full-chunk remesh (GPU direct run):

| Pass | GPU time | Share |
|------|---------:|------:|
| mask | 0.044 ms | 0.3% |
| **merge** | **13.854 ms** | **99.7%** |
| finalize | 0.003 ms | 0.0% |

- A full-chunk remesh uses **13.9 ms, about 83% of a 60 Hz frame budget**, and would
  not fit at 144 Hz (6.9 ms). It is stable today only because the GPU finishes within
  the frame.
- **The merge pass is the entire cost.** It dispatches only 6 workgroups (6 faces ×
  96 slices = 576 threads), each scanning a full 96×96 slice cell by cell. The
  bottleneck is that pass's structure, not the amount of data, so optimizing it is
  likely a better first step than slice-scoped output.
- Open question: the Phase 3 static benchmark measured only 6.33 ms GPU compute for
  the original terrain, versus 13.9 ms here. Candidates are the heavily carved terrain
  and the GPU clocking down between edits that are 250 ms apart; the static ×20
  benchmark now shows the per-pass split, which should separate the two.
