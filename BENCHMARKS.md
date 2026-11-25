# Benchmarks

Results log for the CPU-vs-GPU comparison (Objective 5). Raw exports from the app
(**Export JSON**, **Export sweep JSON**) go in `benchmarks/`; turn them into tables
with `scripts/benchmarks-to-md.mjs` (see [How to run](#how-to-run)).

**Report machine:** AMD Radeon RX 5600 XT (RADV NAVI10, RDNA 1), Linux, Chrome 152, 1920×999 @ 60 Hz (`timestamp-query` available)
**Chunk:** 96³ (884,736 voxels), sine-noise terrain, 50,112 greedy quads / 116,820 naive faces

Every number below was measured on that machine unless a section says otherwise.
Empty tables are waiting for runs on it; nothing in them is estimated.

---

## How to run

All benchmarks run in the app (`npm run dev`, open http://localhost:3000 in Chrome on
the discrete-GPU machine). Keep the tab **visible and in the foreground** for the
whole run: browsers pause rendering in background tabs, which ruins frame times.
The app refuses to start (and the sweep stops) when the tab is hidden, and flags
runs whose frame coverage fell below 80%.

| What | How | Output |
|------|-----|--------|
| Full-chunk meshing | pick a mode, **Benchmark ×20** | stats panel (median of 20 after 2 warm-ups) |
| Merge pass A/B | open `/?mergeImpl=legacy`, then `/` (bit-run, default); GPU Direct, **Benchmark ×20** in each; read *mask · merge · finalize* | stats panel |
| Correctness | **Run correctness check** | panel + console JSON |
| Edit benchmark | choose a world size, **Run edit benchmark**, then **Export JSON** | `edit-benchmark-<date>.json` |
| Sweep | pick the largest world (default 4×4), **Run sweep**, then **Export sweep JSON** | `sweep-<date>.json` |

Then, from the repo root:

```bash
node scripts/benchmarks-to-md.mjs benchmarks/sweep-<date>.json > benchmarks/sweep-<date>.md
```

This prints markdown tables (paste them below) and writes `benchmarks/sweep-<date>.csv`
(one row per grid × mode, with median/min/max columns, for charts). Several files can
be passed at once; `--csv-dir DIR` puts the CSVs elsewhere. Edit-benchmark exports
from Phase 4 work too.

**What the sweep measures**, for every world size (1×1 … N×N chunks), every mode, and
3 back-to-back runs per configuration (median and spread reported):

- **initial**: full meshing of the whole grid from freshly generated terrain (chunk
  times summed; includes the full voxel upload on GPU modes);
- **edit benchmark**: 40 seeded explosions (radius from the slider) every 250 ms over
  the whole world. Each edit remeshes only the chunks it touched (plus neighbours
  whose shared border changed), one chunk per frame. *edit ms* = meshing time summed
  over those chunks; *latency* = edit to last chunk shown; frame times, hitches and
  bytes per edit as in Phase 4;
- **GPU time per pass** (mask / merge / finalize) from timestamp queries.

> Note: `benchmarks/edit-benchmark-2026-10-06-rx5600xt.json`, the raw data behind the
> Phase 4 tables, is linked below but is **not in the repository**. Please add it.

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
(code at `d377c70`/`6d91dea`; file missing from the repo, see note above).

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

### Per-pass GPU time (motivation for Phase 5 stage 1)

Per-pass GPU time for a full-chunk remesh (GPU direct run):

| Pass | GPU time | Share |
|------|---------:|------:|
| mask | 0.044 ms | 0.3% |
| **merge** | **13.854 ms** | **99.7%** |
| finalize | 0.003 ms | 0.0% |

A full-chunk remesh used 13.9 ms, about 83% of a 60 Hz frame budget, and the merge
pass was the entire cost: it dispatched only 6 workgroups (576 threads), each scanning
a 96×96 slice cell by cell from a 288-word private array.

---

## Phase 5, stage 1: bit-run merge pass

The merge pass was rewritten (`merge_bitrun_main`): rows of the slice mask are scanned
3 words at a time with `countTrailingZeros`, quads grow down by ANDing whole words,
the slice lives in workgroup memory, and the pass runs 72 workgroups of 8 threads.
Output is identical to the CPU greedy rule (unit-tested on 10 chunk types, and checked
on the GPU by **Run correctness check**). The old pass stays available as
`?mergeImpl=legacy` for this A/B.

**RX 5600 XT** (GPU Direct, Benchmark ×20, default terrain, median):

| Merge | mask ms | merge ms | finalize ms | GPU total ms | Total ms |
|-------|--------:|---------:|------------:|-------------:|---------:|
| legacy (`?mergeImpl=legacy`) | | | | | |
| bit-run (default) | | | | | |

Development check only, **not the report machine**: on the laptop used for development
(NVIDIA Turing laptop GPU, through the Claude desktop app's built-in Chromium pane),
running both merges back to back on the same chunk (median of 55, mask steady at
0.066 ms): legacy merge 2.82 ms, bit-run merge 1.25 ms. Workgroup size (`?mergeWG=1..14`)
made little difference there. What remains is serial work per slice: quads on this
terrain are small (1.29 × 1.61 cells on average) and the busiest slice has 366 quads,
about 17× the average, so that one thread sets the pass time. How the RX 5600 XT
compares is unknown until measured: its legacy merge was relatively much slower
(315× the mask pass, vs 43× on the laptop), so the gain there may differ.

---

## Phase 5: multi-chunk world (sweep)

Paste the output of `scripts/benchmarks-to-md.mjs` here. Expected layout (median
[min–max] over 3 runs):

### Initial full meshing of the grid (ms)

| grid (chunks) | CPU greedy | GPU + readback | GPU direct |
|---------------|-----------:|---------------:|-----------:|
| 1×1 (1) | | | |
| 2×2 (4) | | | |
| 3×3 (9) | | | |
| 4×4 (16) | | | |

### Edit benchmark (radius 24)

| grid (chunks) | mode | edit ms | latency ms | p99 frame | max frame | hitches | ↑ KB/edit | ↓ KB/edit |
|---------------|------|--------:|-----------:|----------:|----------:|--------:|----------:|----------:|
| 1×1 (1) | CPU greedy | | | | | | | |
| 1×1 (1) | GPU + readback | | | | | | | |
| 1×1 (1) | GPU direct | | | | | | | |
| 4×4 (16) | … | | | | | | | |

### GPU time per edit by pass (ms)

| grid (chunks) | mode | mask | merge | finalize |
|---------------|------|-----:|------:|---------:|
| 1×1 (1) | GPU direct | | | |

### Memory per chunk (from the code, not measured)

| | before (Phase 4) | now |
|-|-----------------:|----:|
| GPU output buffer, direct mode | 8 MB (500,000 vertices) | 6 MB default (393,216 vertices), grows on overflow to max 24 MB |
| CPU copy of that buffer (Three's attribute array) | 8 MB | 0 (released after allocation) |
| GPU voxel mirror | 108 KB | 108 KB |
| Shared scratch (masks + neighbour planes) | 648 KB | 655 KB (once, not per chunk) |

For 16 chunks of the default terrain: about 96 MB of GPU output buffers and ~1.7 MB of
voxel mirrors; the app shows the live total under the world selector.
