# Testing on the discrete-GPU PC

Checklists for the RX 5600 XT machine, one per stage of Phase 5. Each stage has its own
branch, so a problem can be traced to the stage that introduced it:

| Branch | Contains |
|--------|----------|
| `phase5-stage1` | bit-run merge pass, `?mergeImpl` A/B flag, correctness check |
| `phase5-stage2` | multi-chunk world, neighbour-aware meshing, buffer sizing, free-fly camera |
| `phase5-multichunk` | everything: stage 2 + sweep benchmark, script, docs |

**What was checked before handing over** (development laptop, NVIDIA Turing laptop GPU
through the Claude desktop app's built-in Chromium, so *not* your machine or browser):
`npm run build`; Node unit tests of all pure logic (mesher equivalence on 10 chunk
types, chunk grid, border planes, raycast, benchmark summary/script, an end-to-end CPU
sweep); in the browser, the correctness check (GPU = CPU, both merges, 1 chunk and
2×2), 3×3 GPU-direct rendering, cross-chunk explosions, and identical quad totals
across all three modes after edits. Frame-time numbers from that laptop are
meaningless (the pane renders at ~2 fps when not in front), so **every timing and
frame-time result must come from your PC**.

General rules for every run:

- Chrome, tab in the **foreground and visible** for the whole run, nothing else heavy
  running, laptop/PC on mains power. The app refuses to start (or stops) a benchmark
  when the tab is hidden and marks runs whose frame coverage fell below 80% with ⚠️:
  re-run those.
- After pulling a branch: `npm install` (only if `package-lock.json` changed), then
  `npm run dev` and open http://localhost:3000.
- Send back: console output (right-click → *Save as…* in DevTools console, or copy), a
  screenshot of the left panel, and any exported JSON (put it in `benchmarks/`).

---

## Stage 1: bit-run merge pass (`git checkout phase5-stage1`)

1. **Correctness.** Click **Run correctness check**.
   - Expect: `✅`, CPU 50,112 quads, *GPU bitrun 50,112 · same quad set*, *GPU legacy 50,112 · same quad set*.
   - Send: screenshot + the `Correctness check:` JSON from the console.
2. **Merge A/B timing.** Select **GPU Direct**.
   - Open `http://localhost:3000/?mergeImpl=legacy`, click **Benchmark ×20** twice, note
     the second result's *mask · merge · finalize* line and *Total Time*.
   - Open `http://localhost:3000/` (bit-run), same thing.
   - Then also `/?mergeWG=4` and `/?mergeWG=14` (bit-run with other workgroup sizes).
   - Send: screenshots of the four stats panels. These fill the stage 1 table in
     BENCHMARKS.md.
3. **Visual.** In GPU Direct with bit-run, **Generate Mesh**, orbit around. Expect the same
   terrain as CPU mode: no holes, no stray quads, no flicker.
4. **Edit benchmark (regression).** Radius 24, **Run edit benchmark**, **Export JSON**.
   Compare with the Phase 4 table: GPU modes should still hold 16.7 ms frames, and the
   per-pass *merge* time per edit should drop.
   - Send: the JSON.

If anything in 1 fails, stop and send the console output: the check prints the first
mismatching quads.

## Stage 2: multi-chunk world (`git checkout phase5-stage2`)

1. **Correctness.** **Run correctness check**. Expect `✅` with 50,112 quads for 1 chunk and
   **204,826** quads for the 2×2 world, for both merges.
2. **1×1 unchanged.** World 1×1; **Generate Mesh** in each of the three modes: 50,112 quads
   each.
3. **Worlds.** For 2×2, 3×3, 4×4 and 6×6, in **GPU Direct**: pick the size, wait, then
   check the *World* line (chunks, quads shown, GPU meshing memory) and orbit around.
   Expect no visible seams or extra walls on chunk borders, and ~6 MB output per chunk.
   - Send: screenshot of the panel for each size, and the console if anything is logged.
4. **Same world, all modes.** World 3×3, GPU Direct. Set radius 24, click **Random
   explosion** ~10 times (some will span 2–4 chunks: the edit line says *over N chunks*).
   Note *quads shown*. Switch to CPU, **Generate Mesh**; then GPU + Readback,
   **Generate Mesh**. Expect the same *quads shown* all three times.
5. **Editing across borders.** Blocks tool: remove and place blocks right at a chunk
   border (chunk borders are at x and z = 96, 192, …; floor grid lines are 4 voxels
   apart). Both chunks must
   update; no holes appear in the neighbour.
6. **Free-fly.** Camera → Free-fly. WASD/Space/Shift move, drag to look, wheel changes
   speed. **Lock mouse**: a crosshair appears; clicks edit at the crosshair; Esc releases.
   Switch back to Orbit.
7. **Frame stability.** World 4×4, radius 24, **Run edit benchmark**, **Export JSON**.
   - Send: the JSON and a screenshot of the results table (note the *latency* and
     *chunks* columns).

## Stage 3: sweep (`git checkout phase5-multichunk`)

1. Close other tabs/apps. Radius 24. Sweep up to **4×4** (default), **Run sweep**; 4 grid
   sizes × 3 modes × 3 runs take roughly 10 minutes (the panel shows an estimate; CPU mode
   on big worlds runs longer because its edits fall behind the 250 ms cadence). Do not
   touch the window.
2. **Export sweep JSON** into `benchmarks/`.
3. Optional, if time and memory allow: a second sweep up to 6×6.
4. Convert and check:
   ```bash
   node scripts/benchmarks-to-md.mjs benchmarks/sweep-<date>.json > benchmarks/sweep-<date>.md
   ```
   This also writes `benchmarks/sweep-<date>.csv`. Paste the tables into BENCHMARKS.md
   (Phase 5 section).
   - Send: the JSON (and the .md/.csv if you ran the script).
5. Add the missing Phase 4 file `benchmarks/edit-benchmark-2026-10-06-rx5600xt.json`
   (it is linked from BENCHMARKS.md but not in the repository).

What to look at in the results: CPU *initial ms* should grow roughly linearly with chunk
count; GPU *edit ms* should stay near one or two chunk remeshes regardless of world size
(only touched chunks are remeshed); CPU hitches should persist at every size; ↓ KB/edit
should be ~0 (32 bytes of counters) for GPU Direct and ~MBs for GPU + Readback.

## Stage 4: docs

Read README.md, DATA_FORMAT.md and BENCHMARKS.md once against the running app; report
anything that does not match what you see.
