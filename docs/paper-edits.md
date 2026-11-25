# Paper edits from the Phase 5 runs (RX 5600 XT, 2026-11-25)

Every number below comes from a committed export in `benchmarks/`. Items marked
**STILL OPEN** have no measurement yet; do not fill them from expectations.

Sources:
- `benchmarks/phase5/merge-ab-2026-11-25T13-31-51-920Z.json`: merge A/B, 55 alternating rounds
- `benchmarks/phase5/sweep-2026-11-25T13-58-50-605Z-r24-1to3.json`: sweep 1×1–3×3, radius 24, 3 runs each, bit-run merge
- `benchmarks/phase5/sweep-2026-11-25T14-43-22-210Z-r24-4x4.json`: sweep 4×4 only, same settings, run in a fresh tab after the
  memory-leak fix (commit 898113c). The 1×1–3×3 runs used the code from before that fix; the fix changes memory use,
  not the meshing work, but say so if a reviewer asks why the 4×4 row was measured separately.
- In-app correctness check (screenshot, same session)
- `benchmarks/phase4/edit-benchmark-2026-10-06T18-04-57-411Z.json`: the data behind Tables V–VI (legacy merge)

All three Phase 5 exports report Chrome/155 in the user agent. The browser was Brave (Chromium 155);
the Phase 4 data was Chrome 152.

---

## 1. Section VI.A: replace the red TODO "GPU 2×2 neighbour check on the RX 5600 XT"

> On the RX 5600 XT the in-app check gave the same result: both GPU merge passes returned
> 50,112 quads for the single chunk and 204,826 for the 2×2 terrain, with quad sets
> identical to the CPU mesher's.

## 2. Table VII: fill the RX 5600 XT row

| GPU | Legacy merge | Bit-run merge | Speed-up |
|---|---|---|---|
| Laptop NVIDIA (Turing) | 2.82 ms | 1.25 ms | 2.3× |
| RX 5600 XT | **6.30 ms** | **2.23 ms** | **2.8×** |

New footnote: *Medians of 55 alternating rounds on each GPU; mask pass 0.066 ms (laptop) and 0.033 ms (RX 5600 XT).*

Spread on the RX 5600 XT, if wanted: legacy 6.26–6.42 ms and bit-run 2.16–2.28 ms (25th–75th percentile).
The median of the per-round ratios is also 2.83×.

## 3. Section VI.D: add after the laptop sentence

> On the RX 5600 XT, timed the same way, the bit-run merge took 2.23 ms against 6.30 ms,
> a 2.8× reduction.

**Watch out for the 13.85 ms.** Section IV.E and Table VII both say the legacy merge took 13.85 ms on the
RX 5600 XT. That number was measured *during the edit benchmark* (Table VI). Timed on its own, the same
pass took 6.30 ms. Do not put 13.85 in Table VII; it would make the speed-up look like 6×. Suggested
wording for IV.E:

> On the RX 5600 XT this pass took 13.85 ms of a 13.9 ms remesh during the edit benchmark (6.3 ms when
> timed on its own), so it was the whole cost.

## 4. Table VIII: fill all four rows

Bit-run merge, radius 24, medians of 3 runs. Every run had 100% frame coverage (0.999–1.001).

| Grid | CPU full | GPU direct full | CPU max frame | GPU max frame |
|---|---|---|---|---|
| 1×1 | 120.6 | 6.2 | 116.7 | 16.7 |
| 2×2 | 490.8 | 17.3 | 500.0 | 16.7 |
| 3×3 | 1064.5 | 36.5 | 500.0 | 16.7 |
| 4×4 | 1892.7 | 61.6 | 483.3 | 16.7 |

Values are in ms. "GPU max frame" is GPU direct. GPU + readback also stayed at 16.7 ms, except one
33.3 ms frame in one of the three 3×3 runs (0–1 hitches).

Suggested text to replace the VI.D TODO:

> Full meshing grew in proportion to the number of chunks on both paths: about 118–123 ms per chunk on
> the CPU (120.6, 490.8, 1064.5 and 1892.7 ms for 1, 4, 9 and 16 chunks) against about 4–6 ms per chunk
> for GPU direct (6.2, 17.3, 36.5 and 61.6 ms), a 31× gap at 4×4. During edits the CPU's worst frame
> rose from 117 ms on one chunk to about 500 ms on the larger worlds, where one explosion dirties
> several chunks (2.3 on average at 4×4). On the 4×4 world the CPU path never recovered between edits:
> its median frame was 125 ms and 52 frames exceeded 33 ms. Every frame of the GPU direct path stayed at
> 16.7 ms on every world size, and it moved 70–78 KB per edit, against 9.4–13 MB on the CPU path.

Hitch count caveat for 4×4 CPU: "hitches" are frames slower than twice the *median* frame. Because the
median CPU frame was itself 125 ms, only 13 frames counted as hitches. Quote ">33 ms frames" (52) for
that row instead.

Supporting medians (same file):

| Grid | Mode | Edit ms | ↑ KB/edit | GPU ms/edit |
|---|---|---|---|---|
| 1×1 | CPU | 103.2 | 3,943 | — |
| 1×1 | GPU + readback | 11.6 | 3,189 | 3.91 |
| 1×1 | GPU direct | 4.9 | 34.8 | 3.90 |
| 2×2 | CPU | 121.4 | 9,418 | — |
| 2×2 | GPU direct | 5.7 | 70.4 | 4.14 |
| 3×3 | CPU | 223.0 | 9,967 | — |
| 3×3 | GPU direct | 10.0 | 70.5 | 7.25 |

| 4×4 | CPU | 231.3 | 13,177 | — |
| 4×4 | GPU direct | 11.2 | 77.6 | 7.77 |

GPU + readback on 2×2, 3×3 and 4×4: 13.8, 27.2 and 29.9 ms per edit, about 7.5–10.6 MB up and down.
Full meshing with readback on 4×4: 231.5 ms.

## 5. Tables IV–VI: optional bit-run rows (1×1, radius 24, final code)

Tables IV–VI are labelled "legacy merge pass". The 1×1 rows of the sweep are the same edit benchmark on
the final code, so they can be added as bit-run rows or quoted in the text.

For Table VI (medians of 3 runs):

| Mode | Edit ms | CPU→GPU | GPU→CPU | GPU ms |
|---|---|---|---|---|
| CPU greedy | 103.2 | 3,943 KB | 0 | n/a |
| GPU + readback | 11.6 | 3,189 KB | 3,155 KB | 3.91 |
| GPU direct | 4.9 | 34.8 KB | 0 | 3.90 |

For Table V: CPU avg 23.1, p99 100.0, max 116.7, 39 hitches, 39 frames over 33 ms. Both GPU modes ran
avg, p99 and max at 16.67 with 0 hitches.

KB per edit is averaged over all 40 edits. Two of them landed while a remesh was running and were folded
into it (0.95 chunks remeshed per edit), which is why CPU→GPU here is 3,943 KB rather than Table VI's
4,122 KB. The meshes themselves are identical.

Table IV static numbers on the final code: CPU greedy, from the app's Benchmark ×20 panel (screenshot, 1×1):
median 119.98 ms, best 117.80 ms. The two GPU rows are **STILL OPEN** (run Benchmark ×20 in GPU + Readback
and GPU Direct). Benchmark ×20 has no export button, so keep a screenshot of each panel as the record.

## 6. Section VI.E "Where the system lags": update

Replace "and we have not yet shown that the bit-run pass closes that gap on the main GPU" with:

> With the bit-run merge the RX 5600 XT finished an edit in 4.9 ms (3.9 ms of GPU time), which fits in
> a 144 Hz frame; a full 13.9 ms legacy remesh would not.

Keep the point that the merge is still serial within a slice.

## 7. Abstract and Conclusion

- **Abstract.** "cut merge time from 2.82 ms to 1.25 ms on a second GPU" becomes "cut merge time
  2.8-fold on the RX 5600 XT (6.30 to 2.23 ms) and 2.3-fold on a second GPU".
- **Conclusion, Outcome.** Same change.
- **Conclusion, Limitations.** Drop "the multi-chunk scaling results and the bit-run timing on the main GPU
  are still to be measured". Scaling was measured up to 4×4 chunks (16 chunks, 817,291 quads); larger
  worlds (5×5, 6×6) were not benchmarked.

## 8. Text that no longer matches the code

- **Section IV.F**, "so frustum culling is turned off for it". The code gives each chunk's geometry an
  explicit bounding box (`src/render/DirectRenderBridge.js:63`), so chunks are frustum-culled. Replace
  with:
  > The mesher therefore gives each chunk's geometry its bounding box explicitly, the chunk's own
  > 96³ cube, so chunks are still frustum-culled.
- **Section VII, Future enhancement**, "add frustum culling and level of detail for larger worlds"
  becomes "skip meshing chunks outside the view and add level of detail for larger worlds".
- **Table III.** Add the Phase 5 browser (Brave, Chromium 155) next to Chrome 152, or state which tables
  used which.

## 9. STILL OPEN

- **IV.G memory per chunk.** Read the *World* line in GPU Direct at 1×1, 2×2, 4×4 and 6×6. From the
  code, expect 6.0 MiB output and 108 KB of voxels per chunk, plus about 0.64 MiB of shared scratch.
  This must be confirmed by measurement before it goes in.
- **Table IV** on the final code (see section 5).
- **Citations.** These have not been checked, especially the 2025 references.
