# Voxel Data Format Specification

Formats shared by the CPU code (`src/voxel`, `src/meshing/*.js`) and the WGSL in
`src/meshing/gpuMeshShader.js`. `CHUNK_SIZE` is 96; several layouts below rely on it
(a mask row is exactly 3 words).

## 1. Chunk occupancy (`ChunkData`)

- **Type**: `Uint32Array`, 1 bit per voxel (0 = air, 1 = solid), 32 voxels per `u32`.
- **Size**: 96³ = 884,736 voxels = 27,648 words = **110,592 bytes (108 KB)** per chunk.
- **Index** for local `(x, y, z)`, each in `0..95`:
  ```javascript
  const flatIndex = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
  const wordIndex = flatIndex >>> 5;  // flatIndex / 32
  const bitOffset = flatIndex & 31;   // flatIndex % 32
  // set solid: data[wordIndex] |= 1 << bitOffset
  // set air:   data[wordIndex] &= ~(1 << bitOffset)
  // read:      (data[wordIndex] >>> bitOffset) & 1
  ```
- **Dirty range**: every change widens a `[start, end)` word range; the GPU mirror uploads
  only that range (`consumeDirtyRange`, single consumer: `GpuMesher`).

WGSL reads the same buffer as `array<u32>`:
```wgsl
let flatIndex = u32(x) + u32(y) * CHUNK_SIZE + u32(z) * CHUNK_SIZE * CHUNK_SIZE;
let isSolid = (voxels.data[flatIndex >> 5u] & (1u << (flatIndex & 31u))) != 0u;
```

## 2. World and chunk coordinates (`ChunkManager`)

An N×N grid (N = 1..6) of chunks at y = 0. Chunk `(cx, 0, cz)` covers world voxels
`[96·cx, 96·cx + 96) × [0, 96) × [96·cz, 96·cz + 96)`; its mesh is drawn at that world
offset (vertex positions are chunk-local). Picking and editing use world voxel
coordinates.

## 3. Face ids

Used for mask slices, neighbour planes, border lookups and the vertex `w` channel:

| id | face | slice axis | mask `u` | mask `v` |
|---:|------|-----------|----------|----------|
| 0 | +X | x | y | z |
| 1 | −X | x | y | z |
| 2 | +Y | y | x | z |
| 3 | −Y | y | x | z |
| 4 | +Z | z | x | y |
| 5 | −Z | z | x | y |

## 4. Slice masks (GPU scratch, `mask_main` → `merge_*`)

For each face and each of the 96 slices, a 96×96 bit mask of cells whose face is
exposed (solid voxel, air neighbour in that direction):

```
word = face * 96 * 288 + slice * 288 + (bit >> 5)      // 288 words per slice
bit  = u + v * 96
```

6 × 96 × 288 words = 663,552 bytes, shared by all chunks (meshed one at a time).
Row `v` of a slice is the 3 whole words `slice*288 + 3v .. +2`, which is what the bit-run
merge scans.

## 5. Neighbour border planes (`neighbourBorders.js`, binding 5)

The layer of each neighbouring chunk that touches the chunk being meshed, so a face
on the chunk border is emitted only if the neighbouring voxel is air:

```
word = face * 288 + (bit >> 5),  bit = u + v * 96        // (u, v) as in the table above
```

Plane `face` holds the neighbour in that direction: for +X the neighbour's x = 0
column, for −X its x = 95 column, for +Y/−Y its y = 0 / y = 95 rows, for +Z/−Z its
z = 0 / z = 95 layer. A missing neighbour is all zeros (air), which reproduces the
single-chunk behaviour. 6 × 288 words = 6,912 bytes, uploaded before each mesh of a chunk
that has neighbours.

## 6. Mesh output

**Positions** (`array<vec4f>`, 16 bytes per vertex): 6 vertices per quad as two
triangles `(c0 c1 c2) (c0 c2 c3)`, chunk-local `xyz`, `w` = face id. No normals: the
material uses flat shading. Corner order per face is the same as `cpuGreedyMesh.js`
(see `quadCorners` in `bitGreedyMerge.js`).

**Capacity**: per mesh target, in vertices, a multiple of 6. Default 393,216 (6 MB;
the default terrain needs 300,672). If a mesh overflows, the target is replaced by
one of `growCapacity(requested)` (requested × 1.25, in steps of 98,304 vertices, max
1,572,864 = 24 MB) and the chunk is meshed again.

**Indirect / counter buffer** (8 × `u32`):

| word | name | written by |
|-----:|------|-----------|
| 0 | `vertexCount` | atomic counter in merge; clamped to `vertexCap` by finalize |
| 1 | `instanceCount` = 1 | CPU |
| 2 | `firstVertex` = 0 | CPU |
| 3 | `firstInstance` = 0 | CPU |
| 4 | `requestedVertices` | finalize: total before clamping (> word 0 means overflow) |
| 5 | `vertexCap` | CPU: capacity of this target's position buffer |
| 6–7 | padding | |

Words 0–3 are exactly the `drawIndirect` arguments, so the same buffer drives the draw
in direct mode. The CPU reads this buffer (32 bytes) only for statistics and overflow
detection, after the timed region.
