// Mesher equivalence: the JS reference of the GPU bit-run merge (bitGreedyMesh)
// against the CPU greedy mesher, compared as full quad sets (4 corners in emission
// order), not only counts. These are the ten cases of the paper's Section V (RQ1),
// plus the neighbour-aware cases. Run with `npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ChunkData, CHUNK_SIZE } from '../src/voxel/ChunkData.js';
import { ChunkManager } from '../src/voxel/ChunkManager.js';
import { TerrainGenerator } from '../src/voxel/TerrainGenerator.js';
import { cpuGreedyMeshGeneration } from '../src/meshing/cpuGreedyMesh.js';
import { bitGreedyMesh, quadCorners } from '../src/meshing/bitGreedyMerge.js';
import { buildBorderPlanes } from '../src/meshing/neighbourBorders.js';
import { compareQuadKeys, quadKeysFromCorners, quadKeysFromCpuPositions } from '../src/meshing/meshCompare.js';
import { DEFAULT_CAPACITY } from '../src/meshing/gpuMeshPipeline.js';
import { mulberry32 } from '../src/benchmark/editBenchmark.js';

const N = CHUNK_SIZE;

/** Meshes `chunk` both ways and asserts identical quad sets; returns the quad count. */
function assertSameMesh(chunk, neighbours = null) {
  const cpu = cpuGreedyMeshGeneration(chunk, neighbours);
  const cpuKeys = quadKeysFromCpuPositions(cpu.geometry.attributes.position.array);
  cpu.geometry.dispose();
  const borders = neighbours ? buildBorderPlanes(neighbours) : null;
  const bit = bitGreedyMesh(chunk, borders);
  const bitKeys = quadKeysFromCorners(bit.map((q) => quadCorners(q.face, q.slice, q.u, q.v, q.w, q.h)));
  const cmp = compareQuadKeys(cpuKeys, bitKeys);
  assert.equal(cpu.quadCount, cpuKeys.length);
  assert.ok(cmp.equal, `quad sets differ: cpu ${cmp.countA}, bit-run ${cmp.countB}; ` +
    `only CPU ${JSON.stringify(cmp.onlyInA)}, only bit-run ${JSON.stringify(cmp.onlyInB)}`);
  return cpu.quadCount;
}

const fill = (chunk, x0, y0, z0, x1, y1, z1) => {
  for (let z = z0; z < z1; z++) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) chunk.set(x, y, z, true);
  return chunk;
};

const terrainChunk = () => {
  const c = new ChunkData();
  new TerrainGenerator().generate(c);
  return c;
};

test('1. default terrain: 50,112 quads', () => {
  assert.equal(assertSameMesh(terrainChunk()), 50112);
});

test('2. empty chunk: no quads', () => {
  assert.equal(assertSameMesh(new ChunkData()), 0);
});

test('3. full chunk: 6 quads', () => {
  assert.equal(assertSameMesh(fill(new ChunkData(), 0, 0, 0, N, N, N)), 6);
});

test('4. 4^3 cube crossing a word boundary: 6 quads', () => {
  assert.equal(assertSameMesh(fill(new ChunkData(), 30, 30, 30, 34, 34, 34)), 6);
});

test('5. runs ending and starting at bits 31/32 and 63/64', () => {
  const c = new ChunkData();
  fill(c, 0, 10, 10, 32, 11, 11); // run 0..31 ends at a word edge
  fill(c, 32, 12, 12, 64, 13, 13); // run 32..63 is exactly the middle word
  fill(c, 31, 14, 14, 33, 15, 15); // run straddles bits 31/32
  fill(c, 63, 16, 16, 65, 17, 17); // run straddles bits 63/64
  fill(c, 20, 20, 20, 96, 24, 24); // run through all three words to the end of the row
  assertSameMesh(c);
});

test('6. 40^3 checkerboard: 192,000 quads, more than one output buffer holds', () => {
  const c = new ChunkData();
  for (let z = 0; z < 40; z++) for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) {
    if ((x + y + z) % 2 === 0) c.set(x, y, z, true);
  }
  const quads = assertSameMesh(c);
  assert.equal(quads, 192000);
  // The GPU path must detect the overflow and grow, not silently drop quads.
  assert.ok(quads * 6 > DEFAULT_CAPACITY, 'checkerboard should overflow the default GPU buffer');
});

test('7. random chunks, density 0.05 to 0.95', () => {
  for (const [i, density] of [0.05, 0.25, 0.5, 0.75, 0.95].entries()) {
    const rnd = mulberry32(100 + i);
    const c = new ChunkData();
    for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      if (rnd() < density) c.set(x, y, z, true);
    }
    assertSameMesh(c);
  }
});

test('8. terrain carved by ten explosions', () => {
  const c = terrainChunk();
  const rnd = mulberry32(1337);
  for (let i = 0; i < 10; i++) {
    c.carveSphere(Math.floor(rnd() * N), Math.floor(rnd() * N), Math.floor(rnd() * N), 3 + Math.floor(rnd() * 22));
  }
  assertSameMesh(c);
});

test('9. block spanning two chunks: 5 quads per chunk, no face on the shared plane', () => {
  const world = new ChunkManager({ gridSize: 2 });
  for (let z = 10; z < 14; z++) for (let y = 10; y < 14; y++) for (let x = N - 2; x < N + 2; x++) world.set(x, y, z, true);
  const [a, b] = [world.getChunk(0, 0, 0), world.getChunk(1, 0, 0)];
  assert.equal(assertSameMesh(a, world.neighbours(a)), 5);
  assert.equal(assertSameMesh(b, world.neighbours(b)), 5);
});

test('10. 2x2 terrain with neighbours: 204,826 quads (535 fewer than without)', () => {
  const world = new ChunkManager({ gridSize: 2, generator: new TerrainGenerator() });
  let withNeighbours = 0;
  let without = 0;
  for (const chunk of world.chunks) {
    withNeighbours += assertSameMesh(chunk, world.neighbours(chunk));
    without += assertSameMesh(chunk);
  }
  assert.equal(withNeighbours, 204826);
  assert.equal(without, 205361);
});
