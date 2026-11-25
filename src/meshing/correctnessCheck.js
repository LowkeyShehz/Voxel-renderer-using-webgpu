import { ChunkData } from '../voxel/ChunkData';
import { ChunkManager } from '../voxel/ChunkManager.js';
import { TerrainGenerator } from '../voxel/TerrainGenerator';
import { cpuGreedyMeshGeneration } from './cpuGreedyMesh';
import { MERGE_IMPLS } from './gpuMeshPipeline';
import { compareQuadKeys, quadKeysFromCpuPositions, quadKeysFromGpuVec4 } from './meshCompare.js';

/**
 * Meshes `chunk` on the CPU and with every GPU merge implementation, reads the GPU
 * vertices back, and compares quad count and the exact quad set (corners in
 * emission order). `neighbours` are passed to both meshers.
 */
async function checkChunk(mesher, target, chunk, neighbours, label) {
  const cpu = cpuGreedyMeshGeneration(chunk, neighbours);
  const cpuKeys = quadKeysFromCpuPositions(cpu.geometry.attributes.position.array);
  cpu.geometry.dispose();

  const results = [];
  for (const impl of MERGE_IMPLS) {
    mesher.mesh(chunk, target, { mergeImpl: impl, neighbours });
    const counters = await mesher.readCounters(target);
    const data = await mesher.readPositions(target, counters.vertexCount);
    const cmp = compareQuadKeys(cpuKeys, quadKeysFromGpuVec4(data, counters.vertexCount));
    results.push({
      label,
      impl,
      cpuQuads: cpu.quadCount,
      gpuQuads: counters.quadCount,
      overflowed: counters.overflowed,
      countMatch: counters.quadCount === cpu.quadCount,
      setsEqual: cmp.equal,
      onlyInCpu: cmp.onlyInA,
      onlyInGpu: cmp.onlyInB,
    });
  }
  return results;
}

/**
 * Two checks, on chunks and buffers of its own (the world on screen is untouched):
 *  1. a single default-terrain chunk (no neighbours): expect 50,112 quads;
 *  2. every chunk of a 2x2 default-terrain world, meshed with its neighbours, so
 *     faces on shared chunk borders must be culled identically on CPU and GPU.
 *
 * @param {import('./gpuMeshPipeline').GpuMesher} mesher
 */
export async function runCorrectnessCheck(mesher) {
  const generator = new TerrainGenerator();
  const single = new ChunkData();
  generator.generate(single);
  const world = new ChunkManager({ gridSize: 2, generator });

  const target = mesher.createReadbackTarget();
  const results = [];
  try {
    results.push(...(await checkChunk(mesher, target, single, null, 'single chunk')));
    for (const chunk of world.chunks) {
      results.push(...(await checkChunk(mesher, target, chunk, world.neighbours(chunk), `2x2 chunk ${chunk.id}`)));
    }
  } finally {
    mesher.destroyTarget(target);
    mesher.releaseChunk(single);
    for (const chunk of world.chunks) mesher.releaseChunk(chunk);
  }

  const summarize = (rows) => ({
    cpuQuads: rows.filter((r) => r.impl === MERGE_IMPLS[0]).reduce((a, r) => a + r.cpuQuads, 0),
    impls: MERGE_IMPLS.map((impl) => {
      const mine = rows.filter((r) => r.impl === impl);
      return {
        impl,
        gpuQuads: mine.reduce((a, r) => a + r.gpuQuads, 0),
        pass: mine.every((r) => r.countMatch && r.setsEqual && !r.overflowed),
      };
    }),
  });

  const report = {
    mergeWorkgroupSize: mesher.mergeWorkgroupSize,
    single: summarize(results.filter((r) => r.label === 'single chunk')),
    world2x2: summarize(results.filter((r) => r.label !== 'single chunk')),
    results,
    pass: results.every((r) => r.countMatch && r.setsEqual && !r.overflowed),
  };
  console.log('Correctness check:', JSON.stringify(report, null, 2));
  return report;
}
