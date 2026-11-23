import { ChunkData } from '../voxel/ChunkData';
import { TerrainGenerator } from '../voxel/TerrainGenerator';
import { cpuGreedyMeshGeneration } from './cpuGreedyMesh';
import { MERGE_IMPLS } from './gpuMeshPipeline';
import { compareQuadKeys, quadKeysFromCpuPositions, quadKeysFromGpuVec4 } from './meshCompare.js';

/**
 * Meshes a fresh default-terrain chunk on the CPU and with every GPU merge
 * implementation, reads the GPU vertices back, and compares quad count and the
 * exact quad set (corners in emission order) against the CPU result.
 * Uses its own chunk and buffers, so the chunk on screen is not touched.
 *
 * @param {import('./gpuMeshPipeline').GpuMesher} mesher
 */
export async function runCorrectnessCheck(mesher) {
  const chunk = new ChunkData();
  new TerrainGenerator().generate(chunk);

  const cpu = cpuGreedyMeshGeneration(chunk);
  const cpuKeys = quadKeysFromCpuPositions(cpu.geometry.attributes.position.array);
  cpu.geometry.dispose();

  const target = mesher.createReadbackTarget();
  const results = [];
  try {
    for (const impl of MERGE_IMPLS) {
      mesher.mesh(chunk, target, { mergeImpl: impl });
      const counters = await mesher.readCounters(target);
      const data = await mesher.readPositions(target, counters.vertexCount);
      const cmp = compareQuadKeys(cpuKeys, quadKeysFromGpuVec4(data, counters.vertexCount));
      results.push({
        impl,
        gpuQuads: counters.quadCount,
        overflowed: counters.overflowed,
        countMatch: counters.quadCount === cpu.quadCount,
        setsEqual: cmp.equal,
        onlyInCpu: cmp.onlyInA,
        onlyInGpu: cmp.onlyInB,
      });
    }
  } finally {
    target.positions.destroy();
    target.indirect.destroy();
    mesher.releaseChunk(chunk);
  }

  const report = {
    cpuQuads: cpu.quadCount,
    mergeWorkgroupSize: mesher.mergeWorkgroupSize,
    results,
    pass: results.every((r) => r.countMatch && r.setsEqual && !r.overflowed),
  };
  console.log('Correctness check:', JSON.stringify(report, null, 2));
  return report;
}
