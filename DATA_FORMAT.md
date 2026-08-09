# Voxel Data Format Specification

## Overview
To optimize memory bandwidth and enable efficient GPU compute parallelism (especially for greedy meshing), we use a bit-packed occupancy format for voxel chunks.

## Storage Format
- **Type**: `Uint32Array`
- **Packing**: 32 binary voxels (solid or air) are packed into a single 32-bit unsigned integer (`u32`).
- **Memory Footprint**: A 96³ chunk requires 884,736 voxels. At 1 bit per voxel, this requires 27,648 `u32` words, consuming exactly **110,592 bytes (~110 KB)** instead of ~3.5 MB.

## Coordinate Mapping
Given a `CHUNK_SIZE` and a 3D coordinate `(x, y, z)` within the chunk:

1. **Flat Index Calculation**:
   ```javascript
   const flatIndex = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
   ```

2. **Word Index and Bit Offset**:
   ```javascript
   const wordIndex = flatIndex >>> 5; // Math.floor(flatIndex / 32)
   const bitOffset = flatIndex & 31;  // flatIndex % 32
   ```

3. **Bitwise Operations (Read/Write)**:
   - **Set Solid (1)**: `data[wordIndex] |= (1 << bitOffset)`
   - **Set Air (0)**: `data[wordIndex] &= ~(1 << bitOffset)`
   - **Read Occupancy**: `(data[wordIndex] & (1 << bitOffset)) !== 0`

## WGSL (WebGPU) Equivalents
In the compute shader, the storage buffer is defined as:
```wgsl
struct Voxel {
    data: array<u32>
}
```

To read occupancy:
```wgsl
let flatIndex = u32(x) + u32(y) * CHUNK_SIZE + u32(z) * CHUNK_SIZE * CHUNK_SIZE;
let wordIndex = flatIndex >> 5u;
let bitOffset = flatIndex & 31u;
let isSolid = (voxels.data[wordIndex] & (1u << bitOffset)) != 0u;
```

This format ensures that we send the absolute minimum amount of data over the PCIe bus and aligns perfectly with bitwise greedy meshing strategies in future phases.
