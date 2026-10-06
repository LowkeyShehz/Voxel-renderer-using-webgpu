import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

/**
 * VOXEL ENGINE PROTOTYPE - WebGPU Compute Shader Meshing
 * Validates: GPU compute can generate voxel geometry faster than CPU
 */

const CHUNK_SIZE = 96;
const VOXEL_COUNT = CHUNK_SIZE ** 3;

// === VOXEL DATA GENERATOR ===
function generateVoxelData() {
  const data = new Uint8Array(VOXEL_COUNT);
  for (let i = 0; i < VOXEL_COUNT; i++) {
    const x = i % CHUNK_SIZE;
    const y = Math.floor(i / CHUNK_SIZE) % CHUNK_SIZE;
    const z = Math.floor(i / (CHUNK_SIZE * CHUNK_SIZE));

    // Create a complex noise pattern (High Density)
    // This creates "caves" and "mountains" - extremely high surface area
    const scale = 0.15;
    const noise = Math.sin(x * scale) + Math.sin(y * scale) + Math.sin(z * scale);

    if (noise > 0) {
      data[i] = 1; // Solid voxel
    }
  }
  return data;
}

// === CPU MESHING (BASELINE) ===
function cpuMeshGeneration(voxelData) {
  const start = performance.now();

  const vertices = [];
  const indices = [];

  const getVoxel = (x, y, z) => {
    if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) {
      return 0; // Air outside bounds
    }
    return voxelData[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE];
  };

  // Face definitions: [normal, offset vectors for 4 corners]
  const faces = [
    { dir: [1, 0, 0], corners: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]] },  // Right
    { dir: [-1, 0, 0], corners: [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]] }, // Left
    { dir: [0, 1, 0], corners: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]] },  // Top
    { dir: [0, -1, 0], corners: [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]] }, // Bottom
    { dir: [0, 0, 1], corners: [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]] },  // Front
    { dir: [0, 0, -1], corners: [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]] }, // Back
  ];

  for (let z = 0; z < CHUNK_SIZE; z++) {
    for (let y = 0; y < CHUNK_SIZE; y++) {
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const voxel = getVoxel(x, y, z);
        if (!voxel) continue;

        for (const face of faces) {
          const [nx, ny, nz] = face.dir;
          const neighbor = getVoxel(x + nx, y + ny, z + nz);

          if (!neighbor) {
            // Emit face
            const baseIdx = vertices.length / 6;

            for (const [ox, oy, oz] of face.corners) {
              vertices.push(x + ox, y + oy, z + oz, nx, ny, nz);
            }

            indices.push(
              baseIdx, baseIdx + 1, baseIdx + 2,
              baseIdx, baseIdx + 2, baseIdx + 3
            );
          }
        }
      }
    }
  }

  const elapsed = performance.now() - start;

  const geometry = new THREE.BufferGeometry();
  const positionsAndNormals = new Float32Array(vertices);
  const positions = new Float32Array(vertices.length / 2);
  const normals = new Float32Array(vertices.length / 2);

  for (let i = 0; i < vertices.length / 6; i++) {
    positions[i * 3] = vertices[i * 6];
    positions[i * 3 + 1] = vertices[i * 6 + 1];
    positions[i * 3 + 2] = vertices[i * 6 + 2];
    normals[i * 3] = vertices[i * 6 + 3];
    normals[i * 3 + 1] = vertices[i * 6 + 4];
    normals[i * 3 + 2] = vertices[i * 6 + 5];
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(indices);

  return { geometry, time: elapsed, faceCount: indices.length / 3 };
}

// === WEBGPU COMPUTE MESHING ===
let cachedDevice = null;
let cachedPipeline = null;
let cachedStagingBuffer = null;

async function gpuMeshGeneration(voxelData) {
  const start = performance.now();

  if (!navigator.gpu) {
    throw new Error('WebGPU not available. Enable it in chrome://flags/#enable-unsafe-webgpu');
  }

  // Use cached device to avoid initialization overhead
  let device = cachedDevice;
  if (!device) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('WebGPU adapter not found. Your GPU may not support WebGPU.');
    }
    device = await adapter.requestDevice();
    cachedDevice = device;
    console.log('WebGPU device initialized');
  }

  // Create input buffer (voxel data)
  // Convert Uint8Array to Uint32Array because WGSL 'array<u32>' expects 4 bytes per element
  const voxelDataU32 = new Uint32Array(voxelData);
  const voxelBuffer = device.createBuffer({
    size: voxelDataU32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(voxelBuffer, 0, voxelDataU32);

  // Output buffer (vertex data: position + normal = 8 floats per vertex due to alignment)
  // WGSL struct Vertex { position: vec3f, normal: vec3f } aligns to 32 bytes (8 floats)
  // Clamp to 3.5M vertices (~112MB) to fit within default 128MB storage buffer limit
  const maxVertices = Math.min(VOXEL_COUNT * 6 * 6, 3500000);
  const floatsPerVertex = 8; // vec3(3) + pad(1) + vec3(3) + pad(1)
  const vertexBuffer = device.createBuffer({
    size: maxVertices * floatsPerVertex * 4, // 8 floats * 4 bytes
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Counter buffer (atomic counter for vertex writes)
  const counterBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  // Initialize counter to 0
  device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]));

  // Get or create compute pipeline
  let pipeline = cachedPipeline;
  if (!pipeline) {
    // Compute shader
    const shaderCode = `
      struct Voxel {
      data: array<u32>
    }
    
    struct Vertex {
      position: vec3f,
      normal: vec3f,
    }
    
    struct Vertices {
      data: array<Vertex>
    }
    
    struct Counter {
      value: atomic<u32>
    }
    
    @group(0) @binding(0) var<storage, read> voxels: Voxel;
    @group(0) @binding(1) var<storage, read_write> vertices: Vertices;
    @group(0) @binding(2) var<storage, read_write> counter: Counter;
    
    const CHUNK_SIZE: u32 = ${CHUNK_SIZE}u;
    const LIMIT: u32 = 3500000u;
    
    fn getVoxel(x: i32, y: i32, z: i32) -> u32 {
      if (x < 0 || x >= i32(CHUNK_SIZE) || y < 0 || y >= i32(CHUNK_SIZE) || z < 0 || z >= i32(CHUNK_SIZE)) {
        return 0u;
      }
      let idx = u32(x) + u32(y) * CHUNK_SIZE + u32(z) * CHUNK_SIZE * CHUNK_SIZE;
      return voxels.data[idx];
    }
    
    fn emitFace(x: f32, y: f32, z: f32, normal: vec3f, corners: array<vec3f, 4>) {
      let baseIdx = atomicAdd(&counter.value, 6u);
      
      if (baseIdx + 6u >= LIMIT) {
        return;
      }
      
      // Emit 2 triangles (6 vertices) for the quad
      // Triangle 1: 0, 1, 2
      vertices.data[baseIdx + 0u].position = vec3f(x, y, z) + corners[0];
      vertices.data[baseIdx + 0u].normal = normal;
      vertices.data[baseIdx + 1u].position = vec3f(x, y, z) + corners[1];
      vertices.data[baseIdx + 1u].normal = normal;
      vertices.data[baseIdx + 2u].position = vec3f(x, y, z) + corners[2];
      vertices.data[baseIdx + 2u].normal = normal;
      
      // Triangle 2: 0, 2, 3
      vertices.data[baseIdx + 3u].position = vec3f(x, y, z) + corners[0];
      vertices.data[baseIdx + 3u].normal = normal;
      vertices.data[baseIdx + 4u].position = vec3f(x, y, z) + corners[2];
      vertices.data[baseIdx + 4u].normal = normal;
      vertices.data[baseIdx + 5u].position = vec3f(x, y, z) + corners[3];
      vertices.data[baseIdx + 5u].normal = normal;
    }
    
    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
      let x = i32(global_id.x);
      let y = i32(global_id.y);
      let z = i32(global_id.z);
      
      if (x >= i32(CHUNK_SIZE) || y >= i32(CHUNK_SIZE) || z >= i32(CHUNK_SIZE)) {
        return;
      }
      
      let voxel = getVoxel(x, y, z);
      if (voxel == 0u) {
        return;
      }
      
      let fx = f32(x);
      let fy = f32(y);
      let fz = f32(z);
      
      // Check each face
      if (getVoxel(x + 1, y, z) == 0u) { // Right
        emitFace(fx, fy, fz, vec3f(1, 0, 0), array<vec3f, 4>(
          vec3f(1, 0, 0), vec3f(1, 1, 0), vec3f(1, 1, 1), vec3f(1, 0, 1)
        ));
      }
      if (getVoxel(x - 1, y, z) == 0u) { // Left
        emitFace(fx, fy, fz, vec3f(-1, 0, 0), array<vec3f, 4>(
          vec3f(0, 0, 1), vec3f(0, 1, 1), vec3f(0, 1, 0), vec3f(0, 0, 0)
        ));
      }
      if (getVoxel(x, y + 1, z) == 0u) { // Top
        emitFace(fx, fy, fz, vec3f(0, 1, 0), array<vec3f, 4>(
          vec3f(0, 1, 0), vec3f(0, 1, 1), vec3f(1, 1, 1), vec3f(1, 1, 0)
        ));
      }
      if (getVoxel(x, y - 1, z) == 0u) { // Bottom
        emitFace(fx, fy, fz, vec3f(0, -1, 0), array<vec3f, 4>(
          vec3f(0, 0, 1), vec3f(0, 0, 0), vec3f(1, 0, 0), vec3f(1, 0, 1)
        ));
      }
      if (getVoxel(x, y, z + 1) == 0u) { // Front
        emitFace(fx, fy, fz, vec3f(0, 0, 1), array<vec3f, 4>(
          vec3f(1, 0, 1), vec3f(1, 1, 1), vec3f(0, 1, 1), vec3f(0, 0, 1)
        ));
      }
      if (getVoxel(x, y, z - 1) == 0u) { // Back
        emitFace(fx, fy, fz, vec3f(0, 0, -1), array<vec3f, 4>(
          vec3f(0, 0, 0), vec3f(0, 1, 0), vec3f(1, 1, 0), vec3f(1, 0, 0)
        ));
      }
    }
  `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
    cachedPipeline = pipeline;
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: voxelBuffer } },
      { binding: 1, resource: { buffer: vertexBuffer } },
      { binding: 2, resource: { buffer: counterBuffer } },
    ],
  });

  // Execute compute shader
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(
    Math.ceil(CHUNK_SIZE / 4),
    Math.ceil(CHUNK_SIZE / 4),
    Math.ceil(CHUNK_SIZE / 4)
  );
  passEncoder.end();

  // Read back counter
  // Read back counter
  const counterReadBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);

  // Submit first pass (Compute + Counter Read)
  device.queue.submit([commandEncoder.finish()]);

  // Map and read counter
  await counterReadBuffer.mapAsync(GPUMapMode.READ);
  const counterData = new Uint32Array(counterReadBuffer.getMappedRange());
  const vertexCount = counterData[0];
  counterReadBuffer.unmap();

  console.log(`GPU generated ${vertexCount} vertices (${vertexCount / 6} faces)`);

  if (vertexCount === 0) {
    throw new Error('GPU compute shader produced 0 vertices. Check voxel data.');
  }

  // Read back vertices (2nd pass - only read what we need)
  // We need to read back the full aligned size (8 floats per vertex)
  const byteSize = vertexCount * floatsPerVertex * 4;
  const maxByteSize = maxVertices * floatsPerVertex * 4;

  // Ensure staging buffer is large enough for WORST case (just in case) or at least current size
  if (!cachedStagingBuffer || cachedStagingBuffer.size < byteSize) {
    if (cachedStagingBuffer) cachedStagingBuffer.destroy();
    // Allocate max size once to avoid re-allocating if count fluctuates
    cachedStagingBuffer = device.createBuffer({
      size: maxByteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  // Important: Buffer must be unmapped to be used in copyBufferToBuffer
  if (cachedStagingBuffer.mapState === 'mapped') {
    cachedStagingBuffer.unmap();
  }

  const gpuReadEncoder = device.createCommandEncoder();
  gpuReadEncoder.copyBufferToBuffer(vertexBuffer, 0, cachedStagingBuffer, 0, byteSize);
  device.queue.submit([gpuReadEncoder.finish()]);

  await cachedStagingBuffer.mapAsync(GPUMapMode.READ, 0, byteSize);
  const vertexData = new Float32Array(cachedStagingBuffer.getMappedRange(0, byteSize));
  const vertexDataCopy = new Float32Array(vertexData);
  cachedStagingBuffer.unmap();

  const elapsed = performance.now() - start;

  // Build Three.js geometry
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);

  // Pack data, skipping padding
  for (let i = 0; i < vertexCount; i++) {
    const base = i * 8; // 8 floats per vertex in GPU buffer due to alignment

    // Position (floats 0, 1, 2)
    positions[i * 3] = vertexDataCopy[base];
    positions[i * 3 + 1] = vertexDataCopy[base + 1];
    positions[i * 3 + 2] = vertexDataCopy[base + 2];

    // Normal (floats 4, 5, 6 - skip index 3 which is padding)
    normals[i * 3] = vertexDataCopy[base + 4];
    normals[i * 3 + 1] = vertexDataCopy[base + 5];
    normals[i * 3 + 2] = vertexDataCopy[base + 6];
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  // No indices needed - already triangles

  return { geometry, time: elapsed, faceCount: vertexCount / 3 };
}

// === REACT COMPONENT ===
export default function VoxelEnginePrototype() {
  const mountRef = useRef(null);
  const [mode, setMode] = useState('cpu');
  const [stats, setStats] = useState({ time: 0, faces: 0 });
  const [error, setError] = useState(null);
  const [webgpuAvailable, setWebgpuAvailable] = useState(true);
  const sceneRef = useRef(null);

  useEffect(() => {
    // Check WebGPU availability
    setWebgpuAvailable(!!navigator.gpu);

    // Setup Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(CHUNK_SIZE * 1.5, CHUNK_SIZE * 1.5, CHUNK_SIZE * 1.5);
    camera.lookAt(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(50, 50, 50);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x404040));

    // Grid helper
    const gridHelper = new THREE.GridHelper(CHUNK_SIZE, CHUNK_SIZE);
    gridHelper.position.set(CHUNK_SIZE / 2, 0, CHUNK_SIZE / 2);
    scene.add(gridHelper);

    sceneRef.current = { scene, camera, renderer };

    // Animation loop
    let frameId;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const radius = CHUNK_SIZE * 1.5;
      camera.position.x = Math.cos(Date.now() * 0.0001) * radius + CHUNK_SIZE / 2;
      camera.position.z = Math.sin(Date.now() * 0.0001) * radius + CHUNK_SIZE / 2;
      camera.lookAt(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, []);

  const generateMesh = async () => {
    setError(null);
    const voxelData = generateVoxelData();

    try {
      let result;
      if (mode === 'cpu') {
        result = cpuMeshGeneration(voxelData);
      } else {
        result = await gpuMeshGeneration(voxelData);
      }

      setStats({ time: result.time.toFixed(2), faces: result.faceCount });

      // Clear old mesh and add new one
      const { scene } = sceneRef.current;
      const oldMesh = scene.getObjectByName('voxelMesh');
      if (oldMesh) scene.remove(oldMesh);

      const material = new THREE.MeshStandardMaterial({
        color: mode === 'cpu' ? 0x3498db : 0xe74c3c,
        flatShading: true,
      });

      const mesh = new THREE.Mesh(result.geometry, material);
      mesh.name = 'voxelMesh';
      scene.add(mesh);

    } catch (err) {
      setError(err.message);
      console.error(err);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <div ref={mountRef} />

      <div style={{
        position: 'absolute',
        top: 20,
        left: 20,
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '20px',
        borderRadius: '8px',
        fontFamily: 'monospace',
      }}>
        <h2>🎮 Voxel Engine Prototype</h2>

        {!webgpuAvailable && (
          <div style={{
            background: 'rgba(231, 76, 60, 0.2)',
            border: '1px solid #e74c3c',
            borderRadius: '4px',
            padding: '10px',
            marginBottom: '15px',
            fontSize: '12px',
            lineHeight: '1.6'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
              ⚠️ WebGPU Not Available
            </div>
            <div style={{ fontSize: '11px', color: '#ddd' }}>
              GPU Compute mode won't work. Enable WebGPU:
              <br />• <code>chrome://flags/#enable-unsafe-webgpu</code>
              <br />• Set to "Enabled" → Relaunch
            </div>
          </div>
        )}

        <div style={{ marginBottom: '10px' }}>
          <label>
            <input
              type="radio"
              value="cpu"
              checked={mode === 'cpu'}
              onChange={(e) => setMode(e.target.value)}
            />
            CPU Meshing
          </label>
          <label style={{ marginLeft: '10px' }}>
            <input
              type="radio"
              value="gpu"
              checked={mode === 'gpu'}
              onChange={(e) => setMode(e.target.value)}
              disabled={!webgpuAvailable}
            />
            <span style={{ opacity: webgpuAvailable ? 1 : 0.5 }}>
              GPU Compute {!webgpuAvailable && '(Unavailable)'}
            </span>
          </label>
        </div>

        <button
          onClick={generateMesh}
          style={{
            padding: '10px 20px',
            background: mode === 'cpu' ? '#3498db' : '#e74c3c',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
          }}
        >
          Generate Mesh ({mode.toUpperCase()})
        </button>

        {stats.time > 0 && (
          <div style={{ marginTop: '15px', fontSize: '14px' }}>
            <div>⏱️ Time: <strong>{stats.time}ms</strong></div>
            <div>📐 Faces: <strong>{stats.faces.toLocaleString()}</strong></div>
          </div>
        )}

        {error && (
          <div style={{
            marginTop: '15px',
            padding: '10px',
            background: 'rgba(231, 76, 60, 0.2)',
            border: '1px solid #e74c3c',
            borderRadius: '4px',
            fontSize: '12px',
            lineHeight: '1.6'
          }}>
            <div style={{ color: '#e74c3c', fontWeight: 'bold', marginBottom: '5px' }}>
              ❌ WebGPU Error
            </div>
            <div style={{ color: '#fff' }}>
              {error}
            </div>
            {error.includes('not available') && (
              <div style={{ marginTop: '8px', color: '#aaa', fontSize: '11px' }}>
                <strong>Fix:</strong> Enable WebGPU in your browser:
                <br />1. Go to <code>chrome://flags/#enable-unsafe-webgpu</code>
                <br />2. Set to "Enabled"
                <br />3. Click "Relaunch"
                <br />4. Refresh this page
              </div>
            )}
          </div>
        )}

        <div style={{ marginTop: '15px', fontSize: '12px', opacity: 0.7 }}>
          {CHUNK_SIZE}×{CHUNK_SIZE}×{CHUNK_SIZE} voxel chunk (Complex Terrain, 884k voxels)
        </div>
      </div>
    </div>
  );
}
