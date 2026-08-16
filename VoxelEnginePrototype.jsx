import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { ChunkData, CHUNK_SIZE } from './src/voxel/ChunkData';
import { TerrainGenerator } from './src/voxel/TerrainGenerator';
import { cpuGreedyMeshGeneration } from './src/meshing/cpuGreedyMesh';
import { gpuGreedyMeshGeneration } from './src/meshing/gpuMeshPipeline';
import { DirectRenderBridge } from './src/render/DirectRenderBridge';

/**
 * VOXEL ENGINE PROTOTYPE - WebGPU Compute Shader Meshing
 * Validates: GPU compute can generate voxel geometry faster than CPU
 */

// === REACT COMPONENT ===
export default function VoxelEnginePrototype() {
  const mountRef = useRef(null);
  const [mode, setMode] = useState('cpu');
  const [stats, setStats] = useState({ time: 0, quads: 0, naiveFaces: 0, readbackTime: 0 });
  const [error, setError] = useState(null);
  const [webgpuAvailable, setWebgpuAvailable] = useState(true);
  const sceneRef = useRef(null);
  const bridgeRef = useRef(null);

  useEffect(() => {
    setWebgpuAvailable(!!navigator.gpu);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(CHUNK_SIZE * 1.5, CHUNK_SIZE * 1.5, CHUNK_SIZE * 1.5);
    camera.lookAt(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2);

    // Initialize WebGPURenderer
    const renderer = new THREE.WebGPURenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(50, 50, 50);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x404040));

    const gridHelper = new THREE.GridHelper(CHUNK_SIZE, CHUNK_SIZE);
    gridHelper.position.set(CHUNK_SIZE / 2, 0, CHUNK_SIZE / 2);
    scene.add(gridHelper);

    sceneRef.current = { scene, camera, renderer, mesh: null };
    bridgeRef.current = new DirectRenderBridge(renderer);

    let frameId;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const radius = CHUNK_SIZE * 1.5;
      camera.position.x = Math.cos(Date.now() * 0.0001) * radius + CHUNK_SIZE / 2;
      camera.position.z = Math.sin(Date.now() * 0.0001) * radius + CHUNK_SIZE / 2;
      camera.lookAt(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2);
      renderer.renderAsync(scene, camera); // WebGPURenderer requires renderAsync
    };
    
    // WebGPURenderer needs initialization
    renderer.init().then(() => bridgeRef.current.init()).then(() => {
        animate();
    });

    return () => {
      cancelAnimationFrame(frameId);
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, []);

  const generateMesh = async () => {
    setError(null);
    const chunk = new ChunkData();
    const generator = new TerrainGenerator();
    generator.generate(chunk);

    try {
      let result;
      if (mode === 'cpu') {
        result = cpuGreedyMeshGeneration(chunk);
        result.readbackTime = 0; // CPU path has no readback
      } else {
        result = await gpuGreedyMeshGeneration(chunk);
      }

      setStats({ 
        time: result.time.toFixed(2), 
        quads: result.quadCount,
        naiveFaces: result.naiveFaceCount || 'N/A (Calculated by CPU)',
        readbackTime: result.readbackTime !== undefined ? result.readbackTime.toFixed(2) : 0,
        gpuComputeTime: result.gpuComputeTime || null
      });

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
        position: 'absolute', top: 20, left: 20,
        background: 'rgba(0,0,0,0.8)', color: 'white', padding: '20px',
        borderRadius: '8px', fontFamily: 'monospace',
      }}>
        <h2>🎮 Voxel Engine Prototype</h2>
        <div style={{ marginBottom: '10px' }}>
          <label>
            <input type="radio" value="cpu" checked={mode === 'cpu'} onChange={(e) => setMode(e.target.value)} /> CPU Greedy Meshing
          </label>
          <label style={{ marginLeft: '10px' }}>
            <input type="radio" value="gpu" checked={mode === 'gpu'} onChange={(e) => setMode(e.target.value)} disabled={!webgpuAvailable} />
            <span style={{ opacity: webgpuAvailable ? 1 : 0.5 }}> GPU Greedy Meshing </span>
          </label>
        </div>
        <button onClick={generateMesh} style={{
          padding: '10px 20px', background: mode === 'cpu' ? '#3498db' : '#e74c3c',
          color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer'
        }}>
          Generate Mesh ({mode.toUpperCase()})
        </button>
        {error && <div style={{ color: '#ff6b6b', marginTop: '10px' }}>❌ {error}</div>}
        {stats.time > 0 && (
          <div style={{ marginTop: '15px' }}>
            <div>⏱️ Total Time: <strong>{stats.time}ms</strong></div>
            {stats.gpuComputeTime && <div>⚡ GPU Compute Time: <strong>{stats.gpuComputeTime}ms</strong></div>}
            <div>📤 Readback Time: <strong>{stats.readbackTime}ms</strong></div>
            <div>📉 Naive Faces (No Merge): <strong>{typeof stats.naiveFaces === 'number' ? stats.naiveFaces.toLocaleString() : stats.naiveFaces}</strong></div>
            <div>🚀 Greedy Quads (Merged): <strong style={{ color: '#2ecc71' }}>{typeof stats.quads === 'number' ? stats.quads.toLocaleString() : stats.quads}</strong></div>
            <div style={{ fontSize: '11px', color: '#95a5a6', marginTop: '5px' }}>
               Notice the massive geometry reduction!
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
