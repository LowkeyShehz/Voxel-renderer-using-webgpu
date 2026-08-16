import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three/webgpu';
import { ChunkManager } from './engine/ChunkManager';
import { CHUNK_SIZE } from './engine/Chunk';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { DirectRenderBridge } from './render/DirectRenderBridge';

export default function VoxelGame() {
    const mountRef = useRef(null);
    const [stats, setStats] = useState({ fps: 0, chunks: 0, gpuTime: 0, readbackTime: 0 });
    const [mode, setMode] = useState('cpu');
    const chunkManagerRef = useRef(null);
    const bridgeRef = useRef(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        // Scene Setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x87CEEB); // Sky blue
        // scene.fog = new THREE.Fog(0x87CEEB, CHUNK_SIZE * 2, CHUNK_SIZE * 4);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 30, 0);

        const renderer = new THREE.WebGPURenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        mountRef.current.appendChild(renderer.domElement);

        const bridge = new DirectRenderBridge(renderer);
        bridgeRef.current = bridge;

        // Lighting
        const light = new THREE.DirectionalLight(0xffffff, 1.2);
        light.position.set(50, 100, 50);
        scene.add(light);
        scene.add(new THREE.AmbientLight(0x404040));

        // Controls
        const controls = new PointerLockControls(camera, document.body);
        scene.add(camera);

        // Engine
        const chunkManager = new ChunkManager(scene, bridge, (statsUpdate) => {
            setStats(prev => ({ ...prev, ...statsUpdate }));
        });
        chunkManagerRef.current = chunkManager;

        // Movement
        const velocity = new THREE.Vector3();
        const direction = new THREE.Vector3();
        const moveState = { forward: false, backward: false, left: false, right: false };

        const onKeyDown = (e) => {
            switch (e.code) {
                case 'ArrowUp':
                case 'KeyW': moveState.forward = true; break;
                case 'ArrowLeft':
                case 'KeyA': moveState.left = true; break;
                case 'ArrowDown':
                case 'KeyS': moveState.backward = true; break;
                case 'ArrowRight':
                case 'KeyD': moveState.right = true; break;
            }
        };
        const onKeyUp = (e) => {
            switch (e.code) {
                case 'ArrowUp':
                case 'KeyW': moveState.forward = false; break;
                case 'ArrowLeft':
                case 'KeyA': moveState.left = false; break;
                case 'ArrowDown':
                case 'KeyS': moveState.backward = false; break;
                case 'ArrowRight':
                case 'KeyD': moveState.right = false; break;
            }
        };
        document.addEventListener('keydown', onKeyDown);
        document.addEventListener('keyup', onKeyUp);
        document.addEventListener('click', () => controls.lock());

        // Loop
        let prevTime = performance.now();
        let frameId;
        let frames = 0;
        let lastStatsUpdate = 0;

        const animate = async () => {
            frameId = requestAnimationFrame(animate);
            const time = performance.now();
            const delta = (time - prevTime) / 1000;
            prevTime = time;

            // Move
            if (controls.isLocked) {
                velocity.x -= velocity.x * 10.0 * delta;
                velocity.z -= velocity.z * 10.0 * delta;
                direction.z = Number(moveState.forward) - Number(moveState.backward);
                direction.x = Number(moveState.left) - Number(moveState.right);
                direction.normalize();

                if (moveState.forward || moveState.backward) velocity.z -= direction.z * 400.0 * delta;
                if (moveState.left || moveState.right) velocity.x -= direction.x * 400.0 * delta;

                controls.moveRight(-velocity.x * delta);
                controls.moveForward(-velocity.z * delta);
            }

            // Engine Update
            try {
                chunkManager.update(camera.position);
                await renderer.renderAsync(scene, camera);
            } catch (err) {
                console.error("Loop error:", err);
                setError(err.message);
                return;
            }

            // Stats
            frames++;
            if (time - lastStatsUpdate > 1000) {
                setStats(prev => ({
                    ...prev,
                    fps: frames,
                    chunks: chunkManager.world.chunks.size
                }));
                frames = 0;
                lastStatsUpdate = time;
            }
        };

        // Initialize WebGPURenderer and DirectRenderBridge before animating
        bridge.init().then(() => {
            animate();
        }).catch(err => {
            console.error("Initialization error:", err);
            setError(err.message);
        });

        return () => {
            cancelAnimationFrame(frameId);
            mountRef.current?.removeChild(renderer.domElement);
            document.removeEventListener('keydown', onKeyDown);
            document.removeEventListener('keyup', onKeyUp);
            chunkManager.world.chunks.clear(); // Cleanup
        };
    }, []);

    const toggleMode = (newMode) => {
        setMode(newMode);
        if (chunkManagerRef.current) {
            chunkManagerRef.current.setMode(newMode);
        }
    };

    return (
        <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
            {error ? (
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', background: '#ff4d4f', color: 'white', padding: '20px', borderRadius: '8px', zIndex: 10 }}>
                    <h2>Error Initializing Engine</h2>
                    <p>{error}</p>
                </div>
            ) : (
                <div ref={mountRef} />
            )}

            <div style={{
                position: 'absolute',
                top: 20,
                left: 20,
                background: 'rgba(0,0,0,0.5)',
                color: 'white',
                padding: '15px',
                borderRadius: '5px',
                fontFamily: 'monospace'
            }}>
                <h2>Voxel Game Engine</h2>
                <div>FPS: {stats.fps}</div>
                <div>Chunks: {stats.chunks}</div>
                {mode === 'gpu' && (
                    <>
                        <div style={{ color: '#2ecc71', marginTop: '5px' }}>GPU Meshing: {stats.gpuTime.toFixed(1)}ms</div>
                        <div style={{ color: '#e74c3c' }}>CPU Readback: {stats.readbackTime.toFixed(1)}ms</div>
                    </>
                )}
                <div style={{ marginTop: '10px' }}>
                    <button
                        onClick={() => toggleMode('cpu')}
                        style={{ fontWeight: mode === 'cpu' ? 'bold' : 'normal', marginRight: '10px' }}
                    >
                        CPU Mode
                    </button>
                    <button
                        onClick={() => toggleMode('gpu')}
                        style={{ fontWeight: mode === 'gpu' ? 'bold' : 'normal' }}
                    >
                        GPU Mode
                    </button>
                </div>
                <div style={{ marginTop: '10px', fontSize: '12px' }}>
                    WASD to Move, Click to Capture Mouse
                </div>
            </div>
        </div>
    );
}
