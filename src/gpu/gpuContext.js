/**
 * One GPUDevice shared by the compute mesher and Three's WebGPURenderer.
 *
 * Buffers cannot cross devices, so zero-copy rendering requires that the compute
 * output and the renderer live on the same device. We create it ourselves and hand
 * it to `new THREE.WebGPURenderer({ device })`.
 *
 * Three's own device request asks for a `featureLevel: 'compatibility'` adapter,
 * which some browsers/drivers do not provide; that makes it fall back to WebGL2
 * even though plain WebGPU works fine. Creating the device here sidesteps that.
 */
let devicePromise = null;

export function getGpuDevice() {
  if (!devicePromise) {
    devicePromise = createDevice().catch((err) => {
      devicePromise = null; // allow a retry after a failure
      throw err;
    });
  }
  return devicePromise;
}

async function createDevice() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported on this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No WebGPU adapter available.');
  }

  // Request every feature the adapter offers (this is what Three does when it
  // creates the device itself), so the renderer behaves identically.
  const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features] });

  device.lost.then((info) => {
    console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
  });

  console.log('GPU device ready:', adapter.info?.description || adapter.info?.vendor || 'unknown adapter');
  return device;
}

export function hasTimestampQuery(device) {
  return device.features.has('timestamp-query');
}
