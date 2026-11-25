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
let gpuInfo = null;

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
  avoidMappedAtCreation(device);

  device.lost.then((info) => {
    console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
  });

  const info = adapter.info || {};
  gpuInfo = {
    vendor: info.vendor || '',
    architecture: info.architecture || '',
    device: info.device || '',
    description: info.description || '',
    timestampQuery: device.features.has('timestamp-query'),
  };
  console.log('GPU device ready:', gpuInfo.description || gpuInfo.vendor || 'unknown adapter');
  return device;
}

/**
 * Works around a renderer-process leak: in Chromium 155 (Chrome and Brave, Linux,
 * RADV) every buffer created with `mappedAtCreation: true` keeps its full size
 * allocated in the page's process after unmap(), destroy() and garbage collection.
 * Three uploads every new vertex/index attribute that way, so each CPU or readback
 * remesh leaked a few MB until the tab hit "Array buffer allocation failed" (a
 * 1x1..4x4 sweep reached 6.5 GB). Buffers filled with queue.writeBuffer do not leak.
 *
 * So such buffers are created unmapped, and getMappedRange()/unmap() are emulated
 * on that buffer: the caller writes into a plain staging ArrayBuffer, and unmap()
 * uploads it with writeBuffer. Same contents and usage for the caller (plus
 * COPY_DST). Buffers that are mappable (MAP_READ/MAP_WRITE) are left alone.
 */
function avoidMappedAtCreation(device) {
  const createBuffer = device.createBuffer.bind(device);
  const mappable = GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE;
  device.createBuffer = (descriptor) => {
    if (!descriptor.mappedAtCreation || descriptor.usage & mappable) return createBuffer(descriptor);
    // Fields are read explicitly: Three passes a reused descriptor object whose
    // properties a spread ({ ...descriptor }) does not copy.
    const { label, size: byteSize, usage } = descriptor;
    const buffer = createBuffer({ label, size: byteSize, usage: usage | GPUBufferUsage.COPY_DST });
    let ranges = []; // [offset, ArrayBuffer] per getMappedRange() call
    buffer.getMappedRange = (offset = 0, size = byteSize - offset) => {
      if (!ranges) throw new Error('getMappedRange() after unmap()');
      const range = new ArrayBuffer(size);
      ranges.push([offset, range]);
      return range;
    };
    buffer.unmap = () => {
      if (!ranges) return;
      for (const [offset, range] of ranges) device.queue.writeBuffer(buffer, offset, range);
      ranges = null;
    };
    return buffer;
  };
}

export function hasTimestampQuery(device) {
  return device.features.has('timestamp-query');
}

/** Adapter details for labelling benchmark exports (null until the device exists). */
export function getGpuInfo() {
  return gpuInfo;
}
