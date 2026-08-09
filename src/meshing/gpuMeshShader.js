export const CHUNK_SIZE = 96;

export const gpuMeshShader = `
struct Voxel {
    data: array<u32>
}

struct MaskBuffer {
    data: array<atomic<u32>>
}

struct FloatArray {
    data: array<f32>
}

struct Counter {
    value: atomic<u32>
}

@group(0) @binding(0) var<storage, read> voxels: Voxel;
@group(0) @binding(1) var<storage, read_write> masks: MaskBuffer;
@group(0) @binding(2) var<storage, read_write> positions: FloatArray;
@group(0) @binding(3) var<storage, read_write> normals: FloatArray;
@group(0) @binding(4) var<storage, read_write> counter: Counter;

const CHUNK_SIZE: u32 = ${CHUNK_SIZE}u;
const LIMIT: u32 = 3500000u; // Safety clamp

fn getVoxel(x: i32, y: i32, z: i32) -> u32 {
    if (x < 0 || x >= i32(CHUNK_SIZE) || y < 0 || y >= i32(CHUNK_SIZE) || z < 0 || z >= i32(CHUNK_SIZE)) {
        return 0u;
    }
    let flatIndex = u32(x) + u32(y) * CHUNK_SIZE + u32(z) * CHUNK_SIZE * CHUNK_SIZE;
    let wordIndex = flatIndex >> 5u;
    let bitOffset = flatIndex & 31u;
    if ((voxels.data[wordIndex] & (1u << bitOffset)) != 0u) {
        return 1u;
    }
    return 0u;
}

fn setMaskBit(face: u32, slice: u32, u: u32, v: u32) {
    let bitIndex = u + v * CHUNK_SIZE;
    let wordIdx = bitIndex >> 5u;
    let bitOffset = bitIndex & 31u;
    
    // Global word index: face * 96 * 288 + slice * 288 + wordIdx
    let globalWordIdx = face * CHUNK_SIZE * 288u + slice * 288u + wordIdx;
    atomicOr(&masks.data[globalWordIdx], 1u << bitOffset);
}

// --- PASS A: MASK GENERATION ---
@compute @workgroup_size(4, 4, 4)
fn mask_main(@builtin(global_invocation_id) global_id: vec3u) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let z = i32(global_id.z);
    
    if (x >= i32(CHUNK_SIZE) || y >= i32(CHUNK_SIZE) || z >= i32(CHUNK_SIZE)) {
        return;
    }
    
    if (getVoxel(x, y, z) == 0u) {
        return;
    }
    
    let ux = u32(x); let uy = u32(y); let uz = u32(z);
    
    // Face 0: +X (Right) -> slice x, u=y, v=z
    if (getVoxel(x + 1, y, z) == 0u) { setMaskBit(0u, ux, uy, uz); }
    // Face 1: -X (Left) -> slice x, u=y, v=z
    if (getVoxel(x - 1, y, z) == 0u) { setMaskBit(1u, ux, uy, uz); }
    // Face 2: +Y (Top) -> slice y, u=x, v=z
    if (getVoxel(x, y + 1, z) == 0u) { setMaskBit(2u, uy, ux, uz); }
    // Face 3: -Y (Bottom) -> slice y, u=x, v=z
    if (getVoxel(x, y - 1, z) == 0u) { setMaskBit(3u, uy, ux, uz); }
    // Face 4: +Z (Front) -> slice z, u=x, v=y
    if (getVoxel(x, y, z + 1) == 0u) { setMaskBit(4u, uz, ux, uy); }
    // Face 5: -Z (Back) -> slice z, u=x, v=y
    if (getVoxel(x, y, z - 1) == 0u) { setMaskBit(5u, uz, ux, uy); }
}

// --- PASS B: GREEDY MERGING ---
fn emitQuad(corners: array<vec3f, 4>, normal: vec3f) {
    let baseVertexIdx = atomicAdd(&counter.value, 6u);
    if (baseVertexIdx + 6u >= LIMIT) { return; }
    
    // Convert vertex index to float array index
    let base = baseVertexIdx * 3u;
    
    // Triangle 1
    positions.data[base + 0u] = corners[0].x; positions.data[base + 1u] = corners[0].y; positions.data[base + 2u] = corners[0].z;
    positions.data[base + 3u] = corners[1].x; positions.data[base + 4u] = corners[1].y; positions.data[base + 5u] = corners[1].z;
    positions.data[base + 6u] = corners[2].x; positions.data[base + 7u] = corners[2].y; positions.data[base + 8u] = corners[2].z;
    
    normals.data[base + 0u] = normal.x; normals.data[base + 1u] = normal.y; normals.data[base + 2u] = normal.z;
    normals.data[base + 3u] = normal.x; normals.data[base + 4u] = normal.y; normals.data[base + 5u] = normal.z;
    normals.data[base + 6u] = normal.x; normals.data[base + 7u] = normal.y; normals.data[base + 8u] = normal.z;
    
    // Triangle 2
    let base2 = base + 9u;
    positions.data[base2 + 0u] = corners[0].x; positions.data[base2 + 1u] = corners[0].y; positions.data[base2 + 2u] = corners[0].z;
    positions.data[base2 + 3u] = corners[2].x; positions.data[base2 + 4u] = corners[2].y; positions.data[base2 + 5u] = corners[2].z;
    positions.data[base2 + 6u] = corners[3].x; positions.data[base2 + 7u] = corners[3].y; positions.data[base2 + 8u] = corners[3].z;
    
    normals.data[base2 + 0u] = normal.x; normals.data[base2 + 1u] = normal.y; normals.data[base2 + 2u] = normal.z;
    normals.data[base2 + 3u] = normal.x; normals.data[base2 + 4u] = normal.y; normals.data[base2 + 5u] = normal.z;
    normals.data[base2 + 6u] = normal.x; normals.data[base2 + 7u] = normal.y; normals.data[base2 + 8u] = normal.z;
}

@compute @workgroup_size(6, 96, 1) // 576 threads
fn merge_main(@builtin(global_invocation_id) global_id: vec3u) {
    let face = global_id.x; // 0 to 5
    let slice = global_id.y; // 0 to 95
    
    if (face >= 6u || slice >= CHUNK_SIZE) {
        return;
    }
    
    // Copy the 2D mask for this slice into private memory for fast access & clearing
    var localMask: array<u32, 288>;
    let offset = face * CHUNK_SIZE * 288u + slice * 288u;
    for (var i = 0u; i < 288u; i++) {
        localMask[i] = atomicLoad(&masks.data[offset + i]);
    }
    
    fn getLocalMaskBit(u: u32, v: u32) -> bool {
        let bitIndex = u + v * CHUNK_SIZE;
        let wordIdx = bitIndex >> 5u;
        let bitOffset = bitIndex & 31u;
        return (localMask[wordIdx] & (1u << bitOffset)) != 0u;
    }
    
    fn clearLocalMaskBits(u_start: u32, v_start: u32, w: u32, h: u32) {
        for (var dv = 0u; dv < h; dv++) {
            for (var du = 0u; du < w; du++) {
                let bitIndex = (u_start + du) + (v_start + dv) * CHUNK_SIZE;
                let wordIdx = bitIndex >> 5u;
                let bitOffset = bitIndex & 31u;
                localMask[wordIdx] &= ~(1u << bitOffset);
            }
        }
    }
    
    // Greedy meshing
    for (var v = 0u; v < CHUNK_SIZE; v++) {
        for (var u = 0u; u < CHUNK_SIZE; u++) {
            if (getLocalMaskBit(u, v)) {
                // Find width
                var w = 1u;
                while (u + w < CHUNK_SIZE && getLocalMaskBit(u + w, v)) {
                    w++;
                }
                
                // Find height
                var h = 1u;
                var done = false;
                while (v + h < CHUNK_SIZE) {
                    for (var k = 0u; k < w; k++) {
                        if (!getLocalMaskBit(u + k, v + h)) {
                            done = true;
                            break;
                        }
                    }
                    if (done) { break; }
                    h++;
                }
                
                // Clear the mask
                clearLocalMaskBits(u, v, w, h);
                
                // Emit Quad based on face
                let f_slice = f32(slice);
                let f_u = f32(u); let f_v = f32(v);
                let f_w = f32(w); let f_h = f32(h);
                var corners: array<vec3f, 4>;
                var normal: vec3f;
                
                if (face == 0u) { // +X
                    normal = vec3f(1, 0, 0);
                    corners = array<vec3f, 4>(
                        vec3f(f_slice + 1.0, f_u, f_v),
                        vec3f(f_slice + 1.0, f_u + f_w, f_v),
                        vec3f(f_slice + 1.0, f_u + f_w, f_v + f_h),
                        vec3f(f_slice + 1.0, f_u, f_v + f_h)
                    );
                } else if (face == 1u) { // -X
                    normal = vec3f(-1, 0, 0);
                    corners = array<vec3f, 4>(
                        vec3f(f_slice, f_u, f_v + f_h),
                        vec3f(f_slice, f_u + f_w, f_v + f_h),
                        vec3f(f_slice, f_u + f_w, f_v),
                        vec3f(f_slice, f_u, f_v)
                    );
                } else if (face == 2u) { // +Y
                    normal = vec3f(0, 1, 0);
                    corners = array<vec3f, 4>(
                        vec3f(f_u, f_slice + 1.0, f_v),
                        vec3f(f_u, f_slice + 1.0, f_v + f_h),
                        vec3f(f_u + f_w, f_slice + 1.0, f_v + f_h),
                        vec3f(f_u + f_w, f_slice + 1.0, f_v)
                    );
                } else if (face == 3u) { // -Y
                    normal = vec3f(0, -1, 0);
                    corners = array<vec3f, 4>(
                        vec3f(f_u, f_slice, f_v + f_h),
                        vec3f(f_u, f_slice, f_v),
                        vec3f(f_u + f_w, f_slice, f_v),
                        vec3f(f_u + f_w, f_slice, f_v + f_h)
                    );
                } else if (face == 4u) { // +Z
                    normal = vec3f(0, 0, 1);
                    corners = array<vec3f, 4>(
                        vec3f(f_u + f_w, f_v, f_slice + 1.0),
                        vec3f(f_u + f_w, f_v + f_h, f_slice + 1.0),
                        vec3f(f_u, f_v + f_h, f_slice + 1.0),
                        vec3f(f_u, f_v, f_slice + 1.0)
                    );
                } else if (face == 5u) { // -Z
                    normal = vec3f(0, 0, -1);
                    corners = array<vec3f, 4>(
                        vec3f(f_u, f_v, f_slice),
                        vec3f(f_u, f_v + f_h, f_slice),
                        vec3f(f_u + f_w, f_v + f_h, f_slice),
                        vec3f(f_u + f_w, f_v, f_slice)
                    );
                }
                
                emitQuad(corners, normal);
            }
        }
    }
}
`;
