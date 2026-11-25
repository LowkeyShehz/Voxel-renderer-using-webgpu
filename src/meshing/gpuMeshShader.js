/**
 * WGSL source for the two-pass greedy mesher plus a finalize step.
 *
 * Output layout (shared with gpuMeshPipeline.js):
 *   positions : array<vec4f>, 6 vertices per quad, w = face id (0..5: +X -X +Y -Y +Z -Z)
 *   indirect  : [vertexCount, instanceCount, firstVertex, firstInstance, requestedVertices,
 *                vertexCap]
 *               The first four words are exactly the GPUDrawIndirect argument layout,
 *               so the same buffer can drive drawIndirect() with no CPU involvement.
 *               vertexCap (written by the CPU, a multiple of 6) is the capacity of
 *               this target's position buffer, so every chunk's buffer can be sized
 *               separately without recompiling the shader.
 *   borders   : 6 bit planes holding the neighbour chunks' touching layers
 *               (neighbourBorders.js); all zero = neighbours are air.
 *
 * Normals are not written: they are fully determined by the face id, and the
 * material uses flat shading.
 */
export const createGpuMeshShader = (CHUNK_SIZE, { mergeWorkgroupSize = 8 } = {}) => {
    const WORDS_PER_SLICE = (CHUNK_SIZE * CHUNK_SIZE) / 32;

    // Bit-run merge: one slice per thread, slices staged in workgroup memory with an
    // odd stride so the threads of a workgroup start on different memory banks.
    const SLICE_STRIDE = WORDS_PER_SLICE + 1;
    const MERGE_WG = mergeWorkgroupSize;
    if (CHUNK_SIZE !== 96) throw new Error('merge_bitrun_main assumes CHUNK_SIZE = 96 (3 words per row)');
    if (!Number.isInteger(MERGE_WG) || MERGE_WG < 1 || MERGE_WG * SLICE_STRIDE * 4 > 16384) {
        throw new Error(`mergeWorkgroupSize ${MERGE_WG} must be 1..${Math.floor(16384 / (SLICE_STRIDE * 4))} (16 KB workgroup storage)`);
    }

    return `
struct Voxel {
    data: array<u32>
}

struct MaskBuffer {
    data: array<atomic<u32>>
}

struct Vec4Array {
    data: array<vec4f>
}

struct IndirectDrawParams {
    vertexCount: atomic<u32>,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
    requestedVertices: atomic<u32>,
    vertexCap: u32
}

@group(0) @binding(0) var<storage, read> voxels: Voxel;
@group(0) @binding(1) var<storage, read_write> masks: MaskBuffer;
@group(0) @binding(2) var<storage, read_write> positions: Vec4Array;
@group(0) @binding(4) var<storage, read_write> indirectParams: IndirectDrawParams;
@group(0) @binding(5) var<storage, read> borders: Voxel;

const CHUNK_SIZE: u32 = ${CHUNK_SIZE}u;
const WORDS_PER_SLICE: u32 = ${WORDS_PER_SLICE}u;

fn getVoxel(x: i32, y: i32, z: i32) -> u32 {
    if (x < 0 || x >= i32(CHUNK_SIZE) || y < 0 || y >= i32(CHUNK_SIZE) || z < 0 || z >= i32(CHUNK_SIZE)) {
        return 0u;
    }
    let flatIndex = u32(x) + u32(y) * CHUNK_SIZE + u32(z) * CHUNK_SIZE * CHUNK_SIZE;

    let wordIdx = flatIndex >> 5u;
    let bitOffset = flatIndex & 31u;

    if ((voxels.data[wordIdx] & (1u << bitOffset)) != 0u) {
        return 1u;
    }
    return 0u;
}

// Occupancy of the voxel next to (x, y, z) in direction \`face\`, whose mask cell is
// (u, v). Inside the chunk it is read directly; outside, from the neighbour's border
// plane (same u, v), so faces against a solid neighbour are not emitted.
fn neighbourVoxel(x: i32, y: i32, z: i32, face: u32, u: u32, v: u32) -> u32 {
    if (x >= 0 && x < i32(CHUNK_SIZE) && y >= 0 && y < i32(CHUNK_SIZE) && z >= 0 && z < i32(CHUNK_SIZE)) {
        return getVoxel(x, y, z);
    }
    let bit = u + v * CHUNK_SIZE;
    return (borders.data[face * WORDS_PER_SLICE + (bit >> 5u)] >> (bit & 31u)) & 1u;
}

fn setMaskBit(face: u32, slice: u32, u: u32, v: u32) {
    let bitIndex = u + v * CHUNK_SIZE;
    let wordIdx = bitIndex >> 5u;
    let bitOffset = bitIndex & 31u;

    // Global word index: face * CHUNK_SIZE * WORDS_PER_SLICE + slice * WORDS_PER_SLICE + wordIdx
    let globalWordIdx = face * CHUNK_SIZE * WORDS_PER_SLICE + slice * WORDS_PER_SLICE + wordIdx;
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
    if (neighbourVoxel(x + 1, y, z, 0u, uy, uz) == 0u) { setMaskBit(0u, ux, uy, uz); }
    // Face 1: -X (Left) -> slice x, u=y, v=z
    if (neighbourVoxel(x - 1, y, z, 1u, uy, uz) == 0u) { setMaskBit(1u, ux, uy, uz); }
    // Face 2: +Y (Top) -> slice y, u=x, v=z
    if (neighbourVoxel(x, y + 1, z, 2u, ux, uz) == 0u) { setMaskBit(2u, uy, ux, uz); }
    // Face 3: -Y (Bottom) -> slice y, u=x, v=z
    if (neighbourVoxel(x, y - 1, z, 3u, ux, uz) == 0u) { setMaskBit(3u, uy, ux, uz); }
    // Face 4: +Z (Front) -> slice z, u=x, v=y
    if (neighbourVoxel(x, y, z + 1, 4u, ux, uy) == 0u) { setMaskBit(4u, uz, ux, uy); }
    // Face 5: -Z (Back) -> slice z, u=x, v=y
    if (neighbourVoxel(x, y, z - 1, 5u, ux, uy) == 0u) { setMaskBit(5u, uz, ux, uy); }
}

// --- PASS B: GREEDY MERGING ---
fn emitQuad(corners: array<vec4f, 4>) {
    let base = atomicAdd(&indirectParams.vertexCount, 6u);
    // The counter keeps counting past the cap so finalize_main can report how much
    // was requested; a quad is only written if all six of its vertices fit.
    if (base + 6u > indirectParams.vertexCap) { return; }

    // Triangle 1
    positions.data[base + 0u] = corners[0];
    positions.data[base + 1u] = corners[1];
    positions.data[base + 2u] = corners[2];
    // Triangle 2
    positions.data[base + 3u] = corners[0];
    positions.data[base + 4u] = corners[2];
    positions.data[base + 5u] = corners[3];
}

@compute @workgroup_size(6, 16, 1)
fn merge_main(@builtin(global_invocation_id) global_id: vec3u) {
    let face = global_id.x; // 0 to 5
    let slice = global_id.y; // 0 to CHUNK_SIZE

    if (face >= 6u || slice >= CHUNK_SIZE) {
        return;
    }

    // Copy the 2D mask for this slice into private memory for fast access & clearing
    var localMask: array<u32, ${WORDS_PER_SLICE}>;
    let offset = face * CHUNK_SIZE * WORDS_PER_SLICE + slice * WORDS_PER_SLICE;
    for (var i = 0u; i < WORDS_PER_SLICE; i++) {
        localMask[i] = atomicLoad(&masks.data[offset + i]);
    }

    let fw = f32(face); // packed into the w component of every vertex

    // Greedy meshing
    for (var v = 0u; v < CHUNK_SIZE; v++) {
        for (var u = 0u; u < CHUNK_SIZE; u++) {
            // Inline: getLocalMaskBit(u, v)
            let startBitIndex = u + v * CHUNK_SIZE;
            let startWordIdx = startBitIndex >> 5u;
            let startBitOffset = startBitIndex & 31u;
            let isSet = (localMask[startWordIdx] & (1u << startBitOffset)) != 0u;

            if (isSet) {
                // Find width
                var w = 1u;
                var keepGoing = true;
                while (u + w < CHUNK_SIZE && keepGoing) {
                    let bi = (u + w) + v * CHUNK_SIZE;
                    let wi = bi >> 5u;
                    let bo = bi & 31u;
                    if ((localMask[wi] & (1u << bo)) != 0u) {
                        w++;
                    } else {
                        keepGoing = false;
                    }
                }

                // Find height
                var h = 1u;
                var done = false;
                while (v + h < CHUNK_SIZE && !done) {
                    for (var k = 0u; k < w; k++) {
                        let bi2 = (u + k) + (v + h) * CHUNK_SIZE;
                        let wi2 = bi2 >> 5u;
                        let bo2 = bi2 & 31u;
                        if ((localMask[wi2] & (1u << bo2)) == 0u) {
                            done = true;
                            break;
                        }
                    }
                    if (!done) {
                        h++;
                    }
                }

                // Clear the mask (inline clearLocalMaskBits)
                for (var dv = 0u; dv < h; dv++) {
                    for (var du = 0u; du < w; du++) {
                        let cbi = (u + du) + (v + dv) * CHUNK_SIZE;
                        let cwi = cbi >> 5u;
                        let cbo = cbi & 31u;
                        localMask[cwi] &= ~(1u << cbo);
                    }
                }

                // Emit Quad based on face
                let f_slice = f32(slice);
                let f_u = f32(u); let f_v = f32(v);
                let f_w = f32(w); let f_h = f32(h);
                var corners: array<vec4f, 4>;

                if (face == 0u) { // +X
                    corners = array<vec4f, 4>(
                        vec4f(f_slice + 1.0, f_u, f_v, fw),
                        vec4f(f_slice + 1.0, f_u + f_w, f_v, fw),
                        vec4f(f_slice + 1.0, f_u + f_w, f_v + f_h, fw),
                        vec4f(f_slice + 1.0, f_u, f_v + f_h, fw)
                    );
                } else if (face == 1u) { // -X
                    corners = array<vec4f, 4>(
                        vec4f(f_slice, f_u, f_v + f_h, fw),
                        vec4f(f_slice, f_u + f_w, f_v + f_h, fw),
                        vec4f(f_slice, f_u + f_w, f_v, fw),
                        vec4f(f_slice, f_u, f_v, fw)
                    );
                } else if (face == 2u) { // +Y
                    corners = array<vec4f, 4>(
                        vec4f(f_u, f_slice + 1.0, f_v, fw),
                        vec4f(f_u, f_slice + 1.0, f_v + f_h, fw),
                        vec4f(f_u + f_w, f_slice + 1.0, f_v + f_h, fw),
                        vec4f(f_u + f_w, f_slice + 1.0, f_v, fw)
                    );
                } else if (face == 3u) { // -Y
                    corners = array<vec4f, 4>(
                        vec4f(f_u, f_slice, f_v + f_h, fw),
                        vec4f(f_u, f_slice, f_v, fw),
                        vec4f(f_u + f_w, f_slice, f_v, fw),
                        vec4f(f_u + f_w, f_slice, f_v + f_h, fw)
                    );
                } else if (face == 4u) { // +Z
                    corners = array<vec4f, 4>(
                        vec4f(f_u + f_w, f_v, f_slice + 1.0, fw),
                        vec4f(f_u + f_w, f_v + f_h, f_slice + 1.0, fw),
                        vec4f(f_u, f_v + f_h, f_slice + 1.0, fw),
                        vec4f(f_u, f_v, f_slice + 1.0, fw)
                    );
                } else if (face == 5u) { // -Z
                    corners = array<vec4f, 4>(
                        vec4f(f_u, f_v, f_slice, fw),
                        vec4f(f_u, f_v + f_h, f_slice, fw),
                        vec4f(f_u + f_w, f_v + f_h, f_slice, fw),
                        vec4f(f_u + f_w, f_v, f_slice, fw)
                    );
                }

                emitQuad(corners);
            }
        }
    }
}

// --- PASS B (bit-run): GREEDY MERGING ---
// Line-for-line port of mergeSliceBitRuns() in bitGreedyMerge.js, which the unit
// tests check against the CPU mesher (same quads). merge_main above is kept as the
// legacy A/B baseline (?mergeImpl=legacy).
//
// Differences from merge_main:
//   - each row (3 words) is scanned with countTrailingZeros instead of cell by cell,
//     and a quad grows down by ANDing whole words against per-word run masks;
//   - the slice lives in workgroup memory, not a 288-word private array;
//   - one slice per thread with a small workgroup, so 576 / MERGE_WG workgroups are
//     spread over the GPU instead of 6.
const ROW_WORDS: u32 = 3u;
const MERGE_WG: u32 = ${MERGE_WG}u;
const SLICE_STRIDE: u32 = ${SLICE_STRIDE}u;
const SLICE_COUNT: u32 = 6u * CHUNK_SIZE;

var<workgroup> sliceMasks: array<u32, ${MERGE_WG * SLICE_STRIDE}>;

// Bits of row word k that fall inside [u0, u1).
fn rangeMask(k: u32, u0: u32, u1: u32) -> u32 {
    if (u1 <= k * 32u || u0 >= k * 32u + 32u) { return 0u; }
    let lo = max(u0, k * 32u) - k * 32u; // 0..31
    let hi = min(u1, k * 32u + 32u) - k * 32u; // lo+1..32
    return (0xFFFFFFFFu >> (32u - hi)) & (0xFFFFFFFFu << lo);
}

// Length of the run of set bits starting at u in the row whose first word is rowBase.
fn runLength(rowBase: u32, u: u32) -> u32 {
    var w = 0u;
    var k = u >> 5u;
    var b = u & 31u;
    loop {
        if (k >= ROW_WORDS) { break; }
        // Shifting zero-fills the top, so the inverted word ends the run by bit 32 - b.
        let run = countTrailingZeros(~(sliceMasks[rowBase + k] >> b));
        w += run;
        if (run < 32u - b) { break; }
        k++;
        b = 0u;
    }
    return w;
}

fn emitFaceQuad(face: u32, slice: u32, u: u32, v: u32, w: u32, h: u32) {
    let fw = f32(face);
    let s = f32(slice);
    let fu = f32(u); let fv = f32(v);
    let uw = f32(u + w); let vh = f32(v + h);
    var c: array<vec4f, 4>;
    switch face {
        case 0u: { c = array<vec4f, 4>(vec4f(s + 1.0, fu, fv, fw), vec4f(s + 1.0, uw, fv, fw), vec4f(s + 1.0, uw, vh, fw), vec4f(s + 1.0, fu, vh, fw)); }
        case 1u: { c = array<vec4f, 4>(vec4f(s, fu, vh, fw), vec4f(s, uw, vh, fw), vec4f(s, uw, fv, fw), vec4f(s, fu, fv, fw)); }
        case 2u: { c = array<vec4f, 4>(vec4f(fu, s + 1.0, fv, fw), vec4f(fu, s + 1.0, vh, fw), vec4f(uw, s + 1.0, vh, fw), vec4f(uw, s + 1.0, fv, fw)); }
        case 3u: { c = array<vec4f, 4>(vec4f(fu, s, vh, fw), vec4f(fu, s, fv, fw), vec4f(uw, s, fv, fw), vec4f(uw, s, vh, fw)); }
        case 4u: { c = array<vec4f, 4>(vec4f(uw, fv, s + 1.0, fw), vec4f(uw, vh, s + 1.0, fw), vec4f(fu, vh, s + 1.0, fw), vec4f(fu, fv, s + 1.0, fw)); }
        default: { c = array<vec4f, 4>(vec4f(fu, fv, s, fw), vec4f(fu, vh, s, fw), vec4f(uw, vh, s, fw), vec4f(uw, fv, s, fw)); }
    }
    emitQuad(c);
}

@compute @workgroup_size(${MERGE_WG})
fn merge_bitrun_main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
    // Slice index = face * CHUNK_SIZE + slice. A workgroup's slices are consecutive in
    // the mask buffer, so load them together (adjacent threads read adjacent words).
    let first = wg.x * MERGE_WG;
    for (var i = lid; i < MERGE_WG * WORDS_PER_SLICE; i += MERGE_WG) {
        let t = i / WORDS_PER_SLICE;
        if (first + t < SLICE_COUNT) {
            sliceMasks[t * SLICE_STRIDE + i % WORDS_PER_SLICE] = atomicLoad(&masks.data[first * WORDS_PER_SLICE + i]);
        }
    }
    workgroupBarrier();

    let idx = first + lid;
    if (idx >= SLICE_COUNT) { return; }
    let face = idx / CHUNK_SIZE;
    let slice = idx % CHUNK_SIZE;
    let base = lid * SLICE_STRIDE;

    for (var v = 0u; v < CHUNK_SIZE; v++) {
        let rowBase = base + v * ROW_WORDS;
        for (var k = 0u; k < ROW_WORDS; k++) {
            // Each quad clears its own cells, so this loop ends once the word is empty.
            loop {
                let word = sliceMasks[rowBase + k];
                if (word == 0u) { break; }
                let u = k * 32u + countTrailingZeros(word);
                let w = runLength(rowBase, u);
                let m0 = rangeMask(0u, u, u + w);
                let m1 = rangeMask(1u, u, u + w);
                let m2 = rangeMask(2u, u, u + w);

                var h = 1u;
                loop {
                    if (v + h >= CHUNK_SIZE) { break; }
                    let r = rowBase + h * ROW_WORDS;
                    if ((sliceMasks[r] & m0) != m0 || (sliceMasks[r + 1u] & m1) != m1 || (sliceMasks[r + 2u] & m2) != m2) { break; }
                    h++;
                }

                for (var dv = 0u; dv < h; dv++) {
                    let r = rowBase + dv * ROW_WORDS;
                    sliceMasks[r] &= ~m0;
                    sliceMasks[r + 1u] &= ~m1;
                    sliceMasks[r + 2u] &= ~m2;
                }

                emitFaceQuad(face, slice, u, v, w, h);
            }
        }
    }
}

// --- PASS C: FINALIZE ---
// Clamps the vertex counter to what was actually written, so drawIndirect() can
// never read past the end of the position buffer. The unclamped total is kept in
// requestedVertices to detect overflow.
@compute @workgroup_size(1)
fn finalize_main() {
    let total = atomicLoad(&indirectParams.vertexCount);
    atomicStore(&indirectParams.requestedVertices, total);
    if (total > indirectParams.vertexCap) {
        atomicStore(&indirectParams.vertexCount, indirectParams.vertexCap);
    }
}
`;
};
