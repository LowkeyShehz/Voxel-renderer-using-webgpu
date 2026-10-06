import * as THREE from 'three';
import { raycastVoxels } from './raycastVoxels.js';

// A press-release counts as a click (not a camera drag) if it is short and barely moved.
const CLICK_MAX_MOVE_PX = 5;
const CLICK_MAX_MS = 400;

/**
 * Turns pointer input on the canvas into voxel picks. It only picks; what to do with
 * a pick (remove, place, explode) is decided by the caller.
 *
 * Picking raycasts the chunk's occupancy data (see raycastVoxels), so it works
 * identically for all meshing modes, including zero-copy where there is no CPU geometry.
 *
 * Coexists with OrbitControls: left/right *drags* move the camera, short *clicks*
 * come here.
 */
export class VoxelEditor {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.domElement
   * @param {THREE.Camera} opts.camera
   * @param {() => import('../voxel/ChunkData').ChunkData} opts.getChunk
   * @param {(e: {button: number, hit: object}) => void} opts.onPick  button 0 = left, 2 = right
   * @param {(hit: object|null) => void} [opts.onHover]
   * @param {() => boolean} [opts.isEnabled]
   */
  constructor({ domElement, camera, getChunk, onPick, onHover, isEnabled = () => true }) {
    this.domElement = domElement;
    this.camera = camera;
    this.getChunk = getChunk;
    this.onPick = onPick;
    this.onHover = onHover;
    this.isEnabled = isEnabled;

    this._raycaster = new THREE.Raycaster();
    this._ndc = new THREE.Vector2();
    this._down = null;
    this._hoverEvent = null;
    this._hoverFrame = 0;

    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerLeave = this._onPointerLeave.bind(this);
    this._onContextMenu = (e) => e.preventDefault();

    domElement.addEventListener('pointerdown', this._onPointerDown);
    domElement.addEventListener('pointerup', this._onPointerUp);
    domElement.addEventListener('pointermove', this._onPointerMove);
    domElement.addEventListener('pointerleave', this._onPointerLeave);
    domElement.addEventListener('contextmenu', this._onContextMenu);
  }

  _pick(event) {
    const rect = this.domElement.getBoundingClientRect();
    this._ndc.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    this._raycaster.setFromCamera(this._ndc, this.camera);
    const { origin, direction } = this._raycaster.ray;
    return raycastVoxels(this.getChunk(), origin, direction);
  }

  _onPointerDown(e) {
    this._down = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button };
  }

  _onPointerUp(e) {
    const down = this._down;
    this._down = null;
    if (!down || down.button !== e.button || !this.isEnabled()) return;
    if (e.button !== 0 && e.button !== 2) return;
    if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > CLICK_MAX_MOVE_PX) return;
    if (performance.now() - down.t > CLICK_MAX_MS) return;

    const hit = this._pick(e);
    if (hit) this.onPick({ button: e.button, hit });
  }

  _onPointerMove(e) {
    if (!this.onHover) return;
    if (e.buttons !== 0 || !this.isEnabled()) {
      this.onHover(null); // dragging the camera, or editing is locked
      return;
    }
    // At most one raycast per frame, however fast the pointer events arrive.
    this._hoverEvent = e;
    if (!this._hoverFrame) {
      this._hoverFrame = requestAnimationFrame(() => {
        this._hoverFrame = 0;
        if (this._hoverEvent) this.onHover(this._pick(this._hoverEvent));
      });
    }
  }

  _onPointerLeave() {
    this._hoverEvent = null;
    this.onHover?.(null);
  }

  dispose() {
    cancelAnimationFrame(this._hoverFrame);
    this.domElement.removeEventListener('pointerdown', this._onPointerDown);
    this.domElement.removeEventListener('pointerup', this._onPointerUp);
    this.domElement.removeEventListener('pointermove', this._onPointerMove);
    this.domElement.removeEventListener('pointerleave', this._onPointerLeave);
    this.domElement.removeEventListener('contextmenu', this._onContextMenu);
  }
}
