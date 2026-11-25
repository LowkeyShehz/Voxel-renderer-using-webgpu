import * as THREE from 'three';

const MOVE_KEYS = {
  KeyW: [0, 0, -1], KeyS: [0, 0, 1], KeyA: [-1, 0, 0], KeyD: [1, 0, 0],
  Space: [0, 1, 0], ShiftLeft: [0, -1, 0], ShiftRight: [0, -1, 0],
};
const LOOK_SPEED = 0.003; // radians per pixel
const MAX_PITCH = Math.PI / 2 - 0.01;

/**
 * Free-fly camera: WASD to move along the view direction, Space / Shift for up /
 * down, mouse wheel to change speed. Look around by dragging with the left mouse
 * button, or, after lockPointer(), by moving the mouse (Esc releases the lock).
 *
 * Drags only look; short clicks are left to VoxelEditor, which picks from the
 * screen centre while the pointer is locked.
 */
export class FreeFlyControls {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {HTMLElement} domElement
   */
  constructor(camera, domElement, { speed = 40 } = {}) {
    this.camera = camera;
    this.domElement = domElement;
    this.speed = speed; // voxels per second
    this.enabled = false;

    this._keys = new Set();
    this._dragging = false;
    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._move = new THREE.Vector3();

    this._onKeyDown = (e) => {
      if (!this.enabled || !(e.code in MOVE_KEYS) || isTyping(e)) return;
      this._keys.add(e.code);
      if (e.code === 'Space') e.preventDefault(); // no page scroll / button activation
    };
    this._onKeyUp = (e) => this._keys.delete(e.code);
    this._onBlur = () => this._keys.clear();
    this._onPointerDown = (e) => {
      if (this.enabled && e.button === 0) this._dragging = true;
    };
    this._onPointerUp = () => {
      this._dragging = false;
    };
    this._onPointerMove = (e) => {
      if (!this.enabled || !(this._dragging || this.isLocked)) return;
      this._euler.setFromQuaternion(this.camera.quaternion);
      this._euler.y -= e.movementX * LOOK_SPEED;
      this._euler.x = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this._euler.x - e.movementY * LOOK_SPEED));
      this.camera.quaternion.setFromEuler(this._euler);
    };
    this._onWheel = (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      this.speed = Math.max(2, Math.min(400, this.speed * (e.deltaY > 0 ? 0.8 : 1.25)));
    };

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('blur', this._onBlur);
    domElement.addEventListener('pointerdown', this._onPointerDown);
    window.addEventListener('pointerup', this._onPointerUp);
    domElement.addEventListener('pointermove', this._onPointerMove);
    domElement.addEventListener('wheel', this._onWheel, { passive: false });
  }

  get isLocked() {
    return document.pointerLockElement === this.domElement;
  }

  /** Optional: capture the mouse for mouse-look (must be called from a user gesture). */
  lockPointer() {
    return this.domElement.requestPointerLock?.();
  }

  /** @param {number} dt seconds since the last frame */
  update(dt) {
    if (!this.enabled || this._keys.size === 0) return;
    this._move.set(0, 0, 0);
    for (const code of this._keys) {
      const [x, y, z] = MOVE_KEYS[code];
      this._move.x += x;
      this._move.y += y;
      this._move.z += z;
    }
    if (this._move.lengthSq() === 0) return;
    // x/z move in the camera's frame; y is world up/down.
    const up = this._move.y;
    this._move.y = 0;
    this._move.applyQuaternion(this.camera.quaternion);
    this._move.y += up;
    this._move.normalize().multiplyScalar(this.speed * Math.min(dt, 0.1));
    this.camera.position.add(this._move);
  }

  dispose() {
    if (this.isLocked) document.exitPointerLock();
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('blur', this._onBlur);
    this.domElement.removeEventListener('pointerdown', this._onPointerDown);
    window.removeEventListener('pointerup', this._onPointerUp);
    this.domElement.removeEventListener('pointermove', this._onPointerMove);
    this.domElement.removeEventListener('wheel', this._onWheel);
  }
}

function isTyping(e) {
  const t = e.target;
  return t instanceof HTMLElement && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
}
