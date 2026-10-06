/**
 * Records requestAnimationFrame-to-requestAnimationFrame times. Call tick() once
 * per frame from the render loop with the rAF timestamp.
 *
 * Two uses:
 *  - recent(): rolling stats over the last few seconds, for the live readout
 *  - begin()/end(): capture every frame time in a window, for the benchmarks
 *
 * Frame-to-frame time is what a user perceives as stutter: it grows both when the
 * main thread is blocked (CPU meshing) and when the GPU falls behind.
 */
const RING_SIZE = 120;

export class FrameRecorder {
  constructor() {
    this._ring = new Float64Array(RING_SIZE);
    this._count = 0;
    this._last = null;
    this._session = null;
  }

  tick(now) {
    if (this._last !== null) {
      const dt = now - this._last;
      this._ring[this._count % RING_SIZE] = dt;
      this._count++;
      if (this._session) this._session.push(dt);
    }
    this._last = now;
  }

  /** @returns {{avg:number, max:number}|null} over the most recent frames */
  recent() {
    const n = Math.min(this._count, RING_SIZE);
    if (n === 0) return null;
    let sum = 0;
    let max = 0;
    for (let i = 0; i < n; i++) {
      sum += this._ring[i];
      if (this._ring[i] > max) max = this._ring[i];
    }
    return { avg: sum / n, max };
  }

  begin() {
    this._session = [];
  }

  /** @returns {number[]} every frame time since begin() */
  end() {
    const frames = this._session ?? [];
    this._session = null;
    return frames;
  }
}
