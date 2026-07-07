// splitmix32 — tiny, fast, seedable. Every game records its seed so any
// game is reproducible.
export type Rand = () => number

export function splitmix32(seed: number): Rand {
  let s = seed | 0
  return () => {
    s = (s + 0x9e3779b9) | 0
    let t = s ^ (s >>> 16)
    t = Math.imul(t, 0x21f0aaad)
    t ^= t >>> 15
    t = Math.imul(t, 0x735a2d97)
    t ^= t >>> 15
    return (t >>> 0) / 4294967296
  }
}

/** Fresh unpredictable 32-bit seed. */
export function randomSeed(): number {
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const a = new Uint32Array(1)
    crypto.getRandomValues(a)
    return a[0] | 0
  }
  return (Math.random() * 0x100000000) | 0
}
