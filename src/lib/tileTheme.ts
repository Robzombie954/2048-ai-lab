// The classic 2048 warm palette (it pops beautifully on a dark surface),
// with a gold glow ramp for the big tiles and a dark+gold treatment for
// the legendary ≥4096 tiles.
export interface TileTheme {
  bg: string
  fg: string
  boxShadow?: string
  border?: string
}

const CLASSIC: Record<number, TileTheme> = {
  1: { bg: '#eee4da', fg: '#776e65' },
  2: { bg: '#ede0c8', fg: '#776e65' },
  3: { bg: '#f2b179', fg: '#f9f6f2' },
  4: { bg: '#f59563', fg: '#f9f6f2' },
  5: { bg: '#f67c5f', fg: '#f9f6f2' },
  6: { bg: '#f65e3b', fg: '#f9f6f2' },
  7: { bg: '#edcf72', fg: '#f9f6f2' },
  8: { bg: '#edcc61', fg: '#f9f6f2' },
  9: { bg: '#edc850', fg: '#f9f6f2' },
  10: {
    bg: '#edc53f',
    fg: '#f9f6f2',
    boxShadow: '0 0 24px rgb(237 194 46 / 0.35)',
  },
  11: {
    bg: '#edc22e',
    fg: '#f9f6f2',
    boxShadow: '0 0 32px rgb(237 194 46 / 0.55)',
  },
}

const SUPER: TileTheme = {
  bg: '#221f18',
  fg: '#f4d872',
  border: '1px solid rgb(237 194 46 / 0.6)',
  boxShadow: '0 0 36px rgb(237 194 46 / 0.5)',
}

export function tileTheme(exp: number): TileTheme {
  return CLASSIC[exp] ?? SUPER
}

export function tileFontSize(exp: number): string {
  const digits = String(1 << exp).length
  if (digits <= 2) return 'clamp(1.2rem, 5.4cqw, 2.1rem)'
  if (digits === 3) return 'clamp(1.05rem, 4.6cqw, 1.8rem)'
  if (digits === 4) return 'clamp(0.9rem, 3.8cqw, 1.5rem)'
  return 'clamp(0.75rem, 3.2cqw, 1.2rem)'
}
