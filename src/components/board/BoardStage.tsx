import { useLabStore } from '../../state/labStore'
import { tileTheme, tileFontSize } from '../../lib/tileTheme'
import { formatInt } from '../../lib/format'

const STEP = 111.4286 // one cell of travel, in % of tile size

export function BoardStage() {
  const tiles = useLabStore((s) => s.tiles)
  const instant = useLabStore((s) => s.instantBoard)
  const mode = useLabStore((s) => s.mode)
  const running = useLabStore((s) => s.running)
  const mps = useLabStore((s) => s.movesPerSec)
  const activeDoc = useLabStore((s) => s.activeDoc)

  const slideMs = mode === 'watch' ? Math.max(40, Math.min(100, 550 / mps)) : 100

  return (
    <div
      className="board-frame w-full max-w-md rounded-xl"
      style={
        {
          aspectRatio: '1 / 1',
          containerType: 'inline-size',
          '--slide-ms': `${slideMs}ms`,
        } as React.CSSProperties
      }
    >
      <div className="grid h-full grid-cols-4 gap-[2.5%] p-[2.5%]">
        {Array.from({ length: 16 }, (_, i) => (
          <div key={i} className="board-cell" />
        ))}
      </div>
      {tiles.map((t) => {
        const theme = tileTheme(t.exp)
        const row = (t.idx / 4) | 0
        const col = t.idx % 4
        const tf = `translate(${(col * STEP).toFixed(3)}%, ${(row * STEP).toFixed(3)}%)`
        return (
          <div
            key={t.id}
            className={[
              'tile',
              instant ? 'instant' : '',
              t.justSpawned ? 'tile-spawn' : '',
              t.justMerged ? 'tile-merge' : '',
            ].join(' ')}
            style={
              {
                '--tf': tf,
                background: theme.bg,
                color: theme.fg,
                border: theme.border,
                boxShadow: theme.boxShadow,
                fontSize: tileFontSize(t.exp),
                zIndex: t.ghost ? 1 : 2,
              } as React.CSSProperties
            }
          >
            {formatInt(1 << t.exp)}
          </div>
        )
      })}
      {!activeDoc && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-xl bg-black/40 backdrop-blur-[2px]">
          <div className="text-lg font-semibold text-zinc-200">No model loaded</div>
          <div className="text-sm text-zinc-400">
            Create a new model or pick one from the library
          </div>
        </div>
      )}
      {activeDoc && mode === 'turbo' && running && (
        <div className="absolute right-3 top-3 rounded-md bg-black/50 px-2 py-1 text-xs font-medium text-gold-300 backdrop-blur-sm">
          TURBO
        </div>
      )}
    </div>
  )
}
