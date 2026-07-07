// The educational centerpiece: the model's live learned value for each of
// the four directions, exactly as used to pick the move.
import { useLabStore } from '../../state/labStore'
import { DIR_ARROWS, DIR_NAMES } from '../../engine/rules'
import { formatCompact } from '../../lib/format'

const ORDER = [0, 3, 1, 2] // up, left, right, down — reading order

export function DirectionViz() {
  const values = useLabStore((s) => s.lastValues)
  const lastAction = useLabStore((s) => s.lastAction)
  const exploring = useLabStore((s) => s.exploring)
  const activeDoc = useLabStore((s) => s.activeDoc)

  const finite = values.filter((v): v is number => v !== null)
  const min = finite.length ? Math.min(...finite) : 0
  const max = finite.length ? Math.max(...finite) : 1
  const span = max - min || 1

  return (
    <div className="w-full max-w-md rounded-xl border border-white/5 bg-surface-1 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-500">
          {activeDoc?.kind === 'dqn' ? 'Q-values per direction' : 'Learned value per direction'}
        </span>
        <span
          className={`rounded bg-sky-500/15 px-1.5 py-0.5 text-[11px] font-medium text-sky-300 transition-opacity ${
            exploring ? 'opacity-100' : 'opacity-0'
          }`}
          aria-hidden={!exploring}
        >
          ε exploration
        </span>
      </div>
      <div className="space-y-2">
        {ORDER.map((dir) => {
          const v = values[dir]
          const isChosen = lastAction === dir && v !== null
          const width = v === null ? 0 : 12 + (88 * (v - min)) / span
          return (
            <div key={dir} className="flex items-center gap-3">
              <span
                className={`w-6 text-center text-base ${
                  isChosen ? 'text-gold-400' : v === null ? 'text-zinc-700' : 'text-zinc-400'
                }`}
                title={DIR_NAMES[dir]}
              >
                {DIR_ARROWS[dir]}
              </span>
              <div className="h-5 flex-1 overflow-hidden rounded bg-white/[0.04]">
                {v !== null && (
                  <div
                    className={`dir-bar h-full rounded ${
                      isChosen ? 'chosen-pulse bg-gold-500/80' : 'bg-zinc-600/60'
                    }`}
                    style={{ width: `${width}%` }}
                  />
                )}
              </div>
              <span
                className={`w-14 text-right font-mono text-xs tabular-nums ${
                  isChosen ? 'text-gold-300' : v === null ? 'text-zinc-700' : 'text-zinc-500'
                }`}
              >
                {v === null ? '—' : formatCompact(v)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

