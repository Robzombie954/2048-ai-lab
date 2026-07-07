// UI-side twin of the trainer worker. Holds a TraceGame mirror that replays
// each moveEvent (same proven rules, plus tile identities) to drive the
// two-phase board animation: slide → merge-pop + spawn.
import { TraceGame } from '../engine/traceEngine'
import type { Grade } from '../agents/types'
import type { HighScoreGameRecord, ModelConfig, TrainingMode, TrajectoryPoint } from '../shared/types'
import { getDB } from '../persistence/db'
import { getHighScoreGames } from '../worker/checkpointer'
import {
  deleteModel as dbDeleteModel,
  forkModel as dbForkModel,
  importModelFromExport,
  listModels,
  renameModel as dbRenameModel,
} from '../persistence/modelStore'
import { parseExport } from '../persistence/exportImport'
import { getBuckets } from '../persistence/statStore'
import { normalizeBuckets } from '../stats/chartSeries'
import type { MoveEventPayload, WorkerCommand, WorkerEvent } from '../worker/protocol'
import { getLab, setLab, type RenderTile } from './labStore'

class WorkerBridge {
  private worker: Worker
  private game = new TraceGame(1)
  private phase2Timer: ReturnType<typeof setTimeout> | null = null
  private pendingPhase2: (() => void) | null = null
  private disposeWaiters: (() => void)[] = []
  private refreshTimer: ReturnType<typeof setTimeout> | null = null
  private toastTimer: ReturnType<typeof setTimeout> | null = null

  // High-score replay player (client-side, uses same TraceGame for perfect animation)
  private replayGame = new TraceGame(1)
  private replayTimer: ReturnType<typeof setTimeout> | null = null
  private replayRecord: HighScoreGameRecord | null = null
  private replayIndex = 0
  private replayPlaying = false
  private replaySpeed = 10 // actions/sec during replay playback
  private pendingReplayPhase2: (() => void) | null = null
  private replayPhase2Timer: ReturnType<typeof setTimeout> | null = null

  // Playlist for "play all replays in order"
  private replayPlaylist: HighScoreGameRecord[] = []
  private replayPlaylistIndex = 0

  constructor() {
    this.worker = new Worker(new URL('../worker/trainer.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.worker.onmessage = (e: MessageEvent<WorkerEvent>) => this.onEvent(e.data)
    this.worker.onerror = (e) => setLab({ errorMsg: `worker: ${e.message}` })
    window.addEventListener('beforeunload', () => {
      // Best effort — frequent auto-checkpoints are the real safety net.
      this.send({ type: 'saveCheckpoint', reason: 'stop' })
    })
  }

  private send(cmd: WorkerCommand): void {
    this.worker.postMessage(cmd)
  }

  // ── boot & library ─────────────────────────────────────────────────

  async boot(): Promise<void> {
    await this.refreshModels()
    try {
      const db = await getDB()
      const session = await db.get('session', 'current')
      if (session?.modelId) {
        const models = getLab().models
        const doc = models.find((m) => m.id === session.modelId)
        if (doc) {
          if (session.wasTraining) {
            setLab({
              resumeHint: {
                modelId: doc.id,
                name: doc.name,
                mode: session.mode,
                movesPerSec: session.movesPerSec,
                ageMs: Date.now() - session.updatedAt,
              },
            })
          } else {
            this.loadModel(doc.id)
          }
        }
      }
    } catch {
      // No session — fresh boot.
    }
    setLab({ ready: true })
  }

  async refreshModels(): Promise<void> {
    const models = await listModels()
    setLab({ models })
    const active = getLab().activeDoc
    if (active) {
      const fresh = models.find((m) => m.id === active.id)
      if (fresh) setLab({ activeDoc: fresh, milestones: fresh.milestones })
    }
  }

  resume(): void {
    const hint = getLab().resumeHint
    if (!hint) return
    setLab({ resumeHint: null, mode: hint.mode, movesPerSec: hint.movesPerSec })
    this.loadModel(hint.modelId)
    // Start once the model reports in.
    const unsub = setInterval(() => {
      if (getLab().activeDoc?.id === hint.modelId) {
        clearInterval(unsub)
        this.start()
      }
    }, 100)
    setTimeout(() => clearInterval(unsub), 15_000)
  }

  dismissResume(): void {
    const hint = getLab().resumeHint
    setLab({ resumeHint: null })
    if (hint) this.loadModel(hint.modelId)
  }

  private freshTrajectory(score = 0, move = 0): TrajectoryPoint[] {
    if (move <= 0) return [{ move: 0, score }]
    return [
      { move: 0, score: 0 },
      { move, score },
    ]
  }

  private nextTrajectory(move: number, score: number): TrajectoryPoint[] {
    const prev = getLab().trajectory
    const last = prev[prev.length - 1]
    const base = !last || move <= 1 || move <= last.move ? [{ move: 0, score: 0 }] : prev
    const next = [...base, { move, score }]
    return next.length > 4096 ? next.slice(next.length - 4096) : next
  }

  // ── commands ───────────────────────────────────────────────────────

  createModel(name: string, config: ModelConfig): void {
    setLab({
      planningDepth: config.kind === 'ntuple' ? config.planningDepth : 1,
      buckets: [],
      partialBucket: null,
      gameOverSummary: null,
      trajectory: this.freshTrajectory(),
      highScoreGames: [],
      replay: null,
      targetGameCount: null,
    })
    this.send({ type: 'createModel', name, config })
  }

  loadModel(modelId: string): void {
    this.send({ type: 'loadModel', modelId })
  }

  start(): void {
    const { mode, movesPerSec } = getLab()
    this.stopReplay()
    this.send({ type: 'start', mode, movesPerSec })
  }

  stop(): void {
    this.send({ type: 'stop' })
  }

  setMode(mode: TrainingMode): void {
    setLab({ mode })
    this.send({ type: 'setMode', mode })
  }

  setSpeed(movesPerSec: number): void {
    setLab({ movesPerSec })
    this.send({ type: 'setSpeed', movesPerSec })
  }

  setPlanningDepth(depth: number): void {
    setLab({ planningDepth: depth })
    this.send({ type: 'setPlanningDepth', depth })
  }

  setAnnealing(enabled: boolean, halfLifeGames?: number): void {
    this.send({ type: 'setAnnealing', enabled, halfLifeGames })
  }

  grade(grade: Grade): void {
    if (!getLab().awaitingGrade) return
    setLab({ awaitingGrade: false })
    this.send({ type: 'grade', grade })
  }

  newGame(): void {
    if (!getLab().activeDoc) return
    this.finishPhase2()
    setLab({ gameOverSummary: null, trajectory: this.freshTrajectory() })
    this.send({ type: 'newGame' })
  }

  playOneGame(): void {
    if (!getLab().activeDoc) return
    this.finishPhase2()
    const { movesPerSec } = getLab()
    setLab({
      mode: 'watch',
      running: true,
      oneGameActive: true,
      gameOverSummary: null,
      trajectory: this.freshTrajectory(),
      targetGameCount: null,
      replay: null,
    })
    this.send({ type: 'playOneGame', movesPerSec })
  }

  trainForGames(count: number): void {
    if (!getLab().activeDoc || count <= 0) return
    this.finishPhase2()
    setLab({
      mode: 'turbo',
      running: true,
      oneGameActive: false,
      gameOverSummary: null,
      trajectory: this.freshTrajectory(),
      targetGameCount: count,
      replay: null,
    })
    this.send({ type: 'trainForGames', count })
  }

  saveCheckpoint(): void {
    this.send({ type: 'saveCheckpoint', reason: 'manual' })
  }

  exportModel(): void {
    this.send({ type: 'exportModel' })
  }

  // ── high score replay (uses recorded seed + action recipe) ─────────

  startReplay(record: HighScoreGameRecord, keepPlaylist = false): void {
    this.stopAnyTrainingOrReplay(keepPlaylist)
    this.replayRecord = record
    this.replayIndex = 0
    this.replayPlaying = false
    this.replaySpeed = 10
    this.replayGame.reset(record.summary.seed)

    const isPlaylist = this.replayPlaylist.length > 0
    const playlistInfo = isPlaylist ? {
      playlistTotal: this.replayPlaylist.length,
      playlistCurrent: this.replayPlaylistIndex + 1,
    } : {}

    setLab({
      replay: {
        record,
        index: 0,
        playing: false,
        speed: this.replaySpeed,
        ...playlistInfo,
      },
      mode: 'watch',
      running: false,
      oneGameActive: false,
      targetGameCount: null,
      gameOverSummary: null,
      // Seed the visual board with start state
      tiles: this.replayGame.tiles().map((t) => ({ ...t, justSpawned: true })),
      score: this.replayGame.score,
      gameIndex: record.summary.game,
      lastAction: null,
    })
    const prefix = isPlaylist ? `(${this.replayPlaylistIndex + 1}/${this.replayPlaylist.length}) ` : ''
    this.toast(`${prefix}Replaying game #${record.summary.game + 1} (score ${record.summary.score})`)
  }

  /** Play every high-score replay in sequence, lowest score to highest. */
  playAllHighScores(records: HighScoreGameRecord[]): void {
    if (!records || records.length === 0) return
    // Sort lowest to highest score (the order the model improved)
    const sorted = [...records].sort((a, b) => a.summary.score - b.summary.score)

    // Stop (this will clear playlist), then immediately re-establish playlist mode
    this.stopAnyTrainingOrReplay()
    this.replayPlaylist = sorted
    this.replayPlaylistIndex = 0

    // Now perform the start logic (without triggering another stop)
    const record = sorted[0]
    this.replayRecord = record
    this.replayIndex = 0
    this.replayPlaying = false
    this.replaySpeed = 10
    this.replayGame.reset(record.summary.seed)

    setLab({
      replay: {
        record,
        index: 0,
        playing: false,
        speed: this.replaySpeed,
        playlistTotal: sorted.length,
        playlistCurrent: 1,
      },
      mode: 'watch',
      running: false,
      oneGameActive: false,
      targetGameCount: null,
      gameOverSummary: null,
      tiles: this.replayGame.tiles().map((t) => ({ ...t, justSpawned: true })),
      score: this.replayGame.score,
      gameIndex: record.summary.game,
      lastAction: null,
    })
    this.toast(`(1/${sorted.length}) Replaying game #${record.summary.game + 1} (score ${record.summary.score})`)

    // Auto-start playback after a brief moment
    setTimeout(() => {
      if (this.replayRecord && this.replayPlaylist.length > 0) {
        this.toggleReplayPlay()
      }
    }, 300)
  }

  stopReplay(keepPlaylist = false): void {
    this.finishReplayPhase2()
    if (this.replayTimer) {
      clearTimeout(this.replayTimer)
      this.replayTimer = null
    }
    this.replayPlaying = false
    this.replayRecord = null
    if (!keepPlaylist) {
      this.replayPlaylist = []
      this.replayPlaylistIndex = 0
    }
    setLab({ replay: null, lastAction: null })
  }

  toggleReplayPlay(): void {
    if (!this.replayRecord) return
    this.replayPlaying = !this.replayPlaying
    setLab((s) => {
      if (!s.replay) return {}
      return { replay: { ...s.replay, playing: this.replayPlaying } }
    })
    if (this.replayPlaying) {
      this.scheduleReplayStep()
    } else if (this.replayTimer) {
      clearTimeout(this.replayTimer)
      this.replayTimer = null
    }
  }

  replayStep(): void {
    if (!this.replayRecord || this.replayIndex >= this.replayRecord.actions.length) return
    this.finishReplayPhase2()
    const action = this.replayRecord.actions[this.replayIndex] as any
    const events = this.replayGame.move(action)
    this.replayIndex++
    const rec = this.replayRecord

    const base = {
      score: this.replayGame.score,
      gameIndex: rec.summary.game,
      lastAction: action,
      instantBoard: false,
    }

    if (!events) {
      // should not happen
      this.replayGame.forceState(this.replayGame.board as any, this.replayGame.score)
      setLab({
        ...base,
        tiles: this.replayGame.tiles().map((t) => ({ ...t })),
        replay: this.replayRecord ? { record: this.replayRecord, index: this.replayIndex, playing: this.replayPlaying, speed: this.replaySpeed } : null,
      })
      return
    }

    // Mirror the nice animation from onMove
    const spawnedId = events.spawn?.id
    const ghosts: any[] = events.merges.map((m: any) => ({
      id: m.victimId,
      idx: m.at,
      exp: m.exp - 1,
      ghost: true,
    }))
    const phase1 = [
      ...ghosts,
      ...this.replayGame
        .tiles()
        .filter((t: any) => t.id !== spawnedId)
        .map((t: any) => ({ ...t })),
    ]
    setLab({
      ...base,
      tiles: phase1,
      replay: this.replayRecord ? { record: this.replayRecord, index: this.replayIndex, playing: this.replayPlaying, speed: this.replaySpeed } : null,
    })

    const phase2 = () => {
      const mergedIds = new Set(events.merges.map((m: any) => m.survivorId))
      setLab({
        tiles: this.replayGame.tiles().map((t: any) => ({
          ...t,
          justMerged: mergedIds.has(t.id),
          justSpawned: t.id === spawnedId,
        })),
        replay: this.replayRecord ? { record: this.replayRecord, index: this.replayIndex, playing: this.replayPlaying, speed: this.replaySpeed } : null,
      })
    }

    const slideMs = Math.max(30, Math.min(120, 800 / this.replaySpeed))
    this.pendingReplayPhase2 = phase2
    this.replayPhase2Timer = setTimeout(() => {
      this.replayPhase2Timer = null
      this.pendingReplayPhase2 = null
      phase2()
      if (this.replayPlaying) this.scheduleReplayStep()
    }, slideMs)

    // auto stop at end (or advance playlist)
    if (this.replayIndex >= this.replayRecord.actions.length) {
      this.handleReplayFinished()
    }
  }

  setReplaySpeed(speed: number): void {
    this.replaySpeed = Math.max(1, Math.min(30, speed))
    setLab((s) => (s.replay ? { replay: { ...s.replay, speed: this.replaySpeed } } : {}))
  }

  private handleReplayFinished(): void {
    this.replayPlaying = false
    setLab((s) => (s.replay ? { replay: { ...s.replay, playing: false } } : {}))

    // If in a playlist, advance to the next after a nice pause on the final board
    if (this.replayPlaylist.length > 0 && this.replayPlaylistIndex < this.replayPlaylist.length - 1) {
      this.replayPlaylistIndex++
      const nextRecord = this.replayPlaylist[this.replayPlaylistIndex]
      // Pause ~1.2s so user can see the final position of the previous record
      setTimeout(() => {
        // Still in playlist mode? (stopReplay clears the array)
        if (this.replayPlaylist.length > 0) {
          this.startReplay(nextRecord, true)  // keep the playlist
          // Auto-resume playing the next one
          setTimeout(() => {
            if (this.replayRecord && this.replayPlaylist.length > 0) {
              this.toggleReplayPlay()
            }
          }, 200)
        }
      }, 1200)
    } else if (this.replayPlaylist.length > 0) {
      // Finished the last one
      this.toast('Finished all top score replays (lowest → highest)')
      this.replayPlaylist = []
      this.replayPlaylistIndex = 0
    }
  }

  private finishReplayPhase2(): void {
    if (this.replayPhase2Timer) {
      clearTimeout(this.replayPhase2Timer)
      this.replayPhase2Timer = null
    }
    if (this.pendingReplayPhase2) {
      const fn = this.pendingReplayPhase2
      this.pendingReplayPhase2 = null
      fn()
    }
  }

  private scheduleReplayStep(): void {
    if (!this.replayPlaying || !this.replayRecord) return
    if (this.replayIndex >= this.replayRecord.actions.length) {
      this.handleReplayFinished()
      return
    }
    this.replayTimer = setTimeout(() => {
      this.replayTimer = null
      this.replayStep()
    }, 1000 / this.replaySpeed)
  }

  private stopAnyTrainingOrReplay(keepPlaylist = false): void {
    if (getLab().running) {
      this.send({ type: 'stop' })
    }
    this.stopReplay(keepPlaylist)
  }

  async deleteModel(modelId: string): Promise<void> {
    if (getLab().activeDoc?.id === modelId) {
      await new Promise<void>((resolve) => {
        this.disposeWaiters.push(resolve)
        this.send({ type: 'dispose' })
        setTimeout(resolve, 3000)
      })
    }
    await dbDeleteModel(modelId)
    await this.refreshModels()
    this.toast('Model deleted')
  }

  async forkModel(modelId: string, name: string): Promise<void> {
    const doc = await dbForkModel(modelId, name)
    await this.refreshModels()
    if (doc) this.toast(`Forked → "${doc.name}"`)
  }

  async renameModel(modelId: string, name: string): Promise<void> {
    await dbRenameModel(modelId, name)
    await this.refreshModels()
  }

  async importFile(file: File): Promise<void> {
    try {
      const parsed = parseExport(await file.arrayBuffer())
      const doc = await importModelFromExport(parsed)
      await this.refreshModels()
      this.toast(`Imported "${doc.name}"`)
    } catch (err) {
      setLab({ errorMsg: err instanceof Error ? err.message : String(err) })
    }
  }

  // ── event handling ─────────────────────────────────────────────────

  private onEvent(ev: WorkerEvent): void {
    switch (ev.type) {
      case 'ready':
        break
      case 'modelLoaded': {
        this.finishPhase2()
        this.game.forceState(Uint8Array.from(ev.board), ev.score)
        setLab({
          activeDoc: ev.doc,
          diagnostics: ev.diagnostics,
          milestones: ev.doc.milestones,
          tiles: this.game.tiles().map((t) => ({ ...t, justSpawned: true })),
          instantBoard: false,
          score: ev.score,
          gameIndex: ev.doc.games,
          lastValues: [null, null, null, null],
          lastAction: null,
          awaitingGrade: false,
          oneGameActive: false,
          gameOverSummary: null,
          trajectory: this.freshTrajectory(ev.score),
          planningDepth:
            ev.doc.config.kind === 'ntuple' ? ev.doc.config.planningDepth : 1,
        })
        void getBuckets(ev.doc.id).then((buckets) => {
          if (getLab().activeDoc?.id === ev.doc.id) setLab({ buckets: normalizeBuckets(buckets) })
        })
        void getHighScoreGames(ev.doc.id).then((hs) => {
          if (getLab().activeDoc?.id === ev.doc.id) setLab({ highScoreGames: hs })
        })
        this.scheduleRefresh()
        // "Level up" (and any seeded create) can request the fresh model start
        // training the moment it's ready — race-free, since it's loaded now.
        if (getLab().autostartOnLoad) {
          setLab({ autostartOnLoad: false })
          this.start()
        }
        break
      }
      case 'modelDisposed':
        setLab({
          activeDoc: null,
          diagnostics: null,
          live: null,
          tiles: [],
          buckets: [],
          partialBucket: null,
          milestones: [],
          running: false,
          oneGameActive: false,
          gameOverSummary: null,
          trajectory: this.freshTrajectory(),
          highScoreGames: [],
          replay: null,
          targetGameCount: null,
        })
        for (const w of this.disposeWaiters.splice(0)) w()
        break
      case 'trainingState':
        setLab({
          running: ev.running,
          mode: ev.mode,
          movesPerSec: ev.movesPerSec,
          oneGameActive: ev.oneGameActive,
          // Clear target indicator when training stops
          ...( !ev.running ? { targetGameCount: null } : {} ),
        })
        break
      case 'boardSync':
        this.finishPhase2()
        this.game.forceState(Uint8Array.from(ev.board), ev.score)
        setLab({
          tiles: this.game.tiles().map((t) => ({ ...t, justSpawned: true })),
          instantBoard: false,
          score: ev.score,
          gameIndex: ev.gameIndex,
          gameOverSummary: null,
          trajectory: this.freshTrajectory(ev.score, ev.moveIndex),
        })
        break
      case 'moveEvent':
        this.onMove(ev.move)
        break
      case 'snapshot':
        this.finishPhase2()
        this.game.forceState(Uint8Array.from(ev.board), ev.score)
        setLab({
          tiles: this.game.tiles(),
          instantBoard: true,
          score: ev.score,
          gameIndex: ev.gameIndex,
          lastValues: ev.values,
          lastAction: ev.lastAction,
        })
        break
      case 'stats': {
        if (ev.completedBucket) {
          const done = ev.completedBucket
          const buckets = normalizeBuckets([...getLab().buckets.filter((b) => b.bucket !== done.bucket), done])
          setLab({ live: ev.live, partialBucket: ev.partialBucket, buckets })
        } else {
          setLab({ live: ev.live, partialBucket: ev.partialBucket })
        }
        break
      }
      case 'episodeEnd':
        setLab({ gameOverSummary: ev.summary })
        break
      case 'milestone': {
        const milestones = [...getLab().milestones, ev.milestone]
        setLab({ milestones })
        this.toast(`🏆 First ${1 << ev.milestone.exp} tile — game ${ev.milestone.game + 1}`)
        break
      }
      case 'highScoreGame': {
        const current = getLab()
        if (current.activeDoc && current.activeDoc.id === ev.record.modelId) {
          const existing = current.highScoreGames || []
          // Avoid dups by game+score
          const already = existing.some((r) => r.summary.game === ev.record.summary.game && r.summary.score === ev.record.summary.score)
          const hs = already ? existing : [...existing, ev.record].sort((a, b) => a.summary.game - b.summary.game)
          setLab({ highScoreGames: hs })
          this.toast(`🏆 New high score: ${ev.record.summary.score} (game ${ev.record.summary.game + 1})`)
        }
        // Also feed the archive (it will queue if no folder chosen yet)
        import('../persistence/highScoreArchive').then((m) => m.recordHighScoreGame(ev.record)).catch(() => {})
        break
      }
      case 'checkpointSaved':
        setLab({ lastCheckpointAt: Date.now(), lastCheckpointBytes: ev.sizeBytes })
        break
      case 'exportReady': {
        const blob = new Blob([ev.data], { type: 'application/octet-stream' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = ev.fileName
        a.click()
        setTimeout(() => URL.revokeObjectURL(url), 10_000)
        this.toast(`Exported ${ev.fileName}`)
        break
      }
      case 'modelsChanged':
        this.scheduleRefresh()
        break
      case 'error':
        setLab({ errorMsg: ev.message })
        break
    }
  }

  private onMove(move: MoveEventPayload): void {
    this.finishPhase2()
    const spawn = move.spawn ? { idx: move.spawn.idx, exp: move.spawn.exp } : null
    const events = this.game.move(move.action, spawn)
    const base = {
      score: move.score,
      gameIndex: move.gameIndex,
      lastValues: move.values,
      lastAction: move.action,
      exploring: move.exploring,
      awaitingGrade: move.awaitingGrade,
      instantBoard: false,
      gameOverSummary: null,
      trajectory: this.nextTrajectory(move.moveIndex + 1, move.score),
    }
    if (!events) {
      // Mirror out of sync — hard resync (fresh ids, no animation).
      this.game.forceState(Uint8Array.from(move.postBoard), move.score)
      setLab({ ...base, tiles: this.game.tiles(), instantBoard: true })
      return
    }
    // Verify the mirror against the worker's authoritative board.
    let inSync = true
    for (let i = 0; i < 16; i++) {
      if (this.game.board[i] !== move.postBoard[i]) {
        inSync = false
        break
      }
    }
    if (!inSync) {
      this.game.forceState(Uint8Array.from(move.postBoard), move.score)
      setLab({ ...base, tiles: this.game.tiles(), instantBoard: true })
      return
    }

    // Phase 1 — everything slides: survivors at new positions, merge victims
    // as ghosts sliding to the merge cell. The spawned tile is withheld.
    const spawnedId = events.spawn?.id
    const ghosts: RenderTile[] = events.merges.map((m) => ({
      id: m.victimId,
      idx: m.at,
      exp: m.exp - 1,
      ghost: true,
    }))
    const phase1: RenderTile[] = [
      ...ghosts,
      ...this.game
        .tiles()
        .filter((t) => t.id !== spawnedId)
        .map((t) => ({ ...t })),
    ]
    setLab({ ...base, tiles: phase1 })

    // Phase 2 — drop ghosts, pop merged survivors, reveal the spawn.
    const mergedIds = new Set(events.merges.map((m) => m.survivorId))
    const phase2 = () => {
      setLab({
        tiles: this.game.tiles().map((t) => ({
          ...t,
          justMerged: mergedIds.has(t.id),
          justSpawned: t.id === spawnedId,
        })),
      })
    }
    const mps = getLab().movesPerSec
    const slideMs =
      getLab().mode === 'watch' ? Math.max(40, Math.min(100, 550 / mps)) : 100
    this.pendingPhase2 = phase2
    this.phase2Timer = setTimeout(() => {
      this.phase2Timer = null
      this.pendingPhase2 = null
      phase2()
    }, slideMs)
  }

  /** Fast-forward a pending phase 2 so a rapid next move never overlaps it. */
  private finishPhase2(): void {
    if (this.phase2Timer) {
      clearTimeout(this.phase2Timer)
      this.phase2Timer = null
    }
    if (this.pendingPhase2) {
      const fn = this.pendingPhase2
      this.pendingPhase2 = null
      fn()
    }
  }

  private scheduleRefresh(): void {
    if (this.refreshTimer) return
    this.refreshTimer = setTimeout(() => {
      this.refreshTimer = null
      void this.refreshModels()
    }, 400)
  }

  private toast(msg: string): void {
    setLab({ toast: msg })
    if (this.toastTimer) clearTimeout(this.toastTimer)
    this.toastTimer = setTimeout(() => setLab({ toast: null }), 3500)
  }
}

export const bridge = new WorkerBridge()

