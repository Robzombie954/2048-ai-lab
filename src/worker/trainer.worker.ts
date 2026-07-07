// The training worker: owns the game engine, the active agent, stat
// aggregation, and checkpointing. The UI is a pure renderer of the events
// this worker emits — it never advances game state on its own.
import { newBoard, maxExp } from '../engine/board'
import type { Dir } from '../engine/rules'
import { applyMove, isGameOver, legalMask, spawnTile, startBoard } from '../engine/fastEngine'
import { splitmix32, randomSeed, type Rand } from '../engine/rng'
import type { Agent, Grade } from '../agents/types'
import { createAgent, restoreAgent } from '../agents/factory'
import { DEFAULT_ANNEAL_FLOOR, DEFAULT_ANNEAL_HALFLIFE } from '../agents/ntuple/anneal'
import type { GameSummary, HighScoreGameRecord, ModelConfig, ModelDoc, TrainingMode } from '../shared/types'
import { BucketAggregator } from '../stats/buckets'
import { MilestoneTracker } from '../stats/milestones'
import { getDB } from '../persistence/db'
import { newModelDoc } from '../persistence/modelStore'
import { buildExport } from '../persistence/exportImport'
import { getBuckets } from '../persistence/statStore'
import {
  loadResumeData,
  persistBucket,
  persistGame,
  persistHighScoreGame,
  rollbackBeyond,
  saveCheckpoint,
  updateSession,
} from './checkpointer'
import type { WorkerCommand, WorkerEvent } from './protocol'

const post = (ev: WorkerEvent, transfer?: Transferable[]) =>
  (self as unknown as Worker).postMessage(ev, transfer ?? [])

const TURBO_CHUNK_MS = 12
const STATS_INTERVAL_MS = 500
const SNAPSHOT_INTERVAL_MS = 250

class Trainer {
  private agent: Agent | null = null
  private doc: ModelDoc | null = null

  private board = newBoard()
  private after = newBoard()
  private next = newBoard()

  // Grade-mode pending transition (dedicated copies).
  private pendState = newBoard()
  private pendAfter = newBoard()
  private pendNext = newBoard()
  private pendAction: Dir = 0
  private pendReward = 0
  private pendTerminal = false
  private awaitingGrade = false

  private gameSeed = 0
  private gameRand: Rand = splitmix32(0)
  private score = 0
  private gameMoves = 0
  private curMaxExp = 0
  private gameStartTrainMs = 0
  private lastValues: (number | null)[] = [null, null, null, null]
  private lastAction: Dir | null = null
  private oneGameActive = false

  private running = false
  private mode: TrainingMode = 'turbo'
  private movesPerSec = 6
  private watchTimer: ReturnType<typeof setTimeout> | null = null
  private turboScheduled = false
  private readonly turboChannel = new MessageChannel()

  private games = 0
  private totalMoves = 0
  private trainMs = 0
  private lastMark = 0
  private recent: number[] = []
  private aggregator = new BucketAggregator()
  private milestones = new MilestoneTracker()

  private gamesSinceCp = 0
  private lastCpWall = 0
  private checkpointing = false
  private writeQueue: Promise<unknown> = Promise.resolve()

  private lastStatsEmit = 0
  private lastSnapshotEmit = 0
  private mpsEma = 0
  private lastStepAt = 0

  // High score recipe collection
  private currentActions: number[] = []

  // Target batch training
  private targetGames: number | null = null

  // Games/sec tracking
  private gamesPerSecEma = 0
  private lastGameWall = 0

  constructor() {
    this.turboChannel.port1.onmessage = () => {
      this.turboScheduled = false
      this.turboChunk()
    }
  }

  async handle(cmd: WorkerCommand): Promise<void> {
    try {
      switch (cmd.type) {
        case 'createModel':
          await this.createModel(cmd.name, cmd.config)
          break
        case 'loadModel':
          await this.loadModel(cmd.modelId)
          break
        case 'start':
          this.start(cmd.mode, cmd.movesPerSec)
          break
        case 'stop':
          await this.stop()
          break
        case 'setMode':
          this.setMode(cmd.mode)
          break
        case 'setSpeed':
          this.movesPerSec = Math.max(0.5, Math.min(20, cmd.movesPerSec))
          this.emitTrainingState()
          break
        case 'setPlanningDepth':
          this.setPlanningDepth(cmd.depth)
          break
        case 'setAnnealing':
          this.setAnnealing(cmd.enabled, cmd.halfLifeGames)
          break
        case 'grade':
          this.applyGrade(cmd.grade)
          break
        case 'newGame':
          this.freshGame()
          break
        case 'playOneGame':
          this.playOneGame(cmd.movesPerSec)
          break
        case 'trainForGames':
          this.trainForGames(cmd.count)
          break
        case 'saveCheckpoint':
          await this.checkpoint(cmd.reason, true)
          break
        case 'exportModel':
          await this.exportModel()
          break
        case 'dispose':
          await this.dispose()
          break
      }
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) })
    }
  }

  // ── model lifecycle ────────────────────────────────────────────────

  private async createModel(name: string, config: ModelConfig): Promise<void> {
    await this.unloadCurrent()
    const doc = newModelDoc(name, config)
    this.doc = doc
    this.agent = createAgent(config)
    this.games = 0
    this.totalMoves = 0
    this.trainMs = 0
    this.recent = []
    this.aggregator.reset()
    this.milestones.reset([])
    this.gamesSinceCp = 0
    this.targetGames = null
    this.currentActions = []
    this.gamesPerSecEma = 0
    this.lastGameWall = 0
    const db = await getDB()
    await db.put('models', doc)
    await this.checkpoint('manual', true)
    this.startGame()
    this.emitModelLoaded()
    post({ type: 'modelsChanged' })
    await updateSession(doc.id, false, this.mode, this.movesPerSec)
  }

  private async loadModel(modelId: string): Promise<void> {
    await this.unloadCurrent()
    const db = await getDB()
    const doc = await db.get('models', modelId)
    if (!doc) throw new Error('model not found')
    let cp = doc.latestCheckpointId ? await db.get('checkpoints', doc.latestCheckpointId) : null
    if (!cp && doc.bestCheckpointId) cp = await db.get('checkpoints', doc.bestCheckpointId)
    this.doc = doc
    if (cp) {
      this.agent = restoreAgent(cp.metaJson, cp.buffers)
      this.games = cp.gamesAt
      this.totalMoves = cp.movesAt
      this.trainMs = cp.trainMsAt
    } else {
      this.agent = createAgent(doc.config)
      this.games = 0
      this.totalMoves = 0
      this.trainMs = 0
    }
    this.targetGames = null
    this.currentActions = []
    this.gamesPerSecEma = 0
    this.lastGameWall = 0
    if (doc.config.kind === 'ntuple') this.agent.planningDepth = doc.config.planningDepth

    // Align stats with the restored weights.
    await rollbackBeyond(doc.id, this.games)
    doc.games = this.games
    doc.moves = this.totalMoves
    doc.trainMs = this.trainMs
    doc.milestones = doc.milestones.filter((m) => m.game < this.games)
    this.milestones.reset(doc.milestones)
    const resume = await loadResumeData(doc.id, this.games)
    this.recent = resume.recentScores
    this.aggregator.reset()
    const diag = this.agent.getDiagnostics()
    this.aggregator.seed(
      resume.partialBucketGames,
      diag.meanAbsTdError,
      diag.epsilon ?? 0,
      diag.learningRate,
    )
    await db.put('models', doc)
    this.gamesSinceCp = 0
    this.startGame()
    this.emitModelLoaded()
    await updateSession(doc.id, false, this.mode, this.movesPerSec)
  }

  private async unloadCurrent(): Promise<void> {
    if (this.running) await this.stop()
    this.agent = null
    this.doc = null
  }

  private async dispose(): Promise<void> {
    await this.unloadCurrent()
    post({ type: 'modelDisposed' })
    await updateSession(null, false, this.mode, this.movesPerSec)
  }

  // ── training control ───────────────────────────────────────────────

  private start(mode: TrainingMode, movesPerSec: number): void {
    if (!this.agent || !this.doc) throw new Error('no model loaded')
    this.mode = mode
    this.movesPerSec = movesPerSec
    this.running = true
    this.oneGameActive = false
    this.lastMark = performance.now()
    this.lastCpWall = this.lastCpWall || performance.now()
    this.emitTrainingState()
    void updateSession(this.doc.id, true, mode, movesPerSec)
    if (mode === 'watch') this.scheduleWatch()
    else if (mode === 'turbo') this.scheduleTurbo()
    else this.stepGrade()
  }

  private setPlanningDepth(depthInput: number): void {
    const depth = Math.max(1, Math.min(3, Math.trunc(depthInput)))
    if (this.agent) this.agent.planningDepth = depth
    const doc = this.doc
    if (doc?.config.kind === 'ntuple') {
      doc.config = { ...doc.config, planningDepth: depth }
      doc.updatedAt = Date.now()
      this.queueWrite(async () => {
        const db = await getDB()
        await db.put('models', doc)
      })
      post({ type: 'modelsChanged' })
    }
  }

  private setAnnealing(enabled: boolean, halfLifeGames?: number): void {
    const agent = this.agent
    const doc = this.doc
    if (!agent || !doc || doc.config.kind !== 'ntuple') return
    const anneal = enabled
      ? {
          halfLifeGames: halfLifeGames && halfLifeGames > 0 ? halfLifeGames : DEFAULT_ANNEAL_HALFLIFE,
          startGame: this.games,
          floor: DEFAULT_ANNEAL_FLOOR,
        }
      : null
    agent.setAnneal?.(anneal, this.games)
    doc.config = { ...doc.config, anneal: anneal ?? undefined }
    doc.updatedAt = Date.now()
    this.queueWrite(async () => {
      const db = await getDB()
      await db.put('models', doc)
    })
    post({ type: 'modelsChanged' })
    this.emitStats(true)
  }

  private playOneGame(movesPerSec: number): void {
    if (!this.agent || !this.doc) throw new Error('no model loaded')
    if (this.awaitingGrade) this.applyGrade(0)
    if (this.watchTimer) {
      clearTimeout(this.watchTimer)
      this.watchTimer = null
    }
    this.markTime()
    this.mode = 'watch'
    this.movesPerSec = Math.max(0.5, Math.min(20, movesPerSec))
    this.running = true
    this.oneGameActive = true
    this.targetGames = null
    this.lastMark = performance.now()
    this.freshGame()
    this.emitTrainingState()
    if (this.doc) void updateSession(this.doc.id, true, this.mode, this.movesPerSec)
    this.scheduleWatch()
  }

  private trainForGames(count: number): void {
    if (!this.agent || !this.doc || count <= 0) return
    if (this.awaitingGrade) this.applyGrade(0)
    if (this.watchTimer) {
      clearTimeout(this.watchTimer)
      this.watchTimer = null
    }
    this.targetGames = this.games + count
    this.markTime()
    this.mode = 'turbo' // batch training is fastest headless
    this.running = true
    this.oneGameActive = false
    this.lastMark = performance.now()
    this.lastCpWall = this.lastCpWall || performance.now()
    this.emitTrainingState()
    if (this.doc) void updateSession(this.doc.id, true, this.mode, this.movesPerSec)
    this.scheduleTurbo()
  }

  private async stop(): Promise<void> {
    if (!this.running && !this.awaitingGrade) {
      this.emitTrainingState()
      return
    }
    this.markTime()
    this.running = false
    this.oneGameActive = false
    this.targetGames = null
    if (this.watchTimer) {
      clearTimeout(this.watchTimer)
      this.watchTimer = null
    }
    if (this.awaitingGrade) this.applyGrade(0)
    this.emitTrainingState()
    this.emitStats(true)
    if (this.doc) {
      await this.checkpoint('stop', true)
      await updateSession(this.doc.id, false, this.mode, this.movesPerSec)
    }
  }

  private setMode(mode: TrainingMode): void {
    if (mode === this.mode) return
    if (this.awaitingGrade) this.applyGrade(0)
    this.markTime()
    this.mode = mode
    this.oneGameActive = false
    if (this.watchTimer) {
      clearTimeout(this.watchTimer)
      this.watchTimer = null
    }
    this.emitTrainingState()
    if (this.doc) void updateSession(this.doc.id, this.running, mode, this.movesPerSec)
    if (!this.running) return
    this.lastMark = performance.now()
    if (mode === 'watch') {
      this.emitBoardSync()
      this.scheduleWatch()
    } else if (mode === 'turbo') {
      this.scheduleTurbo()
    } else {
      this.emitBoardSync()
      this.stepGrade()
    }
  }

  // ── loops ──────────────────────────────────────────────────────────

  private scheduleWatch(): void {
    if (!this.running || this.mode !== 'watch') return
    this.watchTimer = setTimeout(() => {
      if (!this.running || this.mode !== 'watch' || !this.agent) return
      this.markTime()
      this.stepCore(true)
      this.emitStats(false)
      void this.checkpoint('auto', false)
      this.scheduleWatch()
    }, 1000 / this.movesPerSec)
  }

  private scheduleTurbo(): void {
    if (!this.turboScheduled) {
      this.turboScheduled = true
      this.turboChannel.port2.postMessage(0)
    }
  }

  private turboChunk(): void {
    if (!this.running || this.mode !== 'turbo' || !this.agent) return
    const start = performance.now()
    let moves = 0
    while (this.running && this.mode === 'turbo') {
      this.stepCore(false)
      moves++
      if (performance.now() - start >= TURBO_CHUNK_MS) break
    }
    const dt = performance.now() - start
    this.trainMs += dt
    this.lastMark = performance.now()
    if (dt > 0) {
      const inst = (moves / dt) * 1000
      this.mpsEma = this.mpsEma === 0 ? inst : this.mpsEma * 0.9 + inst * 0.1
    }
    this.emitSnapshot()
    this.emitStats(false)
    void this.checkpoint('auto', false)
    this.scheduleTurbo()
  }

  private stepGrade(): void {
    if (!this.running || this.mode !== 'grade' || !this.agent || this.awaitingGrade) return
    this.markTime()
    this.stepCore(true)
    this.emitStats(false)
  }

  private markTime(): void {
    const now = performance.now()
    if (this.lastMark > 0) this.trainMs += Math.min(now - this.lastMark, 2000)
    this.lastMark = now
  }

  // ── the actual step ────────────────────────────────────────────────

  private stepCore(emit: boolean): void {
    const agent = this.agent
    const doc = this.doc
    if (!agent || !doc) return
    const mask = legalMask(this.board)
    if (mask === 0) {
      // Shouldn't happen (finishGame starts a fresh board), but stay safe.
      this.finishGame()
      return
    }
    const ev = agent.evaluateMoves(this.board, mask, this.gameRand)
    this.currentActions.push(ev.chosen)
    const r = applyMove(this.board, ev.chosen, this.after)
    this.next.set(this.after)
    const spawn = spawnTile(this.next, this.gameRand)
    const terminal = isGameOver(this.next)

    const newMax = maxExp(this.next)
    if (newMax > this.curMaxExp) {
      this.curMaxExp = newMax
      if (newMax > doc.bestExp) doc.bestExp = newMax
      const m = this.milestones.check(newMax, this.games, this.totalMoves + 1, this.trainMs)
      if (m) {
        doc.milestones.push(m)
        post({ type: 'milestone', milestone: m })
      }
    }

    if (this.mode === 'grade') {
      this.pendState.set(this.board)
      this.pendAfter.set(this.after)
      this.pendNext.set(this.next)
      this.pendAction = ev.chosen
      this.pendReward = r.score
      this.pendTerminal = terminal
      this.awaitingGrade = true
    } else {
      agent.observeTransition({
        state: this.board,
        action: ev.chosen,
        reward: r.score,
        afterstate: this.after,
        next: this.next,
        terminal,
      })
    }

    this.score += r.score
    this.gameMoves++
    this.totalMoves++
    this.board.set(this.next)
    this.lastValues = ev.values.map((v) => (Number.isFinite(v) ? v : null))
    this.lastAction = ev.chosen

    if (this.mode !== 'turbo') {
      const now = performance.now()
      if (this.lastStepAt > 0) {
        const inst = 1000 / Math.max(now - this.lastStepAt, 1)
        this.mpsEma = this.mpsEma === 0 ? inst : this.mpsEma * 0.8 + inst * 0.2
      }
      this.lastStepAt = now
    }

    if (emit) {
      post({
        type: 'moveEvent',
        move: {
          gameIndex: this.games,
          moveIndex: this.gameMoves - 1,
          action: ev.chosen,
          values: this.lastValues,
          exploring: ev.exploring,
          spawn: { idx: spawn.idx, exp: spawn.exp },
          postBoard: Array.from(this.board),
          score: this.score,
          terminal,
          awaitingGrade: this.mode === 'grade',
        },
      })
    }

    if (terminal && this.mode !== 'grade') this.finishGame()
  }

  private applyGrade(grade: Grade): void {
    if (!this.awaitingGrade || !this.agent) return
    this.awaitingGrade = false
    this.agent.observeTransition({
      state: this.pendState,
      action: this.pendAction,
      reward: this.pendReward,
      afterstate: this.pendAfter,
      next: this.pendNext,
      terminal: this.pendTerminal,
      grade,
    })
    if (this.pendTerminal) {
      this.finishGame()
      if (this.running && this.mode === 'grade') this.stepGrade()
    } else if (this.running && this.mode === 'grade') {
      this.stepGrade()
    }
  }

  private finishGame(): void {
    const agent = this.agent
    const doc = this.doc
    if (!agent || !doc) return

    const prevBestScore = doc.bestScore
    const prevBestExp = doc.bestExp

    const summary: GameSummary = {
      game: this.games,
      score: this.score,
      moves: this.gameMoves,
      maxExp: this.curMaxExp,
      seed: this.gameSeed,
    }

    // Track games/sec
    const nowWall = performance.now()
    if (this.lastGameWall > 0) {
      const dt = Math.max((nowWall - this.lastGameWall) / 1000, 0.001)
      const inst = 1 / dt
      this.gamesPerSecEma = this.gamesPerSecEma > 0 ? this.gamesPerSecEma * 0.75 + inst * 0.25 : inst
    }
    this.lastGameWall = nowWall

    const isNewHighScore = this.score > prevBestScore
    if (isNewHighScore) {
      doc.bestScore = this.score
      const record: HighScoreGameRecord = {
        version: 1,
        endedAt: Date.now(),
        modelId: doc.id,
        modelName: doc.name,
        modelKind: doc.kind,
        config: doc.config,
        summary: { ...summary },
        previousBestScore: prevBestScore,
        previousBestExp: prevBestExp,
        finalBoard: Array.from(this.board),
        actions: [...this.currentActions],
        actionLegend: ['up', 'right', 'down', 'left'] as const,
        totalMovesAt: this.totalMoves,
        trainMsAt: this.trainMs,
      }
      post({ type: 'highScoreGame', record })
      this.queueWrite(() => persistHighScoreGame(record))
      // Also persist the updated bestScore on the model promptly
      this.queueWrite(async () => {
        const db = await getDB()
        await db.put('models', doc)
      })
      // Feed the optional FS archive (non-blocking, handles pending queue)
      import('../persistence/highScoreArchive')
        .then((m) => m.recordHighScoreGame(record))
        .catch(() => {})
    }

    this.games++
    this.gamesSinceCp++
    this.recent.push(this.score)
    if (this.recent.length > 100) this.recent.shift()
    const diag = agent.getDiagnostics()
    const completed = this.aggregator.addGame(
      summary,
      diag.meanAbsTdError,
      diag.epsilon ?? 0,
      diag.learningRate,
      this.trainMs - this.gameStartTrainMs,
      Date.now(),
    )
    this.queueWrite(() => persistGame(doc.id, summary))
    if (completed) {
      this.queueWrite(() => persistBucket(doc.id, completed))
      this.emitStats(true, completed)
    }
    agent.onEpisodeEnd(summary)
    if (this.mode !== 'turbo') post({ type: 'episodeEnd', summary })

    // Target batch stop?
    if (this.targetGames !== null && this.games >= this.targetGames) {
      this.targetGames = null
      this.running = false
      this.oneGameActive = false
      if (this.watchTimer) {
        clearTimeout(this.watchTimer)
        this.watchTimer = null
      }
      this.emitTrainingState()
      this.emitStats(true)
      if (this.doc) {
        void this.checkpoint('stop', true)
        void updateSession(this.doc.id, false, this.mode, this.movesPerSec)
      }
      this.startGame()
      return
    }

    if (this.oneGameActive) {
      this.oneGameActive = false
      this.running = false
      if (this.watchTimer) {
        clearTimeout(this.watchTimer)
        this.watchTimer = null
      }
      this.emitTrainingState()
      this.emitStats(true)
      if (this.doc) {
        void this.checkpoint('stop', true)
        void updateSession(this.doc.id, false, this.mode, this.movesPerSec)
      }
      this.startGame()
      return
    }
    this.startGame()
  }

  /**
   * Abandon the current in-progress game and deal a fresh one immediately,
   * keeping all learned weights. The abandoned game is NOT recorded as a
   * completed game — a truncated score would be a misleading data point.
   * Learning already happened online move-by-move, so nothing is lost.
   */
  private freshGame(): void {
    if (!this.agent || !this.doc) return
    this.awaitingGrade = false
    // Clear the agent's per-episode carry-over (n-tuple's prev-afterstate link)
    // without doing a terminal update — the game didn't actually end in a loss.
    this.agent.onEpisodeEnd({
      score: this.score,
      moves: this.gameMoves,
      maxExp: this.curMaxExp,
    })
    this.startGame()
    this.emitBoardSync()
    // Re-kick modes that step on demand; watch's pending timer and turbo's
    // loop both simply continue on the new board.
    if (this.running && this.mode === 'grade') this.stepGrade()
  }

  private startGame(): void {
    // Refresh any per-game schedule (learning-rate annealing) before the game.
    this.agent?.refreshSchedule?.(this.games)
    this.gameSeed = randomSeed()
    this.gameRand = splitmix32(this.gameSeed)
    startBoard(this.board, this.gameRand)
    this.score = 0
    this.gameMoves = 0
    this.curMaxExp = maxExp(this.board)
    this.gameStartTrainMs = this.trainMs
    this.currentActions = []
    if (this.mode !== 'turbo') this.emitBoardSync()
  }

  // ── persistence ────────────────────────────────────────────────────

  private queueWrite(fn: () => Promise<unknown>): void {
    this.writeQueue = this.writeQueue.then(fn).catch((err) => {
      post({ type: 'error', message: `persistence: ${err instanceof Error ? err.message : err}` })
    })
  }

  private async checkpoint(reason: 'auto' | 'manual' | 'stop', force: boolean): Promise<void> {
    const agent = this.agent
    const doc = this.doc
    if (!agent || !doc || this.checkpointing) return
    if (!force) {
      const interval =
        doc.config.kind === 'ntuple' && doc.config.preset === 'expert' ? 1000 : 250
      const now = performance.now()
      const since = now - this.lastCpWall
      const due =
        (this.gamesSinceCp >= interval && since >= 60_000) ||
        (this.gamesSinceCp >= 1 && since >= 300_000)
      if (!due) return
    }
    this.checkpointing = true
    try {
      const snap = agent.serialize()
      const rollingAvg = this.recent.length
        ? this.recent.reduce((a, b) => a + b, 0) / this.recent.length
        : 0
      const cp = await saveCheckpoint(
        doc,
        snap,
        { games: this.games, moves: this.totalMoves, trainMs: this.trainMs, rollingAvg },
        reason,
      )
      this.gamesSinceCp = 0
      this.lastCpWall = performance.now()
      post({
        type: 'checkpointSaved',
        checkpointId: cp.id,
        sizeBytes: cp.sizeBytes,
        reason,
        gamesAt: cp.gamesAt,
      })
      post({ type: 'modelsChanged' })
    } finally {
      this.checkpointing = false
    }
  }

  private async exportModel(): Promise<void> {
    const agent = this.agent
    const doc = this.doc
    if (!agent || !doc) throw new Error('no model loaded')
    const snap = agent.serialize()
    const buckets = await getBuckets(doc.id)
    const rollingAvg = this.recent.length
      ? this.recent.reduce((a, b) => a + b, 0) / this.recent.length
      : 0
    const data = buildExport(
      { ...doc, games: this.games, moves: this.totalMoves, trainMs: this.trainMs },
      {
        gamesAt: this.games,
        movesAt: this.totalMoves,
        trainMsAt: this.trainMs,
        rollingAvg,
        metaJson: snap.metaJson,
        buffers: snap.buffers,
      },
      buckets,
    )
    const safeName = doc.name.replace(/[^a-z0-9-_ ]/gi, '').trim() || 'model'
    post(
      { type: 'exportReady', fileName: `${safeName}-g${this.games}.2048model`, data },
      [data],
    )
  }

  // ── event emission ─────────────────────────────────────────────────

  private emitModelLoaded(): void {
    if (!this.agent || !this.doc) return
    post({
      type: 'modelLoaded',
      doc: this.doc,
      diagnostics: this.agent.getDiagnostics(),
      board: Array.from(this.board),
      score: this.score,
    })
    this.emitStats(true)
  }

  private emitTrainingState(): void {
    post({
      type: 'trainingState',
      running: this.running,
      mode: this.mode,
      movesPerSec: this.movesPerSec,
      oneGameActive: this.oneGameActive,
    })
  }

  private emitBoardSync(): void {
    post({
      type: 'boardSync',
      gameIndex: this.games,
      moveIndex: this.gameMoves,
      board: Array.from(this.board),
      score: this.score,
      terminal: isGameOver(this.board),
    })
  }

  private emitSnapshot(): void {
    const now = performance.now()
    if (now - this.lastSnapshotEmit < SNAPSHOT_INTERVAL_MS) return
    this.lastSnapshotEmit = now
    post({
      type: 'snapshot',
      board: Array.from(this.board),
      score: this.score,
      gameIndex: this.games,
      values: this.lastValues,
      lastAction: this.lastAction,
    })
  }

  private emitStats(force: boolean, completedBucket?: import('../shared/types').StatBucket): void {
    const doc = this.doc
    const agent = this.agent
    if (!doc || !agent) return
    const now = performance.now()
    if (!force && now - this.lastStatsEmit < STATS_INTERVAL_MS) return
    this.lastStatsEmit = now
    const diag = agent.getDiagnostics()
    post({
      type: 'stats',
      live: {
        games: this.games,
        totalMoves: this.totalMoves,
        trainMs: this.trainMs,
        movesPerSec: this.running ? this.mpsEma : 0,
        gamesPerSec: this.running ? this.gamesPerSecEma : 0,
        currentScore: this.score,
        currentMaxExp: this.curMaxExp,
        bestScore: doc.bestScore,
        bestExp: doc.bestExp,
        recentAvg: this.recent.length
          ? this.recent.reduce((a, b) => a + b, 0) / this.recent.length
          : 0,
        paramCount: diag.paramCount,
        memoryBytes: diag.memoryBytes,
        learningRate: diag.learningRate,
        epsilon: diag.epsilon,
        meanAbsTdError: diag.meanAbsTdError,
        replayFill: diag.replayFill,
      },
      partialBucket: this.aggregator.partial(Date.now()),
      ...(completedBucket ? { completedBucket } : {}),
    })
  }
}

const trainer = new Trainer()
self.onmessage = (e: MessageEvent<WorkerCommand>) => {
  void trainer.handle(e.data)
}
post({ type: 'ready' })

