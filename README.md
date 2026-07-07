# 2048 AI Lab

Watch real reinforcement-learning agents learn to play 2048 from scratch — in
your browser, with no hardcoded strategy. A fresh model flails at random, then
genuinely improves and stays improved. Everything a nerd could want to watch
while an AI gets good is on screen: live per-direction value estimates, a
learning curve, a max-tile histogram, TD-error and rate charts, and a
first-tile milestone ribbon.

## What makes it honest

There are **no board-quality heuristics anywhere** in the agents. A move is
chosen only from learned parameters plus the game's legality rules. A brand-new
model provably plays at the random baseline (~1,090 avg score); the improvement
you watch is entirely learned.

## Two architectures (pick at model creation)

- **N-Tuple Network + TD(0) afterstate learning** — "the fast learner"
  (Szubert & Jaśkowski 2014). Millions of pattern weights, temporal-difference
  updates, Temporal Coherence adaptive learning rates, optional optimistic
  init, and optional expectimax planning depth (which still uses *only* the
  learned value function). Presets: Starter 17×4-tuple (4.5 MB), Balanced
  8×5-tuple (33 MB), Expert 4×6-tuple (268 MB — the literature config that
  masters the game). In turbo it hits the 2048 tile within a couple hundred
  games and the 8192 tile within minutes.
- **Neural Network (DQN)** — "the slow real thing". A from-scratch, hand-rolled
  MLP (256→hidden→4) trained with experience replay, a target network, Huber
  loss, and an ε-greedy schedule. Learns slowly over hours — the honest
  neural-net experience.

## Training modes

- **Turbo** — max-speed headless training in a Web Worker (hundreds of
  thousands of moves/sec for the n-tuple), board snapshotted to the UI a few
  times a second. Leave it running for weeks; stats stay fast via per-100-game
  aggregate buckets.
- **Watch** — animated self-play at an adjustable speed, with the original
  2048 slide/merge/spawn feel and the live direction-value visualization.
- **Grade** — the AI plays one move at a time and you grade each move
  Good / Neutral / Bad (keys 1 / 2 / 3). Your grade is a real, persistent
  update to the model's value function.

## Model management

New · Save · Load · Fork · Rename · Delete, plus file **Export/Import**
(`.2048model` binary container). Everything persists in IndexedDB — weights as
`ArrayBuffer` blobs, full stat history, milestones. Training auto-checkpoints
and survives a tab refresh (it offers to resume).

## Run it

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # engine equivalence, TD-learning, DQN grad-check, export roundtrip
npm run build    # type-check + production build
```

## How it's built

- **Vite + React + TypeScript + Tailwind v4**, dark-mode-first.
- **Engine**: a single canonical `slideLine` rule; a 65,536-entry row LUT for
  the fast headless engine and a tile-identity trace engine for animation, both
  proven behaviorally identical over *all* 65,536 possible rows by test.
- **Training** runs entirely in a Web Worker; the UI is a pure renderer of the
  worker's events. The worker owns the game loop in every mode.
- **Charts** via uPlot (streams long histories at 60fps).

See `src/agents/` for the learning algorithms, `src/engine/` for the game,
`src/worker/` for the training loop and protocol, and `tests/` for the proofs.
