# Roadmap — igo-training

This repo's own phase numbering — distinct from `igo-app`'s, which only
tracks the handoff points that matter to the app (see its Phase 2-4).

**Training runs locally, on the user's own machine — not a rented/cloud
GPU.** How to actually work within that (phased training runs, batch
scheduling, session length) is still to be decided — revisit when Phase
2 starts in earnest.

## Phase 1 (current) — scaffolding
- [x] Repo/folder structure in place (`bootstrap/`, `selfplay/`,
      `export/`, `eval/`, `configs/`)
- [x] Export contract confirmed against `igo-app/docs/MODEL_CONTRACT.md`
      (see `docs/ARCHITECTURE.md`)
- [x] Placeholder/tiny checkpoint exported end-to-end through the
      pipeline just to validate the `igo-app` loading path works — an
      untrained net (`bootstrap/train.py` → `export/to_tflite.py`),
      same idea as `igo-app/tools/build_placeholder_net.py` but produced
      via this repo's actual PyTorch → ONNX → TFLite path. Verified: the
      resulting `.tflite`'s tensors are exactly `board_planes [1,9,9,3]`
      in, `policy [1,82]` / `value [1,1]` out, correctly named — see
      `docs/ARCHITECTURE.md`'s `export/` section for what it took to get
      there. Not yet verified loading inside `igo-app`/on-device
      specifically (only Python-side, same as this pipeline's own
      checks) — should behave identically to the expert-tier checkpoint
      there, since both conform to the same contract, but hasn't been
      separately confirmed.

## Phase 2 — Ray-zeroGo bootstrap
- [x] Collect KataGo self-play games (`selfplay/generate.py`) — samples
      moves directly from igo-app's real KataGo checkpoint's own policy
      output (no tree search needed: it's already strong, this is
      distillation, not reinforcement-learning self-play), ~20ms/move on
      this machine. Public 9x9 game records as an alternative/additional
      source not pursued yet.
- [x] Imitation-learning training loop (`bootstrap/train.py`) — real
      gradient descent now (standard AlphaZero-style loss: policy
      cross-entropy + value MSE against `selfplay/generate.py`'s data),
      not the Phase 1 placeholder. TensorBoard logging
      (`bootstrap/monitoring.py`) and checkpoint-interval saving wired
      in. Smoke-tested end to end (3 games → 214 examples → a real
      training run, loss decreasing) before the first real batch.
- [x] Produce Ray-zeroGo's first bootstrapped checkpoint — first real run:
      300 self-play games → 22,621 examples (~7.6 min), then training
      capped at 20 minutes (`configs/bootstrap_train_batch1.yaml`).
      Policy loss dropped from ~4.4 (the exact value a uniform-random
      82-way guess scores, `ln(82)`) to ~2.3 -- genuine learning, not
      noise. Two things to know about this specific checkpoint before
      reading too much into it:
      - It saw the same fixed 22,621-example batch roughly 780 times
        over (a lot of steps, not much data) -- almost certainly
        overfit/memorized rather than generalized. Proves the loop
        learns; isn't a meaningfully strong net. More self-play data
        (not more epochs on the same data) is the fix for a real
        attempt.
      - A `torch.save` call failed once near the very end of the run
        (`RuntimeError: File ... cannot be opened`) -- looked like a
        transient issue writing to `/mnt/c` (the Windows-mounted
        project path) under heavy repeated I/O (~1300 checkpoint saves
        over the run), not a logic bug; a checkpoint from shortly before
        the failure survived and loads fine.
      - **Hardened (2026-09-03):** `checkpoints/`, `runs/`, and
        `selfplay_games/` are now symlinks (from the repo, still on
        `/mnt/c`) to real directories on native WSL storage
        (`~/igo_training_runs/...`), same fix as the venv itself (see
        `docs/BUILD_NOTES.md`-equivalent notes in igo-app) — repo-relative
        config paths (`checkpoints/foo.pt`, `log_dir: runs`, etc.) work
        unchanged. `.gitignore` entries for these three switched from
        `dir/` to `dir` (no trailing slash), since a trailing-slash
        pattern only matches a real directory, not a symlink to one.
        `bootstrap/train.py`'s checkpoint saves also now retry (3
        attempts, 2s apart) on `RuntimeError` before giving up. Verified:
        full unit test suite (43 tests) and a real smoke-test training
        run both pass with the new symlinked paths.
- [x] **First real (non-overfit) bootstrap attempt (2026-09-03):** 10x
      batch1's scale. `configs/selfplay_generate_batch2.yaml` generated
      3,000 self-play games → 224,730 examples (~78 min) to
      `selfplay_games/batch2.npz`; `configs/bootstrap_train_batch2.yaml`
      trained 50 epochs over the full set (175,550 steps, ~12.6 min, no
      save failures — the hardening above held up) to
      `checkpoints/bootstrap_batch2.pt`. Total loss dropped from ~5.3
      (near the uniform-random baseline) to ~3.3 by epoch 25, then
      plateaued/noisy around 3.1-3.6 through epoch 50 rather than
      continuing to fall — unlike batch1's near-800-epoch run on a fixed
      small batch, this looks like the tiny 2-conv-layer net (see
      `bootstrap/model.py`) hitting its capacity ceiling on real data,
      not memorization. This is now a genuine (if weak, architecture-
      limited) bootstrapped checkpoint, not just a proof the loop learns.
- [x] **Evaluated batch2 for actual playing strength (2026-09-03):**
      `configs/eval_batch2_vs_batch1.yaml` and
      `configs/eval_batch2_vs_untrained.yaml` (40 games each,
      50 simulations/move — a 200-simulation timing test took ~101s/game,
      too slow for a meaningful sample size at similar total compute) via
      `eval/promote.py`. Result: `bootstrap_batch2.pt` won **all 40
      games** against both `bootstrap_batch1.pt` and a fresh untrained
      checkpoint (1726.1 vs 1273.9 in both matches — identical numbers
      because Elo's update only depends on the win/loss sequence, not
      opponent identity, and both were 40-0 shutouts). Confirms two
      things: batch2 is a real, working player (clearly beats random),
      and batch1 was exactly as overfit as suspected — it plays no
      better than an untrained net once put in an actual game, despite
      its training loss having dropped much further than batch2's.
- [x] **Exported batch2 to `.tflite` (2026-09-03):** `python -m
      export.to_tflite --checkpoint checkpoints/bootstrap_batch2.pt
      --board-size 9 --out export/ray_zero_batch2.tflite` — first time
      this pipeline ran against a real trained checkpoint rather than an
      untrained placeholder. Tensors match `igo-app/docs/MODEL_CONTRACT.md`
      exactly (`board_planes [1,9,9,3]` in, `policy [1,82]`/`value [1,1]`
      out, correctly named). Also numerically verified this time (the
      placeholder pass only checked shapes): fed one real board position
      through both the PyTorch checkpoint and the exported `.tflite`,
      max policy difference ~1e-7 and value difference ~9e-8 — float32
      rounding noise, not a conversion bug. `export/*.tflite` isn't
      committed (gitignored, same as `igo-app`'s own model asset) —
      regenerate from `checkpoints/bootstrap_batch2.pt` via the command
      above. Not yet copied into `igo-app`'s assets or wired into a
      difficulty picker — that's a real product decision (this
      checkpoint is weak/architecture-limited, see above) better made
      once Phase 3's self-play fine-tuning has actually run, not on this
      bootstrap checkpoint alone.
- [x] **Widened the net and confirmed it was capacity-limited
      (2026-09-03):** batch2's loss plateau (above) was tested as a
      capacity ceiling, not a data ceiling, by widening
      `bootstrap/model.py`'s `RayZeroNet` from 2 conv layers/32 channels
      to 3 layers/64 channels and retraining on the *same* `batch2.npz`
      (`configs/bootstrap_train_batch2_wide.yaml`, no new self-play
      needed) → `checkpoints/bootstrap_batch2_wide.pt`. Loss kept falling
      well past where the old net plateaued (~2.60 final vs. the old
      net's ~3.1-3.6 floor, still trending down at epoch 50 — likely not
      capacity-limited itself yet). Confirmed this was a real strength
      gain, not just a lower loss number: evaluated
      (`configs/eval_batch2wide_vs_batch2.yaml`, same 40-games/50-sims
      settings) and it beat `bootstrap_batch2.pt` **40-0** too.
      - Widening broke loading old checkpoints (`RayZeroPolicyValueNet`
        always built the *current* architecture before calling
        `load_state_dict`, so the old 2-layer checkpoint's missing
        `conv3` key crashed it) — fixed by making `channels` and
        `num_conv_layers` explicit, overridable parameters threaded
        through `RayZeroNet` → `RayZeroPolicyValueNet` →
        `eval/match.py`'s `play_match` → `eval/promote.py`'s config
        (`current_tier_channels`/`current_tier_num_conv_layers`, etc.),
        so checkpoints from before/after an architecture change can
        still be compared without retraining anything. This is a
        workaround, not the real fix `CLAUDE.md`'s own "every checkpoint
        is versioned and logged with: architecture config..." convention
        calls for — checkpoints still don't self-describe their own
        architecture (just a raw `state_dict`), so the caller has to
        already know it. Worth fixing properly (save a small metadata
        dict alongside the `state_dict`) before this bites again on the
        next architecture change.
- [x] **Checkpoints self-describe their architecture (2026-09-03):** the
      real fix flagged above. New module `bootstrap/checkpoint.py`
      (`CheckpointMetadata`, `save_checkpoint`/`load_checkpoint`,
      `build_model`) wraps a `state_dict` with `channels`,
      `num_conv_layers`, `data_source`, `seed`, and `elo` (`None` until
      an `eval/` run estimates one). `bootstrap/train.py`,
      `bootstrap/inference.py`'s `RayZeroPolicyValueNet`, and
      `export/to_tflite.py` all go through it now — a checkpoint saved
      from here on gets its architecture right automatically, no manual
      `channels`/`num_conv_layers` needed (verified: exported
      `bootstrap_smoke_test.pt` and played it in `eval/match.py` against
      a legacy checkpoint with zero manual overrides on the new
      checkpoint's side). `channels`/`num_conv_layers` also moved out of
      `bootstrap/model.py`'s hardcoded defaults into `bootstrap/train.py`
      config keys, per `CLAUDE.md`'s "hyperparameters live in config
      files" convention. Old checkpoints (`bootstrap_batch1/2/2_wide.pt`,
      etc.) are untouched, bare `state_dict`s — they keep loading via the
      legacy path (`metadata=None`), still needing
      `channels`/`num_conv_layers` supplied explicitly the same way as
      before (`eval/promote.py`'s `current_tier_channels`/
      `current_tier_num_conv_layers` config keys). 5 new tests
      (`tests/test_checkpoint.py`), full suite (48 tests) passing.
- [x] **Confirmed the widened net's own ceiling (2026-09-03):**
      `bootstrap_batch2_wide.pt` was only trained 50 epochs and was still
      trending down at the end, so its loss plateau wasn't confirmed yet.
      `configs/bootstrap_train_batch2_wide_long.yaml` — same architecture
      and data, 141 epochs (495,317 steps, hit the 90-min cap) → first
      checkpoint through `bootstrap/checkpoint.py`'s new metadata format
      end to end (`bootstrap_batch2_wide_long.pt`). Loss dropped to
      ~2.6-2.8 by roughly epoch 20 and then stayed there, noisily, for
      the remaining ~120 epochs (3x longer training, no further real
      improvement) — a genuine plateau, not just an under-trained run
      like the 50-epoch checkpoint looked. 3 conv layers/64 channels is
      now a confirmed capacity ceiling on this data, not a "just needed
      more epochs" situation. Next real architecture step (not yet
      done): a small residual tower (a few residual blocks, still
      64-96 channels) — the standard, low-export-risk next step for
      small-board Go nets, preferred over jumping straight to something
      more exotic (attention/transformer-style blocks, which this
      repo's own KataGo-conversion notes already flag as fragile through
      `onnx2tf`).
- [x] **Residual tower implemented and confirmed stronger (2026-09-03):**
      `bootstrap/model.py`'s `RayZeroNet` gained an opt-in residual path
      (`num_residual_blocks > 0`: a stem conv + `ResidualBlock`s, each
      two 3x3 convs with BatchNorm and a skip connection) alongside the
      existing plain-stack path, so every prior checkpoint stays
      loadable unchanged. Trained 4 blocks/64 channels on the same
      `batch2.npz` (`configs/bootstrap_train_batch2_residual.yaml`,
      first real checkpoint using the residual path):
      41 epochs/145,820 steps, hit the same 90-min cap as the plain-stack
      comparison run. Loss fell the *entire* run — 3.34 → 2.52 → 2.41 →
      2.27 → 2.18 → **2.11**, still trending down when time ran out —
      qualitatively different from the plain stack's hard plateau at
      ~2.6-2.8. Confirmed as real strength, not just lower loss:
      evaluated against `bootstrap_batch2_wide_long.pt` (the strongest
      plain-stack checkpoint, `configs/eval_residual_vs_wide_long.yaml`,
      same 40-games/50-sims settings) and won **40-0**. Both checkpoints
      self-described their architecture via `bootstrap/checkpoint.py` —
      no manual `channels`/`num_conv_layers`/`num_residual_blocks`
      overrides needed for either side, first real use of that machinery
      paying off. Not yet done: exporting this checkpoint to `.tflite`,
      and since it was still improving (not plateaued) when the time cap
      hit, a longer run and/or more residual blocks is worth trying
      before concluding this is *this* architecture's ceiling too.

## Phase 3 — Self-play fine-tuning
- [x] **Self-play generation loop (2026-09-03):** `selfplay/self_play.py`
      -- genuine AlphaZero-style self-play via real MCTS search
      (`mcts/mcts.py`), unlike Phase 2's `generate.py` (samples straight
      from KataGo's policy, no search). Records the search's visit-count
      distribution as the policy target, not a raw net output. Same
      `SelfPlayExamples` format `generate.py` produces, so
      `bootstrap/train.py` needed no changes to train on it. Much more
      expensive per move than distillation (a full search per move) but
      still cheap in practice: a 10-game/100-sim timing calibration
      averaged ~17.8s/game (~0.25s/move); the first real batch (100
      games, `configs/selfplay_self_play_gen1.yaml`) took ~34 min for
      7,879 examples.
- [x] **Policy/value update loop (2026-09-03):** no new training code
      needed -- `bootstrap/train.py` already consumed the right data
      format. What *was* needed: warm-starting. Every run before this
      built a fresh, randomly-initialized network, which would have
      made a self-play fine-tune meaningless (training from scratch on
      ~8k examples measures "far less data than Phase 2," not whether
      self-play helps). New `init_from_checkpoint` config key continues
      training an existing checkpoint's weights instead
      (`CheckpointMetadata` gained a matching field to record the
      lineage). 4 new tests (`tests/test_train.py`).
- [x] **First full generation cycle, proven end to end (2026-09-03):**
      self-played 100 games with `bootstrap_batch2_residual.pt` →
      fine-tuned a candidate from it (warm start, 10 conservative epochs,
      5x lower LR than the original bootstrap runs, deliberately
      cautious given the small batch) →
      evaluated the candidate against its own parent
      (`configs/eval_gen1_candidate_vs_residual.yaml`, same
      40-games/50-sims settings as prior comparisons, **both sides
      playing with their own equal-budget MCTS search, not raw
      policy**) → candidate won **40-0** (1726.1 vs 1273.9). Meaningful
      because both sides searched equally hard — the win shows
      fine-tuning toward the search-refined policy made the *network
      itself* stronger, which is the actual mechanism Phase 3 is
      supposed to deliver, not just "more search wins." One caveat: a
      single generation on one fairly small self-play batch is real
      signal, not yet proof this compounds over multiple generations —
      that's the natural next step, not yet done.
- [x] **Caught and guarded against a self-play collapse (2026-09-03):**
      generating generation 2's self-play data (with
      `bootstrap_gen1_candidate.pt`) produced ~30/100 games ending in
      2-16 moves — both players passing on a near-empty board, White
      "winning" purely from komi. Didn't exist in generation 1's
      self-play (1/100 short games, using the pre-fine-tune checkpoint).
      Diagnosed: fine-tuning flattened the policy enough (empty-board
      top-move confidence 16.5% → 6.6%) that Pass picked up a small but
      nonzero prior (0.01% → 1.7%) — small in isolation, but enough that
      a 100-simulation search budget can run away with it in a single
      tree search once the value net (which has barely seen post-pass
      positions) misjudges that branch. Training on these games would
      have taught the exact behavior that produced them, worse each
      generation — a real self-play death spiral, not just noisy data.
      Fixed with `selfplay/self_play.py`'s new `min_moves_to_keep`
      config key: discards any game shorter than the threshold before
      it reaches the saved dataset. 20 moves cleanly separates every
      collapsed game (max 16) from every legitimate one seen so far
      (min 42, across both generations' self-play).
- [x] **Root-caused the collapse further and hardened against it
      (2026-09-03) — full detail in `docs/SELF_PLAY_STABILITY.md`, this
      is the summary:** filtering degenerate games wasn't sufficient —
      fine-tuning on the *cleaned* data still pushed empty-board Pass
      probability to 24.9%. Four fine-tune attempts (varying only mix
      ratio and step count, mixing in the broad `batch2.npz` alongside
      new self-play data via `bootstrap/dataset.py`'s new
      `build_training_loader`) found a 25%-anchor/75%-self-play ratio,
      step count matched to the original run's self-play exposure, was
      both a real win and stable — 50/50 was either a wash or,
      surprisingly, *worse* with more steps. But chaining that same
      recipe into a second warm-started generation still collapsed
      (34.5% Pass) — the real cause is multiplicative bias compounding
      across chained warm-starts, not the ratio itself. Separately,
      research into *why* pure win/loss/tie value targets are
      implicated (AlphaZero's own deliberate choice, but with a known
      flat-near-saturation failure mode) led to adding a KataGo-style
      auxiliary score-margin head (`bootstrap/model.py`'s
      `has_score_head`, training-only, `export/to_tflite.py` strips it)
      — implemented and verified end to end. **Evaluated against the
      actual chaining collapse and it did not fix it**: chaining the
      score-head-equipped recipe into a second generation still
      collapsed (1.46%→27.04% empty-board Pass probability, nearly the
      same magnitude as without the head, 1.22%→34.5%). Useful negative
      result — rules out value-target flatness as the chaining
      collapse's primary cause. Keeping the score head (real signal, no
      downside) but it isn't a fix for this failure mode. Implemented
      the warm-start-chaining restructuring next (fine-tune from the
      pristine checkpoint each generation, mixing in all accumulated
      self-play data — needed no new code, `bootstrap/dataset.py`'s
      existing weighted multi-source mixing already supports it):
      **partial fix.** Empty-board Pass probability came down to 16.09%
      (vs. 27-34% chained), but the checkpoint was no longer a clear
      strength win either (1491.8 vs 1508.2 vs. the pristine baseline,
      "Do not promote" — a statistical wash, not the decisive 40-0 win
      every single-generation fine-tune produced). Trading one problem
      for another, not a full solution. Full detail and live next-step
      options in `docs/SELF_PLAY_STABILITY.md` section 9 — pausing here
      to decide direction (harden self-play data quality further,
      retune the mix, move to a true continuous replay-buffer loop, or
      accept the single-generation result as Phase 3's current
      deliverable) rather than keep varying parameters blindly.
- [x] **Root-caused and fixed the chaining collapse for real (2026-09-04
      into 2026-09-05) — full detail in `docs/SELF_PLAY_STABILITY.md`
      sections 11-20, this is the summary:** the earlier filtering-only
      approach (above) revealed the bias was pervasive (89/100 games
      affected), not just rare bad luck — filtering *output* had hit
      diminishing returns. Root cause: `mcts/mcts.py` had no root
      exploration noise and a low simulation budget, a known AlphaZero
      pathology. Fixed in stages: root Dirichlet noise excluding `Pass`
      from the noise draw (naively including it made things *worse*,
      not better — noise landing on `Pass` sent real search budget down
      the poorly-calibrated post-pass branch), `alpha=0.1` tuned for
      9x9's branching factor (not AlphaZero's 19x19 value of 0.03),
      temperature annealing after move 30 (fixed a separate fine-tune
      quality wash this uncovered), and finally a structural
      `no_pass_before_move=20` guard — the fix that actually closed the
      remaining chaining gap. Also fixed two unrelated but significant
      bugs found along the way: an ~85x self-play performance bug
      (PyTorch thread oversubscription) and an eval-methodology bug
      (`eval/promote.py`'s "40-game matches" were actually only 2 unique
      deterministic games repeated 20x each, undermining the
      statistical confidence of every promotion decision in the
      project's history). **Final result: both generation 1 and a real
      chained generation 2 came back 0/100 short/collapsed games, and
      generation 2 is a confirmed genuine strength win over both the
      pristine baseline and its own generation-1 parent** —
      `bootstrap_gen2_no_pass_guard_candidate.pt` was the best checkpoint
      at that point. The healthy AlphaZero-style improvement loop this
      phase was always meant to produce is now actually working end to
      end.
- [x] **Generation 3, for extra confidence — the pattern holds a third
      time (2026-09-05):** chained the same recipe one more generation
      from `bootstrap_gen2_no_pass_guard_candidate.pt` — **0/100 short
      games** (third consecutive clean generation).
      `bootstrap_gen3_no_pass_guard_candidate.pt` beats both the pristine
      baseline (1597.9 vs. 1402.1) and its own gen2 parent (1582.7 vs.
      1417.3), both PROMOTE. Three chained generations, each a genuine
      strength win, zero collapse at every step. See
      `docs/SELF_PLAY_STABILITY.md` section 21.
- [x] **Generation 4, widening igo-app's difficulty spread (2026-09-06):**
      chained one more generation from `bootstrap_gen3_no_pass_guard_candidate.pt`
      — **0/100 short games** (fourth consecutive clean generation), 9,748
      examples. `bootstrap_gen4_no_pass_guard_candidate.pt` beats both the
      pristine baseline (1614.2 vs. 1385.8) and its own gen3 parent
      (1552.3 vs. 1447.7), both PROMOTE. **This is now the best checkpoint
      overall** — four chained generations, each a genuine strength win,
      zero collapse at every step. See `docs/SELF_PLAY_STABILITY.md`
      section 22. Exported and verified on-device (Phase 4 below), added
      to igo-app as a third difficulty tier.
- [x] **Generation 5, and parallelizing self-play (2026-09-06):** targeting
      generation 10 per the user's request, but generation 4's self-play
      alone took ~180 minutes serial despite 16 CPU cores being available
      (`self_play.py` is deliberately single-threaded per process). New
      `selfplay/run_parallel.py` splits self-play across 4 worker
      subprocesses, then merges their output — same recipe, ~3.3x faster
      wall time. Chained from `bootstrap_gen4_no_pass_guard_candidate.pt`
      — **0/100 short games**, 9,810 examples, 55 minutes wall time (vs.
      gen4's 180).  `bootstrap_gen5_no_pass_guard_candidate.pt` beats
      both the pristine baseline (1601.8 vs. 1398.2) and its own gen4
      parent (1557.4 vs. 1442.6), both PROMOTE. **This is now the best
      checkpoint overall** — five chained generations, each a genuine
      strength win, zero collapse at every step. See
      `docs/SELF_PLAY_STABILITY.md` section 23. Not yet exported to
      igo-app — holding off on exporting every single generation to
      avoid churning the app's difficulty tiers mid-chain; revisiting
      once the chain reaches a natural stopping point near generation 10.
- [x] **Generation 5's plateau resolved — chain resumed (2026-09-06):**
      three independent 100-game generation-6 attempts (two self-play
      samples, two fine-tune data mixes) all beat the pristine baseline
      but failed to promote against gen5. A root-cause investigation
      ranked self-play data volume per generation as the most likely
      contributor (established 9x9 self-play replications use 20-50x
      this project's 100 games/generation), and ruled out the fine-tune
      budget as under-training. A fourth attempt tripling `num_games` to
      300 (keeping the replay-buffer fine-tune mix) resolved it
      decisively on the first try: **0/300 short games**, PROMOTE vs.
      gen5 (1556.8 vs. 1443.2) and vs. the pristine baseline (1655.2 vs.
      1344.8). `bootstrap_gen6_no_pass_guard_candidate.pt` is the new
      best checkpoint — six chained generations overall. See
      `docs/SELF_PLAY_STABILITY.md` section 24 for the full
      investigation and numbers. **New standing recipe going forward:**
      300 self-play games/generation (not 100), a replay-buffer
      fine-tune mix (small legacy anchor + recent generations' self-play,
      favoring the newest), and an 80-game eval against the immediate
      parent (40 remains fine for the baseline comparison).
- [x] **Generation 7, first full cycle under the new recipe (2026-09-06):**
      chained from `bootstrap_gen6_no_pass_guard_candidate.pt`, 300
      self-play games, replay-buffer fine-tune mix. **0/300 short games**,
      30,727 examples. PROMOTE vs. gen6 (1571.9 vs. 1428.1) and vs. the
      pristine baseline (1605.8 vs. 1394.2) — a clean promote on the
      first attempt, confirming the recipe isn't a one-off fix. New best
      checkpoint, seven chained generations overall. See
      `docs/SELF_PLAY_STABILITY.md` section 25.
- [x] **Generation 8, resolved on retry (2026-09-06):** attempt 1 did not
      promote against gen7 (38.2-Elo gap) but still beat the pristine
      baseline decisively — read as ordinary self-play variance, not a
      new plateau. A retry with a fresh seed confirmed that: **0/300
      short games**, PROMOTE vs. gen7 (1565.5 vs. 1434.5) and vs. the
      pristine baseline (1694.4 vs. 1305.6 — the most decisive baseline
      win yet). New best checkpoint, eight chained generations overall.
      See `docs/SELF_PLAY_STABILITY.md` section 26.
- [x] Save checkpoints at intervals — these become candidate difficulty
      tiers, gated on Elo (`eval/`). Three fine-tuned generations are
      exported and on-device in igo-app so far (gen2/gen3/gen4); gen5 and
      beyond exist as checkpoints but aren't exported yet (see above).
- [x] Elo rating math (`eval/elo.py`) and the promotion gate
      (`should_promote`) — fully implemented and tested
      (`tests/test_elo.py`). See `docs/ARCHITECTURE.md`'s "Difficulty-tier
      promotion" section for the design.
- [x] `engine/` — Python port of `igo-app/engine/`'s Go rules (rules
      enforcement: legality, capture/ko, area scoring). Proven against a
      full port of `igo-app/engine/`'s own test suite
      (`tests/test_position.py`, `tests/test_scoring.py`, 20 tests, all
      passing) — not just eyeballed.
- [x] `mcts/` — Python port of `igo-app/mcts/`'s PUCT search, and
      `bootstrap/inference.py`'s `RayZeroPolicyValueNet` (mirrors
      `igo-app/inference/TfLitePolicyValueNet.kt`'s encode/decode).
      Proven against ports of both their test suites (`tests/test_mcts.py`,
      `tests/test_inference.py`).
- [x] Actually playing games between checkpoints (`eval/match.py`) — done,
      using `mcts/` + `engine/`. Verified two ways: a symmetry test
      (`tests/test_match.py`, a checkpoint played against itself averages
      to ~50/50) and a real CLI run (`configs/eval_smoke_test.yaml`, two
      untrained 9x9 checkpoints, 4 games via `python -m eval.promote`,
      produced a real Elo verdict in ~23s). `eval/`'s whole pipeline
      (`elo.py` → `match.py` → `promote.py`) is now implemented end to
      end — what's missing is real checkpoints worth evaluating, i.e.
      Phase 2 actually happening.

## Phase 4 — Handoff to igo-app
- [x] **First real self-trained checkpoint exported (2026-09-05):**
      `python -m export.to_tflite --checkpoint checkpoints/bootstrap_gen2_no_pass_guard_candidate.pt
      --board-size 9 --out export/ray_zero_gen2_no_pass_guard.tflite` —
      tensors match `igo-app/docs/MODEL_CONTRACT.md` exactly
      (`board_planes [1,9,9,3]` in, `policy [1,82]`/`value [1,1]` out).
      Numerically verified against the source PyTorch checkpoint: max
      policy difference 0.000145 (0.0145%), value difference 2.4e-7 —
      float32 rounding noise, not a conversion bug. First time this
      pipeline has exported an actually self-trained, self-play-improved
      Ray-zeroGo checkpoint (not just an untrained placeholder or a
      Phase 2 imitation-only bootstrap) — see `docs/SELF_PLAY_STABILITY.md`
      for how this checkpoint was produced. Not yet copied into
      `igo-app`'s assets, loaded on-device, or wired into a difficulty
      picker — see below.
- [x] **Verified on-device (2026-09-05):** copied
      `ray_zero_gen2_no_pass_guard.tflite` into
      `igo-app/app/src/main/assets/models/`, wrote
      `app/src/androidTest/kotlin/com/igoapp/app/RayZeroModelOnDeviceTest.kt`
      (same pattern as the existing `KatagoModelOnDeviceTest`, but
      without a specific-best-move assertion -- Ray-zeroGo is an early,
      two-generations-in checkpoint, not a professional-strength engine,
      so asserting a specific move would be presumptuous; this only
      confirms the checkpoint loads and produces well-formed output).
      Ran via
      `.\gradlew.bat :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.igoapp.app.RayZeroModelOnDeviceTest`
      on the `MinSdk_API24` emulator (see `docs/BUILD_NOTES.md`): 1 test,
      0 failures, ran in 0.103s. First real self-play-trained Ray-zeroGo
      checkpoint confirmed working under Android's actual on-device
      TFLite runtime, not just desktop/WSL Python.
- [x] **Exported and verified `bootstrap_gen3_no_pass_guard_candidate.pt`
      on-device (2026-09-06):** `export/ray_zero_gen3_no_pass_guard.tflite`,
      numerically verified against the source checkpoint (max policy diff
      ~2.2e-7, value diff ~3.6e-7 — float32 rounding noise). Copied into
      `igo-app/app/src/main/assets/models/` alongside gen2 (kept as
      separate assets — both are intended as difficulty tiers, not a
      replacement of one by the other). `RayZeroModelOnDeviceTest`
      extended to cover both checkpoints: 2 tests, 0 failures on
      `MinSdk_API24`.
- [x] **Exported and verified `bootstrap_gen4_no_pass_guard_candidate.pt`
      on-device (2026-09-06):** `export/ray_zero_gen4_no_pass_guard.tflite`,
      numerically verified against the source checkpoint. Copied into
      `igo-app/app/src/main/assets/models/` alongside gen2/gen3.
      `RayZeroModelOnDeviceTest` extended to cover all three checkpoints.
- [x] Hand off checkpoints to the app for the difficulty picker — gen2,
      gen3, and gen4 are all exported, on-device, and wired into
      igo-app's "Play vs Ray-zeroGo" difficulty picker (see
      `igo-app/docs/ROADMAP.md` Phase 4).
- [ ] Optional Elo-over-generations chart in-app — not started; Elo
      numbers exist per-eval (see Phase 3 above) but aren't surfaced
      anywhere in igo-app's UI.
