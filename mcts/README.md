# mcts/

A Python port of `igo-app/mcts/`'s PUCT search driver (Kotlin), for
`eval/match.py` to actually play moves during checkpoint evaluation
matches with (see `../docs/ARCHITECTURE.md`'s "Difficulty-tier
promotion" section for why: without search, both checkpoints would play
uniformly-random moves and Elo differences between similar-strength
nets wouldn't be measurable). Same rationale as `../engine/`'s port —
avoid a second, independently-written search implementation silently
disagreeing with the first.

| Kotlin (`igo-app/mcts/`) | Python (here)          |
|---------------------------|--------------------------|
| `PolicyValueNet.kt`        | `policy_value_net.py`   |
| `Mcts.kt`                  | `mcts.py`                |

Proven against a full port of `igo-app/mcts/`'s own test suite — see
`../tests/test_mcts.py`.

`bootstrap/inference.py`'s `RayZeroPolicyValueNet` implements this
module's `PolicyValueNet` protocol against a `RayZeroNet` PyTorch
checkpoint, the same way `igo-app/inference/TfLitePolicyValueNet.kt`
implements the Kotlin `PolicyValueNet` interface against a `.tflite`
checkpoint.

**If you change the search, change it on both sides** — same caveat as
`../engine/README.md`.
