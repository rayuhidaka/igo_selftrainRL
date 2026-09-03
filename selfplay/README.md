# selfplay/

`generate.py` produces imitation-learning training data by sampling moves
directly from igo-app's real KataGo-derived checkpoint's own policy output
(no tree search) — see its docstring for why that's the right call here
(distillation from an already-strong teacher, not reinforcement-learning
self-play) and `../docs/ROADMAP.md`'s Phase 2.

Fast: ~20ms/move on this machine (one net evaluation per move), so a few
hundred games is a few minutes' work, not hours.

`self_play.py` is that separate, later work (`../docs/ROADMAP.md`'s
Phase 3): genuine self-play using real MCTS search (`../mcts/mcts.py`)
with Ray-zeroGo's own current-best checkpoint, not KataGo's. The
recorded policy target is the search's visit-count distribution, not a
raw net output. Much more expensive per move than `generate.py` (a full
search per move instead of one net evaluation) -- expect real seconds
per move, not milliseconds. Produces the same
`bootstrap.dataset.SelfPlayExamples` format `generate.py` does, so
`bootstrap/train.py` trains on either's output unmodified. The policy/value
update loop itself is just `bootstrap/train.py` again, pointed at
`self_play.py`'s output — no separate training code needed.
