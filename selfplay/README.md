# selfplay/

`generate.py` produces imitation-learning training data by sampling moves
directly from igo-app's real KataGo-derived checkpoint's own policy output
(no tree search) — see its docstring for why that's the right call here
(distillation from an already-strong teacher, not reinforcement-learning
self-play) and `../docs/ROADMAP.md`'s Phase 2.

Fast: ~20ms/move on this machine (one net evaluation per move), so a few
hundred games is a few minutes' work, not hours.

The self-play generation + policy/value update loop for fine-tuning
Ray-zeroGo *past* its imitation-learning starting point (`../docs/ROADMAP.md`'s
Phase 3) is separate, later work, not this file.
