Simulation of tapered dendrites with branches, implementation of Yihe, Lu & Timofeeva, Yulia. (2020). Exact solutions to cable equations in branching neurons with tapering dendrites. Journal of Mathematical Neuroscience. 10. 10.1186/s13408-020-0078-z.

Project requirements are in `requirements.txt`

`main.py` runs a short simulation and logs output to tensorboard.

Run tensorboard with `tensorboard --logdir=runs`

Structure:
- src/config - Classes with configuration of parameters
- src/dendrites/dendrite_engine.py - Main engine
- src/dendrites/boundary - Strategy for boundary condition
- src/dendrites/forward - Strategy for forward (simulation step-by-step)
- src/dendrites/voltage_cache_tables - Optimization to allow parallel execution in pytorch
- tests/ - Tests for this project
- 