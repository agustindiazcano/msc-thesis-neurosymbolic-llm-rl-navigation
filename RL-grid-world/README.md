# VLNN vs MLP/DQN — Neurosymbolic Testbed for RL and Navigation

Experimental prototype used as a testbed for a thesis on **neurosymbolic AI applied to LLM/RL and robotic navigation**. It includes two independent experiments:

1. **`benchmark_geometries.py`** (static classification, `make_moons`): compares a classic MLP against variants of a geometric-prototype network (RBF-like) using different activation geometries.
2. **`vlnn_rl_navigation.py`** (RL + navigation with obstacles): compares a classic DQN (PyTorch) against a Q-Learning agent built on a dynamic geometric neuron network (VLNN), plus a symbolic layer and an explicit physics-constraint layer.

Author: Agustín Díaz Cano — ORCID: [0009-0001-4336-490X](https://orcid.org/0009-0001-4336-490X)

## Motivation

The goal is not just to measure accuracy/win-rate, but to contrast two paradigms of knowledge representation:

- **MLP/DQN**: approximation via distributed weights, tuned by gradient descent over a dense network. All the "knowledge" lives inside an opaque weight matrix, with no explicit or inspectable mechanism.
- **VLNN (Volumetric Logic/Cognitive Network)**: explicit representation of the state space through geometric prototypes (spheres/RBFs) with their own lifecycle (they are born, mutate via mitosis, merge, and die), plus a symbolic layer that compresses repeated neuron clusters into reusable "concepts" (translation-invariant geometric hashing), plus an explicit physics-constraint layer (action masking) that blocks invalid actions without the agent having to learn them through trial and error.

This second path is the one relevant to the thesis: instead of forcing a dense network to brute-force infer that "walls are solid," that knowledge is injected as a mechanism (known physics), and learning is reserved for what actually needs to be learned (the navigation policy).

---

## Experiment 1 — `benchmark_geometries.py`

Classification on `make_moons` (two interleaved moons, not linearly separable). Compares:

- **Baseline MLP** (`sklearn.neural_network.MLPClassifier`).
- **VolumetricNetwork**: places one prototype per cluster (via `KMeans`) and activates by geometric region around the center. Geometry variants: hard sphere (L2), fuzzy (Gaussian), diamond (L1), 8/14-point star, "marine mine" (many spikes), and a **Smart** variant that adds iterative Kalman-style re-centering, a radius bounded by the distance to the nearest enemy point ("Lagrange constraint"), and a weight proportional to local density ("Bayes weight").
- Prediction is by vote: activations from all neurons of each class are summed, and the class with the higher signal wins.

### How to run it

```bash
pip install numpy matplotlib scikit-learn
python benchmark_geometries.py
```

Output: a console table (accuracy, training time, suggested hardware note per geometry) and a 2×4 figure with the decision boundaries of each variant.

### Reference result (single run, `random_state=42`)

| Model | Accuracy | Time (ms) | Hardware note |
|---|---|---|---|
| Baseline MLP (Deep) | 0.9217 | 2344.57 | MATRIX MUL (GPU) |
| Hard Sphere (L2) | 0.8833 | 4293.03 | INT MULT (CPU) |
| **Soft Fuzzy** | **0.9500** | **177.99** | FLOAT (GPU) |
| Diamond (L1) | 0.8183 | 493.23 | INT ADC (CPU/BitNet) |
| Star 8-Point | 0.8233 | 1953.52 | FLOAT (GPU) |
| Star 14-Point | 0.8450 | 341.51 | FLOAT (GPU) |
| Marine Mine | 0.7450 | 213.71 | FLOAT (GPU) |
| Smart (Bayes+Lagr+MK) | 0.8100 | 186.35 | MIXED (High Reliability) |

**Interpretation note (relevant to the thesis' parsimony principle):** the simplest variant (Soft Fuzzy, just a center + Gaussian sigma) is the one that generalizes best and is also one of the fastest. The three ad hoc refinements in the "Smart" variant (re-centering, nearest-enemy constraint, Bayesian weight) make it worse than both the baseline and Soft Fuzzy — the radius ends up determined by the distance to the nearest enemy point, a quantity sensitive to outliers, which a single noisy point near the center can collapse. This is evidence that stacking "principled-sounding" mechanisms without calibration can move a model further from the real phenomenon rather than closer to it.

**Rigor limitation to resolve before citing in the thesis:** the timings come from a single run (`time.time()`), with no repetitions or standard deviation — a rigorously calibrated timing comparison would need multiple runs averaged (`timeit`, 10–20 repetitions). Accuracy, by contrast, is deterministic here (fixed seed) and comparable as-is.

---

## Experiment 2 — `vlnn_rl_navigation.py`

### Components

**1. Environment (`GridWorld` + `LidarWrapper`)**
- N×N grid with random obstacles (reproducible seed per episode via `MapManager`).
- LIDAR observation: 4 distances to obstacles/walls (up, right, down, left) + normalized vector toward the goal (6 dimensions total).
- Reward: shaping for reducing distance to the goal, penalty for collisions, bonus for reaching the goal.
- **Curriculum learning**: maps go 6×6 → 9×9 → 12×12 during training, with an epsilon reset each time the scale changes.

**2. Baseline — `DQN_Agent` (PyTorch)**
Dense network (6 → 64 → 64 → 5), ReLU, Adam, MSE loss against the Bellman target (`reward + gamma * max_next_q`), epsilon-greedy policy with exponential decay.

**3. VLNN — `VolumetricCognitiveBrain`**
- **`CognitiveNeuron`**: geometric prototype (center + radius) with its own Q-values per action; the center is updated via a sample-count-weighted moving average.
- **Dynamic creation**: if a state doesn't fall inside any existing neuron, a new one is created, inheriting Q-values from the nearest neighbor.
- **Mitosis**: a neuron that accumulates "pain" (persistent negative prediction error) above a threshold splits into smaller-radius daughters, inheriting partial knowledge from the parent.
- **Merging (`fuse_neurons`)**: nearby neurons that also dictate the same optimal policy and have similar Q-values are merged into one.
- **Pruning (`prune_dead_neurons`)**: neurons unused for a while are removed.
- **`SymbolicCortex`**: periodic consolidation of neuron clusters into "symbols" via a quantized, translation-invariant geometric hash — compressed, reusable long-term memory.
- **Physics constraint (`check_physics_feasibility`)**: before evaluating Q-values, actions that would cause a collision (per LIDAR) are discarded — world knowledge injected as an explicit rule, not learned by trial and error.

**4. Experimental protocol**
- **Phase 1 — Training**: 800 episodes with curriculum (6×6 → 9×9 → 12×12).
- **Phase 2 — Deployment ("the jungle")**: 100 episodes on a 24×24 map with more obstacles, never seen during training. Measures **real generalization**, not memorization.
- Metrics: win rate, collisions, steps to goal, parameter/neuron/symbol count, symbolic compression ratio, training time.

### How to run it

```bash
pip install numpy matplotlib torch
python vlnn_rl_navigation.py
```

Outputs:
- Console log every 50 episodes (win rate, collisions, symbols/neurons or parameters depending on the agent).
- `vnn_complete_experiment.png`: win-rate curves (training vs. deployment) for both agents.
- Final JSON printed to console with summary metrics for both phases.

### Configurable parameters (`PARAMS`)

| Parameter | Description |
|---|---|
| `n_episodes` | Training episodes (800) |
| `gamma` | Bellman discount factor (0.95) |
| `epsilon_start/end/decay` | Epsilon-greedy exploration schedule |
| `init_radius` | Initial radius of a new neuron |
| `min_radius` | Minimum radius before mitosis is blocked |
| `hidden_dim` | Hidden layer size of the MLP baseline |

### Known issues / limitations (to resolve before citing results in the thesis)

- `collision_count` is declared inside the episode loop in `train()` but is **never incremented** anywhere in that function (it is incremented in `run_deployment_test`, via `info.get('collision')`). Right now `metrics['collisions']` stays at 0 for the entire training phase — this is a bug, not a design choice. A check on `info['collision']` needs to be added inside the `train()` loop so that the `collisions_6x6/9x9/12x12` metrics are real.
- In `run_deployment_test`, the line `metrics['collisions'].append(collision_count)` is duplicated (it appears twice in a row), which misaligns that metric against `success` and `steps_to_goal` (you end up with twice as many collision entries as episodes). The duplicate line needs to be removed.
- `get_parameter_count()` for the VLNN counts neurons × 14, which is not directly comparable to the MLP's trainable weights — these are different units of complexity; this should be clarified or normalized before presenting it as a "model size" comparison.
- The merge threshold (`q_diff < 0.3`) and the mitosis threshold (`pain_threshold = 8`) are hand-fixed constants with no calibration sweep — a hyperparameter search or experimental justification is needed before presenting them as optimal.
- Runtimes are not averaged over multiple runs or seeds; for a rigorously calibrated performance comparison, run with multiple seeds and report mean ± standard deviation.

### Relation to the thesis

This experiment is the basis for a controlled comparison between a **neurosymbolic** approach (geometric prototypes + symbols + explicit physics constraints) and a **pure subsymbolic** approach (MLP/DQN) on RL navigation tasks, specifically evaluating:

- the ability to generalize to unseen environments (curriculum transfer → deployment in "the jungle"),
- the interpretability/parsimony of the resulting model (inspectable neurons and symbols vs. an opaque weight matrix),
- whether explicit physical knowledge (invalid-action masking) reduces the need for trial-and-error learning compared to letting a dense network infer those constraints on its own.

---

## Requirements

```bash
pip install numpy matplotlib scikit-learn torch
```

## Suggested repo structure

```
.
├── benchmark_geometries.py       # Experiment 1: classification / RBF geometries
├── vlnn_rl_navigation.py         # Experiment 2: RL + navigation (VLNN vs DQN)
└── README.md
```
