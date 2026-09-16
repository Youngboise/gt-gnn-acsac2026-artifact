# GT-GNN ACSAC 2026 Artifact

This repository accompanies the accepted ACSAC 2026 paper:

**Belief-Conditioned Game-Theoretic Graph Neural Networks for Adaptive Cyber Deception**

## Artifact scope

This public release is prepared for the **Artifact Available** badge.

The repository provides:

- source code for the six-stage GT-GNN experimental pipeline;
- selected precomputed outputs corresponding to the paper's main tables, sensitivity analyses, held-out attacker experiments, and scalability evaluation;
- dependency information and documentation for inspecting the implementation and interpreting the released results.

This repository is **not intended to claim a clean-room, end-to-end reproduction of every experiment from raw inputs** unless all required intermediate datasets, configurations, and trained checkpoints are also supplied.

## GT-GNN pipeline

The implementation is organized into six stages:

1. **Belief estimation**  
   Estimates a factorized belief over attacker goal, skill, stealth, and decoy awareness.

2. **Belief-conditioned GNN**  
   Uses a belief-conditioned GraphSAGE-style encoder with separate threat and revelation heads.

3. **Strategic graph extraction**  
   Builds a compact decision-relevant subgraph and candidate deception-action set.

4. **Game / AMC evaluation**  
   Uses an approximate attacker-response model and an absorbing Markov chain (AMC) evaluator to estimate action-level risk reduction, information value, delay, cost, and utility.

5. **Value / policy learning**  
   Learns action-value components and policy scores from the Phase 4 action outcomes.

6. **Closed-loop evaluation**  
   Evaluates GT-GNN and comparison policies, ablations, sensitivity tests, held-out attacker dynamics, and scalability.

## Repository layout

```text
gt-gnn-acsac2026-artifact/
├── README.md
├── LICENSE
├── requirements.txt
├── gt_gnn_phase1_belief_case_study/src                      # Phase 1: belief estimation
├── gt_gnn_phase2_belief_gnn/src/                            # Phase 2: belief-conditioned GNN
├── gt_gnn_phase3_strategic_graph/src/                       # Phase 3: strategic graph extraction
├── gt_gnn_phase4_amc_evaluator/src/                         # Phase 4: Game / AMC evaluator
├── gt_gnn_phase5_value_policy_head/src/                     # Phase 5: value / policy head
├── gt_gnn_phase6_closed_loop_eval/src6/                     # Phase 6: closed-loop evaluation
└── results/
    ├── phase6_policy_summary.csv
    ├── phase6_ablation_summary.csv
    ├── phase6_robustness_summary.csv
    ├── phase6_heldout_attacker_summary.csv
    ├── phase6_large_scale_summary.csv
    ├── phase6_large_scale_benchmark.csv
    ├── phase6_integrated_evaluation_summary.json
    └── phase6_paper_experiment_summary.json
```

## Mapping to paper results

| Paper result | Released file |
|---|---|
| Table III - Closed-loop policy comparison | `results/phase6_policy_summary.csv` |
| Table IV - Ablation analysis | `results/phase6_ablation_summary.csv` |
| Tables V-X - Sensitivity analyses | `results/phase6_robustness_summary.csv` |
| Table XI - Held-out attacker dynamics | `results/phase6_heldout_attacker_summary.csv` |
| Figure 1 - Scalability | `results/phase6_large_scale_summary.csv`, `results/phase6_large_scale_benchmark.csv` |
| Integrated evaluation metadata | `results/phase6_integrated_evaluation_summary.json`, `results/phase6_paper_experiment_summary.json` |

### Main Table III identifiers

The implementation uses the following internal policy identifiers:

| Source identifier | Paper label |
|---|---|
| `proposed_value_policy` | GT-GNN |
| `amc_oracle` | AMC Oracle |
| `pred_info_only` | Info-Only |
| `pred_risk_only` | Risk-Only |
| `pred_delay_only` | Delay-Only |
| `phase3_fused` | GNN-Only |
| `centrality_based` | Centrality |
| `random_placement` | Random |
| `static_low_cost` | Static / Static Low-Cost |

The released Table III output reports a mean utility of approximately:

- GT-GNN: **0.781**
- AMC Oracle: **0.999**
- Info-Only: **0.784**
- Risk-Only: **0.693**
- Delay-Only: **0.549**
- GNN-Only: **0.572**
- Centrality: **0.675**
- Random: **0.453**
- Static Low-Cost: **0.413**

These values correspond to the rounded values reported in the accepted paper.

## Software requirements

The source modules use Python and common scientific-computing packages. A consolidated environment may be installed with:

```bash
python -m pip install -r requirements.txt
```

Recommended baseline requirements:

```text
Python >= 3.9
numpy >= 1.24
pandas >= 2.0
PyYAML >= 6.0
torch >= 2.1
scikit-learn >= 1.3
networkx >= 3.0
matplotlib >= 3.6
tqdm >= 4.66
```

The source tree contains the implementation associated with the experimental pipeline. Re-running every stage from scratch additionally requires the corresponding input datasets, intermediate artifacts, configuration files, and model checkpoints.

## Interpretation and implementation notes

The mathematical framework in the paper describes the decision model at an algorithmic level. The experimental implementation uses tractable approximations for some strategic quantities.

In particular:

- the information-value component is implemented using an action-conditioned approximation rather than exhaustive enumeration of every possible post-action observation;
- the attacker-response layer is an approximate, belief-conditioned bounded-rational response model rather than an exact equilibrium solver;
- the Phase 6 closed-loop evaluation records post-action evidence and belief-state recalibration, but it should not be interpreted as online gradient-based retraining of every model component after every action.

These implementation choices should be considered when comparing individual equations in the paper with the released experimental code.

## Result provenance

The precomputed result files are provided to make the numerical evidence associated with the accepted paper directly inspectable.

Runtime and memory measurements can vary across machines. The scalability claim should therefore be interpreted relative to the bounded-decision setting used in the paper: the strategic subgraph and candidate set are explicitly capped.

## License

This repository is released under the **MIT License**. See `LICENSE`.

## Citation

If you use this artifact, please cite the corresponding ACSAC 2026 paper:

> *Belief-Conditioned Game-Theoretic Graph Neural Networks for Adaptive Cyber Deception*, ACSAC 2026.

A complete bibliographic entry can be added after the final proceedings metadata is available.
