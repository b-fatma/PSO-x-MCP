# Particle Swarm Optimization (PSO) × Maximum Covering Problem (MCP)

This repository contains an academic implementation of Particle Swarm Optimization (PSO) for solving the **Maximum Covering Problem (MCP)**, where the goal is to select **exactly `k` subsets** to maximize coverage of elements. The PSO algorithm is adapted to this **discrete binary setting**, and supports extensive experimentation with multiple PSO variants, selection types, mutation strategies, and performance analysis.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Source Code Details](#source-code-details)
- [How to Run](#how-to-run)
- [Datasets](#datasets)
- [References](#references)

---

## Repository Structure

```
├── readme.md                 # Project overview and documentation
├── requirements.txt          # Python dependencies

├── analysis/                 # Jupyter notebooks for performance analysis
│   └── metah-result-analysis-PSO-MCP.ipynb

├── data/                     # MCP benchmark datasets (Set Cover Problem variants) from https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
│   ├── scp41.txt ... scp410.txt
│   ├── scpa1.txt ... scpa5.txt
│   ├── scpb1.txt ... scpb5.txt
│   └── scpc1.txt ... scpc5.txt

├── src/                      # Source code for algorithms and experiments
│   ├── demo.py               # Script to demo algorithms
│   ├── DFS.py                # Depth-First Search implementation
│   ├── DFSTest.py            # DFS testing script
│   ├── GreedyTest.py         # Greedy method test runner
│   ├── GridSearchPSO.py      # Grid search for tuning PSO parameters
│   ├── MPGridSearchPSO.py    # Grid search with multiprocessing support
│   ├── MaxCoveringProblem.py # MCP problem definition
│   ├── PSO.py                # Main PSO algorithm
│   ├── PSOTest.py            # Test script for PSO
│   └── Particle.py           # Particle class definition for PSO


├── stats/                    # Output metrics, logs, and experiment results
│   ├── dfs_1h.csv
│   ├── greedy.csv
│   ├── mutation.csv
│   ├── pso_results.csv
│   ├── standard_bpso_hdbpso.csv
│   ├── standard_whdbpso.csv
│   ├── stochastic_bpso_hdbpso.csv
│   └── stochastic_whdbpso.csv
```

## Source Code Details

#### `MaxCoveringProblem.py`

- Defines the MCP problem, parses input files, and computes `k = floor(m / 20)` ( number of subsets to select ).

#### `PSO.py`

- Implements **Particle Swarm Optimization (PSO)** for solving MCP:
  - Supports multiple **inertia strategies**: `fixed`, `linear`, `nonlinear`.
  - **Checkpointing**: Saves progress to avoid re-running iterations.

#### `Particle.py`

- Defines particles used in PSO:
  - **Initialization strategies**: `random`, `greedy` (selects larger subsets first) ...
  - **Distance metrics**: `bitwise`, `Hamming`, `Weighted Hamming`.
  - **Selection types**:
    - `standard`
    - `stochastic`
    - `deterministic`
  - **Mutation**: Bit Change Mutation (based on [this paper](https://www.researchgate.net/publication/31208097_Binary_Particle_Swarm_Optimization_with_Bit_Change_Mutation)).
  - **Particle variants**: `ParticleRestructured` implemented BRPSO but did not discuss it nor experiment with it, It's based on [the following paper](https://www.mdpi.com/2313-7673/8/2/266)

#### Other Scripts

- **`demo.py`**: Demonstrates PSO, DFS ( with 30s timeout ), and Greedy on `scp41.txt`.
- **`DFS.py`**: Implements Depth-First Search (DFS).
- **`GreedyTest.py`**: Runs the Greedy algorithm.
- **`GridSearchPSO.py`**: Performs grid search for PSO tuning.
- **`<x>Test.py`**: Test scripts for individual algorithms.

## How to Run

To set up the environment and install required dependencies, run:

```bash
pip install -r requirements.txt
```

to run the demo file run :

```bash
python src/demo.py
```

## Datasets

The repository includes several benchmark datasets for the Maximum Covering Problem (MCP). These datasets are based on the **Set Cover Problem (SCP)** and are available in the `data/` directory. You can find the following datasets:

- **`scp41.txt` to `scp410.txt`**
- **`scpa1.txt` to `scpa5.txt`**
- **`scpb1.txt` to `scpb5.txt`**
- **`scpc1.txt` to `scpc5.txt`**

These datasets are derived from the **OR-Library** (Set Cover Problem instances), and more details about the datasets can be found [here](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html).

The format of each file is as follows:

1. The first number represents the total number of elements (`n`).
2. The second number represents the total number of subsets (`m`).

## References

The following references were used or inspired the development of the algorithms and methodologies in this project:

1. J. Kennedy and R. Eberhart, “Particle swarm optimization,” in _Proceedings of ICNN’95 - International Conference on Neural Networks_, vol. 4 of ICNN-95, p. 1942–1948, IEEE, 1995.
2. W. A. L. W. M. Hatta, C. S. Lim, A. F. Z. Abidin, M. H. Azizan, and S. S. Teoh, “Solving maximal covering location with particle swarm optimization,” _International Journal of Innovation, Management and Technology_, vol. 4, no. 2, pp. 211–215, 2013.
3. J. Kennedy and R. Eberhart, “A discrete binary version of the particle swarm algorithm,” in _1997 IEEE International Conference on Systems, Man, and Cybernetics. Computational Cybernetics and Simulation_, vol. 5, pp. 4104–4108 vol.5, 1997.
4. J. Zhu, J. Liu, Y. Chen, X. Xue, and S. Sun, “Binary restructuring particle swarm optimization and its application,” _Biomimetics_, vol. 8, p. 266, June 2023.
5. H. Banka and S. Dara, “A Hamming distance based binary particle swarm optimization (hdbpso) algorithm for high dimensional feature selection, classification and validation,” _Pattern Recognition Letters_, vol. 52, p. 94–100, Jan. 2015.
6. S. Lee, H. Park, and M. Jeon, “Binary particle swarm optimization with bit change mutation,” _IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences_, E90-A, vol. E90-A, 10 2007.
7. F. Vandenbergh and A. Engelbrecht, “A study of particle swarm optimization particle trajectories,” _Information Sciences_, vol. 176, p. 937–971, Apr. 2006.
8. https://web2.qatar.cmu.edu/~gdicaro/15382-Spring18/hw/hw3-files/pso-book-extract.pdf. Accessed: 2025-04-02.
9. S. J. Russell and P. Norvig, _Artificial Intelligence: A Modern Approach_, 3rd ed., Pearson, 2010. (For foundational AI concepts used in the problem-solving approaches).

The above references have significantly contributed to the development and understanding of the methods used in this project.
