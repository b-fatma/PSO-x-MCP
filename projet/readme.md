# Particle Swarm Optimization (PSO) x Maximum Covering Problem (MCP)
This repository includes implementation of PSO to solve a variant of MCP with an additional constraint of selecting exactly k subsets. Naturally, this entails an adaptation of PSO to the discrete binary nature of the problem.

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

## Relevant files in src/
- demo.py provides a short script to run PSO, Depth-First-Search (DFS) with a timeout limit of 30s, and Greedy Algorithm on scp41.txt.
- MaxCoveringProblem.py defines the problem class, given a file it parses it into a MaxCoveringProblem object.
- DFS.py contains a DFS class with 2 variations of solving the problem (time_bound, and exhaustive)
- Particle.py contains the main implementation of Particles for PSO,. It handles different particle classes,  selection types (stochastic | standard), distance types (wHD | HD| bit-wise), initialization strategies (greedy, random, ...).
- PSO.py contains the main implementation of the PSO algorithm, it handles different hyperparameters and and includes checkpoint-based iteration control instead of treating max_iterations as a fixed hyperparameter.
- GridSearchPSO.py performs grid search on PSO, and GMPridSearchPSO.py makes the process more efficient by using multiprocessing.
- <x>Test.py are for testing <x> algorithm.





