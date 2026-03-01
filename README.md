# Route Finding AI Search Engine

This repository contains a Python search engine built from scratch to solve the Route Finding Problem. It evaluates a 2D spatial graph to find optimal paths from a specified origin node to one or more destination nodes.

The project was developed as part of the COS30019 Introduction to Artificial Intelligence coursework and demonstrates both uninformed and informed graph traversal techniques.

## Algorithms Implemented

The search engine includes six distinct algorithms, operating completely independently of external search libraries:

- **DFS**: Depth-First Search (Uninformed)
- **BFS**: Breadth-First Search (Uninformed)
- **GBFS**: Greedy Best-First Search (Informed)
- **AS**: A\* Search (Informed)
- **CUS1**: Uniform Cost Search (Custom Uninformed)
- **CUS2**: Iterative Deepening A* / IDA* (Custom Informed)

### Technical Details

- **Heuristic Design**: The informed search methods use Euclidean distance as the heuristic. This guarantees both admissibility and consistency on a 2D coordinate plane.
- **Tie-Breaking Rules**: The priority queues enforce strict, deterministic tie-breaking. If two nodes have identical evaluation costs, the engine prioritizes the node with the smaller numerical ID. If the IDs are also identical, it defaults to chronological order (first-in, first-out).

## Repository Structure

```text
Assignment2A-COS30019
├── search.py            # Command-line interface and main entry point.
├── engine.py            # Core algorithmic logic and cycle-prevention sets.
├── graph.py             # Parses input text files and computes spatial heuristics.
├── models.py            # Defines state representation and custom priority queue logic.
├── PathFinder-test.txt  # Sample configuration file provided in the assignment.
└── tests/               # Automated testing and benchmarking suite.
    ├── factory.py       # Script that provisions 10 mathematical edge-case topologies.
    ├── runner.py        # Benchmarking tool that executes algorithms in subprocesses.
    ├── cases/           # Generated directory containing the 10 text files from factory.py.
    └── results.csv      # Generated telemetry report containing execution metrics.

```

## Requirements

The codebase is written in pure Python. It relies entirely on the standard library and requires no external dependencies or virtual environments.

- Python 3.9 or higher

## Usage

You can run the search engine directly from your terminal. The script requires two arguments: the path to the graph configuration file and the acronym of the search algorithm you want to use.

**Syntax:**

```bash
python search.py <filepath> <method>

```

**Example:**

```bash
python search.py PathFinder-test.txt as

```

### Output Format

The program prints the results to the standard output in a strict three-line format:

1. The filename and the method used.
2. The destination node reached and the total number of nodes generated during the search.
3. The sequential path taken from the origin to the destination.

_Note: If the search space is entirely exhausted and no path exists, the second line will read "No solution found."_

## Testing and Benchmarking

The `tests` directory contains an automated suite to validate the algorithms against various edge cases, including unreachable goals, infinite loop traps, and heuristic traps.

**1. Generate the Test Cases**
Before benchmarking, you need to generate the graph topologies. Run the factory script from the root directory:

```bash
python tests/factory.py

```

This command generates a new folder named `cases` located exactly at `tests/cases/`. Inside this folder, it creates 10 distinct text files (`T01_Standard.txt` through `T10_ZeroCost.txt`) representing different spatial configurations.

**2. Run the Benchmark Suite**
Once the text files are provisioned, run the test runner:

```bash
python tests/runner.py

```

This script executes all six algorithms against the 10 generated test cases in isolated subprocesses to prevent memory contamination. After the run completes, it generates a telemetry report named `results.csv` located directly inside the `tests/` directory (`tests/results.csv`). This CSV file contains the execution time, space complexity (nodes expanded), and operational status for every single run.
