# Traffic-based Route Guidance System (TBRGS)

## 1. Project Overview

### 1.1 Project Description

The **Traffic-based Route Guidance System (TBRGS)** is a sophisticated pathfinding application that combines classical Artificial Intelligence search techniques with Deep Learning-based traffic prediction. Developed for the city of Boroondara, the system utilizes historical SCATS (Sydney Coordinated Adaptive Traffic System) data to anticipate road congestion and provide optimized routing. While standard navigation systems often rely on static distance, this project implements a dynamic weight model where edge costs are determined by real-time predicted travel times.

### 1.2 Project Objectives

- **Assignment 2A (Core Search Engine):** To design and implement a standalone Python search engine capable of executing uninformed and informed search algorithms on a 2D spatial graph. The engine must handle complex topologies, enforce strict tie-breaking rules, and optimize pathfinding using Euclidean heuristics.
- **Assignment 2B (ML Integration & Application):** To extend the core Search Engine by integrating Machine Learning models (LSTM, GRU, and FNN) that predict traffic flow. These predictions are used to dynamically calculate travel times, which are then visualized in a production-ready Graphical User Interface (GUI) and an interactive real-world map.

---

## 2. Repository Structure

```markdown
Assignment2-IAI-trang
├── run_app.py               # Main entry point (Terminal/GUI)
├── config.json              # Global settings for ML capacities & UI defaults
├── requirements.txt         # Exact package dependencies
├── data/
│   ├── database/            # SQLite caches for pre-computed traffic states
│   ├── maps/                # Graph topologies and interactive map exports
│   ├── processed/           
│   │   └── Scats_Data_Cleaned_GapSafe.csv # Cleaned, time-series traffic data
│   └── raw/                 
│       ├── Scats Data October 2006.xls     # Original raw VicRoads spreadsheets
|       └── Traffic_Count_Locations_with_LONG_LAT.csv # Original SCATS location and ID data
├── models/
│   ├── saved_models/        # Trained LSTM, GRU, and FNN .keras artifacts
│   └── scalers/             # Scalers for normalizing traffic flow inputs
├── notebooks/
│   └── model_training.ipynb # Jupyter environment for ML training & evaluation
├── src/
│   ├── core/                # Phase 1: Search Engine Core Logic
│   │   ├── engine.py        # Algorithmic orchestrator (DFS, BFS, GBFS, IDA, UCS, A*, Yen's)
│   │   ├── graph.py         # Graph parsing and heuristic (h-value) logic
│   │   ├── models.py        # Define SearchState and PriorityQueue for tie-breaking rules
│   │   └── search.py        # Command-line interface logic
│   ├── data/                # Data Engineering
│   │   ├── data_cleaner.py  # ETL pipeline for SCATS spreadsheets
│   │   └── initialize_traffic.py # Seeds SQLite caches with ML predictions
│   ├── ml/                  # Machine Learning Inference
│   │   └── traffic_flow_predictor.py # Inference engine for real-time flow
│   ├── ui/                  # Visualization Layer
│   │   ├── main_gui.py      # Tkinter-based interactive routing interface
│   │   └── real_map_visualizer.py # Folium-based OpenStreetMap generator
│   └── utils/               # Technical Utilities
│       ├── config.py        # Config loader for application parameters
│       ├── graph_generator.py # Builds connected Boroondara graphs
│       ├── logger_setup.py  # Centralized colored logging
│       └── speed_time_converter.py # Quadratic flow-to-speed math
└── tests/                   # Benchmarking and Stress Testing Suite
    ├── cases/               # 10 Topology test cases (T01-T10) from Assignment 2A
    ├── factory.py           # Generates the 10 automated topology test cases for Assignment 2A
    ├── results.csv          # Telemetry report for algorithm performance
    ├── run_test_suite.py    # Assignment 2B Stress Testing entry point
    └── runner.py            # Executes algorithms against factory.py topologies
```

---

## 3. Implementation & Setup Guide

### 3.1 Environment Setup

The project is built using Python 3.9+. It is highly recommended to use a virtual environment to avoid version conflicts.

1. **Clone and Navigate:**
    
    ```bash
    git clone <repository-url>
    cd Assignment2-IAI-trang
    ```
    
2. **Install Dependencies:**
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    

### 3.2 Running the Application

- **GUI Mode (Full System):** Provides interactive route selection and traffic toggles.
    
    ```bash
    python run_app.py data/maps/map.txt --gui
    ```

    Note: Before running the GUI, you need to generate the graph topologies by the following command.

    ```bash
    python src/utils/graph_generator.py
    ```
    
- **Terminal Mode (Standard Search):** Executes a specific algorithm on a provided map.
    
    ```bash
    python run_app.py tests/test_data/PathFinder-test.txt <method>
    
    # Example:
    python run_app.py tests/test_data/PathFinder-test.txt as
    ```
    
    - **Success Output:**
    ```bash
    <filename> <method> 
    <destination_goal> <total_number_of_nodes>
    <path>
    ```

    - **Failure Output:**  ***If the search space is entirely exhausted and no path exists***
    ```bash
    <filename> <method>
    No solution found.
    ```
- (OPTIONAL) To re-generate all the provided output files, run the following command from the root directory `.../Assignment2-IAI/`:

    - **To re-generate the Interactive Map:** This generates `boroondara_interactive.html` in the `data/maps/` folder.
        
        ```bash
        python src/ui/real_map_visualizer.py
        ```
        
    - **To re-generate 10 topology tests and benchmarking:** This generates 10 TXT files in the `/tests/case` folder and a `result.csv` file that contains the execution time, space complexity (nodes expanded), and operational status for every single run.
        
        ```bash
        # Generate 10 automated test cases
        python tests/factory.py
        
        # Executes all 6 search algorithms against the 10 generated test cases
        python tests/runner.py
        ```
        
    - **To re-generate and evaluate the ML models:** Click the `Run all` button in the Jupyer Notebook file named `model_training.ipynb`, located in the `./notebooks` folder.

    - To re-generate the cleaned dataset and SQLite database for the data cleaning and procesing phase to optimize the app performance:

    ```bash
    # To clean dataset:
    python src/data/data_cleaner.py

    # To pre-compute traffic state:
    python src/data/initialize_traffic.py
    ```
---

## 4. Phase 1: Search Engine Design (Assignment 2A)

The core engine is located in `src/core/engine.py` and implements the following algorithmic logic:

- **Uninformed Search:** DFS (LIFO), BFS (FIFO), and Uniform Cost Search (CUS1).
- **Informed Search:** A* (AS) and Greedy Best-First Search (GBFS) using Euclidean distance heuristics.
- **Custom Optimal Search:** Iterative Deepening A* (CUS2/IDA*) for space-efficient pathfinding.
- **Multi-Path Search:** Yen's Algorithm (Top-K) is a key extension for **Assignment 2B** that provides the top *K* alternative routes by systematically suppressing nodes and edges.

**Topology Testing and Benchmarking:** The `tests/` suite validates the system against 10 specialized scenarios (T01–T10), including unreachable nodes, tie-breaking traps, and zero-cost edges, ensuring the engine is robust under all conditions. 

---

## 5. Phase 2: ML & Software Integration (Assignment 2B)

This phase represents the complete 7-step integration of the system:

1. **Data Cleaning and Preprocessing:** The `data_cleaner.py` script performs a "Matrix Melting" ETL process, converting raw SCATS spreadsheets into vertical time-series data and filling gaps using linear interpolation.
2. **Building and Training Machine Learning Models:** Three distinct architectures (LSTM, GRU, and FNN) were trained using `model_training.ipynb` to recognize temporal traffic patterns.
3. **Model Evaluation and Comprehensive Comparison:** Performance is evaluated based on Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to determine the most accurate predictor for different traffic scenarios.
4. **Converting Traffic Flow to Travel Time:** The `speed_time_converter.py` utility applies a quadratic fundamental traffic diagram to map vehicle flow (veh/hr) to expected speeds (km/h), then calculates travel time based on edge length.
5. **Integration Bridging:** `initialize_traffic.py` script executes a batch prediction across all 40+ SCATS sites and seeds the `data/database/` SQLite caches, allowing the Search Engine to pull dynamic edge weights instantly.
6. **Mapping and Visualisation:** `main_gui.py` offers a dynamic canvas with fanned-out multi-lane paths, while `real_map_visualizer.py` projects the SCATS topology onto an interactive OpenStreetMap layer.
7. **Stress Testing:** Unlike the topology tests in Assignment 2A, `run_test_suite.py` performs **10 Stress Tests** specifically designed to validate the system’s stability when handling ML-driven dynamic weights and concurrent path requests.