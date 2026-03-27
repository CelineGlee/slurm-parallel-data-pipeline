# Distributed Big Data Analyse Engine

## Overview
This project implements a high-performance, parallelized application designed to analyze massive social media datasets. Specifically, it processes a **144GB Mastodon dataset** in NDJSON format to identify the "happiest" and "saddest" hours and users based on aggregated sentiment scores. 

## Key Features
* **Massive Data Handling:** Efficiently processes 144GB of raw NDJSON data without memory overload by using byte-level file slicing.
* **MPI Parallelization:** Utilizes `mpi4py` to distribute workloads across multi-node and multi-core architectures (tested up to 2 nodes and 8 cores).
* **Scalable Architecture:** Implements a static distribution strategy where each process reads a designated byte range directly from the shared file system.
* **Robust Data Parsing:** Built-in error handling for common social media data imperfections, including missing fields, invalid types, and malformed JSON.
* **High Parallel Efficiency:** Achieved a **~8.08x speedup** on 8 cores, demonstrating near-perfect parallelizability ($\alpha \approx 99.9\%$)

## System Architecture
### Master-Worker Orchestration
The engine utilizes a Master-Worker (Root-Leaf) model implemented via mpi4py. The workload is distributed to ensure that no single node becomes a bottleneck, particularly during the heavy I/O phase of reading the 144GB dataset.

### 1. The Master (Rank 0): The Orchestrator
The Master process is responsible for the lifecycle of the distributed job:
* **Metadata Discovery:** Calculates the total file size and determines the optimal byte-offset for each available process.
* **Dynamic Partitioning**: Divides the global dataset into $N$ logical segments, ensuring each worker has a unique, non-overlapping range of data.
* **Result Reduction**: Once workers finish their local computations, the Master executes a comm.reduce() operation. It merges distributed Python dictionaries into a unified global dataset
* **Final Analytics**: Performs the final sorting, top-K filtering (e.g., Top 5 happiest hours), and generates the final execution report.
  
### 2. The Workers (All Ranks): Distributed Executors
Every process (including Rank 0) acts as a worker to ensure 100% CPU utilization:
* **Independent I/O:** Each worker uses `.seek()` to jump directly to its assigned byte position. This allows for parallel file reading, which is critical for 100GB+ datasets
* **Stream Alignment:** Workers perform a "leading-edge" check—reading until the first newline—to ensure they start at a valid JSON record boundary, maintaining 100% data integrity across splits.
* **Local Mapping:** Processes parse raw NDJSON into structured Python objects and perform local sentiment aggregation. This minimizes the amount of data that needs to be transferred over the network during the reduction phase.
* **Fault Isolation:** Individual workers handle malformed JSON or missing fields locally, ensuring that data inconsistencies in a specific segment do not interrupt the global process.

### Distribution Strategy
Instead of a standard `scatter()` approach which can bottle-neck Rank 0, this application uses **Independent Byte Slicing**:
1.  **Metadata Retrieval:** Rank 0 determines total file size via `os.path.getsize()`.
2.  **Range Assignment:** Each process is assigned a specific start/end byte range.
3.  **Pointer Alignment:** Processes use `.seek()` to reach their start point. Non-zero ranks perform a `readline()` to align with the next complete JSON record, ensuring no data is duplicated or lost.

### Reduction Strategy
Local results (Python dictionaries) are merged into global dictionaries at the root rank using `comm.reduce()` with a custom `merge_lists` function for optimal performance.

## Performance Evaluation
The application was benchmarked across three configurations on the Spartan `sapphire` partition:

| Configuration | Real Execution Time | Speedup |
| :--- | :--- | :--- |
| 1 Node, 1 Core | 33.46 minutes | 1.00x |
| 1 Node, 8 Cores | 4.14 minutes | 8.08x |
| 2 Nodes, 8 Cores | 4.27 minutes | 7.83x |

## Deployment Guide

### Environment Setup
* **Language:** Python 3.11.3 
* **Parallel Backend:** `mpi4py` 
* **Cluster Manager:** SLURM

### Infrastructure Orchestration (SLURM)
The engine is deployed via SLURM scripts to manage resource allocation. 

```bash
#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00

# Load optimized modules
module load Python/3.11.3
module load mpi4py/3.1.4-Python-3.11.3

# Execute parallel job
srun -n 8 python3 engine_main.py data/mastodon-144g.ndjson

### Running on Spartan
Submit the job using the provided SLURM scripts:
```bash
sbatch 1-8-144g.slurm
```

* Run with command: `sbatch slurm_name.slurm`
* Check Job status, and save output results (including execution time):
    ```
    sacct -u $USER
    cat mastodon_1n_8c_144g_num.out
    ```