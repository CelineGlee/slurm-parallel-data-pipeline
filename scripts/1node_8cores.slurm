#!/bin/bash
#SBATCH --job-name=mastodon_1n8c
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=mastodon_1n8c_144g_%j.out

# Load required modules
module load Python/3.11.3
module load mpi4py/3.1.4-Python-3.11.3

# Record start time
START_TIME=$(date +%s)

# Run the MPI program
mpiexec -n 8 python3 mpi_test_16m.py /home/yucai5/assign1/mastodon-144g.ndjson

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Job execution time: $DURATION seconds"