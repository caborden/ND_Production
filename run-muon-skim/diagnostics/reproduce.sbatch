#!/bin/bash
#SBATCH --job-name=muon-memory62
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
set -eo pipefail
# Pass absolute paths: diagnostic Python script, campaign directory, output directory.
script=$1
campaign=$2
output=$3
shift 3
module load python/3.11
module load cudatoolkit/12.4
work=/global/cfs/cdirs/dune/users/cborden
source "$work/larnd.venv/bin/activate"
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
env -u SLURM_TRES_PER_TASK srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    "$work/larnd.venv/bin/python" "$script" --campaign "$campaign" \
    --index 62 --output-dir "$output" "$@"
