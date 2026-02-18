#!/usr/bin/env bash
#SBATCH --job-name=zarr_performance_test
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=640G   # Memory pool for all cores (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate zarr-env
python test.py