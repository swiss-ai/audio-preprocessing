#!/bin/bash
#SBATCH --job-name=audiocite
#SBATCH --account=infra01           
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/audio_%j.out
#SBATCH --error=logs/audio_%j.err
#SBATCH --reservation=PA-2338-RL  
#SBATCH --environment=ctranslate2-nemo-cudnn

INPUT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/audiocite/wavs_cc_by"
OUTPUT_DIR="/iopsstor/scratch/cscs/${USER}/audio-output/"
WORK_DIR="/iopsstor/scratch/cscs/${USER}/preprocessor"

export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"

srun --cpu-bind=none bash -c '
    export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:${LD_LIBRARY_PATH}
    exec '"${WORK_DIR}"'/.venv-audioprocessing/bin/python '"${WORK_DIR}"'/audiocite.py \
        --input-dir '"${INPUT_DIR}"' \
        --output-dir '"${OUTPUT_DIR}"' \
        --language fr
'

echo "Job ${SLURM_JOB_ID} finished at $(date)"
