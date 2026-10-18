#!/bin/bash
#SBATCH --job-name=aiden-fast-bench
#SBATCH --output=benchmarks/inference-speed/benchmark-%j.out
#SBATCH --error=benchmarks/inference-speed/benchmark-%j.err
#SBATCH --partition=a100x4
#SBATCH --account=zywang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate aiden
cd "$(dirname "$0")/../.."

python fast-infer/benchmark_5000.py
