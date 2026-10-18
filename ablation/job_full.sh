#!/bin/bash
#SBATCH --job-name=aid-l4-full
#SBATCH --output=ablation/logs_l4_2k_100k/full-%j.out
#SBATCH --error=ablation/logs_l4_2k_100k/full-%j.err
#SBATCH --partition=a100x4
#SBATCH --account=zywang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

cd "$(dirname "$0")/.."
bash ablation/run_variant.sh full
