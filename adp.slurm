#!/bin/bash
#SBATCH --job-name=distributed     # job name
#SBATCH -C a100                     # GPU A100 80 Go
#SBATCH --nodes=4                    # number of nodes
#SBATCH --ntasks-per-node=8          # = number of GPUs per node
#SBATCH --gres=gpu:8                 # number of GPUs per node
#SBATCH --cpus-per-task=8           # number of cpu per task
#SBATCH --hint=nomultithread         # no hyperthreading 
#SBATCH --time=00:05:00              # time asked (HH:MM:SS)
#SBATCH --output=logs/gpu_multi_mpi%j.out # name of the output file
#SBATCH --error=logs/gpu_multi_mpi%j.out  # name of the logs file
 
# clean modules
module purge
 
# Jean Zay specific module for A100 GPUs
module load cpuarch/amd
 
# Load conda environment
source /CONDADIR/miniconda3/etc/profile.d/conda.sh
conda activate ADP

set -x

# Execute code
srun python -u main.py \
                       --n_epoch_if_1_worker 300 \
                       --batch_size 128 \
                       --filter_bias_and_bn \
                       --lr 0.1 \
                       --dataset_name CIFAR10 \
                       --model_name resnet18 \
                       --use_linear_scaling \
                       --rate_com 1 \
                       --graph_topology exponential \
                       --apply_acid \
                       --weight_decay 0.0005 \
                       --normalize_grads
                       
#--deterministic_coms --deterministic_neighbor \
