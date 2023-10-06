#!/bin/bash
#SBATCH --job-name=distributed     # nom du job
# Il est possible d'utiliser une autre partition que celle par défaut
# en activant l'une des 5 directives suivantes :
#SBATCH -A wjm@a100
##SBATCH -A wjm@v100
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go), 8 gpus par noeuds
##SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go), 8 gpus par noeuds, 3 noeuds
#SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
##SBATCH -–qos=qos_gpu-t4            # for jobs lasting more than 20h, only works on v100
##SBATCH --qos=qos_gpu-dev
# Ici, reservation de 3x10=30 CPU (pour 3 taches) et de 3 GPU (1 GPU par tache) sur un seul noeud :
#SBATCH --nodes=4                    # nombre de noeud
#SBATCH --ntasks-per-node=8          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:8                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 ou 1/8 des GPU du noeud suivant
# la partition), l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour chaque tache:
##SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 du noeud ici)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=6           # nombre de CPU par tache pour gpu_p4 (1/8 du noeud 8-GPU)
#SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:05:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=logs/gpu_multi_mpi%j.out # nom du fichier de sortie
#SBATCH --error=logs/gpu_multi_mpi%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
module load cpuarch/amd
 
# Chargement des modules
source /gpfswork/rech/ffy/utl44lf/miniconda3/etc/profile.d/conda.sh
conda activate Distributed_v3
#module load pytorch-gpu/py3/2.0.0
 
# Echo des commandes lancees
set -x

# Code execution
srun python -u main.py \
                       --n_epoch_if_1_worker 300 \
                       --batch_size 128 \
                       --filter_bias_and_bn \
                       --lr 0.1 \
                       --dataset_name CIFAR10 \
                       --model_name resnet18 \
                       --use_linear_scaling \
                       --rate_com 1 \
                       --acid_params 0.07 3.66 \
                       --graph_topology cycle \
                       --weight_decay 0.0005 \
                       --non_iid_data \
                       --apply_acid \
                       --normalize_grads
                       
#--apply_acid --normalize_grads
